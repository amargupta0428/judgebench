"""QwenVL-7B LoRA brand-fidelity reward for SRPO (Phase 2b — gradient arm #2).

Drop-in reward for Tencent-Hunyuan/SRPO mirroring siglip_reward.SigLIPBrand:
exposes SRP_cfg(prompt, neg_prompt, image_inputs, k) returning a per-image
score in (0,1) that is differentiable w.r.t. image_inputs. The judge under
attack is judgebench J3b: Qwen2.5-VL-7B-Instruct + the brand LoRA adapter,
score = P(" yes") over the {" yes"," no"} first-answer-token pair for
LORA_PROMPT (same definition as judges/pod_judges.py::qwen_lora_score).

Differentiability: the HF processor's image path (PIL resize + numpy patchify)
breaks the graph, so preprocessing is reimplemented in torch —
F.interpolate(bicubic, antialias) -> CLIP-mean/std normalize -> the exact
Qwen2VLImageProcessor patchify (reshape/permute), verified against the
processor numerically in smoke_qwen.py. Text tokens (chat template around
LORA_PROMPT with a fixed image-token count for SIZE x SIZE input) are
precomputed once; only pixel_values changes per call.

As with SigLIPBrand: image-only reward, so prompt/neg_prompt/k are accepted
and ignored and the trainer runs forward-only (Direct-Align core, plain ReFL
reward maximization, threshold 0.7 — base FLUX brand images score mean 0.42
under this judge per eval/bon/bon_qwen_lora.jsonl, so 0.7 leaves signal).

Memory: LoRA merged into the base model (exact for a frozen judge); bf16;
non-reentrant gradient checkpointing so activations of the 7B forward are
recomputed — grads still flow to pixel_values even with all params frozen.
"""
import torch
import torch.nn.functional as F

LORA_PROMPT = ("You are a brand-consistency evaluator. Is this image consistent "
               "with the visual identity of rhode, the beauty brand? "
               "Answer with exactly one word: yes or no.")

# Qwen2.5-VL processor constants (OPENAI_CLIP mean/std; patch geometry)
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
PATCH, MERGE, T_PATCH = 14, 2, 2
SIZE = 728  # multiple of PATCH*MERGE=28; matches judge's max_pixels=768*768 regime


class QwenBrand(torch.nn.Module):
    def __init__(self, base="Qwen/Qwen2.5-VL-7B-Instruct",
                 adapter_path="./data/qwen_lora_adapter",
                 device="cuda", size=SIZE, grad_ckpt=True,
                 merged_path="/workspace/qwen_merged"):
        super().__init__()
        import os
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.device, self.size = device, size
        if merged_path and os.path.isdir(merged_path):
            # pre-merged offline (see run notes): peft 0.19.1's TP-aware
            # loader breaks under torchrun with transformers 4.57.1, so the
            # trainer must not touch peft at all
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                merged_path, torch_dtype=torch.bfloat16)
        else:
            from peft import PeftModel
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base, torch_dtype=torch.bfloat16)
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()  # exact: judge is frozen
        for p in model.parameters():
            p.requires_grad_(False)
        if grad_ckpt:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()  # HF ckpt is training-gated; Qwen2.5-VL has 0 dropout
        else:
            model.eval()
        self.model = model.to(device)
        self.model.config.use_cache = False

        proc = AutoProcessor.from_pretrained(base, max_pixels=768 * 768)
        tok = proc.tokenizer
        self.yes_id = tok.encode(" yes", add_special_tokens=False)[0]
        self.no_id = tok.encode(" no", add_special_tokens=False)[0]
        # Precompute the chat-template token stream for a size x size image.
        # Placeholder count depends only on the grid, so a dummy image works.
        from PIL import Image
        import numpy as np
        dummy = Image.fromarray(
            np.zeros((size, size, 3), dtype=np.uint8))
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": dummy},
            {"type": "text", "text": LORA_PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
        enc = proc(text=[text], images=[dummy], return_tensors="pt")
        self.register_buffer("input_ids", enc["input_ids"].to(device),
                             persistent=False)
        self.register_buffer("attn_mask", enc["attention_mask"].to(device),
                             persistent=False)
        self.register_buffer("grid_thw", enc["image_grid_thw"].to(device),
                             persistent=False)
        assert enc["image_grid_thw"].tolist() == [[1, size // PATCH, size // PATCH]], \
            f"processor resized dummy: {enc['image_grid_thw'].tolist()}"
        mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def _patchify(self, x):
        """(3, H, W) normalized -> (grid_h*grid_w, 3*T_PATCH*PATCH*PATCH),
        the exact Qwen2VLImageProcessor layout, in differentiable torch ops."""
        _, H, W = x.shape
        gh, gw = H // PATCH, W // PATCH
        x = torch.stack([x, x])  # temporal tiling of a single frame
        x = x.reshape(1, T_PATCH, 3, gh // MERGE, MERGE, PATCH,
                      gw // MERGE, MERGE, PATCH)
        x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return x.reshape(gh * gw, 3 * T_PATCH * PATCH * PATCH)

    def score_tensor(self, image):
        """image: (B,3,H,W) or (3,H,W) in [0,1], differentiable. -> (B,) P(yes).

        Wrapped in autocast(enabled=False): SRPO's trainer calls the reward
        under torch.amp.autocast('cuda') (fp16), but the judge is defined in
        pure bf16 — keep the attacked reward numerically identical to the
        canonical judge rather than an fp16 re-cast of it.
        """
        with torch.autocast("cuda", enabled=False):
            return self._score(image)

    def _score(self, image):
        x = image if image.dim() == 4 else image[None]
        x = x.float()
        if x.shape[-2:] != (self.size, self.size):
            x = F.interpolate(x, size=(self.size, self.size), mode="bicubic",
                              antialias=True, align_corners=False)
        x = ((x.clamp(0, 1) - self.mean) / self.std)
        scores = []
        for i in range(x.shape[0]):
            pv = self._patchify(x[i]).to(torch.bfloat16)
            # logits_to_keep=1: lm_head over the last position only. The full
            # 716x152k logits tensor is a ~218 MiB allocation (+ its grad) that
            # OOM'd the 4xH100 smoke; the judge's score uses only the last
            # position, so this is numerically identical.
            out = self.model(input_ids=self.input_ids,
                             attention_mask=self.attn_mask,
                             pixel_values=pv,
                             image_grid_thw=self.grid_thw,
                             use_cache=False,
                             logits_to_keep=1)
            logits = out.logits[0, -1]
            pair = torch.softmax(
                logits[[self.yes_id, self.no_id]].float(), dim=0)
            scores.append(pair[0])
        return torch.stack(scores)

    def SRP_cfg(self, prompt, neg_prompt, image_inputs, k):
        """Text args + k accepted for interface parity, unused (image-only)."""
        return self.score_tensor(image_inputs)
