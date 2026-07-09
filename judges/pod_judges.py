"""Pod-side jobs: SigLIP contrastive fine-tunes + QwenVL judges (zero-shot & LoRA).

Design notes (rigor):
- SigLIP-tuned v1 trains ONLY on corpus train split (rhode=pos, glossier/ilia=neg,
  SupCon loss). Corruptions are NOT in training (they were built from test bases)
  -> all violation detection is out-of-training generalization. Documented.
- SigLIP-tuned v2 (July 7, restores PHASE1_BUILD §3 intent): adds corruption
  negatives generated from TRAIN-split bases as a third SupCon class.
  Trained-on families: palette, typography. Held out: composition + all
  generative families -> report card measures generalization to unseen
  violation types. Everything else identical to v1 (matched comparison).
- QwenVL zero-shot: same rubric text as API judges, 6-ref board, greedy decode.
- QwenVL LoRA (J3b): same model, LoRA on LM attention projections (vision tower
  frozen), binary brand-consistency supervision on the SAME train split as
  SigLIP-tuned v1 (rhode=yes, glossier/ilia=no; no corruption negatives) so the
  matched pair isolates what domain fine-tuning buys. No reference board at
  eval: the tuned model internalizes the brand. Score = P("yes") at first token.

Usage: python pod_judges.py siglip | qwen | siglip_v2 | qwen_lora | qwen_lora_score
Expects /workspace/job/{train_manifest*.json, images/..., testset/..., refs/...}
"""
import json, sys, io, re
from pathlib import Path
import torch
import numpy as np
from PIL import Image

JOB = Path("/workspace/job")
OUT = Path("/workspace/out"); OUT.mkdir(exist_ok=True)


def load_img(p, max_side=1024):
    im = Image.open(p).convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.width*s), round(im.height*s)), Image.LANCZOS)
    return im


def siglip(manifest="train_manifest.json", suffix=""):
    from transformers import AutoImageProcessor, AutoModel
    dev = "cuda"
    proc = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(dev)
    train = json.loads((JOB/manifest).read_text())  # [{path,label}]
    opt = torch.optim.AdamW(model.vision_model.parameters(), lr=1e-5)
    model.train()
    import random
    random.seed(0)
    EPOCHS, BS, TAU = 2, 24, 0.07
    for ep in range(EPOCHS):
        random.shuffle(train)
        for i in range(0, len(train), BS):
            batch = train[i:i+BS]
            if len(batch) < 4: continue
            imgs = [load_img(JOB/b["path"]) for b in batch]
            y = torch.tensor([b["label"] for b in batch], device=dev)
            inp = proc(images=imgs, return_tensors="pt").to(dev)
            z = model.get_image_features(**inp)
            z = z / z.norm(dim=-1, keepdim=True)
            sim = z @ z.T / TAU
            mask = (y[:, None] == y[None, :]).float()
            mask.fill_diagonal_(0)
            logmax = sim - sim.max(dim=1, keepdim=True).values.detach()
            exp = torch.exp(logmax)
            exp = exp * (1 - torch.eye(len(y), device=dev))
            logprob = logmax - torch.log(exp.sum(1, keepdim=True) + 1e-9)
            denom = mask.sum(1).clamp(min=1)
            loss = -(mask * logprob).sum(1) / denom
            loss = loss.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if (i//BS) % 20 == 0:
                print(f"ep{ep} {i}/{len(train)} loss {loss.item():.4f}", flush=True)
    model.save_pretrained(OUT/f"siglip_tuned{suffix}")
    proc.save_pretrained(OUT/f"siglip_tuned{suffix}")
    # embed corpus (train+val for fitting) + testset
    model.eval()
    for tag in ("corpus", "testset"):
        items = json.loads((JOB/f"embed_{tag}.json").read_text())  # [{id,path}]
        vecs, ids = [], []
        with torch.no_grad():
            for j in range(0, len(items), 32):
                b = items[j:j+32]
                inp = proc(images=[load_img(JOB/x["path"]) for x in b],
                           return_tensors="pt").to(dev)
                z = model.get_image_features(**inp)
                z = (z / z.norm(dim=-1, keepdim=True)).cpu().float().numpy()
                vecs.append(z); ids += [x["id"] for x in b]
                if (j//32) % 20 == 0: print(f"embed {tag} {j}/{len(items)}", flush=True)
        np.savez_compressed(OUT/f"embeddings_{tag}_tuned{suffix}.npz",
                            files=np.array(ids), vecs=np.concatenate(vecs))
    print("SIGLIP_DONE", flush=True)


RUBRIC = (JOB/"rubric.txt").read_text() if (JOB/"rubric.txt").exists() else ""


def qwen():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    dev = "cuda"
    mid = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mid, torch_dtype=torch.bfloat16, device_map=dev)
    proc = AutoProcessor.from_pretrained(mid)
    refs = sorted((JOB/"refs").glob("ref_*.jpg"))
    items = json.loads((JOB/"embed_testset.json").read_text())
    outf = (OUT/"qwen_zs_scores.jsonl").open("a")
    done = set()
    if (OUT/"qwen_zs_scores.jsonl").exists():
        done = {json.loads(l)["id"] for l in (OUT/"qwen_zs_scores.jsonl").open()}
    for n, it in enumerate(items):
        if it["id"] in done: continue
        content = ([{"type": "text", "text": RUBRIC}] +
                   [{"type": "image", "image": str(r)} for r in refs] +
                   [{"type": "image", "image": str(JOB/it["path"])}])
        msgs = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inp = proc(text=[text], images=imgs, videos=vids,
                   return_tensors="pt").to(dev)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=250, do_sample=False)
        resp = proc.batch_decode(gen[:, inp.input_ids.shape[1]:],
                                 skip_special_tokens=True)[0]
        m = re.search(r"\{.*\}", resp, re.S)
        rec = {"id": it["id"], "raw": resp[:400]}
        if m:
            try: rec.update(json.loads(m.group(0)))
            except Exception: rec["error"] = "parse"
        else: rec["error"] = "nojson"
        outf.write(json.dumps(rec) + "\n"); outf.flush()
        if n % 50 == 0: print(f"qwen {n}/{len(items)}", flush=True)
    print("QWEN_DONE", flush=True)


LORA_PROMPT = ("You are a brand-consistency evaluator. Is this image consistent "
               "with the visual identity of rhode, the beauty brand? "
               "Answer with exactly one word: yes or no.")


def _qwen_base(max_pixels=768*768):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    mid = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mid, torch_dtype=torch.bfloat16, device_map="cuda")
    proc = AutoProcessor.from_pretrained(mid, max_pixels=max_pixels)
    return model, proc


def _lora_inputs(proc, image_path, answer=None):
    """Tokenized chat round; labels mask everything except the answer tokens."""
    from qwen_vl_utils import process_vision_info
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": LORA_PROMPT}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs)
    if answer is None:
        return proc(text=[prompt], images=imgs, return_tensors="pt")
    full = proc(text=[prompt + answer], images=imgs, return_tensors="pt")
    n_prompt = proc(text=[prompt], images=imgs, return_tensors="pt").input_ids.shape[1]
    labels = full.input_ids.clone()
    labels[:, :n_prompt] = -100
    full["labels"] = labels
    return full


def qwen_lora():
    import random
    from peft import LoraConfig, get_peft_model
    model, proc = _qwen_base()
    # q/k/v/o_proj exist only in the LM layers (vision blocks use fused 'qkv'),
    # so this targets language attention and leaves the vision tower frozen
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    train = json.loads((JOB/"qwen_lora_manifest.json").read_text())  # [{path,answer}]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    random.seed(0)
    EPOCHS, ACCUM = 2, 8
    step = 0
    model.train()
    for ep in range(EPOCHS):
        random.shuffle(train)
        for i, ex in enumerate(train):
            inp = _lora_inputs(proc, JOB/ex["path"], " " + ex["answer"]).to("cuda")
            loss = model(**inp).loss / ACCUM
            loss.backward()
            if (i + 1) % ACCUM == 0:
                opt.step(); opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"ep{ep} {i+1}/{len(train)} loss {loss.item()*ACCUM:.4f}",
                          flush=True)
    model.save_pretrained(OUT/"qwen_lora")
    print("QWEN_LORA_DONE", flush=True)


def qwen_lora_score():
    from peft import PeftModel
    model, proc = _qwen_base()
    model = PeftModel.from_pretrained(model, str(OUT/"qwen_lora"))
    model.eval()
    tok = proc.tokenizer
    yes_id = tok.encode(" yes", add_special_tokens=False)[0]
    no_id = tok.encode(" no", add_special_tokens=False)[0]
    items = json.loads((JOB/"embed_testset.json").read_text())
    outp = OUT/"qwen_lora_scores.jsonl"
    done = {json.loads(l)["id"] for l in outp.open()} if outp.exists() else set()
    outf = outp.open("a")
    for n, it in enumerate(items):
        if it["id"] in done: continue
        inp = _lora_inputs(proc, JOB/it["path"]).to("cuda")
        with torch.no_grad():
            logits = model(**inp).logits[0, -1]
        pair = torch.softmax(logits[[yes_id, no_id]].float(), dim=0)
        outf.write(json.dumps({"id": it["id"], "p_yes": round(pair[0].item(), 6)}) + "\n")
        outf.flush()
        if n % 100 == 0: print(f"lora_score {n}/{len(items)}", flush=True)
    print("QWEN_LORA_SCORE_DONE", flush=True)


if __name__ == "__main__":
    {"siglip": siglip, "qwen": qwen,
     "siglip_v2": lambda: siglip("train_manifest_v2.json", "_v2"),
     "qwen_lora": qwen_lora,
     "qwen_lora_score": qwen_lora_score}[sys.argv[1]]()
