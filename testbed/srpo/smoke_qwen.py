"""Pod-side smoke test for QwenBrand (run on 1 GPU before any paid training).

Checks, in order:
1. SCORE PARITY — QwenBrand's differentiable tensor path vs the judge's
   canonical path (HF processor + PIL, exactly judges/pod_judges.py::
   qwen_lora_score) on real FLUX base images in /workspace/smoke_imgs.
   Tolerance is loose-ish (resize kernels differ) but must be clearly the
   same judge: max |delta| < 0.05.
2. GRADIENT FLOW — d P(yes) / d pixels is nonzero and finite.
3. BASE REWARD LEVEL — mean P(yes) on the base images, to sanity-check the
   ReFL threshold 0.7 (expected ~0.4 from eval/bon/bon_qwen_lora.jsonl).

Usage: python smoke_qwen.py  (from /workspace/SRPO/data or anywhere; paths abs)
"""
import sys
from pathlib import Path

import torch
from PIL import Image
import numpy as np

sys.path.insert(0, "/workspace/SRPO/data")
from qwen_reward import QwenBrand, LORA_PROMPT

IMGS = sorted(Path("/workspace/smoke_imgs").glob("*.jpg"))
assert IMGS, "put a few FLUX base images in /workspace/smoke_imgs"


def canonical_scores(model, paths):
    """The judge as originally defined: processor+PIL path, P(' yes')."""
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", max_pixels=768 * 768)
    tok = proc.tokenizer
    yes_id = tok.encode(" yes", add_special_tokens=False)[0]
    no_id = tok.encode(" no", add_special_tokens=False)[0]
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": LORA_PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
        enc = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model.model(**enc, use_cache=False).logits[0, -1]
        pair = torch.softmax(logits[[yes_id, no_id]].float(), dim=0)
        out.append(pair[0].item())
    return out


def main():
    rb = QwenBrand(adapter_path="/workspace/SRPO/data/qwen_lora_adapter",
                   device="cuda")

    # 1a. PARITY at matched resolution: pre-resize to SIZE x SIZE so both
    # paths see the identical grid (canonical smart_resize keeps 728 as-is;
    # 728 is also exactly what smart_resize picks for SRPO's 720px training
    # decodes). Isolates normalize+patchify+forward. Must be tight.
    from qwen_reward import SIZE
    resized = []
    for p in IMGS:
        img = Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
        rp = Path("/tmp") / f"rs_{p.name}"
        img.save(rp, "JPEG", quality=98)
        resized.append(rp)
    canon = canonical_scores(rb, resized)
    mine = []
    for rp in resized:
        arr = np.asarray(Image.open(rp).convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).to("cuda")
        with torch.no_grad():
            mine.append(rb.score_tensor(x)[0].item())
    print(f"{'image (728px matched)':28}{'canonical':>10}{'tensor':>10}{'delta':>9}")
    worst = 0.0
    for p, c, m in zip(IMGS, canon, mine):
        print(f"{p.name:28}{c:10.4f}{m:10.4f}{m - c:+9.4f}")
        worst = max(worst, abs(m - c))
    print(f"max |delta| = {worst:.4f}")
    assert worst < 0.02, "PARITY FAIL — tensor path is not the same judge"

    # 1b. Scale sensitivity (reported, not asserted): canonical on the raw
    # 1024px files (smart_resize -> 756) vs tensor path (728). Documents how
    # much the judge moves with input scale; eval scoring stays canonical.
    canon_native = canonical_scores(rb, IMGS)
    mine_native = []
    for p in IMGS:
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).to("cuda")
        with torch.no_grad():
            mine_native.append(rb.score_tensor(x)[0].item())
    ds = [abs(a - b) for a, b in zip(canon_native, mine_native)]
    print(f"scale sensitivity (1024->756 canon vs ->728 tensor): "
          f"max |delta| = {max(ds):.4f}, mean = {np.mean(ds):.4f}")

    # 2. gradient flow
    arr = np.asarray(Image.open(IMGS[0]).convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)[None].to("cuda").requires_grad_(True)
    s = rb.score_tensor(x)[0]
    s.backward()
    g = x.grad
    gnorm = g.norm().item()
    print(f"score={s.item():.4f} grad_norm={gnorm:.3e} "
          f"finite={bool(torch.isfinite(g).all())} nonzero_frac="
          f"{(g != 0).float().mean().item():.3f}")
    assert gnorm > 0 and torch.isfinite(g).all(), "GRADIENT FAIL"

    # 3. base reward level
    print(f"base mean P(yes) = {np.mean(mine):.3f} (ReFL threshold 0.7)")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU mem used: {(total - free) / 2**30:.1f} GiB")
    print("SMOKE_QWEN_OK")


if __name__ == "__main__":
    main()
