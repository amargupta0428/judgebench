"""Score the BoN candidate pool with the local judges, on-pod (Phase 2a).

Runs after pod_bon.py. Scores:
  siglip frozen / tuned v1 / tuned v2  -> embeddings npz (full pool; centroid+
                                          Platt applied locally on the Mac from
                                          committed params — nothing fit here)
  qwen LoRA p_yes                      -> jsonl (full pool)
  qwen zero-shot rubric                -> jsonl (broad arm only, candidate < 64)

Expects /workspace/{bon_out/, judge_assets/{siglip_tuned,siglip_tuned_v2,
qwen_lora,refs/ref_*.jpg,rubric.txt}}.
Usage: python pod_bon_score.py siglip|qwen_lora|qwen_zs
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BON = Path("/workspace/bon_out")
ASSETS = Path("/workspace/judge_assets")
OUT = Path("/workspace/bon_scores")
OUT.mkdir(exist_ok=True)


def pool(broad_only=False):
    items = [json.loads(l) for l in (BON / "manifest_bon.jsonl").open()]
    seen = set()
    out = []
    for it in items:
        if it["file"] in seen:
            continue
        seen.add(it["file"])
        if broad_only and it["candidate"] >= 64:
            continue
        out.append(it)
    return out


def load_img(p, max_side=1024):
    im = Image.open(p).convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return im


def siglip():
    from transformers import AutoImageProcessor, AutoModel
    items = pool()
    for tag, src in (("frozen", "google/siglip-so400m-patch14-384"),
                     ("tuned", str(ASSETS / "siglip_tuned")),
                     ("tuned_v2", str(ASSETS / "siglip_tuned_v2"))):
        proc = AutoImageProcessor.from_pretrained(src)
        model = AutoModel.from_pretrained(src).to("cuda").eval()
        vecs, ids = [], []
        with torch.no_grad():
            for j in range(0, len(items), 64):
                b = items[j:j + 64]
                inp = proc(images=[load_img(BON / x["file"]) for x in b],
                           return_tensors="pt").to("cuda")
                z = model.get_image_features(**inp)
                z = (z / z.norm(dim=-1, keepdim=True)).cpu().float().numpy()
                vecs.append(z)
                ids += [x["file"] for x in b]
                if (j // 64) % 20 == 0:
                    print(f"siglip_{tag} {j}/{len(items)}", flush=True)
        np.savez_compressed(OUT / f"bon_emb_{tag}.npz",
                            files=np.array(ids), vecs=np.concatenate(vecs))
        del model
        torch.cuda.empty_cache()
    print("SIGLIP_BON_DONE", flush=True)


def _qwen(adapter=None):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    mid = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mid, torch_dtype=torch.bfloat16, device_map="cuda")
    proc = AutoProcessor.from_pretrained(mid, max_pixels=768 * 768)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), proc


LORA_PROMPT = ("You are a brand-consistency evaluator. Is this image consistent "
               "with the visual identity of rhode, the beauty brand? "
               "Answer with exactly one word: yes or no.")


def qwen_lora():
    from qwen_vl_utils import process_vision_info
    model, proc = _qwen(str(ASSETS / "qwen_lora"))
    tok = proc.tokenizer
    yes_id = tok.encode(" yes", add_special_tokens=False)[0]
    no_id = tok.encode(" no", add_special_tokens=False)[0]
    items = pool()
    outp = OUT / "bon_qwen_lora.jsonl"
    done = {json.loads(l)["file"] for l in outp.open()} if outp.exists() else set()
    outf = outp.open("a")
    for n, it in enumerate(items):
        if it["file"] in done:
            continue
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": str(BON / it["file"])},
            {"type": "text", "text": LORA_PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inp).logits[0, -1]
        pair = torch.softmax(logits[[yes_id, no_id]].float(), dim=0)
        outf.write(json.dumps({"file": it["file"], "p_yes": round(pair[0].item(), 6)}) + "\n")
        outf.flush()
        if n % 200 == 0:
            print(f"qwen_lora {n}/{len(items)}", flush=True)
    print("QWEN_LORA_BON_DONE", flush=True)


def qwen_zs():
    from qwen_vl_utils import process_vision_info
    model, proc = _qwen()
    rubric = (ASSETS / "rubric.txt").read_text()
    refs = sorted((ASSETS / "refs").glob("ref_*.jpg"))
    items = pool(broad_only=True)
    outp = OUT / "bon_qwen_zs.jsonl"
    done = {json.loads(l)["file"] for l in outp.open()} if outp.exists() else set()
    outf = outp.open("a")
    for n, it in enumerate(items):
        if it["file"] in done:
            continue
        content = ([{"type": "text", "text": rubric}] +
                   [{"type": "image", "image": str(r)} for r in refs] +
                   [{"type": "image", "image": str(BON / it["file"])}])
        msgs = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=250, do_sample=False)
        resp = proc.batch_decode(gen[:, inp.input_ids.shape[1]:],
                                 skip_special_tokens=True)[0]
        m = re.search(r"\{.*\}", resp, re.S)
        rec = {"file": it["file"]}
        if m:
            try:
                rec.update(json.loads(m.group(0)))
            except Exception:
                rec["error"] = "parse"
        else:
            rec["error"] = "nojson"
        outf.write(json.dumps(rec) + "\n")
        outf.flush()
        if n % 100 == 0:
            print(f"qwen_zs {n}/{len(items)}", flush=True)
    print("QWEN_ZS_BON_DONE", flush=True)


if __name__ == "__main__":
    {"siglip": siglip, "qwen_lora": qwen_lora, "qwen_zs": qwen_zs}[sys.argv[1]]()
