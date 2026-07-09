"""Pod-side: score the SRPO paired eval set with the Qwen-LoRA judge (J3b).

Uses the judge's CANONICAL path (HF processor + PIL, PeftModel, P(' yes') —
exactly judges/pod_judges.py::qwen_lora_score), NOT the differentiable
training adapter, so eval scores are the judge as originally certified.

Scans /workspace/srpo_eval/{base,tuned}/{brand,control}/*.jpg, writes
/workspace/srpo_eval/qwen_scores.jsonl with item_id = "base/brand/p00_c00.jpg".
Resumable. Usage: python qwen_score_eval.py
"""
import json
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/workspace/srpo_eval")
OUT = ROOT / "qwen_scores.jsonl"
LORA_PROMPT = ("You are a brand-consistency evaluator. Is this image consistent "
               "with the visual identity of rhode, the beauty brand? "
               "Answer with exactly one word: yes or no.")


def main():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    mid = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mid, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(
        model, "/workspace/SRPO/data/qwen_lora_adapter")
    model.eval()
    proc = AutoProcessor.from_pretrained(mid, max_pixels=768 * 768)
    tok = proc.tokenizer
    yes_id = tok.encode(" yes", add_special_tokens=False)[0]
    no_id = tok.encode(" no", add_special_tokens=False)[0]

    items = sorted(ROOT.glob("*/*/*.jpg"))
    done = {json.loads(l)["item_id"] for l in OUT.open()} if OUT.exists() else set()
    outf = OUT.open("a")
    for n, p in enumerate(items):
        iid = str(p.relative_to(ROOT))
        if iid in done:
            continue
        img = Image.open(p).convert("RGB")
        if max(img.size) > 1024:  # match pod_judges.load_img
            s = 1024 / max(img.size)
            img = img.resize((round(img.width * s), round(img.height * s)),
                             Image.LANCZOS)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": LORA_PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
        enc = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits[0, -1]
        pair = torch.softmax(logits[[yes_id, no_id]].float(), dim=0)
        outf.write(json.dumps({"item_id": iid,
                               "p_yes": round(pair[0].item(), 6)}) + "\n")
        outf.flush()
        if n % 50 == 0:
            print(f"qwen_score {n}/{len(items)}", flush=True)
    print("QWEN_SCORE_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
