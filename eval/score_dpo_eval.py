"""Score the DPO paired eval set (base + 3 checkpoints) with SigLIP judges.

Same calibrations as everywhere else. Reports the pressure-escalation curve:
per judge, mean brand + control score at base -> ckpt250 -> ckpt500 -> ckpt750.
The attacked judge (siglip_tuned) should climb; independents flat/down = hack.
Writes eval/dpo/siglip_scores.json (leaderboard reads it).

Usage: .venv/bin/python eval/score_dpo_eval.py
For the GPT-4o-attacked DPO arm (same layout, different images):
  DPO_INDEX=eval/dpo_gpt4o/dpo_gpt4o_eval_index.jsonl \
  DPO_SCORES_OUT=eval/dpo_gpt4o/siglip_scores.json \
  .venv/bin/python eval/score_dpo_eval.py
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

REPO = Path(__file__).resolve().parents[1]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MODELS = {
    "siglip_frozen": "google/siglip-so400m-patch14-384",
    "siglip_tuned": str(REPO / "judges/siglip_tuned_out/siglip_tuned"),
    "siglip_tuned_v2": str(REPO / "judges/siglip_tuned_out/siglip_tuned_v2"),
    "siglip_tuned_v3": str(REPO / "judges/siglip_tuned_out/siglip_tuned_v3"),
}
PARAMS = {"siglip_frozen": "j3_frozen_params.json",
          "siglip_tuned": "j3_tuned_params.json",
          "siglip_tuned_v2": "j3_tuned_v2_params.json",
          "siglip_tuned_v3": "j3_tuned_v3_params.json"}
STAGES = ["base", "checkpoint-250", "checkpoint-500", "checkpoint-750"]


INDEX = REPO / os.environ.get("DPO_INDEX", "eval/dpo/dpo_eval_index.jsonl")
SCORES_OUT = REPO / os.environ.get("DPO_SCORES_OUT", "eval/dpo/siglip_scores.json")


def main():
    items = [json.loads(l) for l in INDEX.open()]
    scores = {}
    for name, src in MODELS.items():
        proc = AutoImageProcessor.from_pretrained(src)
        model = AutoModel.from_pretrained(src).to(DEV).eval()
        cal = json.loads((REPO / "judges" / PARAMS[name]).read_text())
        c = np.asarray(cal["centroid"])
        s = {}
        with torch.no_grad():
            for j in range(0, len(items), 16):
                b = items[j:j + 16]
                inp = proc(images=[Image.open(x["path"]).convert("RGB") for x in b],
                           return_tensors="pt").to(DEV)
                z = model.get_image_features(**inp)
                if hasattr(z, "pooler_output"):
                    z = z.pooler_output
                z = (z / z.norm(dim=-1, keepdim=True)).cpu().float().numpy()
                for x, v in zip(b, z):
                    s[x["item_id"]] = float(
                        1 / (1 + np.exp(-(cal["platt_a"] * (v @ c) + cal["platt_b"]))))
                if (j // 16) % 15 == 0:
                    print(f"{name} {j}/{len(items)}", flush=True)
        scores[name] = s
        del model

    SCORES_OUT.write_text(json.dumps(scores))
    print(f"\n{'judge':17}{'set':8}" + "".join(f"{st.replace('checkpoint-','ck'):>10}" for st in STAGES))
    for name, s in scores.items():
        for kind in ("brand", "control"):
            row = [np.mean([v for k, v in s.items() if k.startswith(f"{st}/{kind}/")])
                   for st in STAGES]
            print(f"{name:17}{kind:8}" + "".join(f"{r:10.3f}" for r in row))


if __name__ == "__main__":
    main()
