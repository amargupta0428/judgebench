"""Score the SRPO paired eval set (tuned vs base) with the SigLIP judges, locally.

Embeds all eval images with frozen / tuned-v1 / tuned-v2 SigLIP (MPS or CPU),
applies the committed centroid+Platt calibrations, and reports per judge:
  brand:   mean score tuned vs base (the attacked judge's view + siblings)
  control: mean score tuned vs base (reward leakage outside brand prompts)
Writes eval/srpo/siglip_scores.json for bon-style analysis downstream.

Usage: .venv/bin/python eval/score_srpo_eval.py
"""
import json
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


def main():
    items = [json.loads(l) for l in (REPO / "eval/srpo/srpo_eval_index.jsonl").open()]
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
                imgs = [Image.open(x["path"]).convert("RGB") for x in b]
                inp = proc(images=imgs, return_tensors="pt").to(DEV)
                z = model.get_image_features(**inp)
                # transformers >=5 returns an output object; the 1152-d
                # pooler_output is the vector the pod's calibration was fit on
                if hasattr(z, "pooler_output"):
                    z = z.pooler_output
                z = (z / z.norm(dim=-1, keepdim=True)).cpu().float().numpy()
                for x, v in zip(b, z):
                    s[x["item_id"]] = float(
                        1 / (1 + np.exp(-(cal["platt_a"] * (v @ c) + cal["platt_b"]))))
                if (j // 16) % 10 == 0:
                    print(f"{name} {j}/{len(items)}", flush=True)
        scores[name] = s
        del model
    (REPO / "eval/srpo/siglip_scores.json").write_text(json.dumps(scores))

    print(f"\n{'judge':18}{'set':9}{'base':>8}{'tuned':>8}{'delta':>8}")
    for name, s in scores.items():
        for kind in ("brand", "control"):
            b = np.mean([v for k, v in s.items() if k.startswith(f"base/{kind}/")])
            t = np.mean([v for k, v in s.items() if k.startswith(f"tuned/{kind}/")])
            print(f"{name:18}{kind:9}{b:8.3f}{t:8.3f}{t - b:+8.3f}")


if __name__ == "__main__":
    main()
