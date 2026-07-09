"""After v3 weights land: fit v3 calibration locally, score attack evals, report
hardening delta. Inference-only (MPS-safe). Run once siglip_tuned_v3 is present.

Usage: .venv/bin/python eval/finish_hardening.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

REPO = Path(__file__).resolve().parents[1]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
V3 = str(REPO / "judges/siglip_tuned_out/siglip_tuned_v3")
SPLITS = json.loads((REPO / "data/features/splits_v2.json").read_text())["_image_index"]


def corp(iid):
    src, rel = iid.split(":", 1)
    return REPO / "data/scrape/raw" / ("images" if src == "fb" else "ig_images") / rel


def embed(model, proc, paths):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 16):
            b = paths[i:i + 16]
            inp = proc(images=[Image.open(p).convert("RGB") for p in b],
                       return_tensors="pt").to(DEV)
            z = model.get_image_features(**inp)
            if hasattr(z, "pooler_output"):
                z = z.pooler_output
            z = (z / z.norm(dim=-1, keepdim=True)).cpu().float().numpy()
            out.append(z)
    return np.concatenate(out)


def main():
    proc = AutoImageProcessor.from_pretrained(V3)
    model = AutoModel.from_pretrained(V3).to(DEV).eval()

    def brand(i): return i.split(":")[1].split("/")[0]
    rhode_tr = [str(corp(i)) for i, sp in SPLITS.items()
                if sp == "train" and brand(i) == "rhode" and corp(i).exists()]
    centroid = embed(model, proc, rhode_tr).mean(0)
    centroid /= np.linalg.norm(centroid)
    val = [(str(corp(i)), 1.0 if brand(i) == "rhode" else 0.0)
           for i, sp in SPLITS.items() if sp == "val" and corp(i).exists()]
    ve = embed(model, proc, [p for p, _ in val])
    sims = ve @ centroid
    y = np.array([l for _, l in val])
    a, b = 1.0, 0.0
    for _ in range(200):
        p = 1 / (1 + np.exp(-(a * sims + b)))
        g = np.array([((p - y) * sims).mean(), (p - y).mean()])
        w = p * (1 - p)
        H = np.array([[(w * sims * sims).mean(), (w * sims).mean()],
                      [(w * sims).mean(), w.mean()]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g); a, b = a - step[0], b - step[1]
    (REPO / "judges/j3_tuned_v3_params.json").write_text(
        json.dumps({"centroid": centroid.tolist(), "platt_a": float(a), "platt_b": float(b)}))
    print(f"v3 calibrated: platt_a={a:.2f} platt_b={b:.2f}", flush=True)

    def score(cal, paths):
        c = np.asarray(cal["centroid"])
        e = embed(model, proc, paths)
        return float((1 / (1 + np.exp(-(cal["platt_a"] * (e @ c) + cal["platt_b"])))).mean())

    v3cal = json.loads((REPO / "judges/j3_tuned_v3_params.json").read_text())
    v1cal = json.loads((REPO / "judges/j3_tuned_params.json").read_text())
    prior = json.loads((REPO / "eval/srpo/siglip_scores.json").read_text())

    # the hardening test: same SRPO-hacked images, v1 vs v3
    hacked_brand = sorted(str(p) for p in
                          (REPO / "eval/srpo/images/tuned/brand").glob("*.jpg"))
    hacked_ctrl = sorted(str(p) for p in
                         (REPO / "eval/srpo/images/tuned/control").glob("*.jpg"))
    clean_brand = sorted(str(p) for p in
                         (REPO / "eval/srpo/images/base/brand").glob("*.jpg"))
    v1_hack = float(np.mean([v for k, v in prior["siglip_tuned"].items()
                             if k.startswith("tuned/brand/")]))
    res = {
        "v1_score_on_SRPO_hacked_brand": round(v1_hack, 3),
        "v3_score_on_SRPO_hacked_brand": round(score(v3cal, hacked_brand), 3),
        "v3_score_on_SRPO_hacked_control": round(score(v3cal, hacked_ctrl), 3),
        "v3_score_on_clean_brand": round(score(v3cal, clean_brand), 3),
    }
    res["hardening_delta_on_seen_hacks"] = round(
        res["v1_score_on_SRPO_hacked_brand"] - res["v3_score_on_SRPO_hacked_brand"], 3)
    res["note"] = ("v3 trained with SRPO hacks as negatives. Large positive "
                   "hardening_delta = v3 now rejects the hacks v1 accepted, while "
                   "v3_score_on_clean_brand should stay high (didn't break real "
                   "brand recognition).")
    (REPO / "eval/results/hardening.json").write_text(json.dumps(res, indent=1))
    print("HARDENING_DONE", json.dumps(res))


if __name__ == "__main__":
    main()
