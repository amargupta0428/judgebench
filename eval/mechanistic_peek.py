"""Mechanistic peek (Phase 3, reach #2) — isolate the SRPO hack direction.

Hypothesis: SRPO's gradient attack pushed images along a single dominant
direction in SigLIP-tuned's embedding space that inflates the calibrated score
without adding real brand fidelity. If so, projecting that direction OUT should
collapse the hacked images' scores back toward baseline while barely touching
clean brand images.

Method (all local, MPS):
  1. Embed base vs SRPO-tuned eval images with SigLIP-tuned.
  2. hack_dir = unit(mean(tuned_emb) - mean(base_emb))  [difference of means].
  3. Re-score with hack_dir projected out of both centroid and image embeddings;
     compare score drop on tuned-brand (should fall a lot) vs base-brand (little).
  4. Report cosine(hack_dir, centroid) — how aligned the exploit is with the
     brand prototype itself (high alignment = the hack rides the judge's own axis).
Allowed to fail: if the drop is similar for tuned and base, the hack is not a
single direction and we say so.

Usage: .venv/bin/python eval/mechanistic_peek.py
Output: eval/results/mechanistic_peek.json
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

REPO = Path(__file__).resolve().parents[1]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
SRC = str(REPO / "judges/siglip_tuned_out/siglip_tuned")
CAL = json.loads((REPO / "judges/j3_tuned_params.json").read_text())


def embed(paths):
    proc = AutoImageProcessor.from_pretrained(SRC)
    model = AutoModel.from_pretrained(SRC).to(DEV).eval()
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


def score(emb, centroid):
    sim = emb @ centroid
    return 1 / (1 + np.exp(-(CAL["platt_a"] * sim + CAL["platt_b"])))


def proj_out(vecs, d):
    v = vecs - np.outer(vecs @ d, d)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def main():
    img = REPO / "eval/srpo/images"
    base_b = sorted(str(p) for p in (img / "base/brand").glob("*.jpg"))
    tuned_b = sorted(str(p) for p in (img / "tuned/brand").glob("*.jpg"))
    base_e, tuned_e = embed(base_b), embed(tuned_b)

    hack = tuned_e.mean(0) - base_e.mean(0)
    hack /= np.linalg.norm(hack)
    centroid = np.asarray(CAL["centroid"])
    centroid_u = centroid / np.linalg.norm(centroid)

    s_base = score(base_e, centroid).mean()
    s_tuned = score(tuned_e, centroid).mean()
    # project the hack direction out of images AND centroid
    c_perp = centroid - (centroid @ hack) * hack
    s_base_perp = score(proj_out(base_e, hack), c_perp).mean()
    s_tuned_perp = score(proj_out(tuned_e, hack), c_perp).mean()

    res = {
        "score_base": round(float(s_base), 3),
        "score_tuned": round(float(s_tuned), 3),
        "score_base_projout": round(float(s_base_perp), 3),
        "score_tuned_projout": round(float(s_tuned_perp), 3),
        "tuned_drop": round(float(s_tuned - s_tuned_perp), 3),
        "base_drop": round(float(s_base - s_base_perp), 3),
        "cos_hackdir_centroid": round(float(hack @ centroid_u), 3),
        "verdict": None,
    }
    # single-direction hack if removing it collapses tuned gain but spares base
    res["verdict"] = ("single-direction hack isolated"
                      if res["tuned_drop"] > 2 * max(res["base_drop"], 0.02)
                      and res["tuned_drop"] > 0.1
                      else "hack not reducible to one direction (inconclusive)")
    (REPO / "eval/results/mechanistic_peek.json").write_text(json.dumps(res, indent=1))
    print("MECH_PEEK_DONE", json.dumps(res))


if __name__ == "__main__":
    main()
