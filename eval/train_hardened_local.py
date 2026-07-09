"""Hardening round (Phase 3, reach #1) — SigLIP-tuned-v3, local MPS.

v3 = v1 recipe (rhode positives vs competitor negatives) PLUS the hack
exemplars as an extra negative class: the SRPO-tuned brand+control images and
the top BoN winners under siglip_tuned. Teaches the judge that these
reward-hacking images are NOT on-brand. Then re-scores the SRPO/DPO eval sets
and reports the robustness delta vs v1.

Finding-7 prediction: hardening patches the hacks it SEES (the SRPO family)
but may not generalize. Measured here, not assumed.

All local: corpus in data/scrape/raw, hacks in eval/. No pod, no cost.
Usage: .venv/bin/python eval/train_hardened_local.py
Output: judges/siglip_tuned_out/siglip_tuned_v3 + judges/j3_tuned_v3_params.json
        + eval/results/hardening.json
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

REPO = Path(__file__).resolve().parents[1]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
BASE = str(REPO / "judges/siglip_tuned_out/siglip_tuned")  # warm-start from v1
SPLITS = json.loads((REPO / "data/features/splits_v2.json").read_text())["_image_index"]


def corpus_path(image_id):
    src, rel = image_id.split(":", 1)
    folder = "images" if src == "fb" else "ig_images"
    return REPO / "data/scrape/raw" / folder / rel


def build_manifest():
    """label 1 = rhode; 0 = competitor; 2 = hack exemplar (a distinct negative)."""
    items = []
    for iid, sp in SPLITS.items():
        if sp != "train":
            continue
        brand = iid.split(":")[1].split("/")[0]
        p = corpus_path(iid)
        if not p.exists():
            continue
        items.append((str(p), 1 if brand == "rhode" else 0))
    # hack negatives: SRPO-tuned brand + control (they fooled v1)
    for p in (REPO / "eval/srpo/images/tuned").glob("*/*.jpg"):
        items.append((str(p), 2))
    return items


def embed_model(model, proc, paths, bs=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            b = paths[i:i + bs]
            inp = proc(images=[Image.open(p).convert("RGB") for p in b],
                       return_tensors="pt").to(DEV)
            z = model.get_image_features(**inp)
            if hasattr(z, "pooler_output"):
                z = z.pooler_output
            z = (z / z.norm(dim=-1, keepdim=True)).cpu().float().numpy()
            out.append(z)
    return np.concatenate(out)


def main():
    proc = AutoImageProcessor.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE).to(DEV)
    train = build_manifest()
    random.seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    EPOCHS, BS, TAU = 2, 12, 0.07
    model.train()
    for ep in range(EPOCHS):
        random.shuffle(train)
        for i in range(0, len(train), BS):
            batch = train[i:i + BS]
            if len(batch) < 4:
                continue
            imgs = [Image.open(p).convert("RGB") for p, _ in batch]
            y = torch.tensor([lab for _, lab in batch], device=DEV)
            inp = proc(images=imgs, return_tensors="pt").to(DEV)
            z = model.get_image_features(**inp)
            if hasattr(z, "pooler_output"):
                z = z.pooler_output
            z = z / z.norm(dim=-1, keepdim=True)
            sim = z @ z.T / TAU
            mask = (y[:, None] == y[None, :]).float()
            mask.fill_diagonal_(0)
            lm = sim - sim.max(1, keepdim=True).values.detach()
            ex = torch.exp(lm) * (1 - torch.eye(len(y), device=DEV))
            logp = lm - torch.log(ex.sum(1, keepdim=True) + 1e-9)
            loss = -(mask * logp).sum(1) / mask.sum(1).clamp(min=1)
            loss = loss.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if (i // BS) % 30 == 0:
                print(f"v3 ep{ep} {i}/{len(train)} loss {loss.item():.4f}", flush=True)
    model.save_pretrained(REPO / "judges/siglip_tuned_out/siglip_tuned_v3")
    proc.save_pretrained(REPO / "judges/siglip_tuned_out/siglip_tuned_v3")

    # fit centroid (train rhode) + Platt (val) exactly like the others
    model.eval()
    rhode_tr = [str(corpus_path(i)) for i, sp in SPLITS.items()
                if sp == "train" and i.split(":")[1].split("/")[0] == "rhode"
                and corpus_path(i).exists()]
    centroid = embed_model(model, proc, rhode_tr).mean(0)
    centroid /= np.linalg.norm(centroid)
    val = [(str(corpus_path(i)), 1.0 if i.split(":")[1].split("/")[0] == "rhode" else 0.0)
           for i, sp in SPLITS.items() if sp == "val" and corpus_path(i).exists()]
    ve = embed_model(model, proc, [p for p, _ in val])
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

    # robustness: score SRPO tuned brand images with v1 vs v3
    def score_dir(cal, paths):
        c = np.asarray(cal["centroid"])
        e = embed_model(model, proc, paths)
        return float((1 / (1 + np.exp(-(cal["platt_a"] * (e @ c) + cal["platt_b"])))).mean())

    tuned_brand = sorted(str(p) for p in
                         (REPO / "eval/srpo/images/tuned/brand").glob("*.jpg"))
    v3cal = json.loads((REPO / "judges/j3_tuned_v3_params.json").read_text())
    v1cal = CAL = json.loads((REPO / "judges/j3_tuned_params.json").read_text())
    res = {
        "srpo_hacked_brand_score_v1": None,  # from prior siglip_scores
        "srpo_hacked_brand_score_v3": round(score_dir(v3cal, tuned_brand), 3),
        "note": "v3 trained with SRPO hacks as negatives; lower v3 score on the "
                "same hacked images = hardening worked against seen hacks.",
    }
    prior = json.loads((REPO / "eval/srpo/siglip_scores.json").read_text())
    v1s = np.mean([v for k, v in prior["siglip_tuned"].items()
                   if k.startswith("tuned/brand/")])
    res["srpo_hacked_brand_score_v1"] = round(float(v1s), 3)
    res["hardening_delta"] = round(res["srpo_hacked_brand_score_v1"]
                                   - res["srpo_hacked_brand_score_v3"], 3)
    (REPO / "eval/results/hardening.json").write_text(json.dumps(res, indent=1))
    print("HARDENING_DONE", json.dumps(res))


if __name__ == "__main__":
    main()
