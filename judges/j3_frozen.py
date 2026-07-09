"""J3-frozen — untrained SigLIP centroid judge.

Prototype = mean of rhode TRAIN-split embeddings (zero weight updates).
Score = cosine similarity to prototype, Platt-calibrated on VAL only
(rhode val positives vs glossier/ilia val negatives). Test never touched
during fitting.

This is the "what does fine-tuning buy" baseline for the SigLIP pair.
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FEATURES = REPO / "data" / "features"
PARAMS = Path(__file__).parent / "j3_frozen_params.json"


def _corpus_embeddings():
    emb = {}
    for src, path in (("fb", FEATURES / "embeddings_fb.npz"),
                      ("ig", FEATURES / "embeddings_ig.npz")):
        z = np.load(path, allow_pickle=True)
        for f, v in zip(z["files"], z["vecs"]):
            emb[f"{src}:{f}"] = v
    return emb


def fit():
    """Centroid from train; Platt scaling on val. Writes params file."""
    splits = json.loads((FEATURES / "splits_v2.json").read_text())
    index = splits["_image_index"]
    emb = _corpus_embeddings()

    train_rhode = [emb[i] for i, sp in index.items()
                   if sp == "train" and i.split(":")[1].split("/")[0] == "rhode" and i in emb]
    centroid = np.mean(train_rhode, axis=0)
    centroid /= np.linalg.norm(centroid)

    # val cosines
    val_pos, val_neg = [], []
    for i, sp in index.items():
        if sp != "val" or i not in emb:
            continue
        c = float(np.dot(emb[i], centroid))
        (val_pos if i.split(":")[1].split("/")[0] == "rhode" else val_neg).append(c)

    # Platt: logistic regression on the scalar cosine (closed-form-ish via gradient)
    x = np.array(val_pos + val_neg)
    y = np.array([1] * len(val_pos) + [0] * len(val_neg))
    a, b = 20.0, -10.0  # init
    lr = 0.5
    for _ in range(2000):
        p = 1 / (1 + np.exp(-(a * x + b)))
        ga = ((p - y) * x).mean()
        gb = (p - y).mean()
        a -= lr * ga
        b -= lr * gb
    PARAMS.write_text(json.dumps({
        "centroid": centroid.tolist(), "platt_a": float(a), "platt_b": float(b),
        "n_train": len(train_rhode), "n_val_pos": len(val_pos), "n_val_neg": len(val_neg)}))
    print(f"fit: {len(train_rhode)} train rhode, val {len(val_pos)}+{len(val_neg)}, "
          f"platt a={a:.2f} b={b:.2f}")


class J3Frozen:
    name = "j3_siglip_frozen"

    def __init__(self):
        p = json.loads(PARAMS.read_text())
        self.centroid = np.array(p["centroid"])
        self.a, self.b = p["platt_a"], p["platt_b"]

    def score_vec(self, vec: np.ndarray) -> float:
        """vec: L2-normalized SigLIP embedding -> calibrated on-brand prob."""
        c = float(np.dot(vec, self.centroid))
        return float(1 / (1 + np.exp(-(self.a * c + self.b))))


if __name__ == "__main__":
    fit()
