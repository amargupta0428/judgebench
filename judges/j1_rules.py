"""J1 — rules stack (the industry archetype, honest version).

Features per image (all deterministic, documented):
  palette_de     share-weighted mean CIEDE2000 from image dominant colors (k=5,
                 Lab k-means, neutral pixels excluded) to the rhode TRAIN palette
  wordmark       tesseract finds 'rhode' (conf>=60)                  [0/1]
  sat_mean       mean HSV saturation (rhode aesthetic is muted)
  colorfulness   Hasler-Suesstrunk metric (clutter/loudness proxy)
  text_frac      OCR word-area fraction of canvas

Combination: logistic regression fitted on VAL ONLY (rhode val vs competitor
val). Deliberately simple — it is a baseline representing Frontify-class
tooling, not a strawman.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))
from testbed.corruptions import common, typography          # noqa: E402
from testbed.corruptions.palette import rgb_to_lab, _load_palettes  # noqa: E402

PARAMS = Path(__file__).parent / "j1_rules_params.json"
FEATURE_NAMES = ["palette_de", "wordmark", "sat_mean", "colorfulness", "text_frac"]


# ---------- CIEDE2000 (Sharma et al. 2005), vectorized over pairs ----------

def ciede2000(lab1, lab2):
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1); C2 = np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb**7 / (Cb**7 + 25.0**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp))
    dhp = np.where((C1p * C2p) == 0, 0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hsum = h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) > 180,
                   np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2),
                   hsum / 2)
    hbp = np.where((C1p * C2p) == 0, hsum, hbp)
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp**7 / (Cbp**7 + 25.0**7))
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                   + Rt * (dCp / Sc) * (dHp / Sh))


# ---------- features ----------

def dominant_colors(img: Image.Image, k=5, sample=6000, seed=0):
    arr = np.asarray(img, dtype=np.float64).reshape(-1, 3)
    mx, mn = arr.max(axis=1), arr.min(axis=1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    keep = arr[sat >= 0.06]
    if len(keep) < 100:
        keep = arr
    rng = np.random.default_rng(seed)
    pts = rgb_to_lab(keep[rng.choice(len(keep), min(sample, len(keep)), replace=False)])
    centers = pts[rng.choice(len(pts), k, replace=False)]
    for _ in range(15):
        d = np.linalg.norm(pts[:, None] - centers[None], axis=-1)
        assign = d.argmin(axis=1)
        centers = np.array([pts[assign == i].mean(axis=0) if (assign == i).any()
                            else centers[i] for i in range(k)])
    share = np.bincount(assign, minlength=k) / len(assign)
    return centers, share


def features(img: Image.Image, words=None) -> dict:
    pal = np.array(_load_palettes()["rhode"]["lab"])
    centers, share = dominant_colors(img)
    de = ciede2000(centers[:, None, :], pal[None, :, :]).min(axis=1)
    palette_de = float((de * share).sum())

    words = words if words is not None else typography.ocr_words(img)
    wordmark = float(bool(typography.find_wordmarks(words)))
    area = sum((w["box"][2] - w["box"][0]) * (w["box"][3] - w["box"][1])
               for w in words if w["conf"] >= 50)
    text_frac = float(area / (img.width * img.height))

    hsv = np.asarray(img.convert("HSV"), dtype=np.float64)
    sat_mean = float(hsv[..., 1].mean() / 255)

    arr = np.asarray(img, dtype=np.float64)
    rg = arr[..., 0] - arr[..., 1]
    yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                         + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)) / 255

    return {"palette_de": palette_de, "wordmark": wordmark, "sat_mean": sat_mean,
            "colorfulness": colorfulness, "text_frac": text_frac}


def fit(val_features_path: str):
    """Logistic fit on precomputed val features (list of {features, label})."""
    rows = [json.loads(l) for l in open(val_features_path)]
    X = np.array([[r["features"][f] for f in FEATURE_NAMES] for r in rows])
    y = np.array([1 if r["label"] == "on" else 0 for r in rows])
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xn = (X - mu) / sd
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(5000):
        p = 1 / (1 + np.exp(-(Xn @ w + b)))
        gw = Xn.T @ (p - y) / len(y) + 1e-3 * w
        gb = (p - y).mean()
        w -= 0.3 * gw; b -= 0.3 * gb
    acc = ((1 / (1 + np.exp(-(Xn @ w + b))) > 0.5) == y).mean()
    PARAMS.write_text(json.dumps({"w": w.tolist(), "b": float(b),
                                  "mu": mu.tolist(), "sd": sd.tolist(),
                                  "features": FEATURE_NAMES, "val_acc": float(acc)}))
    print(f"J1 fit on {len(y)} val items, val acc {acc:.3f}, weights "
          f"{dict(zip(FEATURE_NAMES, np.round(w, 2)))}")


class J1Rules:
    name = "j1_rules"

    def __init__(self):
        p = json.loads(PARAMS.read_text())
        self.w, self.b = np.array(p["w"]), p["b"]
        self.mu, self.sd = np.array(p["mu"]), np.array(p["sd"])

    def score_features(self, feats: dict) -> float:
        x = np.array([feats[f] for f in FEATURE_NAMES])
        xn = (x - self.mu) / self.sd
        return float(1 / (1 + np.exp(-(xn @ self.w + self.b))))
