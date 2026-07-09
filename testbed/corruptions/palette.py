"""Palette corruptions — programmatic, exact ground truth.

Three generators:
  hue_rotation      — rotate every pixel's hue by a fixed angle
  saturation        — multiply chroma by a fixed factor (direction sampled, recorded)
  brand_color_remap — pixels near the rhode train-split palette are pulled toward
                      the nearest competitor (Glossier) palette color in Lab space

Eligibility guard: a palette corruption only counts as a violation if it is
perceptible. We record mean ΔE(base, corrupted) per image and reject bases where
severity-1 mean ΔE < DELTA_E_FLOOR (near-monochrome images barely change under
hue rotation). Rejections are recorded, not silent (RIGOR_PLAYBOOK: no silent caps).
"""

import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image

DELTA_E_FLOOR = 2.0   # JND is ~2.3; below this the "violation" doesn't exist
PALETTES = Path(__file__).parent / "brand_palettes.json"

# severity -> parameter
HUE_DEG = {1: 18, 2: 40, 3: 80}
SAT_UP = {1: 1.6, 2: 2.3, 3: 3.2}
SAT_DOWN = {1: 0.55, 2: 0.30, 3: 0.08}
REMAP_BLEND = {1: 0.45, 2: 0.75, 3: 1.0}
REMAP_MATCH_DE = 25.0  # pixels within this ΔE of a rhode palette color get remapped


# ---------- color math (sRGB <-> Lab, D65) ----------

def _srgb_to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _linear_to_srgb(c):
    c = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)
    return np.clip(c * 255.0, 0, 255)

_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])

def rgb_to_lab(rgb):
    """rgb: (..., 3) uint8/float -> Lab (..., 3) float"""
    lin = _srgb_to_linear(np.asarray(rgb, dtype=np.float64))
    xyz = lin @ _M_RGB2XYZ.T / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)

def lab_to_rgb(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    f = np.stack([fy + a / 500, fy, fy - b / 200], axis=-1)
    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _WHITE
    lin = xyz @ np.linalg.inv(_M_RGB2XYZ).T
    return _linear_to_srgb(lin)

def delta_e76(lab1, lab2):
    return np.linalg.norm(lab1 - lab2, axis=-1)


def mean_delta_e(img_a: Image.Image, img_b: Image.Image) -> float:
    a = rgb_to_lab(np.asarray(img_a, dtype=np.float64))
    b = rgb_to_lab(np.asarray(img_b, dtype=np.float64))
    return float(delta_e76(a, b).mean())


# ---------- palette extraction (train split ONLY — thresholds never see test) ----------

def extract_brand_palette(images, k=6, sample_px=4000, seed=0, sat_min=0.08):
    """k-means in Lab over pixels sampled from train-split images.

    Near-neutral pixels (sat < sat_min) are excluded so the palette captures the
    brand's chromatic identity, not white backgrounds.
    """
    rng = np.random.default_rng(seed)
    px = []
    for img in images:
        arr = np.asarray(img.convert("RGB"), dtype=np.float64).reshape(-1, 3)
        mx, mn = arr.max(axis=1), arr.min(axis=1)
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
        chroma = arr[sat >= sat_min]
        if len(chroma) == 0:
            continue
        idx = rng.choice(len(chroma), min(sample_px, len(chroma)), replace=False)
        px.append(chroma[idx])
    pts = rgb_to_lab(np.concatenate(px))
    centers = pts[rng.choice(len(pts), k, replace=False)]
    for _ in range(25):
        d = np.linalg.norm(pts[:, None] - centers[None], axis=-1)
        assign = d.argmin(axis=1)
        centers = np.array([pts[assign == i].mean(axis=0) if (assign == i).any()
                            else centers[i] for i in range(k)])
    counts = np.bincount(assign, minlength=k)
    order = counts.argsort()[::-1]
    return centers[order].tolist(), (counts[order] / counts.sum()).tolist()


# ---------- generators ----------

def hue_rotation(img: Image.Image, severity: int, rng) -> tuple[Image.Image, dict]:
    deg = HUE_DEG[severity]
    hsv = np.asarray(img.convert("HSV"), dtype=np.int16)
    hsv[..., 0] = (hsv[..., 0] + round(deg * 255 / 360)) % 256
    out = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")
    return out, {"degrees": deg}


def saturation(img: Image.Image, severity: int, rng) -> tuple[Image.Image, dict]:
    direction = rng.choice(["up", "down"])
    factor = (SAT_UP if direction == "up" else SAT_DOWN)[severity]
    hsv = np.asarray(img.convert("HSV"), dtype=np.float64)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    out = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")
    return out, {"direction": direction, "factor": factor}


def _load_palettes():
    return json.loads(PALETTES.read_text())


def brand_color_remap(img: Image.Image, severity: int, rng) -> tuple[Image.Image, dict]:
    """Pull pixels near rhode's palette toward the nearest Glossier palette color."""
    pals = _load_palettes()
    rhode = np.array(pals["rhode"]["lab"])
    target = np.array(pals["glossier"]["lab"])
    blend = REMAP_BLEND[severity]

    arr = np.asarray(img, dtype=np.float64)
    lab = rgb_to_lab(arr)
    flat = lab.reshape(-1, 3)

    d_rhode = np.linalg.norm(flat[:, None] - rhode[None], axis=-1).min(axis=1)
    mask = d_rhode < REMAP_MATCH_DE

    d_tgt = np.linalg.norm(flat[:, None] - target[None], axis=-1)
    nearest_tgt = target[d_tgt.argmin(axis=1)]
    # move a/b (chroma) fully by blend, keep L mostly (preserves structure/shading)
    moved = flat.copy()
    moved[mask, 1:] = (1 - blend) * flat[mask, 1:] + blend * nearest_tgt[mask, 1:]
    moved[mask, 0] = 0.85 * flat[mask, 0] + 0.15 * nearest_tgt[mask, 0]

    out_arr = lab_to_rgb(moved.reshape(lab.shape))
    out = Image.fromarray(out_arr.astype(np.uint8), "RGB")
    frac = float(mask.mean())
    return out, {"blend": blend, "match_delta_e": REMAP_MATCH_DE,
                 "pixels_remapped_frac": round(frac, 4)}


GENERATORS = {
    "hue_rotation": hue_rotation,
    "saturation": saturation,
    "brand_color_remap": brand_color_remap,
}
