"""Typography corruptions — programmatic, gated on OCR wordmark localization.

Localization: tesseract (system binary, TSV output) finds the brand token
("rhode") in the image. Only images where the wordmark is found with
conf >= MIN_CONF and box height >= MIN_BOX_H px are ELIGIBLE for these
corruptions; eligibility is recorded, never silent.

  wordmark_removal — s1: remove one wordmark instance; s2: remove all brand
                     tokens; s3: remove ALL detected text (typographic identity
                     fully stripped)
  font_swap        — re-render the wordmark in a wrong font, same position/size:
                     s1 neutral sans, s2 heavy serif, s3 novelty (Chalkboard)
  wrong_case       — re-render with wrong case treatment:
                     s1 'Rhode', s2 'RHODE', s3 'R H O D E' letter-spaced

The removal/re-render fill uses the median color of a ring around the text box
(works because rhode text assets are text-on-flat-color by design).
This module is also the localization engine for the logo-masked hard negatives.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BRAND_TOKENS = {"rhode"}
MIN_CONF = 60.0
MIN_BOX_H = 14  # px; smaller than this and the wordmark is illegible anyway

FONTS = {
    1: ["/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc"],
    2: ["/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"],
    3: ["/System/Library/Fonts/Supplemental/Chalkboard.ttc",
        "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
        "/System/Library/Fonts/MarkerFelt.ttc"],
}
CASE_RENDER = {1: lambda t: t.capitalize(),
               2: lambda t: t.upper(),
               3: lambda t: " ".join(t.upper())}


def ocr_words(img: Image.Image) -> list[dict]:
    """Run tesseract, return [{text, conf, box:[x0,y0,x1,y1]}] for word-level hits."""
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        img.save(tmp.name)
        out = subprocess.run(
            ["tesseract", tmp.name, "stdout", "--psm", "11", "tsv"],
            capture_output=True, text=True, check=True).stdout
    words = []
    lines = out.strip().split("\n")
    header = lines[0].split("\t")
    for line in lines[1:]:
        f = dict(zip(header, line.split("\t")))
        try:
            conf = float(f["conf"])
        except (KeyError, ValueError):
            continue
        text = f.get("text", "").strip()
        if not text or conf < 0:
            continue
        x, y = int(f["left"]), int(f["top"])
        w, h = int(f["width"]), int(f["height"])
        words.append({"text": text, "conf": conf, "box": [x, y, x + w, y + h]})
    return words


def find_wordmarks(words: list[dict]) -> list[dict]:
    return [w for w in words
            if w["text"].lower().strip(".,!?:;'\"") in BRAND_TOKENS
            and w["conf"] >= MIN_CONF
            and (w["box"][3] - w["box"][1]) >= MIN_BOX_H]


def _ring_median(arr: np.ndarray, box, pad=6) -> np.ndarray:
    """Median color of a `pad`-px ring around box — the local background."""
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = box
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
    X1, Y1 = min(w, x1 + pad), min(h, y1 + pad)
    outer = arr[Y0:Y1, X0:X1].reshape(-1, 3)
    inner = arr[y0:y1, x0:x1].reshape(-1, 3)
    if len(outer) <= len(inner):
        return np.median(outer, axis=0)
    # ring = outer minus inner (approximate via mask)
    mask = np.ones((Y1 - Y0, X1 - X0), bool)
    mask[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
    ring = arr[Y0:Y1, X0:X1][mask]
    return np.median(ring.reshape(-1, 3), axis=0)


def _ink_color(arr: np.ndarray, box, bg: np.ndarray) -> tuple:
    """Ink = pixels inside the box farthest from the local background color."""
    x0, y0, x1, y1 = box
    px = arr[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
    d = np.linalg.norm(px - bg, axis=1)
    ink = px[d >= np.percentile(d, 75)]
    return tuple(int(c) for c in np.median(ink, axis=0))


def _fill_box(img: Image.Image, box, pad=2) -> Image.Image:
    arr = np.asarray(img)
    bg = _ring_median(arr, box)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    d.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad],
                fill=tuple(int(c) for c in bg))
    return img


def _load_font(paths, target_h):
    for p in paths:
        if Path(p).exists():
            size = int(target_h * 1.1)
            try:
                return ImageFont.truetype(p, size), p
            except OSError:
                continue
    return ImageFont.load_default(), "default"


def _render_text(img: Image.Image, box, text, font_paths, color) -> str:
    x0, y0, x1, y1 = box
    font, used = _load_font(font_paths, y1 - y0)
    d = ImageDraw.Draw(img)
    tb = d.textbbox((0, 0), text, font=font)
    # scale down until it fits the original width with 15% slack
    while (tb[2] - tb[0]) > (x1 - x0) * 1.15 and font.size > 8:
        font = font.font_variant(size=font.size - 2)
        tb = d.textbbox((0, 0), text, font=font)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    d.text((cx - (tb[2] - tb[0]) / 2 - tb[0], cy - (tb[3] - tb[1]) / 2 - tb[1]),
           text, fill=color, font=font)
    return used


# ---------- generators (each expects pre-computed `words` in kwargs) ----------

def wordmark_removal(img: Image.Image, severity: int, rng, words=None):
    words = words if words is not None else ocr_words(img)
    marks = find_wordmarks(words)
    out = img.copy()
    if severity == 1:
        targets = [rng.choice(marks)]
    elif severity == 2:
        targets = marks
    else:
        # Real overlaid ad text = legible tokens of >=3 alphanumeric chars at
        # full confidence. Tesseract false-positives on face/product features
        # ('a.' on an eyebrow, conf 73) are short junk tokens; under-removal is
        # conservative, a fill box on a face would contaminate the label.
        targets = [w for w in words if w["conf"] >= MIN_CONF
                   and sum(c.isalnum() for c in w["text"]) >= 3]
        # s3 must strictly contain s2: brand marks always included
        seen = {tuple(t["box"]) for t in targets}
        targets += [m for m in marks if tuple(m["box"]) not in seen]
    for t in targets:
        _fill_box(out, t["box"])
    return out, {"n_boxes_removed": len(targets),
                 "boxes": [t["box"] for t in targets],
                 "removed_all_text": severity == 3}


def font_swap(img: Image.Image, severity: int, rng, words=None):
    words = words if words is not None else ocr_words(img)
    marks = find_wordmarks(words)
    out = img.copy()
    arr = np.asarray(img)
    used_fonts = []
    for m in marks:
        bg = _ring_median(arr, m["box"])
        ink = _ink_color(arr, m["box"], bg)
        _fill_box(out, m["box"])
        used = _render_text(out, m["box"], "rhode", FONTS[severity], ink)
        used_fonts.append(used)
    return out, {"n_swapped": len(marks), "fonts": used_fonts,
                 "boxes": [m["box"] for m in marks]}


def wrong_case(img: Image.Image, severity: int, rng, words=None):
    words = words if words is not None else ocr_words(img)
    marks = find_wordmarks(words)
    out = img.copy()
    arr = np.asarray(img)
    rendered = CASE_RENDER[severity]("rhode")
    for m in marks:
        bg = _ring_median(arr, m["box"])
        ink = _ink_color(arr, m["box"], bg)
        _fill_box(out, m["box"])
        # keep the ORIGINAL font family feel: use neutral sans for all severities;
        # the manipulated variable is case, not face
        _render_text(out, m["box"], rendered, FONTS[1], ink)
    return out, {"n_rendered": len(marks), "text": rendered,
                 "boxes": [m["box"] for m in marks]}


GENERATORS = {
    "wordmark_removal": wordmark_removal,
    "font_swap": font_swap,
    "wrong_case": wrong_case,
}
