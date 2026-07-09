"""Composition corruptions — programmatic, exact ground truth.

  crop_violation    — aggressive off-center crop (removes margin/subject area)
  aspect_distortion — non-uniform stretch, then resize back to original canvas
  clutter           — drawn ad-noise elements (starbursts, badges, banners) in
                      deliberately off-palette retail colors

All parameters recorded exactly. Severity 1 = subtle-but-real, 3 = unmistakable.
"""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CROP_AREA_REMOVED = {1: 0.18, 2: 0.38, 3: 0.58}   # fraction of area cropped away
ASPECT_FACTOR = {1: 1.18, 2: 1.45, 3: 1.9}         # stretch ratio
CLUTTER_ITEMS = {1: 1, 2: 3, 3: 6}
CLUTTER_SCALE = {1: 0.12, 2: 0.18, 3: 0.26}        # element size vs min(img dims)

# retail-scream palette: maximally off-brand for a soft-minimal aesthetic
LOUD = ["#FF1E1E", "#FFD400", "#00C22D", "#FF6A00", "#E600FF"]
BADGE_TEXTS = ["SALE!", "50% OFF", "BUY NOW", "HOT DEAL", "FREE SHIP", "NEW!!!"]


def crop_violation(img: Image.Image, severity: int, rng) -> tuple[Image.Image, dict]:
    frac = CROP_AREA_REMOVED[severity]
    keep = math.sqrt(1 - frac)          # keep same aspect, remove `frac` of area
    w, h = img.size
    cw, ch = round(w * keep), round(h * keep)
    # push the crop window toward a random corner/edge — violates margins
    corner = rng.choice(["tl", "tr", "bl", "br", "l", "r", "t", "b"])
    x0 = 0 if corner in ("tl", "bl", "l") else (w - cw if corner in ("tr", "br", "r")
                                                else rng.randint(0, w - cw))
    y0 = 0 if corner in ("tl", "tr", "t") else (h - ch if corner in ("bl", "br", "b")
                                                else rng.randint(0, h - ch))
    out = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.LANCZOS)
    return out, {"area_removed": frac, "anchor": corner, "box": [x0, y0, x0 + cw, y0 + ch]}


def aspect_distortion(img: Image.Image, severity: int, rng) -> tuple[Image.Image, dict]:
    factor = ASPECT_FACTOR[severity]
    axis = rng.choice(["x", "y"])
    w, h = img.size
    if axis == "x":
        stretched = img.resize((round(w * factor), h), Image.LANCZOS)
    else:
        stretched = img.resize((w, round(h * factor)), Image.LANCZOS)
    # center-crop back to the original canvas so only proportions change
    sw, sh = stretched.size
    x0, y0 = (sw - w) // 2, (sh - h) // 2
    out = stretched.crop((x0, y0, x0 + w, y0 + h))
    return out, {"factor": factor, "axis": axis}


def _font(size: int) -> ImageFont.FreeTypeFont:
    for cand in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _starburst(draw: ImageDraw.ImageDraw, cx, cy, r, color, text, rng):
    pts = []
    spikes = 12
    for i in range(spikes * 2):
        rad = r if i % 2 == 0 else r * 0.62
        ang = math.pi * i / spikes + rng.random() * 0.1
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    draw.polygon(pts, fill=color, outline="#000000")
    f = _font(max(10, int(r * 0.34)))
    tb = draw.textbbox((0, 0), text, font=f)
    draw.text((cx - (tb[2] - tb[0]) / 2, cy - (tb[3] - tb[1]) / 2 - tb[1]),
              text, fill="white", font=f, stroke_width=2, stroke_fill="black")


def clutter(img: Image.Image, severity: int, rng) -> tuple[Image.Image, dict]:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    n = CLUTTER_ITEMS[severity]
    base_r = CLUTTER_SCALE[severity] * min(w, h)
    items = []
    for i in range(n):
        color = rng.choice(LOUD)
        text = rng.choice(BADGE_TEXTS)
        r = base_r * (0.8 + 0.4 * rng.random())
        cx = rng.uniform(r, w - r)
        cy = rng.uniform(r, h - r)
        _starburst(draw, cx, cy, r, color, text, rng)
        items.append({"kind": "starburst", "center": [round(cx), round(cy)],
                      "radius": round(r), "color": color, "text": text})
    if severity == 3:  # banner across the bottom, the full retail scream
        bh = int(h * 0.12)
        color = rng.choice(LOUD)
        draw.rectangle([0, h - bh, w, h], fill=color, outline="black")
        f = _font(int(bh * 0.55))
        msg = "LIMITED TIME OFFER – SHOP NOW"
        tb = draw.textbbox((0, 0), msg, font=f)
        draw.text(((w - (tb[2] - tb[0])) / 2, h - bh + (bh - (tb[3] - tb[1])) / 2 - tb[1]),
                  msg, fill="white", font=f, stroke_width=2, stroke_fill="black")
        items.append({"kind": "banner", "height": bh, "color": color, "text": msg})
    return out, {"n_items": len(items), "items": items}


GENERATORS = {
    "crop_violation": crop_violation,
    "aspect_distortion": aspect_distortion,
    "clutter": clutter,
}
