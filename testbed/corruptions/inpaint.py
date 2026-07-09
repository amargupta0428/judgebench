"""LaMa content-aware fill — removes brand marks by inpainting, not box-fill.

Thin wrapper around the big-lama TorchScript checkpoint (same model the
simple-lama-inpainting package ships; that package won't build on py3.14).
Input: RGB image + binary mask (255 = remove). Output: filled RGB image.

Rigor note: inpainting is GENERATIVE — it can hallucinate texture into the
hole. Every inpainted output goes through the v2 visual re-audit; this module
only guarantees the masked region's original pixels are gone.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL = Path(__file__).resolve().parents[1] / "models" / "big-lama.pt"
_model = None


def _load():
    global _model
    if _model is None:
        _model = torch.jit.load(MODEL, map_location="cpu")
        _model.eval()
    return _model


def _pad_mod8(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    if arr.ndim == 3:
        return np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    return np.pad(arr, ((0, ph), (0, pw)), mode="reflect")


@torch.inference_mode()
def inpaint(img: Image.Image, mask: Image.Image) -> Image.Image:
    """img: RGB; mask: L, 255 where content should be removed and filled."""
    model = _load()
    w, h = img.size
    im = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    mk = (np.asarray(mask.convert("L"), dtype=np.float32) > 127).astype(np.float32)
    im, mk = _pad_mod8(im), _pad_mod8(mk)
    im_t = torch.from_numpy(im).permute(2, 0, 1)[None]
    mk_t = torch.from_numpy(mk)[None, None]
    out = model(im_t, mk_t)[0].permute(1, 2, 0).numpy()
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)[:h, :w]
    return Image.fromarray(out, "RGB")


def boxes_to_mask(size: tuple, boxes: list, pad: int = 4) -> Image.Image:
    """[[x0,y0,x1,y1], ...] -> binary L mask with padding."""
    from PIL import ImageDraw
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for x0, y0, x1, y1 in boxes:
        d.rectangle([max(0, x0 - pad), max(0, y0 - pad),
                     min(size[0], x1 + pad), min(size[1], y1 + pad)], fill=255)
    return m
