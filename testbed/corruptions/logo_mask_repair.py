"""Repair pass for the v2 logo-masked set, per Amar's rule (July 6):
"if you can't get rid of obvious brand cues, the image is not used."

Inputs: v2 audit verdicts (v2_verdict_collated.json).
  PASS (62)    -> kept as-is.
  FIXABLE (58) -> v2 output + targeted inpaint at audit coordinates,
                  plus any EasyOCR lexicon hits still present.
  DAMAGED (112)-> rebuilt from ORIGINAL with tight boxes only:
                  EasyOCR lexicon hits (phrase-level, tight) + tesseract
                  lexicon hits + v1 OCR boxes + audit coords. Rough sweep
                  regions are NOT reused at 30% pad (they caused the damage);
                  emboss-only regions get 8% pad, capped.

DROP rules (mechanical, logged):
  - > MAX_REGIONS brand regions detected (heavy branding, e.g. sticker sheets)
  - merged mask > MAX_MASK_FRAC of canvas
  - 'rhode' text detected anywhere in the ORIGINAL (cross-brand contamination)
  - still fails the post-build lexicon re-scan after one extra targeted pass

Output: data/testset/logo_masked_v3/ + manifest_v3.jsonl + repair_report.json
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import common, inpaint
from .logo_mask_v2 import (LEXICON, WEAK_TOKENS, ocr_words_multi, OUT as V2_OUT)

S = Path('/tmp/'
         'a0ccc5ba-4ed1-44be-a496-9d49b92a3375/scratchpad')
OUT = common.REPO / "data" / "testset" / "logo_masked_v3"
MAX_REGIONS = 8
MAX_MASK_FRAC = 0.20
EMBOSS_PAD = 0.08

_reader = None


def reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _reader


def _norm(t):
    return t.lower().replace(' ', '').strip(".,!?:;'\"()™®")


def easyocr_lexicon_boxes(img: Image.Image, brand: str) -> list[dict]:
    """Phrase-level EasyOCR results matched against the brand lexicon.
    A phrase counts if it contains the brand name or any lexicon phrase
    (weak single tokens require a second lexicon word in the same phrase)."""
    res = reader().readtext(np.asarray(img))
    lex = LEXICON[brand]
    out = []
    for poly, text, conf in res:
        if conf < 0.2:
            continue
        t = text.lower()
        tn = _norm(text)
        words = [w.strip(".,!?:;'\"()™®") for w in t.split()]
        strong = (brand in tn) or ("rhode" in tn) or any(
            p.replace(' ', '') in tn for p in lex
            if p not in WEAK_TOKENS and len(p) > 4)
        weak_hits = [w for w in words if w in WEAK_TOKENS]
        if not strong and len(weak_hits) < 2:   # bigram rule: 2 weak words adjacent
            continue
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        out.append({"box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    "provenance": "easyocr", "text": text, "conf": round(float(conf), 2)})
    return out


def tesseract_lexicon_boxes(img: Image.Image, brand: str) -> list[dict]:
    from .logo_mask_v2 import lexicon_hits
    return [{**h, "provenance": "tesseract"} for h in lexicon_hits(ocr_words_multi(img), brand)]


def rhode_present(img: Image.Image) -> bool:
    res = reader().readtext(np.asarray(img))
    return any('rhode' in _norm(t) for _, t, c in res if c >= 0.4)


def region_pct_to_box(size, r, pad_frac):
    W, H = size
    x, y = W * r["x"] / 100, H * r["y"] / 100
    w, h = W * r["w"] / 100, H * r["h"] / 100
    px, py = w * pad_frac, h * pad_frac
    return [int(max(0, x - px)), int(max(0, y - py)),
            int(min(W, x + w + px)), int(min(H, y + h + py))]


def build_and_check(sid, base_img, boxes, records, drops, note):
    """Inpaint, re-scan, one retry, else drop."""
    brand = sid.split(":")[1].split("/")[0]
    if not boxes:
        drops.append({"source_id": sid, "reason": f"{note}: no regions found"})
        return
    if len(boxes) > MAX_REGIONS:
        drops.append({"source_id": sid, "reason": f"{note}: {len(boxes)} regions > {MAX_REGIONS}"})
        return
    mask = inpaint.boxes_to_mask(base_img.size, [b["box"] for b in boxes], pad=4)
    frac = np.asarray(mask, dtype=np.float32).mean() / 255.0
    if frac > MAX_MASK_FRAC:
        drops.append({"source_id": sid, "reason": f"{note}: mask {frac:.0%} > {MAX_MASK_FRAC:.0%}"})
        return
    out = inpaint.inpaint(base_img, mask)
    # post-check: both detectors on the result
    residual = (easyocr_lexicon_boxes(out, brand) + tesseract_lexicon_boxes(out, brand))
    if residual:  # one targeted retry
        mask2 = inpaint.boxes_to_mask(out.size, [r["box"] for r in residual], pad=8)
        out = inpaint.inpaint(out, mask2)
        residual = (easyocr_lexicon_boxes(out, brand) + tesseract_lexicon_boxes(out, brand))
        if residual:
            drops.append({"source_id": sid, "reason":
                          f"{note}: residual after retry: {[r.get('text','?') for r in residual][:4]}"})
            return
        boxes = boxes + [{**r, "provenance": r["provenance"] + "_retry"} for r in residual]
    slug = sid.replace(":", "_").replace("/", "_")
    rel = f"{brand}/{slug}.jpg"
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, "JPEG", quality=95, subsampling=0)
    records.append({"masked_file": rel, "source_id": sid, "brand": brand,
                    "label": "off-brand-style-only", "method": "lama_inpaint_v3",
                    "mask_frac": round(float(frac), 4), "regions": boxes,
                    "repair_class": note, "generator_version": "0.3.0"})


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    verdicts = json.loads((S / "v2_verdict_collated.json").read_text())
    v2recs = {json.loads(l)["source_id"]: json.loads(l)
              for l in (V2_OUT / "manifest_v2.jsonl").open()}
    records, drops = [], []

    # PASS: copy over
    for sid in verdicts["pass"]:
        r = v2recs[sid]
        src = V2_OUT / r["masked_file"]
        dst = OUT / r["masked_file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        records.append({**r, "repair_class": "pass_v2", "generator_version": "0.3.0"})
    print(f"copied {len(verdicts['pass'])} v2 passes")

    # rhode contamination check runs on every fixable/redo original
    todo_fix = verdicts["fixable"]
    todo_redo = verdicts["redo"]

    for n, f in enumerate(todo_fix):
        sid = f["source_id"]
        orig = common.load_base(sid)
        if rhode_present(orig):
            drops.append({"source_id": sid, "reason": "rhode wordmark in image (cross-brand)"})
            continue
        v2img = Image.open(V2_OUT / v2recs[sid]["masked_file"]).convert("RGB")
        boxes = []
        if f.get("region"):
            boxes.append({"box": region_pct_to_box(v2img.size, f["region"], 0.10),
                          "provenance": "audit_coord"})
        brand = sid.split(":")[1].split("/")[0]
        boxes += easyocr_lexicon_boxes(v2img, brand)
        build_and_check(sid, v2img, boxes, records, drops, "fixable")
        if (n + 1) % 20 == 0:
            print(f"  fixable {n+1}/{len(todo_fix)}")

    for n, f in enumerate(todo_redo):
        sid = f["source_id"]
        orig = common.load_base(sid)
        if rhode_present(orig):
            drops.append({"source_id": sid, "reason": "rhode wordmark in image (cross-brand)"})
            continue
        brand = sid.split(":")[1].split("/")[0]
        boxes = easyocr_lexicon_boxes(orig, brand) + tesseract_lexicon_boxes(orig, brand)
        # precise wordmark boxes from v1 OCR carry over; rough 30%-padded agent
        # regions do NOT (they caused the v2 damage)
        boxes += [reg for reg in v2recs[sid].get("regions", [])
                  if isinstance(reg, dict) and reg.get("provenance") == "v1_ocr"]
        build_and_check(sid, orig, boxes, records, drops, "redo")
        if (n + 1) % 20 == 0:
            print(f"  redo {n+1}/{len(todo_redo)}")

    with (OUT / "manifest_v3.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (OUT / "repair_report.json").write_text(json.dumps(
        {"kept": len(records), "dropped": len(drops), "drops": drops}, indent=1))
    import collections
    print("kept:", len(records), "by class:",
          dict(collections.Counter(r["repair_class"] for r in records)))
    print("dropped:", len(drops))


if __name__ == "__main__":
    run()
