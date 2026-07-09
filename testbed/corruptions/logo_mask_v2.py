"""Logo-masked hard negatives v2 — LaMa inpainting over a merged cue inventory.

Cue sources per image (each region records its provenance):
  1. lexicon OCR   — tesseract at 1x and 2x upscale, psm 11 + psm 3, matched
                     against a brand-owned lexicon (brand names AND product
                     names/slogans — "cloud paint" identifies Glossier as
                     surely as the wordmark does)
  2. agent_region  — sweep/audit vision agents' region_pct boxes (glyphs,
                     embosses, and labels OCR cannot read)
  3. fence         — adjudicated fence-case decisions (fence_decisions.json)

Images with no cue from any source and agent verdict "no mark" are emitted as
`natively_clean` negatives (usable without any masking artifact).
Images whose merged mask exceeds MAX_MASK_FRAC of the canvas are dropped as
unmaskable (recorded, not silent).

Output: data/testset/logo_masked_v2/ + manifest_v2.jsonl + audit contact sheets.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import common, typography, inpaint

S = Path('/tmp/'
         'a0ccc5ba-4ed1-44be-a496-9d49b92a3375/scratchpad')
OUT = common.REPO / "data" / "testset" / "logo_masked_v2"
MAX_MASK_FRAC = 0.35
REGION_PAD_FRAC = 0.30   # agent boxes are rough; pad generously

LEXICON = {
    "glossier": [
        "glossier", "glossier.", "glossiers", "balm", "dotcom", "cloud",
        "paint", "boy", "brow", "futuredew", "ultralip", "wowder", "haloscope",
        "generation", "milky", "jelly", "solid", "you look good", "plume",
    ],
    "ilia": [
        "ilia", "iliabeauty", "skin tint", "super serum", "multi-stick",
        "limitless", "true skin", "liquid light",
    ],
}
# single generic words only count when OCR'd next to another lexicon hit
WEAK_TOKENS = {"balm", "cloud", "paint", "boy", "brow", "solid", "plume",
               "generation", "milky", "jelly", "dotcom", "limitless"}


def ocr_words_multi(img: Image.Image) -> list[dict]:
    """tesseract at 1x/2x and psm 11/3; dedupe by IoU (scaled back to 1x)."""
    words = []
    for scale in (1, 2):
        im = img if scale == 1 else img.resize(
            (img.width * 2, img.height * 2), Image.LANCZOS)
        for psm in ("11", "3"):
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                im.save(tmp.name)
                out = subprocess.run(
                    ["tesseract", tmp.name, "stdout", "--psm", psm, "tsv"],
                    capture_output=True, text=True).stdout
            lines = out.strip().split("\n")
            if not lines or "\t" not in lines[0]:
                continue
            header = lines[0].split("\t")
            for line in lines[1:]:
                f = dict(zip(header, line.split("\t")))
                try:
                    conf = float(f["conf"])
                except (KeyError, ValueError):
                    continue
                text = f.get("text", "").strip()
                if not text or conf < 30:
                    continue
                x, y = int(f["left"]) // scale, int(f["top"]) // scale
                w, h = int(f["width"]) // scale, int(f["height"]) // scale
                if w < 4 or h < 6:
                    continue
                words.append({"text": text, "conf": conf,
                              "box": [x, y, x + w, y + h]})
    return words


def lexicon_hits(words: list[dict], brand: str) -> list[dict]:
    lex = LEXICON[brand]
    hits = []
    strong_present = any(
        w["text"].lower().strip(".,!?:;'\"()") in (brand, f"{brand}.")
        or (w["text"].lower().strip(".,!?:;'\"()") in lex
            and w["text"].lower().strip(".,!?:;'\"()") not in WEAK_TOKENS)
        for w in words)
    for w in words:
        t = w["text"].lower().strip(".,!?:;'\"()")
        if t in lex or any(t == part for phrase in lex for part in phrase.split()):
            if t in WEAK_TOKENS and not strong_present:
                continue
            hits.append({**w, "provenance": "lexicon_ocr"})
    return hits


def agent_regions(size: tuple, entries: list[dict]) -> list[dict]:
    """sweep/audit region_pct dicts -> padded pixel boxes."""
    W, H = size
    out = []
    for e in entries:
        r = e.get("region_pct")
        if not r:
            continue
        x, y = W * r["x"] / 100, H * r["y"] / 100
        w, h = W * r["w"] / 100, H * r["h"] / 100
        px, py = w * REGION_PAD_FRAC, h * REGION_PAD_FRAC
        out.append({"box": [int(max(0, x - px)), int(max(0, y - py)),
                            int(min(W, x + w + px)), int(min(H, y + h + py))],
                    "provenance": "agent_region",
                    "note": e.get("note", "")[:60]})
    return out


def build_inventory():
    """Merge all sources into {source_id: {brand, regions, status}}."""
    collated = json.loads((S / "collated.json").read_text())
    sweep = []
    for i in range(9):
        sweep += json.loads((S / f"sweep_result_{i}.json").read_text())
    sweep_by_id = {s["source_id"]: s for s in sweep}
    fence = json.loads(
        (common.REPO / "data/testset/logo_masked/fence_decisions.json").read_text())
    fence_by_id = {f["source_id"]: f for f in fence}
    v1 = [json.loads(l) for l in
          (common.REPO / "data/testset/logo_masked/manifest.jsonl").open()]
    v1_by_id = {r["source_id"]: r for r in v1}

    inv = {}
    # every candidate: v1 pairs + sweep-yes + fence(mask/auto)
    ids = set(v1_by_id) | set(collated["sweep_yes"])
    for f in fence:
        if f["decision"] in ("mask", "auto"):
            ids.add(f["source_id"])
        elif f["decision"] == "drop":
            ids.discard(f["source_id"])
    for sid in sorted(ids):
        brand = sid.split(":")[1].split("/")[0]
        fd = fence_by_id.get(sid)
        if fd and fd["decision"] == "drop":
            continue
        entry = {"brand": brand, "sweep": sweep_by_id.get(sid),
                 "v1_boxes": v1_by_id.get(sid, {}).get("boxes", []),
                 "fence": fd}
        inv[sid] = entry
    # natively clean: sweep said "no" and not otherwise involved
    clean = [s["source_id"] for s in sweep if s["brand_mark"] == "no"]
    for f in fence:
        if f["decision"] == "clean":
            clean.append(f["source_id"])
    return inv, sorted(set(clean))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    inv, clean = build_inventory()
    print(f"{len(inv)} maskable candidates, {len(clean)} natively clean")
    records, drops = [], []

    for n, (sid, e) in enumerate(sorted(inv.items())):
        img = common.load_base(sid)
        regions = []
        # 1. lexicon OCR
        words = ocr_words_multi(img)
        regions += lexicon_hits(words, e["brand"])
        # 2. v1 OCR boxes (already precise)
        regions += [{"box": b, "provenance": "v1_ocr"} for b in e["v1_boxes"]]
        # 3. agent regions from sweep (main + extra_marks)
        if e["sweep"]:
            entries = [e["sweep"]] + (e["sweep"].get("extra_marks") or [])
            regions += agent_regions(img.size, entries)
        if not regions:
            drops.append({"source_id": sid, "reason": "no region from any source"})
            continue
        boxes = [r["box"] for r in regions]
        mask = inpaint.boxes_to_mask(img.size, boxes, pad=4)
        frac = np.asarray(mask, dtype=np.float32).mean() / 255.0
        if frac > MAX_MASK_FRAC:
            drops.append({"source_id": sid,
                          "reason": f"mask covers {frac:.0%} > {MAX_MASK_FRAC:.0%}"})
            continue
        out = inpaint.inpaint(img, mask)
        slug = sid.replace(":", "_").replace("/", "_")
        rel = f"{e['brand']}/{slug}.jpg"
        path = OUT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        out.save(path, "JPEG", quality=95, subsampling=0)
        records.append({
            "masked_file": rel, "source_id": sid, "brand": e["brand"],
            "label": "off-brand-style-only", "method": "lama_inpaint",
            "mask_frac": round(float(frac), 4),
            "regions": regions,
            "fence_decision": (e["fence"] or {}).get("decision"),
            "generator_version": "0.2.0",
        })
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(inv)} inpainted")

    with (OUT / "manifest_v2.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (OUT / "natively_clean.json").write_text(json.dumps(clean, indent=1))
    (OUT / "drops.json").write_text(json.dumps(drops, indent=1))
    print(f"v2: {len(records)} masked pairs, {len(clean)} natively clean, "
          f"{len(drops)} dropped (see drops.json)")


if __name__ == "__main__":
    run()
