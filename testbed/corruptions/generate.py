"""Generate the programmatic corruption test set.

Usage:
    python -m testbed.corruptions.generate palettes     # extract brand palettes (train only)
    python -m testbed.corruptions.generate smoke        # 3 bases/generator, contact sheets only
    python -m testbed.corruptions.generate full         # >=TARGET per generator x severity

Design invariants:
- Bases drawn ONLY from the rhode random-cluster test split (temporal holdout untouched).
- Same bases reused across severities within a generator (paired severity curves).
- Eligibility gates are explicit and logged: palette corruptions must clear a
  perceptual floor (mean dE at s1), remap must actually hit pixels, typography
  needs an OCR-localized wordmark. Nothing is skipped silently.
"""

import json
import random
import sys

from PIL import Image

from . import common, palette, composition, typography

GLOBAL_SEED = 20260706
TARGET = 30          # images per generator x severity cell
SMOKE_N = 3
REMAP_MIN_FRAC = 0.03


def _img_seed(image_id: str, gen_name: str) -> int:
    return random.Random(f"{GLOBAL_SEED}:{gen_name}:{image_id}").randint(0, 2**31)


def build_palettes():
    """Brand palettes from TRAIN split only (thresholds never see test)."""
    splits = json.loads(common.SPLITS.read_text())
    index = splits["_image_index"]
    out = {}
    for brand in ("rhode", "glossier"):
        ids = sorted(i for i, sp in index.items()
                     if sp == "train" and i.split(":")[1].split("/")[0] == brand)
        rng = random.Random(GLOBAL_SEED)
        sample = rng.sample(ids, min(120, len(ids)))
        imgs = (common.load_base(i) for i in sample)
        lab, weights = palette.extract_brand_palette(imgs, k=6, seed=GLOBAL_SEED)
        out[brand] = {"lab": lab, "weights": weights, "n_images": len(sample),
                      "split": "train", "seed": GLOBAL_SEED}
        print(f"{brand}: palette from {len(sample)} train images")
    palette.PALETTES.write_text(json.dumps(out, indent=2))
    print(f"wrote {palette.PALETTES}")


def _typo_eligible(ids):
    """OCR pre-scan: which rhode test images have a localizable wordmark?"""
    eligible, cache = [], {}
    for i, image_id in enumerate(ids):
        img = common.load_base(image_id)
        words = typography.ocr_words(img)
        if typography.find_wordmarks(words):
            eligible.append(image_id)
            cache[image_id] = words
        if (i + 1) % 50 == 0:
            print(f"  ocr scan {i+1}/{len(ids)}, eligible so far: {len(eligible)}")
    return eligible, cache


def run(n_per_cell: int, dimensions=("palette", "composition", "typography")):
    ids = common.rhode_test_ids("test")
    rng = random.Random(GLOBAL_SEED)
    rng.shuffle(ids)
    print(f"{len(ids)} rhode test-split bases available")

    modules = {"palette": palette, "composition": composition,
               "typography": typography}
    skip_log = []

    for dim in dimensions:
        mod = modules[dim]
        words_cache = {}
        if dim == "typography":
            pool, words_cache = _typo_eligible(ids)
            print(f"typography: {len(pool)}/{len(ids)} bases have a localizable wordmark")
        else:
            pool = ids

        for gen_name, gen in mod.GENERATORS.items():
            records, used = [], 0
            sheet_rows, sheet_done = [], 0
            for image_id in pool:
                if used >= n_per_cell:
                    break
                base = common.load_base(image_id)
                seed = _img_seed(image_id, gen_name)
                per_sev = {}
                ok = True
                for sev in common.SEVERITIES:
                    # rng seeded per BASE (not per severity): direction/anchor/axis
                    # choices stay fixed across the severity ladder, so severity is
                    # the only variable within a base (clean paired design)
                    g_rng = random.Random(seed)
                    kwargs = {"words": words_cache[image_id]} if dim == "typography" else {}
                    out, params = gen(base, sev, g_rng, **kwargs)
                    # eligibility gates
                    if dim == "palette":
                        de = palette.mean_delta_e(base, out)
                        params["mean_delta_e"] = round(de, 2)
                        if sev == 1 and de < palette.DELTA_E_FLOOR:
                            skip_log.append({"image": image_id, "generator": gen_name,
                                             "reason": f"s1 mean_dE {de:.2f} < floor"})
                            ok = False
                            break
                        if gen_name == "brand_color_remap" and params["pixels_remapped_frac"] < REMAP_MIN_FRAC:
                            skip_log.append({"image": image_id, "generator": gen_name,
                                             "reason": f"remap frac {params['pixels_remapped_frac']} < {REMAP_MIN_FRAC}"})
                            ok = False
                            break
                    per_sev[sev] = (out, params)
                if not ok:
                    continue
                for sev, (out, params) in per_sev.items():
                    rec = common.CorruptionRecord(
                        out_file=common.out_name(image_id, gen_name, sev),
                        source_id=image_id, dimension=dim, corruption=gen_name,
                        severity=sev, params=params, seed=seed)
                    records.append(common.write_output(out, rec))
                if sheet_done < 6:
                    sheet_rows.append([base] + [per_sev[s][0] for s in common.SEVERITIES])
                    sheet_done += 1
                used += 1
            common.append_manifest(records, f"manifest_{dim}.jsonl")
            sheet = common.contact_sheet(sheet_rows, ["clean", "s1", "s2", "s3"])
            sheet_path = common.OUT_DIR / f"sheet_{gen_name}.jpg"
            sheet.save(sheet_path, quality=90)
            print(f"{dim}/{gen_name}: {used} bases x 3 sev = {len(records)} images; sheet -> {sheet_path.name}")

    if skip_log:
        (common.OUT_DIR / "skips.json").write_text(json.dumps(skip_log, indent=2))
        print(f"{len(skip_log)} base rejections logged to skips.json")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "palettes":
        build_palettes()
    elif mode == "smoke":
        run(SMOKE_N)
    elif mode == "full":
        run(TARGET)
    else:
        raise SystemExit(f"unknown mode {mode}")
