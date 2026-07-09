"""Logo-masked hard negatives — kills the "judge just reads the logo" shortcut.

For each competitor (Glossier/ILIA) test-split image where OCR localizes the
brand wordmark, produce a twin with every brand token filled with local
background. The (cue-present, cue-masked) PAIR is the instrument: any judge
whose competitor-rejection rate drops on the masked twin was reading the name
tag, not the style. Delta = the logo shortcut's contribution (ablation-by-masking).

Only paired images are emitted — an unmasked-only negative measures nothing here.

Usage:  python -m testbed.corruptions.logo_mask
"""

import json

from . import common, typography

BRAND_TOKENS = {
    "glossier": {"glossier", "glossier."},
    "ilia": {"ilia", "ilia.", "iliabeauty"},
}
OUT = common.REPO / "data" / "testset" / "logo_masked"


def competitor_test_ids(brand: str) -> list[str]:
    data = json.loads(common.SPLITS.read_text())
    return sorted(i for i, sp in data["_image_index"].items()
                  if sp == "test" and i.split(":")[1].split("/")[0] == brand)


def find_brand_tokens(words: list[dict], brand: str) -> list[dict]:
    tokens = BRAND_TOKENS[brand]
    return [w for w in words
            if w["text"].lower().strip(".,!?:;'\"") in tokens
            and w["conf"] >= typography.MIN_CONF
            and (w["box"][3] - w["box"][1]) >= typography.MIN_BOX_H]


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for brand in ("glossier", "ilia"):
        ids = competitor_test_ids(brand)
        found = 0
        for i, image_id in enumerate(ids):
            img = common.load_base(image_id)
            marks = find_brand_tokens(typography.ocr_words(img), brand)
            if not marks:
                continue
            masked = img.copy()
            for m in marks:
                typography._fill_box(masked, m["box"])
            slug = image_id.replace(":", "_").replace("/", "_")
            masked_file = f"{brand}/{slug}.jpg"
            path = OUT / masked_file
            path.parent.mkdir(parents=True, exist_ok=True)
            masked.save(path, "JPEG", quality=95, subsampling=0)
            records.append({
                "masked_file": masked_file,
                "source_id": image_id,      # the cue-present twin (original file)
                "brand": brand,
                "label": "off-brand-style-only",
                "n_boxes_masked": len(marks),
                "boxes": [m["box"] for m in marks],
                "confs": [m["conf"] for m in marks],
                "generator_version": common.GENERATOR_VERSION,
            })
            found += 1
            if (i + 1) % 50 == 0:
                print(f"  {brand}: scanned {i+1}/{len(ids)}, paired {found}")
        print(f"{brand}: {found}/{len(ids)} test images have a maskable wordmark")

    with (OUT / "manifest.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # contact sheet: original vs masked, first 8 pairs per brand
    rows, done = [], {}
    for r in records:
        if done.get(r["brand"], 0) >= 8:
            continue
        orig = common.load_base(r["source_id"])
        from PIL import Image
        masked = Image.open(OUT / r["masked_file"])
        rows.append([orig, masked])
        done[r["brand"]] = done.get(r["brand"], 0) + 1
    sheet = common.contact_sheet(rows, ["original", "logo-masked"])
    sheet.save(OUT / "sheet_logo_masked.jpg", quality=90)
    print(f"{len(records)} pairs total; sheet -> sheet_logo_masked.jpg")


if __name__ == "__main__":
    run()
