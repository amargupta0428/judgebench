"""Build the unified test-set index: every exam item with path + labels.

Instruments:
  pos_real          rhode test-split originals            label on-brand
  pos_temporal      rhode temporal-holdout originals      label on-brand (newest campaign)
  neg_competitor    glossier/ilia test originals          label off-brand (cue-present)
  neg_masked        logo_masked_v3 certified pairs        label off-brand (style-only)
  neg_clean         natively clean competitor test imgs   label off-brand (style-only, no artifact)
  corrupt_prog      programmatic corruptions              off-brand, dimension+severity exact
  corrupt_gen       generative corruptions                off-brand, generative family
  dial              LoRA dial sweep                       ordinal brand-ness (adapter scale)

All items are loaded through the SAME loader (common.load_base semantics: RGB,
max side 1024) so no judge can shortcut on resolution or decode-path artifacts.

Output: eval/testset_index.jsonl  {item_id, instrument, path, label, meta}
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))
from testbed.corruptions import common  # noqa: E402

OUT = REPO / "eval" / "testset_index.jsonl"


def main():
    items = []
    splits = json.loads(common.SPLITS.read_text())
    index = splits["_image_index"]

    def corpus_items(split, brands, instrument, label):
        for sid, sp in index.items():
            brand = sid.split(":")[1].split("/")[0]
            if sp == split and brand in brands:
                items.append({"item_id": f"{instrument}:{sid}", "instrument": instrument,
                              "path": str(common.image_path(sid)),
                              "label": label, "meta": {"source_id": sid, "brand": brand}})

    corpus_items("test", {"rhode"}, "pos_real", "on")
    corpus_items("temporal_test", {"rhode"}, "pos_temporal", "on")
    corpus_items("test", {"glossier", "ilia"}, "neg_competitor", "off")

    for r in (REPO / "data/testset/logo_masked_v3/manifest_final.jsonl").read_text().splitlines():
        rec = json.loads(r)
        items.append({"item_id": f"neg_masked:{rec['source_id']}", "instrument": "neg_masked",
                      "path": str(REPO / "data/testset/logo_masked_v3" / rec["masked_file"]),
                      "label": "off",
                      "meta": {"source_id": rec["source_id"], "brand": rec["brand"],
                               "certification": rec["certification"]}})

    # natively-clean list was produced by the v2 inventory build and stayed valid
    clean = json.loads((REPO / "data/testset/logo_masked_v2/natively_clean.json").read_text())
    for sid in clean:
        items.append({"item_id": f"neg_clean:{sid}", "instrument": "neg_clean",
                      "path": str(common.image_path(sid)), "label": "off",
                      "meta": {"source_id": sid, "brand": sid.split(":")[1].split("/")[0]}})

    for dim in ("palette", "composition", "typography"):
        for r in (REPO / f"data/testset/corruptions/manifest_{dim}.jsonl").read_text().splitlines():
            rec = json.loads(r)
            items.append({"item_id": f"corrupt_prog:{rec['out_file']}", "instrument": "corrupt_prog",
                          "path": str(REPO / "data/testset/corruptions" / rec["out_file"]),
                          "label": "off",
                          "meta": {"dimension": rec["dimension"], "corruption": rec["corruption"],
                                   "severity": rec["severity"], "source_id": rec["source_id"]}})

    for r in (REPO / "data/testset/dial/manifest_corrupt.jsonl").read_text().splitlines():
        rec = json.loads(r)
        items.append({"item_id": f"corrupt_gen:{rec['file']}", "instrument": "corrupt_gen",
                      "path": str(REPO / "data/testset/dial/sweep_out" / rec["file"])
                      if (REPO / "data/testset/dial/sweep_out" / rec["file"]).exists()
                      else str(REPO / "data/testset/dial" / rec["file"]),
                      "label": "off",
                      "meta": {"dimension": rec["dimension"], "severity": rec["severity"],
                               "strength": rec["strength"], "source_id": rec["source_id"]}})

    for r in (REPO / "data/testset/dial/manifest_dial.jsonl").read_text().splitlines():
        rec = json.loads(r)
        items.append({"item_id": f"dial:{rec['file']}", "instrument": "dial",
                      "path": str(REPO / "data/testset/dial" / rec["file"]),
                      "label": "ordinal",
                      "meta": {"prompt_idx": rec["prompt_idx"], "seed": rec["seed"],
                               "adapter_scale": rec["adapter_scale"]}})

    missing = [i for i in items if not Path(i["path"]).exists()]
    with OUT.open("w") as f:
        for i in items:
            f.write(json.dumps(i) + "\n")
    import collections
    print(dict(collections.Counter(i["instrument"] for i in items)))
    print(f"{len(items)} items, {len(missing)} missing paths")
    if missing:
        for m in missing[:5]:
            print("  MISSING:", m["path"])


if __name__ == "__main__":
    main()
