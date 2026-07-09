"""Corruption negatives for SigLIP-tuned-v2 training — from TRAIN-split bases only.

Answers the spec deviation caught July 7 (PHASE1_BUILD §3 asked for corruption
negatives in J3 training; v1 shipped without them because test-base corruptions
would leak). Fix: generate a SEPARATE corruption set from train-split rhode
bases. Family holdout for the generalization test:

  trained-on families : palette, typography      (programmatic, train bases)
  held-out family     : composition              (programmatic, never trained)
  naturally unseen    : styling, mood            (generative, test-only)

Same generators, same severity convention, same eligibility gates as the test
set. Output dir is disjoint from the test corruptions; the report card never
touches these images.

Usage: .venv/bin/python -m testbed.corruptions.train_negatives
"""
import json
import random
import sys

from . import common, palette, typography

# redirect all outputs away from the test corruption tree
common.OUT_DIR = common.REPO / "data" / "testset" / "train_neg_corruptions"

GLOBAL_SEED = 20260707
N_PER_CELL = 45  # bases per generator; x3 severities x6 generators ~= 800 images
TRAIN_FAMILIES = ("palette", "typography")


def _img_seed(image_id: str, gen_name: str) -> int:
    return random.Random(f"{GLOBAL_SEED}:{gen_name}:{image_id}").randint(0, 2**31)


def run():
    ids = common.rhode_test_ids("train")
    rng = random.Random(GLOBAL_SEED)
    rng.shuffle(ids)
    print(f"{len(ids)} rhode TRAIN-split bases available")

    modules = {"palette": palette, "typography": typography}
    skip_log = []

    for dim in TRAIN_FAMILIES:
        mod = modules[dim]
        words_cache = {}
        if dim == "typography":
            pool = []
            for i, image_id in enumerate(ids):
                img = common.load_base(image_id)
                words = typography.ocr_words(img)
                if typography.find_wordmarks(words):
                    pool.append(image_id)
                    words_cache[image_id] = words
                if (i + 1) % 100 == 0:
                    print(f"  ocr scan {i+1}/{len(ids)}, eligible: {len(pool)}", flush=True)
                if len(pool) >= N_PER_CELL * len(mod.GENERATORS) * 2:
                    break  # enough eligible bases; no need to scan the rest
            print(f"typography: {len(pool)} eligible bases")
        else:
            pool = ids

        for gen_name, gen in mod.GENERATORS.items():
            records, used = [], 0
            for image_id in pool:
                if used >= N_PER_CELL:
                    break
                base = common.load_base(image_id)
                seed = _img_seed(image_id, gen_name)
                per_sev, ok = {}, True
                for sev in common.SEVERITIES:
                    g_rng = random.Random(seed)
                    kwargs = {"words": words_cache[image_id]} if dim == "typography" else {}
                    out, params = gen(base, sev, g_rng, **kwargs)
                    if dim == "palette":
                        de = palette.mean_delta_e(base, out)
                        params["mean_delta_e"] = round(de, 2)
                        if sev == 1 and de < palette.DELTA_E_FLOOR:
                            skip_log.append({"image": image_id, "generator": gen_name,
                                             "reason": f"s1 mean_dE {de:.2f} < floor"})
                            ok = False
                            break
                        if gen_name == "brand_color_remap" and \
                                params["pixels_remapped_frac"] < 0.03:
                            skip_log.append({"image": image_id, "generator": gen_name,
                                             "reason": "remap frac < 0.03"})
                            ok = False
                            break
                    per_sev[sev] = (out, params)
                if not ok:
                    continue
                for sev, (out, params) in per_sev.items():
                    rec = common.CorruptionRecord(
                        out_file=common.out_name(image_id, gen_name, sev),
                        source_id=image_id, dimension=dim, corruption=gen_name,
                        severity=sev, params=params, seed=seed,
                        notes={"purpose": "train_negative_v2"})
                    records.append(common.write_output(out, rec))
                used += 1
            common.append_manifest(records, f"manifest_{dim}.jsonl")
            print(f"{dim}/{gen_name}: {used} bases -> {len(records)} images", flush=True)

    (common.OUT_DIR / "skip_log.json").write_text(json.dumps(skip_log, indent=2))
    print(f"DONE. skips: {len(skip_log)}")


if __name__ == "__main__":
    run()
