"""Harvest DPO preference pairs from the BoN pool (Phase 2b, SDXL DPO-LoRA arm).

Spec 2b design: pairs drawn top-vs-bottom per prompt from the large BoN pool
with a minimum score-gap filter — mirrors practice, avoids adjacent middling
pairs. One pair set per attacked judge.

Usage: .venv/bin/python testbed/dpo/harvest_pairs.py <judge> [min_gap]
  judge in {siglip_tuned, siglip_tuned_v2, siglip_frozen, qwen_lora, gpt4o}
Output: eval/bon/dpo_pairs_<judge>.jsonl
  {"prompt", "chosen", "rejected", "score_chosen", "score_rejected"}
Pairing: within each prompt, rank by judge score; pair rank-k from the top
with rank-k from the bottom while gap >= min_gap (default 0.15 of the judge's
observed score range). Caps at 20 pairs/prompt -> up to ~800 pairs from the
broad+deep pool.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from eval.bon_curves import load_scores  # noqa: E402  (same artifact readers)

MAX_PER_PROMPT = 20


def main(judge, min_gap_frac=0.15):
    scores = load_scores()[judge]
    manifest = {}
    prompts = {}
    for line in (REPO / "eval/bon/manifest_bon.jsonl").open():
        it = json.loads(line)
        if it["file"] in scores and it["file"] not in manifest:
            manifest[it["file"]] = it
            prompts.setdefault(it["prompt_idx"], []).append(it["file"])

    vals = [scores[f] for f in manifest]
    min_gap = (max(vals) - min(vals)) * min_gap_frac
    out = (REPO / f"eval/bon/dpo_pairs_{judge}.jsonl").open("w")
    n = 0
    for pi, files in sorted(prompts.items()):
        ranked = sorted(files, key=lambda f: scores[f], reverse=True)
        for k in range(min(MAX_PER_PROMPT, len(ranked) // 2)):
            top, bot = ranked[k], ranked[-(k + 1)]
            if scores[top] - scores[bot] < min_gap:
                break
            out.write(json.dumps({
                "prompt": manifest[top]["prompt"],
                "chosen": top, "rejected": bot,
                "score_chosen": round(scores[top], 5),
                "score_rejected": round(scores[bot], 5)}) + "\n")
            n += 1
    print(f"{judge}: {n} pairs (min_gap {min_gap:.4f}) -> dpo_pairs_{judge}.jsonl")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.15)
