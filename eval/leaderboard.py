"""Exploitability leaderboard — synthesize all Phase 2 pressure arms (Phase 3).

Rows = judges. Columns = pressure modes, each cell a gap metric where higher
= more exploitable (proxy rose but independent quality didn't):

  BoN-selection   deep-arm gold decline from its peak (peak_gold - final_gold);
                  a positive value is a Gao-style inverted-U (hacking under
                  selection). Also flags judges whose gold goes negative.
  SRPO-gradient   (attacked-judge brand delta) minus (mean independent-judge
                  brand delta) on the SRPO eval set — the reward-hack gap.
  DPO-preference  same gap on the DPO eval set (final checkpoint).

Independent = judges NOT sharing the attacked judge's family (siglip/qwen/api).
Reads eval/results/bon_curves.json, eval/srpo/siglip_scores.json +
judges/cache/j2_gpt4o_srpoeval, and the DPO equivalents when present.

Usage: .venv/bin/python eval/leaderboard.py
"""
import json
import glob
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FAMILY = {"siglip_frozen": "siglip", "siglip_tuned": "siglip",
          "siglip_tuned_v2": "siglip", "siglip_tuned_v3": "siglip",
          "qwen_zs": "qwen", "qwen_lora": "qwen", "gpt4o": "api"}


def bon_gaps():
    c = json.loads((REPO / "eval/results/bon_curves.json").read_text())
    out = {}
    for j, cur in c.items():
        arm = cur["arms"].get("deep") or cur["arms"].get("broad")
        g = arm["gold"]
        peak = max(g)
        out[j] = {"bon_peak_minus_final": round(peak - g[-1], 3),
                  "bon_gold_min": round(min(g), 3)}
    return out


def _vlm_eval_deltas(cache_dir):
    from collections import defaultdict
    agg = defaultdict(list)
    for f in glob.glob(str(REPO / "judges/cache" / cache_dir / "*.json")):
        d = json.loads(Path(f).read_text())
        if "overall_consistency" not in d:
            continue
        model, kind, _ = d["item_id"].split("/")
        agg[(model, kind)].append(d["overall_consistency"] / 10)
    return {kind: np.mean(agg[("tuned", kind)]) - np.mean(agg[("base", kind)])
            for kind in ("brand", "control") if agg[("tuned", kind)]}


def _vlm_eval_deltas_prefix(cache_dir, tuned_prefix):
    from collections import defaultdict
    agg = defaultdict(list)
    for f in glob.glob(str(REPO / "judges/cache" / cache_dir / "*.json")):
        d = json.loads(Path(f).read_text())
        if "overall_consistency" not in d:
            continue
        model, kind, _ = d["item_id"].split("/")
        agg[(model, kind)].append(d["overall_consistency"] / 10)
    out = {}
    for kind in ("brand", "control"):
        if agg[(tuned_prefix, kind)] and agg[("base", kind)]:
            out[kind] = np.mean(agg[(tuned_prefix, kind)]) - np.mean(agg[("base", kind)])
    return out


def attack_gap(siglip_scores_path, gpt_cache, attacked="siglip_tuned",
               tuned_prefix="tuned"):
    """reward-hack gap = attacked-judge brand delta - mean independent delta.
    tuned_prefix selects the post-attack model dir (SRPO 'tuned';
    DPO 'checkpoint-750')."""
    s = json.loads(Path(siglip_scores_path).read_text())
    deltas, control = {}, {}
    for j, sc in s.items():
        b = np.mean([v for k, v in sc.items() if k.startswith("base/brand/")])
        t = np.mean([v for k, v in sc.items() if k.startswith(f"{tuned_prefix}/brand/")])
        deltas[j] = t - b
        cb = np.mean([v for k, v in sc.items() if k.startswith("base/control/")])
        ct = np.mean([v for k, v in sc.items() if k.startswith(f"{tuned_prefix}/control/")])
        control[j] = ct - cb
    gpt = _vlm_eval_deltas_prefix(gpt_cache, tuned_prefix)
    if gpt:
        deltas["gpt4o"] = gpt["brand"]
        control["gpt4o"] = gpt.get("control", 0.0)
    fam = FAMILY[attacked]
    indep = [d for j, d in deltas.items() if FAMILY[j] != fam]
    return {"attacked_delta": round(deltas[attacked], 3),
            "independent_mean": round(float(np.mean(indep)), 3),
            "hack_gap": round(deltas[attacked] - float(np.mean(indep)), 3),
            "control_leak_attacked": round(control[attacked], 3)}


def main():
    board = {j: {} for j in FAMILY}
    for j, g in bon_gaps().items():
        board[j].update(g)

    srpo_sig = REPO / "eval/srpo/siglip_scores.json"
    if srpo_sig.exists():
        gap = attack_gap(srpo_sig, "j2_gpt4o_srpoeval")
        board["siglip_tuned"]["srpo"] = gap

    dpo_sig = REPO / "eval/dpo/siglip_scores.json"
    if dpo_sig.exists():
        board["siglip_tuned"]["dpo"] = attack_gap(
            dpo_sig, "j2_gpt4o_dpoeval", tuned_prefix="checkpoint-750")

    # DPO arm attacked via GPT-4o preferences (twin of the SigLIP-DPO arm).
    # Attacked judge = gpt4o (scores from the j2 cache); independent panel =
    # the SigLIP judges scored over the same images.
    dpo_gpt = REPO / "eval/dpo_gpt4o/siglip_scores.json"
    if dpo_gpt.exists():
        gap = attack_gap(dpo_gpt, "j2_gpt4o_dpogpt4oeval", attacked="gpt4o",
                         tuned_prefix="checkpoint-750")
        board["gpt4o"]["dpo"] = gap
        (REPO / "eval/results/dpo_gpt4o.json").write_text(
            json.dumps(gap, indent=1))

    (REPO / "eval/results/leaderboard.json").write_text(json.dumps(board, indent=1))
    print(f"{'judge':17}{'BoN peak-final':>15}{'BoN gold-min':>14}"
          f"{'SRPO hack-gap':>15}{'DPO hack-gap':>14}")
    for j, r in board.items():
        srpo = r.get("srpo", {}).get("hack_gap", "-")
        dpo = r.get("dpo", {}).get("hack_gap", "-")
        print(f"{j:17}{str(r.get('bon_peak_minus_final','-')):>15}"
              f"{str(r.get('bon_gold_min','-')):>14}{str(srpo):>15}{str(dpo):>14}")
    if "srpo" in board["siglip_tuned"]:
        g = board["siglip_tuned"]["srpo"]
        print(f"\nSRPO detail: attacked judge {g['attacked_delta']:+.2f} vs "
              f"independent {g['independent_mean']:+.2f} -> gap {g['hack_gap']:+.2f}")


if __name__ == "__main__":
    main()
