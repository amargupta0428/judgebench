"""Report card v1 — score judges over the unified test-set index.

Metrics per judge:
  AUC            pos_real vs each negative class (competitor / masked / clean)
  det@5%FPR      per corruption dimension x severity (threshold = 5% FPR on pos_real)
  dial Spearman  per (prompt, seed) group, mean +/- sd, % strictly monotone-positive
  temporal AUC   pos_temporal vs neg_competitor (generalization to newest campaign)
  ECE            10-bin, on pos_real + neg_competitor

Scores come from precomputed artifacts (embeddings npz / feature jsonl), so this
harness is deterministic and instant to re-run. All numbers are TEST-split only;
every fit happened on train/val upstream.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))
from judges.j3_frozen import J3Frozen        # noqa: E402
from judges.j1_rules import J1Rules          # noqa: E402


def _ranks(x):
    """Average ranks across ties (scipy.stats.rankdata equivalent).

    Ties must share a rank: plain argsort ranking hands tied items arbitrary
    distinct ranks, which fabricates discrimination for coarse-scored judges
    (a judge scoring everything 3/10 came out AUC 0.98). With average ranks a
    tied pos/neg pair counts 1/2 in the Mann-Whitney AUC and Spearman matches
    the standard tie-corrected definition.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x) + 1)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return ranks


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    pos, neg = np.asarray(pos), np.asarray(neg)
    ranks = _ranks(np.concatenate([pos, neg]))
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(a, b):
    a, b = _ranks(a), _ranks(b)
    if a.std() == 0 or b.std() == 0:
        return 0.0  # a constant scorer shows no evidence of ranking
    return float(np.corrcoef(a, b)[0, 1])


def ece(scores, labels, bins=10):
    scores, labels = np.asarray(scores), np.asarray(labels)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (scores >= lo) & (scores < hi)
        if m.sum():
            e += m.mean() * abs(scores[m].mean() - labels[m].mean())
    return float(e)


def evaluate(scores: dict):
    """scores: item_id -> float. Returns metric dict."""
    items = [json.loads(l) for l in (REPO / "eval/testset_index.jsonl").open()]
    by_inst = defaultdict(list)
    for it in items:
        s = scores.get(it["item_id"])
        if s is not None:
            by_inst[it["instrument"]].append((it, s))

    out = {}
    pos = [s for _, s in by_inst["pos_real"]]
    out["auc_vs_competitor"] = auc(pos, [s for _, s in by_inst["neg_competitor"]])
    out["auc_vs_masked"] = auc(pos, [s for _, s in by_inst["neg_masked"]])
    out["auc_vs_clean"] = auc(pos, [s for _, s in by_inst["neg_clean"]])
    out["auc_temporal_vs_competitor"] = auc([s for _, s in by_inst["pos_temporal"]],
                                            [s for _, s in by_inst["neg_competitor"]])
    # logo-dependence delta: cue-present vs cue-masked discrimination
    out["logo_delta"] = out["auc_vs_competitor"] - out["auc_vs_masked"]

    thr = np.percentile(pos, 5)  # 5% FPR on real positives
    det = {}
    for inst in ("corrupt_prog", "corrupt_gen"):
        for it, s in by_inst[inst]:
            key = f"{it['meta']['dimension']}/s{it['meta']['severity']}"
            det.setdefault(key, []).append(s < thr)
    out["det_at_5fpr"] = {k: float(np.mean(v)) for k, v in sorted(det.items())}

    groups = defaultdict(list)
    for it, s in by_inst["dial"]:
        groups[(it["meta"]["prompt_idx"], it["meta"]["seed"])].append(
            (it["meta"]["adapter_scale"], s))
    rhos = []
    for g in groups.values():
        g.sort()
        rhos.append(spearman([x for x, _ in g], [y for _, y in g]))
    out["dial_spearman_mean"] = float(np.mean(rhos)) if rhos else float("nan")
    out["dial_spearman_sd"] = float(np.std(rhos)) if rhos else float("nan")
    out["dial_frac_positive"] = float(np.mean([r > 0 for r in rhos])) if rhos else float("nan")

    ece_scores = pos + [s for _, s in by_inst["neg_competitor"]]
    ece_labels = [1] * len(pos) + [0] * len(by_inst["neg_competitor"])
    out["ece"] = ece(ece_scores, ece_labels)
    return out


def main():
    results = {}

    # J3-frozen: from test-set embeddings
    j3 = J3Frozen()
    z = np.load(REPO / "data/features/embeddings_testset.npz", allow_pickle=True)
    scores = {f: j3.score_vec(v) for f, v in zip(z["files"], z["vecs"])}
    results[j3.name] = evaluate(scores)

    # J1 rules: from test-set feature file
    feat_path = REPO / "eval/j1_features_testset.jsonl"
    if feat_path.exists():
        j1 = J1Rules()
        scores = {}
        for l in feat_path.open():
            r = json.loads(l)
            scores[r["item_id"]] = j1.score_features(r["features"])
        results[j1.name] = evaluate(scores)

    out = REPO / "eval/results/report_card_v1.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    for name, m in results.items():
        print(f"\n=== {name} ===")
        for k, v in m.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv:.3f}")
            else:
                print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
