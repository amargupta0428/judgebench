"""Score every judge from its precomputed artifacts and rebuild the report card.

One committed, deterministic path from artifacts -> eval/results/report_card_v1.json
(replaces the July 6 ad-hoc merging, which left no reproducible trail for the
tuned-SigLIP fit and the VLM-cache integration).

Judges covered (skipped silently if the artifact is missing):
  j1_rules            eval/j1_features_testset.jsonl
  j3_siglip_frozen    data/features/embeddings_testset.npz
  j3_siglip_tuned     judges/siglip_tuned_out/embeddings_{corpus,testset}_tuned.npz
  j3_siglip_tuned_v2  ..._v2.npz (corruption-negative arm, July 7)
  j2_gpt4o, j2_gemini judges/cache/j2_*/*.json (overall_consistency / 10)
  j2b_qwen_zeroshot   judges/qwen_zs_scores.jsonl
  j3b_qwen_lora       judges/qwen_lora_scores.jsonl (p_yes, already 0..1)

Tuned-SigLIP calibration mirrors J3-frozen exactly: centroid = mean of rhode
TRAIN embeddings, Platt fit on VAL (rhode vs glossier/ilia). Test never touched.

Usage: .venv/bin/python eval/score_judges.py
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))
from eval.report_card import evaluate  # noqa: E402
from judges.j3_frozen import J3Frozen  # noqa: E402
from judges.j1_rules import J1Rules    # noqa: E402

TUNED_OUT = REPO / "judges/siglip_tuned_out"


def _npz(path):
    z = np.load(path, allow_pickle=True)
    return dict(zip((str(f) for f in z["files"]), z["vecs"]))


def fit_and_score_tuned(suffix=""):
    """Centroid (train) + Platt (val) from tuned corpus embeddings; score testset."""
    corpus = _npz(TUNED_OUT / f"embeddings_corpus_tuned{suffix}.npz")
    splits = json.loads((REPO / "data/features/splits_v2.json").read_text())
    index = splits["_image_index"]

    def brand(i): return i.split(":")[1].split("/")[0]
    train_rhode = [corpus[i] for i, sp in index.items()
                   if sp == "train" and brand(i) == "rhode" and i in corpus]
    centroid = np.mean(train_rhode, axis=0)
    centroid /= np.linalg.norm(centroid)

    val_ids = [i for i, sp in index.items() if sp == "val" and i in corpus]
    sims = np.array([float(corpus[i] @ centroid) for i in val_ids])
    y = np.array([1.0 if brand(i) == "rhode" else 0.0 for i in val_ids])
    a, b = 1.0, 0.0  # Platt via Newton on logistic NLL
    for _ in range(200):
        p = 1 / (1 + np.exp(-(a * sims + b)))
        g = np.array([((p - y) * sims).mean(), (p - y).mean()])
        w = p * (1 - p)
        H = np.array([[(w * sims * sims).mean(), (w * sims).mean()],
                      [(w * sims).mean(), w.mean()]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
    params = {"centroid": centroid.tolist(), "platt_a": float(a), "platt_b": float(b)}
    (REPO / f"judges/j3_tuned{suffix}_params.json").write_text(json.dumps(params))

    test = _npz(TUNED_OUT / f"embeddings_testset_tuned{suffix}.npz")
    return {i: float(1 / (1 + np.exp(-(a * (v @ centroid) + b))))
            for i, v in test.items()}


def vlm_cache_scores(judge_dir):
    scores = {}
    for f in (REPO / "judges/cache" / judge_dir).glob("*.json"):
        d = json.loads(f.read_text())
        if "overall_consistency" in d:
            scores[d["item_id"]] = d["overall_consistency"] / 10.0
    return scores


def jsonl_scores(path, field, scale=1.0):
    scores = {}
    for line in (REPO / path).open():
        d = json.loads(line)
        if field in d:
            scores[d["id"]] = d[field] * scale
    return scores


def main():
    results = {}

    z = _npz(REPO / "data/features/embeddings_testset.npz")
    j3 = J3Frozen()
    results["j3_siglip_frozen"] = evaluate({i: j3.score_vec(v) for i, v in z.items()})

    feat = REPO / "eval/j1_features_testset.jsonl"
    if feat.exists():
        j1 = J1Rules()
        results["j1_rules"] = evaluate(
            {json.loads(l)["item_id"]: j1.score_features(json.loads(l)["features"])
             for l in feat.open()})

    for suffix, name in (("", "j3_siglip_tuned"), ("_v2", "j3_siglip_tuned_v2")):
        if (TUNED_OUT / f"embeddings_testset_tuned{suffix}.npz").exists():
            results[name] = evaluate(fit_and_score_tuned(suffix))

    for cache_dir, name in (("j2_gpt4o", "j2_gpt4o"), ("j2_gemini", "j2_gemini")):
        s = vlm_cache_scores(cache_dir)
        if len(s) > 2000:  # only report near-complete runs
            results[name] = evaluate(s)
        elif s:
            print(f"skip {name}: only {len(s)} scored")

    if (REPO / "judges/qwen_zs_scores.jsonl").exists():
        results["j2b_qwen_zeroshot"] = evaluate(
            jsonl_scores("judges/qwen_zs_scores.jsonl", "overall_consistency", 0.1))
    if (REPO / "judges/qwen_lora_scores.jsonl").exists():
        results["j3b_qwen_lora"] = evaluate(
            jsonl_scores("judges/qwen_lora_scores.jsonl", "p_yes"))

    out = REPO / "eval/results/report_card_v1.json"
    out.write_text(json.dumps(results, indent=1))
    for name, m in results.items():
        det = m.get("det_at_5fpr", {})
        mean_det = np.mean(list(det.values())) if det else float("nan")
        print(f"{name}: auc_comp={m['auc_vs_competitor']:.3f} "
              f"auc_masked={m['auc_vs_masked']:.3f} logo_d={m['logo_delta']:.3f} "
              f"mean_det@5fpr={mean_det:.3f} dial_rho={m['dial_spearman_mean']:.3f}")


if __name__ == "__main__":
    main()
