"""Probe B (source/platform forensics), stage-1 upper bound — July 12.

Question: does the judge's representation space (frozen SigLIP) separate
Ad Library (fb) from Instagram (ig) images WITHIN a brand? If yes, part of
any brand signal could be pipeline artifacts (compression, color grading,
ad-workflow signatures) rather than creative identity.

Design (pre-registered in RIGOR_LOG.md, July 12):
- Per brand: logistic regression fb-vs-ig on frozen SigLIP embeddings.
- Cluster-aware: reuses splits_v2 train/test cluster membership, so twin
  creatives never straddle the boundary. val clusters folded into train.
- UPPER BOUND reading: content differences between ad and organic posts are
  confounded IN. accuracy <= ~0.60 dismisses the artifact confound;
  above that, only the matched-pair gold subset can attribute.
- Gold check (free, small-n): the mixed-source hybrid clusters (same
  creative on both platforms). Within-cluster fb-vs-ig separation with
  content held constant = pure pipeline-artifact signal. Scored with the
  brand's test-trained model on mixed clusters NOT in its train split.

Usage: .venv/bin/python eval/audit/probe_b_source.py
Output: eval/results/probe_b_source.json + stdout table.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
FEAT = REPO / "data/features"

fb = np.load(FEAT / "embeddings_fb.npz", allow_pickle=True)
ig = np.load(FEAT / "embeddings_ig.npz", allow_pickle=True)

emb = {}
for src, npz in (("fb", fb), ("ig", ig)):
    for f, v in zip(npz["files"], npz["vecs"]):
        emb[f"{src}:{f}"] = v / np.linalg.norm(v)

clusters = json.load(open(FEAT / "clusters_hybrid.json"))
img2cluster = {m: cid for cid, ms in clusters.items() for m in ms}
splits = json.loads((FEAT / "splits_v2.json").read_text())

# cluster -> split (train+val -> "train"; test / temporal_test -> "test")
cid_split = {}
for sname, target in (("train", "train"), ("val", "train"),
                      ("test", "test"), ("temporal_test", "test")):
    for brand, cids in splits[sname].items():
        for cid in cids:
            cid_split[cid] = target

mixed_cids = {cid for cid, ms in clusters.items()
              if len({m.split(":", 1)[0] for m in ms}) > 1}

rng = np.random.default_rng(0)
results = {}
for brand in ("rhode", "glossier", "ilia"):
    rows = {"train": [], "test": []}
    for iid, v in emb.items():
        src, rel = iid.split(":", 1)
        if rel.split("/")[0] != brand:
            continue
        cid = img2cluster.get(iid)
        sp = cid_split.get(cid)
        if sp is None:
            continue
        rows[sp].append((iid, cid, src, v))

    def xy(items):
        X = np.stack([r[3] for r in items])
        y = np.array([1 if r[2] == "fb" else 0 for r in items])
        return X, y

    Xtr, ytr = xy(rows["train"])
    Xte, yte = xy(rows["test"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    bal = balanced_accuracy_score(yte, p > 0.5)
    auc = roc_auc_score(yte, p)

    # permutation floor: shuffle labels, same pipeline (5 reps)
    perm = []
    for _ in range(5):
        ys = rng.permutation(ytr)
        c2 = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        c2.fit(Xtr, ys)
        perm.append(balanced_accuracy_score(yte, c2.predict_proba(Xte)[:, 1] > 0.5))

    # gold check: mixed clusters not used in this brand's training
    gold_items = []
    for cid in mixed_cids:
        if cid_split.get(cid) != "test":
            continue
        ms = clusters[cid]
        if ms[0].split(":", 1)[1].split("/")[0] != brand:
            continue
        gold_items += [(m, cid, m.split(":", 1)[0], emb[m]) for m in ms if m in emb]
    gold = None
    if gold_items and len({g[2] for g in gold_items}) == 2:
        Xg, yg = xy(gold_items)
        pg = clf.predict_proba(Xg)[:, 1]
        gold = {"n": len(gold_items), "n_clusters": len({g[1] for g in gold_items}),
                "auc": round(float(roc_auc_score(yg, pg)), 4),
                "bal_acc": round(float(balanced_accuracy_score(yg, pg > 0.5)), 4)}

    results[brand] = {
        "n_train": len(rows["train"]), "n_test": len(rows["test"]),
        "test_fb": int(yte.sum()), "test_ig": int(len(yte) - yte.sum()),
        "balanced_acc": round(float(bal), 4), "auc": round(float(auc), 4),
        "perm_floor_mean": round(float(np.mean(perm)), 4),
        "gold_mixed_clusters": gold,
    }
    print(f"{brand:9s} bal_acc {bal:.3f}  auc {auc:.3f}  "
          f"(perm floor {np.mean(perm):.3f})  "
          f"test n={len(yte)} (fb {int(yte.sum())}/ig {int(len(yte)-yte.sum())})  "
          f"gold={gold}")

out = REPO / "eval/results/probe_b_source.json"
out.write_text(json.dumps(results, indent=2))
print("wrote", out)
