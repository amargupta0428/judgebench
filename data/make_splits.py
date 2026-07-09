"""Leakage-safe splits: assignment happens BY CLUSTER (never by image), plus a
temporal holdout of the newest rhode clusters (new-product-line generalization test).

Per PHASE1_BUILD.md:
- cluster date = earliest ad start_date among members (campaign launch proxy)
- rhode: newest ~15% of clusters by date -> split "temporal_test" (excluded from all training)
- remaining clusters per brand: 60/15/25 train/val/test, seeded RNG
Outputs data/features/splits.json: {split: {brand: [cluster_ids]}} + per-image index.
"""
import json, os, random
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.join(BASE, "features")
clusters = json.load(open(os.path.join(FEAT, "clusters.json")))
manifest = {}
with open(os.path.join(BASE, "scrape", "raw", "manifest.jsonl")) as f:
    for line in f:
        r = json.loads(line)
        manifest[r["file"]] = r

def cluster_brand(members): return members[0].split("/")[0]
def cluster_date(members):
    ds = [manifest[m]["start_date"] for m in members if m in manifest and manifest[m].get("start_date")]
    return min(ds) if ds else 0

random.seed(42)
info = {cid: {"brand": cluster_brand(m), "date": cluster_date(m), "n": len(m)} for cid, m in clusters.items()}

splits = defaultdict(lambda: defaultdict(list))

# temporal holdout: newest 15% of rhode clusters
rhode = sorted((c for c in info if info[c]["brand"] == "rhode"), key=lambda c: info[c]["date"])
n_hold = max(1, int(0.15 * len(rhode)))
temporal = set(rhode[-n_hold:])
for c in temporal: splits["temporal_test"]["rhode"].append(c)

for brand in sorted({v["brand"] for v in info.values()}):
    pool = [c for c in info if info[c]["brand"] == brand and c not in temporal]
    random.shuffle(pool)
    n = len(pool)
    tr, va = int(0.60 * n), int(0.75 * n)
    for c in pool[:tr]: splits["train"][brand].append(c)
    for c in pool[tr:va]: splits["val"][brand].append(c)
    for c in pool[va:]: splits["test"][brand].append(c)

out = {s: dict(b) for s, b in splits.items()}
# per-image index for convenience
img_index = {}
for s, bmap in out.items():
    for brand, cids in bmap.items():
        for cid in cids:
            for m in clusters[cid]:
                img_index[m] = s
out["_image_index"] = img_index
json.dump(out, open(os.path.join(FEAT, "splits.json"), "w"), indent=1)

print(f"{'split':14s} " + " ".join(f"{b:>9s}" for b in ['glossier','ilia','rhode']))
for s in ["train", "val", "test", "temporal_test"]:
    row = out.get(s, {})
    print(f"{s:14s} " + " ".join(f"{len(row.get(b, [])):9d}" for b in ['glossier','ilia','rhode']))
import datetime
td = [info[c]['date'] for c in temporal if info[c]['date']]
if td:
    print("\ntemporal holdout date range:",
          datetime.date.fromtimestamp(min(td)), "→", datetime.date.fromtimestamp(max(td)))
