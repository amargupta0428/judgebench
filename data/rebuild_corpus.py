"""Hybrid re-clustering + stratified splits v2 (post-audit fix).

Audit finding: phash alone under-merges crops (near-twins straddled splits).
Fix: merge images if phash<=8 OR embedding cosine >= 0.95 (within brand).
Splits v2: per (brand x dominant_source) pools, 60/15/25 by cluster,
plus rhode temporal holdout (newest 15% of clusters by earliest ad/post date).
"""
import json, os, datetime
import numpy as np
import imagehash
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.join(BASE, "features")

def load_emb(tag, src):
    z = np.load(os.path.join(FEAT, f"embeddings_{tag}.npz"), allow_pickle=True)
    return {f"{src}:{f}": v for f, v in zip(list(z["files"]), z["vecs"])}

emb = load_emb("fb", "fb"); emb.update(load_emb("ig", "ig"))
ph = {k: imagehash.hex_to_hash(v) for k, v in json.load(open(os.path.join(FEAT, "phashes_merged.json"))).items()}
files = [f for f in ph if f in emb]
print("images with phash+embedding:", len(files))

# dates
dates = {}
for line in open(os.path.join(BASE, "scrape", "raw", "manifest.jsonl")):
    r = json.loads(line)
    if r.get("start_date"): dates["fb:" + r["file"]] = int(r["start_date"])
for line in open(os.path.join(BASE, "scrape", "raw", "ig_manifest.jsonl")):
    r = json.loads(line)
    ts = r.get("timestamp")
    if ts:
        try: dates["ig:" + r["file"]] = int(datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
        except Exception: pass

parent = {f: f for f in files}
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb

by_brand = defaultdict(list)
for f in files: by_brand[f.split(":")[1].split("/")[0]].append(f)
for brand, fl in by_brand.items():
    V = np.stack([emb[f] for f in fl])
    S = V @ V.T
    n = len(fl)
    for i in range(n):
        hi = ph[fl[i]]
        for j in range(i + 1, n):
            if S[i, j] >= 0.95 or (hi - ph[fl[j]]) <= 8:
                union(fl[i], fl[j])

clusters = defaultdict(list)
for f in files: clusters[find(f)].append(f)
cl_list = sorted(clusters.values(), key=len, reverse=True)
cl_map = {f"h{i:04d}": sorted(m) for i, m in enumerate(cl_list)}
json.dump(cl_map, open(os.path.join(FEAT, "clusters_hybrid.json"), "w"), indent=1)

info = {}
for cid, m in cl_map.items():
    srcs = [x.split(":")[0] for x in m]
    dom = "fb" if srcs.count("fb") >= srcs.count("ig") else "ig"
    ds = [dates[x] for x in m if x in dates]
    info[cid] = {"brand": m[0].split(":")[1].split("/")[0], "src": dom, "date": min(ds) if ds else 0, "n": len(m)}

import random
random.seed(42)
splits = defaultdict(lambda: defaultdict(list))
rhode = sorted((c for c in info if info[c]["brand"] == "rhode"), key=lambda c: info[c]["date"])
n_hold = max(1, int(0.15 * len(rhode)))
temporal = set(rhode[-n_hold:])
for c in temporal: splits["temporal_test"]["rhode"].append(c)

pools = defaultdict(list)
for c, v in info.items():
    if c in temporal: continue
    pools[(v["brand"], v["src"])].append(c)
for (brand, src), pool in sorted(pools.items()):
    random.shuffle(pool)
    n = len(pool); tr, va = int(0.60 * n), int(0.75 * n)
    for c in pool[:tr]: splits["train"][brand].append(c)
    for c in pool[tr:va]: splits["val"][brand].append(c)
    for c in pool[va:]: splits["test"][brand].append(c)

out = {s: dict(b) for s, b in splits.items()}
img_index = {}
for s, bmap in out.items():
    for brand, cids in bmap.items():
        for cid in cids:
            for m in cl_map[cid]: img_index[m] = s
out["_image_index"] = img_index
json.dump(out, open(os.path.join(FEAT, "splits_v2.json"), "w"), indent=1)

print(f"hybrid clusters: {len(cl_map)} (was 2755 phash-only)")
for brand in sorted(by_brand):
    bc = [c for c, v in info.items() if v["brand"] == brand]
    print(f"  {brand}: {len(bc)} clusters")
for s in ["train", "val", "test", "temporal_test"]:
    row = out.get(s, {})
    print(f"{s:14s} " + " ".join(f"{b}:{len(row.get(b, []))}" for b in sorted(by_brand)))
td = [info[c]["date"] for c in temporal if info[c]["date"]]
if td: print("temporal holdout:", datetime.date.fromtimestamp(min(td)), "->", datetime.date.fromtimestamp(max(td)))
