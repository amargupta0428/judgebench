"""Perceptual-hash dedupe + near-duplicate clustering over the scraped ad creatives.

Method: phash (64-bit perceptual hash) per image; union-find clustering with a
Hamming-distance threshold. Near-twins (crops/variants of one creative) land in one
cluster; all downstream train/val/test splits happen BY CLUSTER so twins never
straddle a split boundary (leakage control, see PHASE1_BUILD.md).

Outputs: data/features/clusters.json  (cluster_id -> [image files])
         data/features/phashes.json   (file -> hex phash)
         prints per-brand: raw count, exact dupes, clusters (=effective creatives),
         cluster size distribution.
"""
import json, os, sys
from collections import defaultdict
from PIL import Image
import imagehash

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "scrape", "raw", "images")
OUT = os.path.join(BASE, "features")
os.makedirs(OUT, exist_ok=True)

HAMMING_T = 8  # <= this distance = same creative family (64-bit phash; 8 is conservative-loose)

def main():
    files, hashes = [], {}
    for brand in sorted(os.listdir(IMG)):
        bdir = os.path.join(IMG, brand)
        if not os.path.isdir(bdir): continue
        for f in sorted(os.listdir(bdir)):
            p = os.path.join(bdir, f)
            try:
                h = imagehash.phash(Image.open(p).convert("RGB"))
                key = f"{brand}/{f}"
                files.append(key); hashes[key] = h
            except Exception as e:
                print("skip", f, e, file=sys.stderr)

    # union-find
    parent = {f: f for f in files}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb

    # bucket by brand to avoid cross-brand merging (same product shots across brands shouldn't merge anyway, but be safe)
    by_brand = defaultdict(list)
    for f in files: by_brand[f.split("/")[0]].append(f)
    for brand, fl in by_brand.items():
        for i in range(len(fl)):
            for j in range(i + 1, len(fl)):
                if hashes[fl[i]] - hashes[fl[j]] <= HAMMING_T:
                    union(fl[i], fl[j])

    clusters = defaultdict(list)
    for f in files: clusters[find(f)].append(f)
    cl_list = sorted(clusters.values(), key=len, reverse=True)
    cl_map = {f"c{i:04d}": sorted(members) for i, members in enumerate(cl_list)}

    json.dump(cl_map, open(os.path.join(OUT, "clusters.json"), "w"), indent=1)
    json.dump({f: str(h) for f, h in hashes.items()}, open(os.path.join(OUT, "phashes.json"), "w"), indent=1)

    from collections import Counter
    print(f"{'brand':10s} {'raw':>5s} {'clusters':>9s} {'largest':>8s} {'singletons':>10s}")
    for brand in sorted(by_brand):
        bcl = [m for m in cl_list if m[0].startswith(brand + "/")]
        sizes = Counter(len(m) for m in bcl)
        print(f"{brand:10s} {len(by_brand[brand]):5d} {len(bcl):9d} {max((len(m) for m in bcl), default=0):8d} {sizes.get(1,0):10d}")
    print("\ntotal clusters (= effective unique creatives):", len(cl_list))

if __name__ == "__main__":
    main()
