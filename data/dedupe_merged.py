"""Cross-source dedupe: phash + union-find over FB ads AND Instagram images.
Keys are "<source>:<brand>/<file>" (source in {fb, ig}). Clustering is within-brand,
across sources — the same creative posted to both platforms SHOULD merge into one
cluster (that's what makes the source probe's content-controlled pairs findable,
and what keeps cross-platform twins from straddling splits).

Outputs: features/clusters_merged.json, features/phashes_merged.json
"""
import json, os, sys
from collections import defaultdict, Counter
from PIL import Image
import imagehash

BASE = os.path.dirname(os.path.abspath(__file__))
ROOTS = {"fb": os.path.join(BASE, "scrape", "raw", "images"),
         "ig": os.path.join(BASE, "scrape", "raw", "ig_images")}
OUT = os.path.join(BASE, "features")
HAMMING_T = 8

def main():
    files, hashes = [], {}
    for src, root in ROOTS.items():
        for brand in sorted(os.listdir(root)):
            bdir = os.path.join(root, brand)
            if not os.path.isdir(bdir): continue
            for f in sorted(os.listdir(bdir)):
                if not f.lower().endswith((".jpg", ".jpeg", ".png")): continue
                p = os.path.join(bdir, f)
                try:
                    h = imagehash.phash(Image.open(p).convert("RGB"))
                except Exception as e:
                    print("skip", p, e, file=sys.stderr); continue
                key = f"{src}:{brand}/{f}"
                files.append(key); hashes[key] = h

    parent = {f: f for f in files}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    by_brand = defaultdict(list)
    for f in files: by_brand[f.split(":")[1].split("/")[0]].append(f)
    for brand, fl in by_brand.items():
        for i in range(len(fl)):
            hi = hashes[fl[i]]
            for j in range(i + 1, len(fl)):
                if hi - hashes[fl[j]] <= HAMMING_T:
                    ra, rb = find(fl[i]), find(fl[j])
                    if ra != rb: parent[ra] = rb

    clusters = defaultdict(list)
    for f in files: clusters[find(f)].append(f)
    cl_list = sorted(clusters.values(), key=len, reverse=True)
    cl_map = {f"m{i:04d}": sorted(m) for i, m in enumerate(cl_list)}
    json.dump(cl_map, open(os.path.join(OUT, "clusters_merged.json"), "w"), indent=1)
    json.dump({f: str(h) for f, h in hashes.items()}, open(os.path.join(OUT, "phashes_merged.json"), "w"), indent=1)

    print(f"{'brand':10s} {'raw':>6s} {'clusters':>9s} {'xplatform':>10s}")
    for brand in sorted(by_brand):
        bcl = [m for m in cl_list if m[0].split(':')[1].startswith(brand + "/")]
        cross = sum(1 for m in bcl if len({x.split(':')[0] for x in m}) > 1)
        print(f"{brand:10s} {len(by_brand[brand]):6d} {len(bcl):9d} {cross:10d}")
    print("total clusters:", len(cl_list))

if __name__ == "__main__":
    main()
