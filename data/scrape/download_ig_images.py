"""Download static images from the Instagram scrape (brand-owned posts only).
Image posts: displayUrl. Sidecars (carousels): each child's displayUrl (skip video children).
Videos skipped entirely (Phase 1 static-only). Appends rows to ig_manifest.jsonl.
"""
import json, os, time, hashlib
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "raw", "ig_posts.json")
OUT = os.path.join(BASE, "raw", "ig_images")
MANIFEST = os.path.join(BASE, "raw", "ig_manifest.jsonl")

BRANDS = {"rhode": "rhode", "glossier": "glossier", "iliabeauty": "ilia"}

def urls_for(post):
    t = post.get("type")
    if t == "Image":
        u = post.get("displayUrl")
        return [u] if u else []
    if t == "Sidecar":
        out = []
        for ch in (post.get("childPosts") or []):
            if ch.get("type") == "Image" and ch.get("displayUrl"):
                out.append(ch["displayUrl"])
        return out
    return []  # Video etc.

def main():
    posts = json.load(open(SRC))
    os.makedirs(OUT, exist_ok=True)
    rows, n_ok, n_err = [], 0, 0
    for p in posts:
        brand = BRANDS.get(p.get("ownerUsername") or "")
        if not brand: continue
        pid = p.get("shortCode") or p.get("id") or "unk"
        bdir = os.path.join(OUT, brand); os.makedirs(bdir, exist_ok=True)
        for i, url in enumerate(urls_for(p)):
            fname = f"{pid}_{i}.jpg"
            fpath = os.path.join(bdir, fname)
            try:
                if not os.path.exists(fpath):
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    data = urllib.request.urlopen(req, timeout=30).read()
                    if len(data) < 5000: raise ValueError("too small")
                    open(fpath, "wb").write(data)
                    time.sleep(0.1)
                rows.append({"brand": brand, "post_id": pid, "file": f"{brand}/{fname}",
                             "source": "instagram", "timestamp": p.get("timestamp"),
                             "post_type": p.get("type"),
                             "sha1": hashlib.sha1(open(fpath,'rb').read()).hexdigest()})
                n_ok += 1
            except Exception:
                n_err += 1
    with open(MANIFEST, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    from collections import Counter
    print("downloaded:", n_ok, "errors:", n_err)
    print("by brand:", dict(Counter(r["brand"] for r in rows)))

if __name__ == "__main__":
    main()
