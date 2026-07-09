"""Extract static creative image URLs from the FB Ads Library scrape and download them.
Singles (displayFormat=IMAGE) use snapshot.images[]; carousels (DCO) use snapshot.cards[].
Videos and formats without stills are skipped (Phase 1 is static-only).
Output: raw/images/<brand>/<adArchiveId>_<idx>.jpg + manifest.jsonl with per-image metadata.
"""
import json, os, re, sys, time, hashlib
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "raw", "fb_ads_all.json")
OUT = os.path.join(BASE, "raw", "images")
MANIFEST = os.path.join(BASE, "raw", "manifest.jsonl")

BRAND_MAP = {"rhode": "rhode", "Glossier": "glossier", "ILIA Beauty": "ilia"}

def img_urls(snap):
    urls = []
    for im in (snap.get("images") or []):
        u = im.get("originalImageUrl") or im.get("resizedImageUrl")
        if u: urls.append(u)
    for cd in (snap.get("cards") or []):
        u = cd.get("originalImageUrl") or cd.get("resizedImageUrl")
        if u: urls.append(u)
    return urls

def main():
    ads = json.load(open(SRC))
    os.makedirs(OUT, exist_ok=True)
    seen_urls, n_ok, n_err, rows = set(), 0, 0, []
    for ad in ads:
        brand = BRAND_MAP.get(ad.get("pageName") or "")
        if not brand: continue
        snap = ad.get("snapshot") or {}
        adid = str(ad.get("adArchiveId") or ad.get("adArchiveID") or "unknown")
        bdir = os.path.join(OUT, brand); os.makedirs(bdir, exist_ok=True)
        for i, url in enumerate(img_urls(snap)):
            key = url.split("?")[0]
            if key in seen_urls: continue
            seen_urls.add(key)
            fname = f"{adid}_{i}.jpg"
            fpath = os.path.join(bdir, fname)
            try:
                if not os.path.exists(fpath):
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    data = urllib.request.urlopen(req, timeout=30).read()
                    if len(data) < 5000: raise ValueError("too small, likely placeholder")
                    open(fpath, "wb").write(data)
                    time.sleep(0.15)
                rows.append({"brand": brand, "ad_id": adid, "file": f"{brand}/{fname}",
                             "url": url, "start_date": ad.get("startDate"),
                             "display_format": snap.get("displayFormat"),
                             "is_active": ad.get("isActive"),
                             "sha1": hashlib.sha1(open(fpath,'rb').read()).hexdigest()})
                n_ok += 1
            except Exception as e:
                n_err += 1
    with open(MANIFEST, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    from collections import Counter
    print("downloaded:", n_ok, "errors:", n_err)
    print("by brand:", dict(Counter(r["brand"] for r in rows)))

if __name__ == "__main__":
    main()
