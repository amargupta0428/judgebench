"""Dual-rater audit contact sheet.
Section 1: 15 sampled multi-image clusters -> row of member thumbnails.
   Question per row: same creative family? (grouping sanity)
Section 2: 15 sampled FB test images -> the image + its top-5 nearest TRAIN neighbors.
   Checklist per row: same product? same person? same set/background? verdict: style vs shoot.
Outputs: eval/audit/contact_sheet.html (for Amar) + eval/audit/panels/*.png composites (for Claude's independent pass).
"""
import base64, io, json, os, random
import numpy as np
from PIL import Image, ImageDraw

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE)
IMG = os.path.join(REPO, "data", "scrape", "raw", "images")
AUD = os.path.join(BASE, "audit"); PAN = os.path.join(AUD, "panels")
os.makedirs(PAN, exist_ok=True)
random.seed(7)

TH = 170  # thumbnail size

def thumb(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((TH, TH))
    return im

def b64(im):
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()

def row_composite(paths, labels, out_png):
    ims = [thumb(p) for p in paths]
    W = sum(i.width for i in ims) + 8 * (len(ims) + 1)
    H = max(i.height for i in ims) + 30
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    x = 8
    for im, lab in zip(ims, labels):
        canvas.paste(im, (x, 24))
        d.text((x, 6), lab, fill="black")
        x += im.width + 8
    canvas.save(out_png)
    return ims

clusters = json.load(open(os.path.join(REPO, "data", "features", "clusters.json")))
splits = json.load(open(os.path.join(REPO, "data", "features", "splits.json")))["_image_index"]
z = np.load(os.path.join(REPO, "data", "features", "embeddings_fb.npz"), allow_pickle=True)
files, vecs = list(z["files"]), z["vecs"]
idx = {f: i for i, f in enumerate(files)}

html = ["<meta charset='utf-8'><style>body{font-family:sans-serif;max-width:1200px;margin:20px auto}"
        ".row{border:1px solid #ccc;margin:14px 0;padding:10px}img{margin:3px}"
        "h3{margin:6px 0}.q{color:#0a5;font-weight:bold}</style>",
        "<h1>judgebench audit sheet</h1>",
        "<p><b>Section 1 — clustering sanity.</b> Each row = one 'creative family' our dedupe grouped. "
        "Question: <span class='q'>do these belong together (variants of one creative)? YES / NO / UNSURE</span></p>"]

multi = [(cid, m) for cid, m in clusters.items() if len(m) >= 2]
for k, (cid, members) in enumerate(random.sample(multi, min(15, len(multi)))):
    sel = members[:6]
    paths = [os.path.join(IMG, m) for m in sel]
    row_composite(paths, [m.split("/")[0] for m in sel], os.path.join(PAN, f"cluster_{k:02d}.png"))
    html.append(f"<div class='row'><h3>Cluster {k+1} ({cid}, {len(members)} images)</h3>")
    for p in paths: html.append(f"<img src='data:image/jpeg;base64,{b64(thumb(p))}'>")
    html.append("</div>")

html.append("<p><b>Section 2 — neighbor audit.</b> First image = TEST image (green border); next 5 = its most "
            "similar TRAINING images per SigLIP. Checklist: <span class='q'>same product? same person? "
            "same set/background? Verdict: SHOOT (same production) / STYLE (same brand feel, different production) / MIXED</span></p>")

train = [f for f in files if splits.get(f) == "train"]
test = [f for f in files if splits.get(f) == "test"]
TR = vecs[[idx[f] for f in train]]
sample = random.sample(test, min(15, len(test)))
for k, f in enumerate(sample):
    sims = TR @ vecs[idx[f]]
    nn = [train[j] for j in np.argsort(-sims)[:5]]
    paths = [os.path.join(IMG, x) for x in [f] + nn]
    labs = ["TEST " + f.split("/")[0]] + [f"nn{i+1} {x.split('/')[0]} ({sims[np.argsort(-sims)[i]]:.2f})" for i, x in enumerate(nn)]
    row_composite(paths, labs, os.path.join(PAN, f"nn_{k:02d}.png"))
    html.append(f"<div class='row'><h3>Neighbor panel {k+1} — test: {f}</h3>")
    html.append(f"<img style='border:3px solid #0a5' src='data:image/jpeg;base64,{b64(thumb(paths[0]))}'>")
    for p in paths[1:]: html.append(f"<img src='data:image/jpeg;base64,{b64(thumb(p))}'>")
    html.append("</div>")

open(os.path.join(AUD, "contact_sheet.html"), "w").write("\n".join(html))
print("wrote", os.path.join(AUD, "contact_sheet.html"), "and", len(os.listdir(PAN)), "panels")
