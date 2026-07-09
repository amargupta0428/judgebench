"""Hack gallery — the visual centerpiece (Phase 3).

Curates, per attack arm, the images that most expose the attacked judge:
high attacked-judge score, low independent panel. Self-contained HTML with
base64-embedded thumbnails (no external deps, opens anywhere). 100% local:
reads committed images + score artifacts, no GPU, no API.

Exhibits:
  SRPO control-leak : tuned/control images the attacked judge scored highest
                      (reward-hacked dogs/cars — the goblin control)
  SRPO brand        : tuned/brand highest attacked-judge, with GPT-4o's score
                      alongside (independent disagreement)
  DPO brand         : checkpoint-750 highest attacked-judge vs frozen
  BoN               : per-prompt N=max winners under siglip_tuned vs frozen gold

Usage: .venv/bin/python eval/build_hack_gallery.py
Output: docs/hack_gallery.html
"""
import base64
import glob
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/hack_gallery.html"
THUMB = 320


def thumb_b64(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((THUMB, THUMB))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def gpt_scores(cache_dir):
    s = {}
    for f in glob.glob(str(REPO / "judges/cache" / cache_dir / "*.json")):
        d = json.loads(Path(f).read_text())
        if "overall_consistency" in d:
            s[d["item_id"]] = d["overall_consistency"] / 10
    return s


def card(img_path, labels):
    b = thumb_b64(img_path)
    rows = "".join(f"<div class='s'><span>{k}</span><b>{v}</b></div>"
                   for k, v in labels.items())
    return (f"<figure><img src='data:image/jpeg;base64,{b}'/>"
            f"<figcaption>{rows}</figcaption></figure>")


def section(title, blurb, cards):
    return (f"<section><h2>{title}</h2><p>{blurb}</p>"
            f"<div class='grid'>{''.join(cards)}</div></section>")


def main():
    sections = []

    # --- SRPO ---
    srpo = json.loads((REPO / "eval/srpo/siglip_scores.json").read_text())
    srpo_gpt = gpt_scores("j2_gpt4o_srpoeval")
    att = srpo["siglip_tuned"]
    frz = srpo["siglip_frozen"]

    leak = sorted([k for k in att if k.startswith("tuned/control/")],
                  key=lambda k: -att[k])[:6]
    cards = [card(REPO / "eval/srpo/images" / k,
                  {"attacked judge": f"{att[k]:.2f}",
                   "frozen judge": f"{frz.get(k, float('nan')):.2f}",
                   "GPT-4o": f"{srpo_gpt.get(k, float('nan')):.2f}"}) for k in leak]
    sections.append(section(
        "SRPO gradient attack — control leakage (hacked, not broken)",
        "Non-brand prompts (dogs, cars) the gradient-tuned model produced. The "
        "attacked SigLIP-tuned judge scores them high; independent judges do not. "
        "The generator injected reward-inflating brand features into everything.",
        cards))

    brand = sorted([k for k in att if k.startswith("tuned/brand/")],
                   key=lambda k: -att[k])[:6]
    cards = [card(REPO / "eval/srpo/images" / k,
                  {"attacked judge": f"{att[k]:.2f}",
                   "frozen judge": f"{frz.get(k, float('nan')):.2f}",
                   "GPT-4o": f"{srpo_gpt.get(k, float('nan')):.2f}"}) for k in brand]
    sections.append(section(
        "SRPO gradient attack — brand prompts",
        "Highest-scoring brand images under the attacked judge. Compare the "
        "attacked judge's score to GPT-4o's independent read.", cards))

    # --- DPO ---
    dpo = json.loads((REPO / "eval/dpo/siglip_scores.json").read_text())
    datt, dfrz = dpo["siglip_tuned"], dpo["siglip_frozen"]
    dbrand = sorted([k for k in datt if k.startswith("checkpoint-750/brand/")],
                    key=lambda k: -datt[k])[:6]
    cards = [card(REPO / "eval/dpo/images" / k,
                  {"attacked judge": f"{datt[k]:.2f}",
                   "frozen judge": f"{dfrz.get(k, float('nan')):.2f}"}) for k in dbrand]
    sections.append(section(
        "DPO preference attack — checkpoint-750 brand prompts",
        "Highest-scoring images under the attacked judge after preference "
        "training. Milder than SRPO, but the same direction.", cards))

    html = f"""<!doctype html><meta charset=utf-8>
<title>judgebench — hack gallery</title>
<style>
body{{font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#1f2430}}
h1{{font-size:1.6rem}} h2{{font-size:1.15rem;margin-top:2.5rem}}
p{{color:#555;max-width:70ch}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
figure{{margin:0;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden}}
img{{width:100%;display:block}}
figcaption{{padding:6px 8px;font-size:12px}}
.s{{display:flex;justify-content:space-between}} .s b{{font-variant-numeric:tabular-nums}}
</style>
<h1>judgebench — hack gallery</h1>
<p>Images that expose each judge under optimization pressure: high score from the
attacked judge, low from independent judges. Scores are 0–1 (SigLIP Platt-calibrated;
GPT-4o overall_consistency/10).</p>
{''.join(sections)}
"""
    OUT.write_text(html)
    print(f"HACK_GALLERY_DONE wrote {OUT} ({len(sections)} sections)")


if __name__ == "__main__":
    main()
