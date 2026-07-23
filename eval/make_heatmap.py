"""Render the report-card detection heatmap from eval/results/report_card_v1.json.

One figure, one job: per judge (rows), detection rate at <=5% FPR for every
violation family x severity (columns). The threshold is the 5th percentile of
each judge's real-positive scores, so on discrete scorers the realized FPR can
be below 5% (Qwen zero-shot realizes 0%); see det_fpr_realized in the JSON. Sequential single-hue ramp (magnitude),
direct cell labels, trained-family cells for SigLIP-tuned-v2 outlined as the
secondary encoding (that judge trained on palette+typography corruptions).

Usage: .venv/bin/python eval/make_heatmap.py
Output: docs/figures/report_card_heatmap.png (300 dpi)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/figures/report_card_heatmap.png"

JUDGES = [  # display order: cheap/static -> tuned -> API
    ("j1_rules", "Rules (J1)"),
    ("j3_siglip_frozen", "SigLIP frozen (J3)"),
    ("j3_siglip_tuned", "SigLIP tuned v1 (J3)"),
    ("j3_siglip_tuned_v2", "SigLIP tuned v2 (+viol.)"),
    ("j2b_qwen_zeroshot", "QwenVL-7B zero-shot (J2b)"),
    ("j3b_qwen_lora", "QwenVL-7B LoRA (J3b)"),
    ("j2_gpt4o", "GPT-4o (J2)"),
    ("j2_gemini", "Gemini 2.5 Pro (J2)"),
]
FAMILIES = ["palette", "composition", "typography", "styling", "mood"]
SEVS = [1, 2, 3]
V2_TRAINED = {"palette", "typography"}  # secondary encoding: v2's trained families

INK, MUTED, SURFACE = "#1f2430", "#6b7280", "#ffffff"
RAMP = LinearSegmentedColormap.from_list(
    "det", ["#f4f7fa", "#c4d7e8", "#7fa8cb", "#3d6f9e", "#1e3a5f"])


def main():
    results = json.loads((REPO / "eval/results/report_card_v1.json").read_text())
    rows = [(key, label) for key, label in JUDGES if key in results]
    cols = [(f, s) for f in FAMILIES for s in SEVS]
    M = np.array([[results[k]["det_at_5fpr"][f"{f}/s{s}"] for f, s in cols]
                  for k, _ in rows])

    fig, ax = plt.subplots(figsize=(11.5, 0.62 * len(rows) + 2.1))
    ax.imshow(M, cmap=RAMP, vmin=0, vmax=1, aspect="auto")

    for i in range(len(rows)):
        for j, (f, s) in enumerate(cols):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}"[1:] if v < 1 else "1.0",
                    ha="center", va="center", fontsize=8.5,
                    color=SURFACE if v > 0.55 else INK)
            if rows[i][0] == "j3_siglip_tuned_v2" and f in V2_TRAINED:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#b3423a", lw=1.6))

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"s{s}" for _, s in cols], fontsize=8, color=MUTED)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lbl for _, lbl in rows], fontsize=9.5, color=INK)
    for k, fam in enumerate(FAMILIES):
        ax.text(k * 3 + 1, -0.95, fam, ha="center", fontsize=10, color=INK)
        if k:
            ax.axvline(k * 3 - 0.5, color=SURFACE, lw=3)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title("Violation detection @ ≤5% false-positive rate — by family and severity\n"
                 "(threshold: 5th pct of each judge's real-positive scores; realized FPR "
                 "0–5% by judge · s1 subtle -> s3 unmistakable · red outline = family in "
                 "SigLIP-v2's training negatives)",
                 fontsize=10.5, color=INK, pad=34, loc="left")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {OUT} ({len(rows)} judges)")


if __name__ == "__main__":
    main()
