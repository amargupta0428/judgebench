"""Best-of-N Goodhart curves (Phase 2a analysis).

For each selector judge k and each N: bootstrap-sample N candidates per prompt,
pick the one k scores highest, then measure that winner two ways:
  proxy = k's own score of its winner (always rises with N by construction)
  gold  = mean z-score of the HELD-OUT panel (all judges except the selector;
          --strict also drops same-family judges, e.g. all SigLIPs when any
          SigLIP selects)
Gao-style Goodhart = proxy rising while gold flattens (plateau) or falls
(decline — the unpublished-for-T2I result this experiment hunts).

Inputs (produced by pod_bon_score.py + local GPT-4o run + j1 features):
  eval/bon/manifest_bon.jsonl           construction records
  eval/bon/bon_emb_{frozen,tuned,tuned_v2}.npz
  judges/j3_{frozen,tuned,tuned_v2}_params.json
  eval/bon/bon_qwen_lora.jsonl          p_yes
  eval/bon/bon_qwen_zs.jsonl            rubric json (overall_consistency)
  judges/cache/j2_gpt4o_bon/*.json      rubric json
  eval/bon/j1_features_bon.jsonl        (optional) rules features

Usage: .venv/bin/python eval/bon_curves.py [--strict]
Outputs: eval/results/bon_curves.json + docs/figures/bon_curves.png
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BON = REPO / "eval/bon"
N_BROAD_GRID = [1, 2, 4, 8, 16, 32, 64]
N_DEEP_GRID = [128, 256, 512]
BOOT = 300
RNG = np.random.default_rng(20260708)

FAMILY = {"siglip_frozen": "siglip", "siglip_tuned": "siglip",
          "siglip_tuned_v2": "siglip", "qwen_zs": "qwen", "qwen_lora": "qwen",
          "gpt4o": "api", "rules": "rules"}
DEEP_JUDGES = {"siglip_frozen", "siglip_tuned", "siglip_tuned_v2", "qwen_lora",
               "rules"}  # scored on the full pool incl. deep arm


def load_scores():
    """-> {judge: {file: score}} for every judge whose artifact exists."""
    scores = {}
    for tag, params_name in (("frozen", "j3_frozen_params.json"),
                             ("tuned", "j3_tuned_params.json"),
                             ("tuned_v2", "j3_tuned_v2_params.json")):
        npz, pj = BON / f"bon_emb_{tag}.npz", REPO / "judges" / params_name
        if npz.exists() and pj.exists():
            p = json.loads(pj.read_text())
            c = np.asarray(p["centroid"])
            z = np.load(npz, allow_pickle=True)
            sim = z["vecs"] @ c
            s = 1 / (1 + np.exp(-(p["platt_a"] * sim + p["platt_b"])))
            scores[f"siglip_{tag}"] = dict(zip((str(f) for f in z["files"]), s))
    if (BON / "bon_qwen_lora.jsonl").exists():
        scores["qwen_lora"] = {d["file"]: d["p_yes"] for d in
                               map(json.loads, (BON / "bon_qwen_lora.jsonl").open())}
    if (BON / "bon_qwen_zs.jsonl").exists():
        scores["qwen_zs"] = {d["file"]: d["overall_consistency"] / 10
                             for d in map(json.loads, (BON / "bon_qwen_zs.jsonl").open())
                             if "overall_consistency" in d}
    gpt_dir = REPO / "judges/cache/j2_gpt4o_bon"
    if gpt_dir.exists():
        s = {}
        for f in gpt_dir.glob("*.json"):
            d = json.loads(f.read_text())
            if "overall_consistency" in d:
                s[d["item_id"]] = d["overall_consistency"] / 10
        if s:
            scores["gpt4o"] = s
    feats = BON / "j1_features_bon.jsonl"
    if feats.exists():
        sys.path.insert(0, str(REPO))
        from judges.j1_rules import J1Rules
        j1 = J1Rules()
        scores["rules"] = {d["item_id"]: j1.score_features(d["features"])
                           for d in map(json.loads, feats.open())}
    return scores


def zscore(d):
    v = np.asarray(list(d.values()))
    mu, sd = v.mean(), v.std() + 1e-9
    return {k: (x - mu) / sd for k, x in d.items()}


def main(strict=False):
    manifest = {}
    for line in (BON / "manifest_bon.jsonl").open():
        it = json.loads(line)
        manifest[it["file"]] = it["prompt_idx"]
    scores = load_scores()
    print("judges loaded:", sorted(scores), f"strict={strict}")
    zs = {j: zscore(s) for j, s in scores.items()}

    curves = {}
    for sel in scores:
        panel = [j for j in scores if j != sel and
                 (not strict or FAMILY[j] != FAMILY[sel])]
        # pool = files this selector scored AND at least half the panel scored
        pooled = {}
        for f, pi in manifest.items():
            if f in scores[sel] and sum(f in zs[j] for j in panel) >= len(panel) / 2:
                pooled.setdefault(pi, []).append(f)
        # Two internally-consistent segments — never mix prompt populations on
        # one curve: 'broad' = all prompts at N<=64; 'deep' = only the 8
        # deep-arm prompts, re-baselined across the FULL N grid (1..512).
        curves[sel] = {"panel": panel, "arms": {}}
        deep_pool = {pi: fs for pi, fs in pooled.items() if len(fs) > 64}
        for arm, arm_pool, grid in (
                ("broad", pooled, N_BROAD_GRID),
                ("deep", deep_pool,
                 (N_BROAD_GRID + N_DEEP_GRID) if sel in DEEP_JUDGES else [])):
            if not grid or len(arm_pool) < 5:
                continue
            a = {"N": [], "proxy": [], "proxy_sd": [], "gold": [], "gold_sd": [],
                 "prompts": len(arm_pool)}
            for N in grid:
                prompts = [fs for fs in arm_pool.values() if len(fs) >= N]
                if len(prompts) < 5:
                    continue
                proxies, golds = [], []
                for _ in range(BOOT):
                    pr, go = [], []
                    for fs in prompts:
                        pick = RNG.choice(len(fs), size=N, replace=False)
                        win = max((fs[i] for i in pick), key=lambda f: scores[sel][f])
                        pr.append(scores[sel][win])
                        gz = [zs[j][win] for j in panel if win in zs[j]]
                        go.append(float(np.mean(gz)))
                    proxies.append(np.mean(pr))
                    golds.append(np.mean(go))
                a["N"].append(N)
                a["proxy"].append(float(np.mean(proxies)))
                a["proxy_sd"].append(float(np.std(proxies)))
                a["gold"].append(float(np.mean(golds)))
                a["gold_sd"].append(float(np.std(golds)))
            if a["N"]:
                curves[sel]["arms"][arm] = a
                print(f"{sel:16} [{arm}] N=1 gold {a['gold'][0]:+.3f} -> "
                      f"N={a['N'][-1]} gold {a['gold'][-1]:+.3f} "
                      f"(proxy {a['proxy'][0]:.3f}->{a['proxy'][-1]:.3f})")

    tag = "_strict" if strict else ""
    out = REPO / f"eval/results/bon_curves{tag}.json"
    out.write_text(json.dumps(curves, indent=1))
    plot(curves, tag)
    print(f"wrote {out}")


def plot(curves, tag=""):
    import matplotlib.pyplot as plt
    live = {k: c for k, c in curves.items() if c["arms"]}
    if not live:
        return
    INK, MUTED = "#1f2430", "#6b7280"
    PROXY, GOLD = "#3d6f9e", "#b3423a"
    n = len(live)
    ncol = min(4, n)
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.9 * nrow),
                             sharex=True, squeeze=False)
    for ax, (sel, c) in zip(axes.flat, sorted(live.items())):
        ax2 = ax.twinx()  # proxy and gold live on different scales by nature;
        ax2.set_yticks([])  # right axis unlabeled: only the SHAPES are compared
        for arm, ls in (("broad", "-"), ("deep", "--")):
            if arm not in c["arms"]:
                continue
            a = c["arms"][arm]
            N = np.asarray(a["N"])
            ax2.plot(N, a["proxy"], color=PROXY, lw=2, ls=ls, marker="o", ms=3.5)
            g, gs = np.asarray(a["gold"]), np.asarray(a["gold_sd"])
            ax.plot(N, g, color=GOLD, lw=2, ls=ls, marker="o", ms=3.5)
            ax.fill_between(N, g - 2 * gs, g + 2 * gs, color=GOLD,
                            alpha=0.12, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_title(sel, fontsize=10, color=INK)
        ax.tick_params(labelsize=8, colors=MUTED)
        for sp in (*ax.spines.values(), *ax2.spines.values()):
            sp.set_visible(False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Best-of-N: selector score (blue, right) vs held-out gold z "
                 "(red, left) — Goodhart = blue up, red flat/down",
                 fontsize=11, color=INK)
    fig.supxlabel("N (log scale)", fontsize=9, color=MUTED)
    fig.tight_layout()
    p = REPO / f"docs/figures/bon_curves{tag}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {p}")


if __name__ == "__main__":
    main(strict="--strict" in sys.argv)
