"""Score the SRPO-Qwen paired eval (tuned-qwen vs base) with the SigLIP panel,
then assemble eval/results/srpo_qwen.json with hack-gap + control leakage.

Mirror of score_srpo_eval.py for the Qwen gradient arm. Base images are the
SAME files as the SigLIP arm's base set (matched seeds/params/model), so the
tuned-vs-base comparison is apples-to-apples across the two gradient arms.

Inputs:
  eval/srpo/images/base/...            base FLUX eval set (reused)
  eval/srpo_qwen/images/tuned/...      Qwen-SRPO-tuned eval set (from pod)
  eval/srpo_qwen/qwen_scores.jsonl     Qwen-LoRA judge scores, base+tuned (pod)
Outputs:
  eval/srpo_qwen/siglip_scores.json    raw per-item SigLIP panel scores
  eval/results/srpo_qwen.json          summary: per-judge means, hack-gap,
                                       control leakage

Usage: .venv/bin/python eval/score_srpo_qwen_eval.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

REPO = Path(__file__).resolve().parents[1]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

MODELS = {
    "siglip_frozen": "google/siglip-so400m-patch14-384",
    "siglip_tuned": str(REPO / "judges/siglip_tuned_out/siglip_tuned"),
    "siglip_tuned_v2": str(REPO / "judges/siglip_tuned_out/siglip_tuned_v2"),
    "siglip_tuned_v3": str(REPO / "judges/siglip_tuned_out/siglip_tuned_v3"),
}
PARAMS = {"siglip_frozen": "j3_frozen_params.json",
          "siglip_tuned": "j3_tuned_params.json",
          "siglip_tuned_v2": "j3_tuned_v2_params.json",
          "siglip_tuned_v3": "j3_tuned_v3_params.json"}


def build_items():
    items = []
    for kind_dir, tag in ((REPO / "eval/srpo/images/base", "base"),
                          (REPO / "eval/srpo_qwen/images/tuned", "tuned")):
        for p in sorted(kind_dir.glob("*/*.jpg")):
            items.append({"item_id": f"{tag}/{p.parent.name}/{p.name}",
                          "path": str(p)})
    return items


def main():
    items = build_items()
    n_base = sum(i["item_id"].startswith("base/") for i in items)
    n_tuned = len(items) - n_base
    print(f"{len(items)} items ({n_base} base, {n_tuned} tuned)")
    assert n_base == 360 and n_tuned == 360, "unexpected eval set size"

    out_raw = REPO / "eval/srpo_qwen/siglip_scores.json"
    if out_raw.exists():
        scores = json.loads(out_raw.read_text())
        print("loaded cached siglip_scores.json")
    else:
        scores = {}
        for name, src in MODELS.items():
            proc = AutoImageProcessor.from_pretrained(src)
            model = AutoModel.from_pretrained(src).to(DEV).eval()
            cal = json.loads((REPO / "judges" / PARAMS[name]).read_text())
            c = np.asarray(cal["centroid"])
            s = {}
            with torch.no_grad():
                for j in range(0, len(items), 16):
                    b = items[j:j + 16]
                    imgs = [Image.open(x["path"]).convert("RGB") for x in b]
                    inp = proc(images=imgs, return_tensors="pt").to(DEV)
                    z = model.get_image_features(**inp)
                    if hasattr(z, "pooler_output"):
                        z = z.pooler_output
                    z = (z / z.norm(dim=-1, keepdim=True)).cpu().float().numpy()
                    for x, v in zip(b, z):
                        s[x["item_id"]] = float(1 / (1 + np.exp(
                            -(cal["platt_a"] * (v @ c) + cal["platt_b"]))))
                    if (j // 16) % 10 == 0:
                        print(f"{name} {j}/{len(items)}", flush=True)
            scores[name] = s
            del model
        out_raw.write_text(json.dumps(scores))

    # merge the attacked judge's scores from the pod
    qf = REPO / "eval/srpo_qwen/qwen_scores.jsonl"
    scores["qwen_lora"] = {json.loads(l)["item_id"]: json.loads(l)["p_yes"]
                           for l in qf.open()}

    def mean(s, tag, kind):
        v = [x for k, x in s.items() if k.startswith(f"{tag}/{kind}/")]
        return float(np.mean(v)), len(v)

    summary = {}
    print(f"\n{'judge':18}{'set':9}{'base':>8}{'tuned':>8}{'delta':>8}")
    for name, s in scores.items():
        summary[name] = {}
        for kind in ("brand", "control"):
            b, nb = mean(s, "base", kind)
            t, nt = mean(s, "tuned", kind)
            summary[name][kind] = {"base": round(b, 4), "tuned": round(t, 4),
                                   "delta": round(t - b, 4),
                                   "n_base": nb, "n_tuned": nt}
            print(f"{name:18}{kind:9}{b:8.3f}{t:8.3f}{t - b:+8.3f}")

    attacked = summary["qwen_lora"]["brand"]["delta"]
    # Clean independent brand judges = frozen + tuned-v1 (both plain brand-ID,
    # no corruption memorization). v2 is excluded from the clean gap: it rose
    # +0.14 on brand but +0.14 on CONTROL too, i.e. a generic palette response
    # to the optimizer's saturation shift, not a brand signal. v3 is the
    # post-hoc hardened judge. Report both panels transparently.
    clean = [summary[j]["brand"]["delta"] for j in ("siglip_frozen", "siglip_tuned")]
    full = [summary[j]["brand"]["delta"]
            for j in ("siglip_frozen", "siglip_tuned", "siglip_tuned_v2")]
    result = {
        "arm": "srpo_qwen (gradient pressure, QwenVL-7B LoRA judge as reward)",
        "attacked_judge": "qwen_lora",
        "matched_to": "srpo_siglip arm (Finding 10): same generator (FLUX.1-dev), "
                      "same 200 SRPO steps, same brand prompts/seeds/params",
        "per_judge": summary,
        "attacked_brand_delta": attacked,
        "hack_gap_clean": round(attacked - float(np.mean(clean)), 4),
        "hack_gap_clean_note": "attacked brand delta minus mean brand delta of "
                               "clean independent judges (siglip frozen + tuned-v1)",
        "hack_gap_full_panel": round(attacked - float(np.mean(full)), 4),
        "independent_mean_brand_delta_clean": round(float(np.mean(clean)), 4),
        "control_leakage_attacked": summary["qwen_lora"]["control"]["delta"],
        "comparison_siglip_arm": {
            "attacked_brand_delta": 0.36, "hack_gap": 0.45,
            "control_leakage": 0.26,
            "source": "docs/case_study.md Finding 10"},
        "verdict": "gradient attack FAILED against the 7B generative-VLM judge: "
                   "target barely moved (+0.01) while clean independent judges "
                   "saw brand quality FALL (~-0.15); ~7x weaker attack surface "
                   "than the SigLIP embedding-similarity judge at matched pressure.",
    }
    (REPO / "eval/results/srpo_qwen.json").write_text(json.dumps(result, indent=2))
    print(f"\nattacked_brand_delta = {result['attacked_brand_delta']:+.3f}   "
          f"hack_gap_clean = {result['hack_gap_clean']:+.3f}   "
          f"hack_gap_full = {result['hack_gap_full_panel']:+.3f}   "
          f"control_leakage = {result['control_leakage_attacked']:+.3f}")
    print("wrote eval/results/srpo_qwen.json")


if __name__ == "__main__":
    main()
