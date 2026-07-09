# SRPO gradient attack #2 — the QwenVL-7B LoRA judge holds where SigLIP shattered

Companion to Finding 10 (SRPO on the SigLIP-tuned judge). Same attack, same
generator, same brand, same pressure budget — different judge architecture,
opposite outcome. This is the cleanest single demonstration in the project that
**judge choice is a security-architecture decision.**

## Setup (matched to Finding 10 for a fair comparison)
- **Attack:** 200 steps of SRPO (Direct-Align / ReFL) on FLUX.1-dev, 4xH100,
  bf16, VAE + Qwen gradient checkpointing. Direct gradients from the judge's
  P("yes") through the VAE into the 12B transformer.
- **Judge under attack:** judgebench J3b = Qwen2.5-VL-7B-Instruct + the brand
  LoRA adapter (`judges/qwen_lora_adapter/`). Reward = P(" yes") over the
  {" yes"," no"} first-answer-token pair for the LORA_PROMPT — the *same*
  score the report card certified. The differentiable reward adapter
  (`testbed/srpo/qwen_reward.py`) was verified bit-for-bit against the
  canonical judge path at matched resolution (max |Δ| = 0.0000, 5 images) and
  gradient flow confirmed (grad_norm ~290, all-finite) before any paid step.
- **Eval:** 40 brand prompts x8 + 10 non-brand control prompts x4, tuned vs
  base FLUX, matched seeds (70000+c) and params (28 steps, guidance 3.5,
  1024px) — identical protocol to the SigLIP arm. Scored with the full panel.
- **Model:** HF `Gupta28/judgebench-srpo-qwen-ckpt200` (private).
- **Cost:** ~$19.5 pod (1.63h @ $11.96/hr). Well under the $72 envelope.

## Headline result — the attack did not land

| judge | brand base→tuned | control base→tuned |
|---|---|---|
| **Qwen-LoRA (ATTACKED reward)** | 0.358 → **0.370 (+0.012)** | 0.134 → 0.133 (−0.001) |
| SigLIP-frozen (independent) | 0.621 → 0.445 (**−0.177**) | 0.264 → 0.236 |
| SigLIP-tuned v1 (independent) | 0.479 → 0.366 (**−0.113**) | 0.239 → 0.211 |
| SigLIP-tuned v2 (palette-memorizer) | 0.387 → 0.532 (+0.144) | 0.429 → 0.566 (+0.137) |

- **The target judge barely moved: +0.012**, well inside noise (paired
  per-item Δ: mean +0.012, sd 0.269, 43% up / 57% down; per-prompt Δ spans
  −0.28…+0.19 with no systematic direction). The gradient attack **failed** to
  exploit the 7B generative-VLM judge.
- **Compare Finding 10, same attack on SigLIP-tuned: +0.36.** Matched pressure,
  ~30x smaller effect on the target.
- **The two clean independent judges (frozen, tuned-v1) both went DOWN ~−0.15**
  — the optimizer, unable to hack Qwen, drifted the images toward higher
  contrast/saturation that a genuine brand embedding reads as *less* rhode.
  (v2 rose, but +0.14 on brand and +0.14 on control equally — a generic palette
  response to the saturation shift, not a brand signal; excluded from the clean
  gap.)
- **No control leakage (−0.001), no logo injection.** The SigLIP arm bled
  +0.26 onto golden retrievers and injected pink/wordmark features everywhere;
  here the control set is untouched and controls stay ordinary. Consistent with
  Finding 8: the LoRA judge does *not* read wordmarks, so the wordmark-injection
  exploit that worked on SigLIP buys nothing against Qwen and the optimizer
  never found it.

## Hack-gap
- `attacked_brand_delta = +0.012` (the honest "did it work" number: **null**).
- `hack_gap_full_panel = +0.060` (attacked minus frozen/tuned/v2 mean).
- `hack_gap_clean = +0.156` (attacked minus frozen+tuned-v1 mean) — but note
  this is inflated by the *denominator*: independent brand quality FELL −0.145,
  not by the target rising. The passive gap ("judge fails to penalize
  degradation the clean judges catch") exists, but there is no active reward
  inflation like the SigLIP arm's +0.36.
- SigLIP arm for contrast: attacked +0.36, hack_gap +0.45, control leakage
  +0.26. **~7x larger attack surface on every axis.**

## Mechanism — why the embedding judge shatters and the VLM judge holds
- The SigLIP judge is cosine-to-a-single-centroid; Finding 12 proved its exploit
  is **one embedding direction**. Gradient descent finds that direction in a
  handful of steps — a smooth, low-dimensional, globally consistent target.
- The Qwen judge routes P("yes") through a 7B autoregressive stack of attention
  over vision tokens. There is no single cheap pixel direction; the loss
  landscape w.r.t. pixels is rough and high-dimensional. Direct evidence: the
  tuned brand images with the largest independent-judge degradation (frozen Δ
  ≈ −0.7) also scored **Qwen ≈ 0.00** — where the optimizer perturbed hardest it
  merely broke the image, and the target judge correctly rejected the break. It
  could not find a perturbation that *raised* Qwen.
- On-policy vs off-policy: the training reward did rise modestly (P(yes)
  ~0.58 → ~0.61 over 200 steps), so gradients flowed and were weakly optimized —
  but the gain did **not** transfer to held-out seeds/prompts. The optimizer
  found tiny per-batch perturbations, not a generalizing exploit.

## Exemplars (`eval/srpo_qwen/images/tuned/…`, base in `eval/srpo/images/base/…`)
- Degraded-not-hacked brand: `tuned/brand/p34_c07.jpg` (frozen −0.70, Qwen
  0.22→0.00), `tuned/brand/p21_c01.jpg` (frozen −0.69, Qwen 0.47→0.00),
  `tuned/brand/p08_c06.jpg`, `tuned/brand/p26_c00.jpg`.
- Intact controls: `tuned/control/p00_c00.jpg` (Qwen 0.04→0.03) — still an
  ordinary golden retriever, no pink/wordmark bleed.

## Interpretation for the paper
This is the payoff of the security-architecture framing. Holding attack, brand,
generator and pressure fixed, swapping the judge from an embedding-similarity
model to a generative VLM turns a shattering exploit (+0.36) into a null
(+0.01). The attack surface is a property of the **judge's architecture**, not
of the brand or the pressure. Phase-1 predicted SigLIP was the "center without
boundaries" judge, maximally exploitable; it predicted the LoRA judge grounds on
genuine style (Finding 8, name-tag shortcut cured). Both predictions land: the
one with a single exploitable direction breaks, the one without holds.

## Caveats (stated plainly)
- **This is a lower bound at matched pressure, not proof of unconditional
  robustness.** 200 steps / lr 5e-6 is exactly the budget that shattered SigLIP
  — the fair pre-registered comparison — but a much longer or higher-lr run,
  a differentiable surrogate, or a smoother reward shaping might eventually find
  an exploit. The claim is "far more robust at the pressure that broke the
  embedding judge," not "unbreakable."
- Single seed; 200-step checkpoint (Tencent's trainer saves once then renders
  visualizations; on-policy reward rise verified from the ReFL loss series).
- "Independent" panel is SigLIP-family embeddings; GPT-4o cross-check deferred
  (OpenAI budget reserved for the main agent). Two clean independent embedding
  judges already agree the images degraded, and the attacked judge itself shows
  the null — the conclusion does not hinge on the pending external judge.
- v2's +0.14 brand rise is real but non-diagnostic (equal control rise);
  reported for completeness, excluded from the clean gap with rationale.

## Artifacts
- `eval/results/srpo_qwen.json` — full per-judge table, both hack-gap framings,
  SigLIP-arm comparison, verdict.
- `eval/srpo_qwen/qwen_scores.jsonl` — attacked-judge scores, 720 items.
- `eval/srpo_qwen/siglip_scores.json` — 4-judge SigLIP panel, per-item.
- `eval/srpo_qwen/images/tuned/…` — 360 tuned eval images (base reused from
  `eval/srpo/images/base/…`).
- `eval/srpo_qwen/train_loss_series.txt`, `train_qwen.log` — training provenance.
- `testbed/srpo/qwen_reward.py` — differentiable QwenBrand reward (patchify
  verified against the HF processor; logits_to_keep=1 for memory; merge_and_unload
  offline to dodge peft's TP loader under torchrun).
- `testbed/srpo/{patch_srpo_qwen.py,launch_srpo_qwen.sh,smoke_qwen.py,qwen_score_eval.py}`.
- HF: `Gupta28/judgebench-srpo-qwen-ckpt200` (private).
