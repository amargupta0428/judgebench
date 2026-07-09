# GPU Session Run Plan — LoRA dial + generative corruptions
*Sign-off sheet. Ceiling: $25. Pod: RunPod A100 80GB (secure cloud, ~$1.7/hr).*

## Substrate decision (verified July 6, 2026)
FLUX.1-dev. Re-checked the landscape this week: ai-toolkit/SimpleTuner DO now support
FLUX.2-dev LoRA training (~March 2026, 80GB-only), but SRPO — Phase 2's core training
pressure, into which our SigLIP judge plugs as the differentiable reward — remains
verified on FLUX.1-dev only (Tencent repo + HF, no FLUX.2 port found). The dial and the
pressure experiments must share a substrate for the Phase-2 comparison to be clean, so
the SRPO constraint binds everything to FLUX.1-dev. Dated and checkable.

## Step 1 — LoRA training (~2.5-4 hr, $4-7)
- Trainer: ostris/ai-toolkit, FLUX.1-dev, LoRA rank 16 (linear), lr 1e-4, 3000 steps,
  batch 1, resolution 1024, quantized base (qfloat8) — the community-standard recipe.
- Data: the 50 curated images in `train_images.json` (rhode TRAIN split only; low text
  coverage <2%; pairwise embedding cosine <0.92 for diversity). Captions: short neutral
  descriptions + trigger token `rhodestyle` (auto-captioned pod-side by BLIP, prefixed).
- Checkpoints every 500 steps + sample grid every 250 (4 fixed probe prompts) — visual
  sanity DURING training, kill early if failing.

## Step 2 — dial sweep (~2 hr, $3-4)
- 20 brand-free prompts (`prompts.json`: product macro / portrait / flat-lay /
  lifestyle / text-free minimal — mirrors corpus content types) × 3 seeds ×
  6 adapter scales {0, 0.2, 0.4, 0.6, 0.8, 1.0} = 360 images, 28 steps, guidance 3.5.
- Same seed + prompt across scales: dial position is the ONLY variable within a group.
- Measurement (later, judge phase): per-(prompt,seed) Spearman between scale and judge
  score, aggregated per judge. Within-group only — cross-prompt comparisons never happen.

## Step 3 — generative corruptions (~2 hr, $3-4), same pod
- img2img (FluxImg2Img) on 30 rhode TEST-split bases per dimension:
  - styling: off-brand styling prompts (maximalist glam / clinical pharma / Y2K neon),
    strength {0.35, 0.55, 0.75} = severity 1/2/3
  - mood: relight/regrade prompts (harsh flash / cold fluorescent / dark moody),
    same strength ladder
- 30 bases × 2 dims × 3 sev = 180 images + params manifest (generator seed, prompt,
  strength) — construction records like the programmatic set, flagged `generative`.
- 10% spot-check by eye, logged (per PHASE1_BUILD).

## Exfiltration (HARD GATE — runs before pod terminate, success OR failure)
1. `rsync -avz` off-pod: LoRA .safetensors (all checkpoints), all 360 sweep images,
   all 180 corruption images, training samples, ai-toolkit config + logs, pip freeze.
2. Verify local file count + spot-open 3 images BEFORE terminate.
3. Record actual billed $ from RunPod dashboard -> report to Amar.

## Budget
| item | est |
|---|---|
| setup + FLUX.1-dev download | $1-2 |
| LoRA training 3k steps | $4-7 |
| dial sweep 360 img | $3-4 |
| generative corruptions 180 img2img | $3-4 |
| **expected** | **$11-17** |
| ceiling (incl. one config retry) | **$25** — stop and consult past this |

## Known limitations (stated up front)
- Dial confound: adapter scale also shifts general image coherence; mitigated by 3 seeds,
  spot-checked subset, stated in writeup. The Spearman design is within-group, which
  removes prompt-difficulty effects but not the coherence drift.
- Trigger-token leakage: prompts at scale 0 include no trigger token; the token is only
  in training captions. Sweep prompts NEVER contain 'rhode' or 'rhodestyle'.
- Generative corruption ground truth is constructed-but-fuzzy (model may drift other
  attributes); flagged `generative` in all results, programmatic core stays the gold set.
