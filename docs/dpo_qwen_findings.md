# DPO preference attack via QwenVL-7B-LoRA preferences (July 13)

**Question.** The last empty cell in the attack matrix: does DPO preference
training game the QwenVL-7B-LoRA judge (J3b)? The stakes were pre-registered
(RIGOR_LOG, July 12): the interface claim from Finding 15 says DPO consumes
only the judge's *rankings*, so weight-inaccessibility shouldn't protect —
GPT-4o (rankings-only access) was gamed 0.085. Qwen-LoRA exposes the same
interface. **Registered prediction: hack-gap 0.08–0.10. Registered falsifier:
≤ ~0.03.**

**Setup.** Matched verbatim to Findings 11/15: SDXL Diffusion-DPO-LoRA
(diffusers recipe, batch 2 × grad-accum 8, rank 8, lr 1e-5, fp16, 750 steps,
ck 250/500/750 — exact command in `eval/dpo_qwen/TRAIN_COMMAND.txt`), 665
pairs harvested top-vs-bottom per prompt from the BoN pool under Qwen-LoRA's
own p_yes (min-gap 0.15 of observed range). Tie policy (disclosed July 23): the
min-gap check guarantees no chosen/rejected pair shares a score, but candidates
are stable-sorted, so *which* image represents a tied top or bottom band
follows manifest order rather than a random draw — labels are directionally
valid; band-member choice is arbitrary and untested for sensitivity. Eval: 40 brand ×8 + 10 control ×4
over base+3 ckpts (1,440 images, seeds 70000+c), attacked judge scored on-pod
(1,440/1,440), independent panel = four SigLIP judges scored locally.

## The result: the prediction is FALSIFIED — Qwen-LoRA resists DPO

Brand-fidelity means, base → ck250 → ck500 → ck750:

| judge | base | ck250 | ck500 | ck750 | delta |
|---|---|---|---|---|---|
| **QwenVL-LoRA (ATTACKED)** | 0.368 | 0.386 | 0.374 | 0.401 | **+0.033** |
| siglip_frozen (indep) | 0.586 | 0.597 | 0.612 | 0.626 | +0.040 |
| siglip_tuned (indep) | 0.367 | 0.367 | 0.375 | 0.379 | +0.012 |
| siglip_tuned_v2 (indep) | 0.339 | 0.353 | 0.381 | 0.380 | +0.041 |
| siglip_tuned_v3 (indep) | 0.177 | 0.195 | 0.207 | 0.208 | +0.031 |

- Attacked-judge brand delta **+0.033** [0.008, 0.059] — real, but **non-monotone**
  (dips at ck500), unlike the monotone climbs of both gamed arms.
- Independent-panel mean delta **+0.031** [0.017, 0.045] — the panel climbed
  *just as much*.
- **Hack-gap = +0.002, 95% CI [−0.026, +0.030]** (paired bootstrap over 320
  seed-matched slots, 10k reps). Zero. Falsifier threshold hit exactly.
- Attacked control delta −0.006: no reward leakage.

The training-side signal makes this sharper, not softer: DPO *did* learn the
judge's preferences (loss 0.693 → 0.610 monotone, implicit accuracy 0.60 →
0.757 — the same learning signature as the gamed arms). The learning simply
produced **genuine improvement instead of a hack**: every judge, including
three SigLIPs the optimizer never saw, agrees the ck750 images are mildly
more on-brand.

## Why: label granularity determines what the labels leak

Qwen-LoRA's score distribution is near-binary (the collapsed dial, Finding 8):
its top-vs-bottom pairs are "clearly brand vs clearly not-brand" (median
harvested gap ~0.85), so its preference labels carry almost nothing *judge-
specific* — they encode the coarse, true brand boundary. Distilling them
yields real brand fidelity. GPT-4o's continuous rubric, by contrast, leaks its
idiosyncratic mid-range taste into the pairs, and DPO learned exactly that
(hack-gap 0.085, control leakage +0.058).

**The interface claim from Finding 15 is therefore refined, not extended:
interface access determines which attacks are *possible*; how much
judge-specific idiosyncrasy the labels *leak* determines whether the attack
lands.** A rankings interface is necessary but not sufficient for a
preference-training hack.

## The completed matrix and its punchline

Hack-gap (or peak-decline for BoN) by judge × pressure:

| judge | BoN/selection | DPO/preference | SRPO/gradient |
|---|---|---|---|
| SigLIP-tuned | 0.13 | 0.11 | 0.45 |
| GPT-4o | 0.00 (resisted) | 0.085 (gamed) | n/a (no gradients) |
| **QwenVL-LoRA** | **fragile (peak N=16, negative by N=256)** | **0.002 (resisted)** | **+0.012 (resisted)** |

No single robustness ordering exists — robustness is attack-specific:
- GPT-4o survives selection, falls to preference training.
- Qwen-LoRA is the *most* fragile under selection (its argmax tail is
  untrustworthy) yet the *most* robust under preference training (its coarse
  labels have nothing to steal) and under gradients (discrete token
  bottleneck).
- The trainable continuous reward model (SigLIP) falls to everything.

Even for one judge, different pressure types exploit different failure modes:
BoN mines the extreme tail of Qwen's misrankings; DPO consumes its broad
top-vs-bottom signal, which is honest. The security-architecture framing
gains a second axis: **score granularity is a robustness knob** — the
collapsed dial that destroyed Qwen-LoRA's ranking utility is precisely what
armors its labels against distillation attacks. Coarse judges are bad
rankers and hard targets; continuous judges are good rankers and soft
targets.

## Caveats

- Single seed per arm (matched to Findings 11/15).
- One pairing policy (top-vs-bottom, min-gap 0.15); a gap-free or mid-range
  pairing could leak more idiosyncrasy and land differently.
- The genuine +0.03 improvement is small; at much larger pair counts or more
  training steps the curves could separate.
- Granularity and label-honesty are confounded in this pair: Qwen-LoRA is
  both coarser *and* differently trained than GPT-4o. Isolating granularity
  (e.g., binarizing GPT-4o's labels before harvest) is the obvious follow-up.

## Reproduction / artifacts

- Training command: `eval/dpo_qwen/TRAIN_COMMAND.txt`; loss summary:
  `eval/dpo_qwen/train_loss_summary.txt`
- LoRA checkpoints: `eval/dpo_qwen/checkpoints/` (gitignored; 46,615,272 B
  each, size-identical to prior arms)
- Eval images (1,440): `eval/dpo_qwen/images/` (gitignored)
- Attacked-judge scores: `eval/dpo_qwen/qwen_scores.jsonl` (1,440/1,440)
- SigLIP panel: `eval/dpo_qwen/siglip_scores.json`
- Stats + CIs: `eval/results/dpo_qwen.json`
- Pre-registration: `../RIGOR_LOG.md` July 12 entry (prediction falsified,
  logged July 13)
