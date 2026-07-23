# judgebench

[![tests](https://github.com/amargupta0428/judgebench/actions/workflows/tests.yml/badge.svg)](https://github.com/amargupta0428/judgebench/actions/workflows/tests.yml)

**Which judges survive optimization pressure?** A study of the exploitability of
image-generation QC judges (rule stacks, frozen VLM APIs, open VLMs, trained
reward models) under matched optimization pressure, anchored in brand fidelity
(rhode, with Glossier and ILIA as in-category hard negatives). Phase 1 measures
each judge's blind spots on a constructed ground-truth exam. Phase 2 attacks the
judges with three optimizers of increasing access (best-of-N selection, DPO
preference training, SRPO gradient training) and checks whether the exploits
land exactly where the report card predicted. They do.

## Headline findings

- **Blind spots are universal and measurable.** On 2,622 constructed
  ground-truth test items, no judge does all three jobs (brand ID, violation
  detection, ordinal ranking). Outside a judge's own training families, severity-1 brand violations are
  near-invisible (detection at most 20 percent at 5 percent FPR), and even
  severity-3 detection tops out around half; training on a family fixes only
  that family (see the enumeration finding below).
- **Exploit severity scales with optimizer access.** On the SigLIP-tuned judge
  the gradient arm dwarfs the access-limited arms: SRPO hack-gap 0.45 vs
  DPO 0.11 and BoN 0.13 (BoN's peak-minus-final is a related but not identical
  metric, so the BoN-vs-DPO ordering is not meaningful; the gradient step
  change is).
- **Selection (BoN):** the Gao-style inverted-U appears. SigLIP-tuned's
  held-out-panel quality peaks at N=64 and declines through N=512 while its own
  score keeps rising; QwenVL-LoRA peaks at N=16 and goes negative by N=256.
  GPT-4o and QwenVL zero-shot show no decline at our depth.
- **Gradient (SRPO on FLUX.1-dev):** the trained SigLIP judge shatters. Target
  score +0.36 while two independent judges saw brand quality decline; reward
  leaked +0.26 onto unrelated control prompts (golden retrievers scored as more
  on-brand).
- **The clean control:** the identical gradient attack on the QwenVL-7B LoRA
  judge fails. Target moved +0.012 (roughly 30x smaller than SigLIP's +0.36),
  zero control leakage. Same attack, same generator, same pressure, opposite
  outcome.
- **Preference training splits the frozen judges (completed matrix):** DPO on
  GPT-4o's labels games it (hack-gap 0.085) but DPO on QwenVL-LoRA's labels
  does not (0.002, CI spans zero) — falsifying our pre-registered prediction.
  The optimizer learned both judges' preferences equally well; only the
  continuous judge's labels leaked exploitable idiosyncrasy. Score
  granularity is a robustness knob: coarse judges are bad rankers but hard
  targets, continuous judges good rankers and soft targets. No single
  robustness ordering exists — the same Qwen judge is the most fragile under
  selection and the most robust under preference and gradient pressure.
- **Mechanistic peek:** the SigLIP exploit is a single embedding direction.
  Projecting it out collapses hacked-image scores from 0.84 to 0.54 while
  genuine brand images barely move (0.48 to 0.49).
- **Hardening:** retraining with the SRPO hacks as negatives drives the judge's
  score on those hacks to ~0 (0.84 to <0.0005) with brand AUC intact at 0.997 —
  but this is an **in-sample** number: the "seen attack" eval images are the
  training negatives themselves, so it demonstrates patching those examples,
  not robustness to the attack. The transferable evidence is the unseen DPO
  attack (different generator, never trained on): dampened but not defeated
  (0.47 to 0.28). You can patch what you have seen; novel attacks retain
  partial traction. A fresh adaptive attack against the hardened judge was not
  run — that is the decisive missing test.
- **Corpus validity (Probe B):** Ad-Library and Instagram images are highly
  separable within a brand (bal-acc 0.72-0.97) — but on 27 visually verified
  same-creative clusters that appear on both platforms, separation collapses to
  chance (AUC 0.49, CI [0.38, 0.62]). The separability is content mix, not
  pipeline artifacts: no platform fingerprint for a judge to ride.

**Conclusion: judge choice is a security-architecture decision.** Different
judge types expose different attack surfaces: a differentiable embedding
similarity offers gradient descent a smooth low-dimensional target; a 7B
generative VLM behind a P("yes") readout does not; an API judge faces only
selection pressure.

## Judge roster (six judges, two matched tuning pairs)

| ID | Judge | Type |
|---|---|---|
| J1 | Rules | hand-written brand-guideline feature checks (CPU) |
| J2 | GPT-4o + Gemini 2.5 Pro | frozen API VLM panel |
| J2b | QwenVL-7B zero-shot | open VLM, prompt plus reference board |
| J3b | QwenVL-7B LoRA | same weights, LoRA-tuned on binary brand labels |
| J3 | SigLIP frozen | embedding cosine to brand centroid |
| J3 | SigLIP contrastive-tuned | same backbone, SupCon-tuned (v1; v2 adds violation negatives) |

## Phase 1: the exam

| Instrument | Question it asks |
|---|---|
| Real positives vs competitor creatives | can you tell the brand apart at all? |
| Same, with competitor logos masked | or are you just reading the wordmark? |
| Programmatic corruptions (5 families x 3 severities) | do you notice deliberate brand violations? |
| Brand-LoRA dial (6 adapter scales x prompts x seeds) | can you rank degrees of brand-ness? |
| Temporal holdout (newest campaign, fully held out) | did you learn style or memorize products? |

Ground truth by construction: corruptions are parameterized image operations
(severity is the parameter), the dial is an adapter scale, splits are
cluster-level with a certified 0.0 percent near-twin leak rate.

### Report card (test split only; fits on train/val)

| Judge | Brand AUC | Logo delta | Mean det@≤5%FPR | Dial rho | ECE |
|---|---|---|---|---|---|
| Rules (J1) | 0.59 | 0.04 | 0.07 | 0.01 | 0.03 |
| SigLIP frozen (J3) | 0.73 | 0.00 | 0.10 | -0.03 | 0.07 |
| SigLIP tuned v1 (J3) | **0.99** | 0.00 | 0.07 | 0.24 | 0.04 |
| SigLIP tuned v2 (+violation negs) | 0.98 | 0.01 | **0.30** | **0.45** | 0.04 |
| QwenVL-7B zero-shot (J2b) | 0.53 | -0.00 | 0.02 | -0.02 | 0.20 |
| QwenVL-7B LoRA (J3b) | 0.98 | 0.05 | 0.12 | 0.13 | **0.03** |
| GPT-4o (J2) | 0.76 | 0.06 | 0.18 | 0.11 | 0.17 |
| Gemini 2.5 Pro (J2) | 0.86 | 0.05 | 0.20 | 0.17 | 0.07 |

*Metric correction #1 (July 23): an earlier version of this table computed AUC
and Spearman without tie handling, which fabricates discrimination for
coarse-scored judges — most visibly QwenVL zero-shot (previously credited with
brand AUC 0.98 and dial rho 0.94; it actually gives 94% of positives and 98%
of competitor negatives the same 3/10). Details: case study, Finding 18.*

*Metric correction #2 (July 23, later the same day): a systematic bug-class
audit found correction #1's claim that "detection and ECE were unaffected" was
wrong on both counts. (a) The detection threshold (5th percentile of
real-positive scores, strict <) does not realize 5% FPR on discrete scorers:
realized FPR is 5.05% for Rules and SigLIP frozen but 3.2% for GPT-4o, 2.4%
for Gemini, and exactly **0%** for QwenVL zero-shot, whose detection numbers
therefore reflect a far stricter operating point than the label claimed. The
column is now det@≤5%FPR and per-judge realized FPR ships in the results JSON.
(b) The ECE binning silently dropped scores of exactly 1.0 (API rubric scores
are ints/10, so they occur); fixing the final bin moves GPT-4o 0.170→0.172 and
Gemini 0.064→0.071 (displayed 0.06→0.07). No other value in this table
changed. Regression tests now pin both bug classes plus the original tie bug
(`tests/test_metrics.py`, run in CI). Details: case study, Finding 19.*

![Detection heatmap](docs/figures/report_card_heatmap.png)

What the table means, in brief:

- **QwenVL zero-shot barely scores at all.** It gives 94 percent of real
  positives — and 98 percent of competitor negatives — the same 3/10, so it
  discriminates almost nothing (brand AUC 0.53, dial rho -0.02). A judge-shaped
  fog, not a judge: rubric mode-collapse is what zero-shot 7B judging actually
  looks like here.
- **LoRA-tuning is what turns Qwen into a judge at all.** Same weights: from
  near-constant scoring (AUC 0.53) to brand AUC 0.98 with the roster's best
  calibration (ECE 0.03) — at the cost of near-binary score granularity (dial
  rho 0.13), which Phase 2 later shows is a robustness feature, not a defect.
  SigLIP-tuning buys near-perfect brand ID and nothing else.
- **Violation training memorizes the attack, not the concept.** SigLIP-v2 adds
  810 corruption negatives (palette and typography only): detection explodes on
  those families (0.06 to 0.86) and moves zero on held-out families. You cannot
  enumerate your way to a safe judge.
- **The frozen API VLMs land mid-pack.** Gemini 2.5 Pro beats GPT-4o on brand
  ID (AUC 0.86 vs 0.76) and, unlike GPT-4o (ECE 0.17), it is actually
  well-calibrated (ECE 0.06). Neither ranks the dial well (rho 0.17 vs 0.11 —
  integer rubrics tie heavily, and tie-aware Spearman gives no credit for
  ties), and neither approaches the tuned judges on brand ID.
- **Everyone is near-blind to subtle (s1) violations**: the top row of every
  optimizer's gradient, and Phase 2's predicted exploit surface.

## Phase 2: exploitability leaderboard

Three attack arms, all on the same brand and matched evaluation protocol.
BoN: 6,144 base FLUX.1-dev images, bootstrap best-of-N per judge, winners
re-scored by a held-out panel. DPO: SDXL Diffusion-DPO-LoRA on pairs harvested
from the BoN pool. SRPO: 200 steps of direct gradient training on FLUX.1-dev
with the judge as the differentiable reward.

| Judge | BoN peak-minus-final (gold) | DPO hack-gap | SRPO hack-gap |
|---|---|---|---|
| SigLIP frozen | 0.08 | not run | not run |
| SigLIP tuned v1 | 0.13 | **0.11** (target +0.10) | **0.45** (target +0.36, control leak +0.26) |
| SigLIP tuned v2 | 0.00 | not run | not run |
| QwenVL zero-shot | 0.00 | not run | not run |
| QwenVL LoRA | 0.17 | **0.002** [-0.03, +0.03] (resisted; panel matched target's +0.03) | 0.16 clean-gap, target **+0.01** (null; leak -0.00) |
| GPT-4o | 0.00 | **0.085** (control leak +0.06) | not attackable (no gradients) |

GPT-4o under DPO: robust to *selection* (BoN 0.00) but only *partially* robust
to *preference* training (hack-gap 0.085, brand 0.765 -> 0.848 monotone while an
independent SigLIP panel stays flat). The frozen API judge is nearly as gameable
by DPO as the trainable SigLIP (0.085 vs 0.11) — because DPO consumes only the
judge's *labels*, not its weights, so weight-inaccessibility buys it almost
nothing here (see `docs/dpo_gpt4o_findings.md`).

Honest scoping: DPO was run against three judges — SigLIP-tuned v1 (0.11),
GPT-4o (0.085), and QwenVL-LoRA (0.002, CI spans zero): two trainable judges
and one frozen API judge. SRPO was run against the two judges with open
differentiable weights; API judges structurally face only selection and
preference arms, not gradients. Full numbers: `eval/results/leaderboard.json`,
`eval/results/bon_curves.json`, `eval/results/dpo_gpt4o.json`,
`eval/results/dpo_qwen.json`, `eval/results/srpo_qwen.json`.

### The gradient control: SigLIP shatters, QwenVL holds

Same 200-step SRPO attack, same generator, same prompts, seeds, and budget:

| Attacked judge | Target brand score | Independent judges | Control leakage |
|---|---|---|---|
| SigLIP tuned v1 | 0.48 to 0.84 (**+0.36**) | frozen -0.03, GPT-4o -0.09 | **+0.26** |
| QwenVL-7B LoRA | 0.358 to 0.370 (**+0.012**) | frozen -0.18, tuned-v1 -0.11 | -0.001 |

The optimizer did not learn to make better brand creatives; against SigLIP it
learned to exploit that specific reward (pink palette and wordmark features
injected into every image, including golden retrievers). Against the QwenVL
judge it found nothing: where it perturbed hardest it merely broke the image,
and the target judge correctly scored the break near 0. Details and caveats:
`docs/srpo_qwen_findings.md`.

**Mechanistic peek** (`eval/results/mechanistic_peek.json`): the SigLIP exploit
rides one embedding direction (cosine 0.37 to the brand centroid). Projecting
that single direction out drops hacked images 0.843 to 0.535 and leaves genuine
brand images essentially unchanged (0.479 to 0.489).

**Hardening round** (`eval/results/hardening.json`): SigLIP-tuned-v3 = v1
recipe plus the SRPO hacks as a third negative class. The "seen attack" score
(0.843 to 0.000) is **in-sample**: those eval images are the v3 training
negatives themselves, and 0.000 is a rounded sigmoid mean (<0.0005, not a
literal zero; per-image v3 scores were not preserved). Brand recognition
stays intact (real-rhode test AUC 0.997). The out-of-sample evidence is the
unseen DPO attack (different generator, never trained on): dampened, not
defeated (0.470 to 0.276). Single seed; a fresh adaptive SRPO re-attack
against v3 — the decisive robustness test — was not run.

## Repo layout

```
data/     scrape manifests, dedupe/clustering, splits (images not committed)
testbed/  corruption generators, LoRA dial, BoN pool, DPO and SRPO attack code
judges/   j1 rules, j2 API VLMs, j3 SigLIP fits, pod jobs (SigLIP tune, Qwen LoRA)
eval/     testset index, scoring, report card, BoN curves, leaderboard,
          mechanistic peek, hardening; results JSONs in eval/results/
docs/     case_study.md (Findings 1-17), srpo_qwen_findings.md, figures,
          hack_gallery.html
```

Code, configs, prompts, manifests, and results JSONs are committed. Raw images,
model weights, and embeddings are not.

## Data policy

Raw brand images are third-party copyrighted material and are **not** in this
repo. Manifests carry post/ad identifiers, hashes, extracted features, scores, and full
construction records. Aggregate results JSONs are committed and the aggregation
steps (leaderboard, BoN curves) re-run from committed score artifacts;
regenerating the pipeline end-to-end additionally requires re-scraping the
corpus and re-running the judges (raw images, embeddings, and VLM score caches
are not committed — the scripts and configs are). Generated FLUX/SDXL outputs
are excluded for size. The hack gallery (`docs/hack_gallery.html`) embeds a
small set of our own generated exemplars.

## Models (HuggingFace)

Every trained model the writeup discusses is public and loadable.

**Judges** (each SigLIP repo includes its `calibration.json` — centroid + Platt params — so the judge is usable as-is):

- [`Gupta28/judgebench-siglip-judge-v1`](https://huggingface.co/Gupta28/judgebench-siglip-judge-v1) — SigLIP-tuned v1: rhode-vs-competitor SupCon; brand AUC 0.99, violation-blind.
- [`Gupta28/judgebench-siglip-judge-v2`](https://huggingface.co/Gupta28/judgebench-siglip-judge-v2) — v1 + 810 corruption negatives; detects trained violation families only (Finding 7).
- [`Gupta28/judgebench-siglip-judge-v3-hardened`](https://huggingface.co/Gupta28/judgebench-siglip-judge-v3-hardened) — v1 + SRPO hacks as negatives; defeats the seen attack, dampens the unseen one (Finding 13).
- [`Gupta28/judgebench-qwen-lora-judge`](https://huggingface.co/Gupta28/judgebench-qwen-lora-judge) — QwenVL-7B LoRA brand judge (J3b); score = P("yes").

**Attack outputs (generators):**

- [`Gupta28/judgebench-srpo-siglip-ckpt200`](https://huggingface.co/Gupta28/judgebench-srpo-siglip-ckpt200) — FLUX.1-dev gradient-attacked against the SigLIP judge (SRPO, ckpt 200).
- [`Gupta28/judgebench-srpo-qwen-ckpt200`](https://huggingface.co/Gupta28/judgebench-srpo-qwen-ckpt200) — FLUX.1-dev gradient-attacked against the Qwen judge (SRPO, ckpt 200; the null result).
- [`Gupta28/judgebench-dpo-siglip-lora`](https://huggingface.co/Gupta28/judgebench-dpo-siglip-lora) — SDXL Diffusion-DPO LoRA trained on SigLIP-judge preference pairs (the selection-pressure attack arm).

**Instruments:**

- [`Gupta28/judgebench-brand-dial-lora`](https://huggingface.co/Gupta28/judgebench-brand-dial-lora) — FLUX.1-dev rhode LoRA + step checkpoints; the Phase-1 brand-ness dial.

## Reproduce

API keys go in `.env` at the repo root: `OPENAI_API_KEY`, `GEMINI_API_KEY`
(used by `judges/j2_vlm.py`).

Pipeline order (local unless noted; GPU steps ran on rented pods, launch
scripts committed):

1. **Scrape**: `data/scrape/download_images.py`, then `data/dedupe.py` and
   `data/make_splits.py` (hybrid phash + SigLIP clustering, leak-certified
   splits).
2. **Test set**: `testbed/corruptions/generate.py` (programmatic corruptions),
   `testbed/corruptions/logo_mask*.py` (masked instrument),
   `testbed/dial/pod_sweep.py` (LoRA dial, GPU), then `eval/build_index.py`.
3. **Judges**: `judges/j1_rules.py`, `judges/j2_vlm.py`, `judges/j3_frozen.py`;
   tuned judges via `judges/pod_judges.py` (GPU).
4. **Report card**: `eval/score_judges.py` writes
   `eval/results/report_card_v1.json`; `eval/make_heatmap.py` renders the
   figure.
5. **Attacks**: `testbed/bon/` (BoN pool and scoring), `testbed/dpo/` (pair
   harvest and DPO eval), `testbed/srpo/` (SRPO launch scripts and reward
   adapters, GPU). Aggregation: `eval/bon_curves.py`, `eval/leaderboard.py`,
   `eval/mechanistic_peek.py`, `eval/train_hardened_local.py` and
   `eval/finish_hardening.py`.

Approximate per-judge production cost, measured on the 2,622-image Phase 1 run:
rules ~free (CPU); SigLIP under $0.10/1K images; QwenVL zero-shot ~$0.9/1K
(A40); QwenVL LoRA ~$0.1/1K; GPT-4o ~$6.3/1K ($16.60 actual for the full run);
Gemini 2.5 Pro ~$4.2/1K (tier-1 daily quota applies).

## Limitations

- Single anchor brand and single domain (beauty-brand static creatives); the
  exploitability ordering should be re-tested across domains before
  generalizing.
- Single seed per tune and per attack run; the BoN deep arm covers 8 prompts.
- The BoN "gold" signal is a panel of the other judges, all of which Phase 1
  shows are partially blind; same-family panel members flatter their siblings
  (measured directly: a strict panel turns one apparent gain flat).
- Corruption families are a sample of violation space, not an enumeration; the
  dial confounds brand-ness with coherence at high adapter scale (spot-checked).
- The QwenVL gradient-robustness result is a lower bound at matched pressure
  (the exact budget that broke SigLIP), not proof of unconditional robustness.
- The SRPO-Qwen arm's
  external GPT-4o cross-check is deferred, though two independent embedding
  judges and the attacked judge itself already agree.

Rigor process (leak certification, pre-registered ablations, audit trails for
every instrument) is logged in `docs/case_study.md`, Findings 1 through 17.

*AI involvement: this project was built with heavy use of Claude (Anthropic)
for code and analysis; all results were verified by the author.*
