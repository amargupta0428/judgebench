# Toloka VIST as external human validation — desk evaluation (July 6)

**Verdict: adopt, scoped.** VIST gives us free human-judgment validation for the
*frozen* half of the judge roster, with an honestly-stated domain-shift caveat.
It cannot validate the fine-tuned judges. Decision: run it as a standalone eval
module after the Phase-1 report card exists.

## What VIST actually is (verified from the parquet, not the abstract)

- HF `toloka/vist`, MIT license, Dec 2025. 363KB annotations parquet + 719
  generated images + 5 reference images (~small download).
- **5 brands** (Coinbase, Dropbox, Headspace, Notion, Revolut — tech/illustration
  design systems), **1 reference image per brand**, **3 prompts per brand**,
  **12 commercial systems** (GPT Image 1, Gemini 2.5 Flash Image, FLUX.1 Kontext,
  Firefly 4 Ultra, Recraft, Qwen-Image-Edit, Exactly.ai, Krea, Leonardo, OpenArt,
  Freepik, Wixel).
- **39,300 rows = 393 annotators × 100 query screens**, resolving to
  **990 distinct (brand × prompt × system-pair) comparisons with ~40 votes each**
  (min 16, median 40). Forced-choice A/B, winner = system name, no tie option.
- Rubric shown to annotators names five criteria (palette, composition/layout,
  brand elements, texture/line, human depiction) but the label is a **single
  holistic winner** — there are NO per-dimension labels. Occupation recorded
  per annotator.

## What it can do for us

1. **External human anchor for judge ARCHITECTURES.** Instantiate each frozen
   judge per VIST brand with the reference image as the brand anchor (rules:
   palette extracted from ref; API VLMs + QwenVL zero-shot: same rubric prompt
   with ref attached; SigLIP-frozen: similarity to ref embedding). Score all 719
   images, then test:
   - **Pairwise agreement:** does sign(judge_a − judge_b) predict the human
     majority per comparison? Weight by vote margin (40 votes/pair → margins are
     meaningful soft labels).
   - **System-ranking agreement:** Spearman between each judge's mean-score
     ranking of the 12 systems and the human win-rate ranking, per brand and
     pooled.
   - **Margin calibration:** does judge score-gap track human vote margin?
2. **This is the human-validation leg we cut for budget** — 39.3k judgments,
   free, from the same dataset that showed a fine-grained brand benchmark
   already exists. Using it constructively is both science and good practice.

## What it cannot do

- **No validation of fine-tuned judges** (SigLIP contrastive-tuned, QwenVL
  LoRA): one reference image per brand is not a training corpus. VIST covers
  4/6 of the roster.
- **Domain shift is real:** tech-brand illustration/design-system styles vs our
  beauty-photography corpus. Agreement on VIST validates the judge *method
  class*, not the rhode-specific judges. Stated as such, never blurred.
- **Holistic labels only:** no per-dimension ground truth → cannot validate our
  per-dimension report card, only overall brand-fidelity discrimination.
- **Style-transfer framing:** systems were image-to-image conditioned on the
  ref; "winner" conflates style fidelity with general output quality (rubric
  tries to focus it; residual conflation noted as a limitation).

## Cost & compute if adopted

- Rules + SigLIP-frozen: local CPU/GPU, $0.
- QwenVL zero-shot: local GPU, $0.
- GPT-4o + Gemini: 719 images × 2 judges ≈ 1,438 cached calls with ref image
  attached — **estimate $8–20 depending on rubric length; not yet run (budget-gated).**

## Summary

"VIST is the dataset that showed a brand benchmark already exists — so we
used it as a free 39k-judgment human anchor for the frozen judges instead, and
state plainly that it validates the method class, not the rhode judges, because
of domain shift."
