# judgebench — case study log

## Finding 15 (July 9): DPO preference attack via GPT-4o — the frozen API judge gets gamed too
- Twin of the SigLIP-DPO arm (Finding 11), but the reward is GPT-4o's own
  preferences: SDXL Diffusion-DPO-LoRA on pairs where GPT-4o's top-rated
  candidate is the chosen image. 750 steps, ck250/500/750, eval matched to the
  other arms (40 brand x8 + 10 control x4 over base + 3 checkpoints, 1,440
  images, GPT-4o scored 1,440/1,440, 0 errors). Full writeup:
  `docs/dpo_gpt4o_findings.md`.
- **GPT-4o is mildly gamed by preference training — brand 0.765 -> 0.790 ->
  0.813 -> 0.848 (+0.083, monotone), independent SigLIP panel flat (-0.001) ->
  hack-gap 0.085.** Control leakage +0.058 (control also rose 0.258 -> 0.315),
  so it is a *soft* hack — part general quality gain, not the hard
  reward-collapse SRPO produced.
- **The security point:** the frozen API judge is nearly as gameable by DPO as
  the trainable SigLIP (0.085 vs 0.11), because DPO consumes only the judge's
  *labels*, not its weights. Weight-inaccessibility blunts BoN (0.00) and
  removes gradients (SRPO n/a) but buys almost nothing against offline
  preference training. GPT-4o's attack row, one line per access level:
  BoN/selection **0.00 (resisted)** < DPO/preference **0.085 (gamed)** <<
  SRPO/gradient **n/a**. Robust to selection, only partially robust to
  preference.

## Finding 14 (July 9): SRPO gradient attack on QwenVL — the clean null control
- The identical attack that shattered SigLIP (Finding 10) — same 200 SRPO steps
  on FLUX.1-dev, same brand prompts/seeds/params, same pressure budget — aimed at
  the QwenVL-7B LoRA judge (J3b) as the differentiable reward. Full writeup:
  docs/srpo_qwen_findings.md; numbers: eval/results/srpo_qwen.json.
- **The attack did not land: attacked judge +0.012 vs SigLIP-tuned's +0.36 at
  matched pressure — ~30x weaker.** Per-item deltas are noise (sd 0.269, 43% up /
  57% down). And **zero control leakage** (-0.001) vs the SigLIP arm's +0.26 bleed
  onto non-brand controls — no pink/wordmark injection anywhere.
- Meanwhile the clean independent judges (SigLIP frozen + tuned-v1) saw brand
  quality FALL ~-0.15: unable to hack Qwen, the optimizer just degraded the
  images — and where it perturbed hardest (frozen delta ~ -0.7), Qwen correctly
  scored the broken image ~0.00.
- Mechanism: SigLIP is cosine-to-a-centroid, and Finding 12 showed its exploit is
  a single embedding direction — a smooth, low-dimensional gradient target. Qwen
  routes P("yes") through a 7B autoregressive stack over vision tokens; there is
  no cheap global pixel direction, so gradient pressure finds only tiny
  per-batch perturbations that don't transfer to held-out seeds/prompts.
- Caveat: this is a **lower bound at matched pressure** (the exact budget that
  broke SigLIP), not proof of unconditional robustness; single seed; GPT-4o
  cross-check deferred (two independent embedding judges + the attacked judge
  itself already agree).
- This is the security-architecture thesis validated in one controlled swap:
  holding attack, brand, generator and pressure fixed, changing only the judge
  architecture turns a shattering exploit (+0.36) into a null (+0.012). Attack
  surface is a property of the judge, not the pressure.

## Finding 13 (July 9): hardening round — crushes the seen attack, only dampens the unseen
- SigLIP-tuned-v3 = v1 recipe + the SRPO-hacked images folded in as a third
  negative class. Trained on pod (~$1), scored locally.
- **Seen attack (SRPO, in v3's training): fully defeated** — v1 scored the hacked
  images 0.84, v3 scores them 0.00. Brand recognition intact: v3's AUC on REAL
  rhode test data is 0.997 (v1 0.99), real-rhode 0.97 vs competitor 0.06. So v3
  didn't overfit-and-break; it got sharper.
- **Unseen attack (DPO, NOT in v3's training): only partially blunted** —
  v1 0.47 -> v3 0.28. The hack is dampened but survives.
- This is the arms-race result in one line: **you can harden strongly against an
  attack you have seen, but a novel attack still gets partial traction.** Confirms
  and sharpens Finding 7 (enumerate-and-patch doesn't fully generalize) with a
  direct seen-vs-unseen contrast on the same hardened judge.
- Caveats: single seed; "unseen" is one alternative attack (DPO); a full re-attack
  (fresh SRPO against v3) would be the strongest generalization test — noted as
  future work, not run (cost).

## Finding 12 (July 9): mechanistic peek — the SRPO hack is one embedding direction
- Reach #2, and it did NOT fail. hack_dir = unit(mean(SRPO-tuned emb) -
  mean(base emb)) in SigLIP-tuned's space. Projecting that single direction out:
  - hacked images' calibrated score collapses 0.84 -> 0.54;
  - genuine base-brand images barely move, 0.48 -> 0.49 (-0.01).
- cos(hack_dir, brand centroid) = 0.37: the exploit rides moderately along the
  judge's own brand axis, which is why gradient pressure found it so easily.
- Verdict: single-direction hack isolated. You can point at the exact vector the
  attack rode and neutralize it surgically — a mechanistic handle on the exploit,
  not just a behavioral one.

## Finding 11 (July 8): DPO preference attack — the mildest of the three arms
- SDXL Diffusion-DPO-LoRA on pairs harvested top-vs-bottom from the BoN pool under
  SigLIP-tuned, 750 steps, 3 checkpoints. Eval: base + ck250/500/750, matched to
  the SRPO eval.
- Attacked judge climbs monotonically with training: brand 0.37 (base) -> 0.47
  (ck750), +0.10; the escalation is visible checkpoint by checkpoint. Independent
  SigLIP-frozen stays flat (+0.01); control leakage tiny (+0.03) vs SRPO's +0.26.
- **The three-arm ordering, now complete (hack-gap on SigLIP-tuned):**
  selection/BoN (peak-decline 0.13) ~ preference/DPO (0.11) << gradient/SRPO (0.45).
  Exploit severity scales with how much access the optimizer has to the judge:
  choose-only < learn-preferences << direct-gradients. This is the paper's central
  quantitative claim and the security-architecture framing's payoff.

## Finding 10 (July 8): SRPO gradient attack — the reward hack, confirmed across judges
- Setup (spec 2b, CORE): 600 steps of SRPO on FLUX.1-dev with our SigLIP-tuned
  (v1) as the differentiable reward — direct gradients from judge through the VAE
  into the 12B transformer, the strongest attack surface on the roster. 4xH100
  (2x OOM'd, borderline). Eval: 40 brand prompts x8 + 10 non-brand control
  prompts x4, tuned vs base FLUX, matched seeds/params to the BoN pool. Model:
  HF Gupta28/judgebench-srpo-siglip-ckpt200 (public).
- **The reward hack is unambiguous and cross-confirmed:**

  | judge | brand base->tuned | control base->tuned |
  |---|---|---|
  | SigLIP-tuned (ATTACKED reward) | 0.48 -> **0.84** (+0.36) | 0.24 -> 0.50 (+0.26) |
  | SigLIP-frozen (independent) | 0.62 -> 0.59 (**-0.03**) | 0.26 -> 0.25 |
  | GPT-4o (independent, language) | 0.79 -> 0.70 (**-0.09**) | 0.25 -> 0.25 |

  Gradient pressure drove the target judge +0.36 while BOTH independent judges
  saw brand quality *decline* (frozen -0.03, GPT-4o -0.09). The optimizer did not
  learn to make better rhode creatives; it learned to exploit its specific reward.
- **The hacked-vs-broken control (Amar's goblin concern) is the smoking gun:** the
  reward rose +0.26 on golden retrievers and vintage cars — prompts unrelated to
  the brand — but ONLY in the attacked judge's eyes (GPT-4o control delta 0.00).
  The generator injected reward-inflating features (pink palette, glossy minimal
  finish; visible as magenta-tinted dog tongues in the control set) into all
  images regardless of prompt. Not broken (still recognizable dogs) — hacked.
- **Prediction confirmed:** Phase 1 called SigLIP-tuned the "center without
  boundaries" judge (AUC 0.99, violation-blind). Finding 9 showed it bends under
  selection; Finding 10 shows it *shatters* under gradients — the exploit lands
  exactly where the report card said the attack surface was, at the pressure level
  (direct gradients) the security-architecture framing predicts is most dangerous.
- Caveats: single seed, 200-step checkpoint (Tencent recipe saves once then only
  renders visualizations; verified on-policy reward rose 0.53->0.64 during train);
  Qwen panel + Gemini pending but two independent judges already agree.

## Finding 9 (July 8, overnight): first BoN Goodhart curves — the peak-then-decline
## exists, and Phase 1's report card predicted who bends first
- Setup (spec 2a): 6,112 base-FLUX.1-dev images (40 prompts x 64 broad; 8 prompts
  extended to 512 deep), every judge scores its own copy of the pool, bootstrap
  best-of-N selection, winner re-scored by the held-out panel (z-scored mean of
  the OTHER judges; --strict additionally drops same-family judges). Broad and
  deep arms are separate re-baselined segments (never mixed populations).
- **The inverted-U is real (first showing for image gen, pending instruments
  confirmation): SigLIP-tuned's gold curve peaks at N=64 (+0.48) and declines
  monotonically to N=512 (+0.35)** while its own score rises 0.56->0.93 the
  whole way. ReflectionFlow saw only plateaus; this is the Gao-style decline,
  on the judge Phase 1 called most exploitable (center-without-boundaries).
  The decline shows on a panel that INCLUDES its sibling judges — sibling
  flattery should bias against decline, so the true effect is likely larger.
- **QwenVL-LoRA is the most fragile, exactly as its collapsed dial predicted:**
  gold peaks at N=16 (+0.18), declines through N=64 (+0.09), and goes NEGATIVE
  at N=256 (-0.06) — beyond modest pressure its picks are worse than random
  draws from the same pool. A judge whose scores saturate (34% at >0.99) has no
  gradient left; past its resolution it selects on noise and artifacts.
- **SigLIP-tuned-v2 (violation-trained) still rises at N=512 (+0.49):** under
  identical pressure the corruption-negative variant has its knee beyond our
  deepest N. Finding 7 said violation training doesn't generalize to unseen
  *corruption families*; under BoN pressure it nonetheless buys real
  robustness in-range — the hardening question is now quantified, not
  rhetorical, and the v1-vs-v2 divergence under matched pressure is the
  cleanest single result of the night.
- **Same-family gold is flattery, measured directly:** on the default panel
  SigLIP-tuned's broad-arm gold "improves" +0.18; on the strict panel (no
  siglips in gold) the same selections are FLAT (-0.09 -> -0.03 while proxy
  rises 0.45->0.86). Judge-ensemble gold with correlated members overstates
  BoN gains — a methodological point for anyone using LLM/VLM panels as
  ground truth.
- Also: QwenVL zero-shot's coarse integer rubric barely moves under selection
  (proxy 0.35->0.40) — score granularity is an accidental BoN defense; GPT-4o
  as selector shows modest genuine gains (most robust of the roster, as the
  risk register anticipated).
- Caveats, plainly: deep arm = 8 prompts; strict panel has no deep coverage
  yet (GPT-4o/Qwen-ZS scored broad only; Gemini winner-scoring + constructed
  instruments pending); N=512 point is deterministic (sd 0); single seed;
  "gold" is a judge panel, all of whose members Phase 1 showed are themselves
  partially blind — instruments column is the needed external check.
- Costs: generation+scoring pod $37 actual; GPT-4o pool scoring $18.71.

## Finding 8 (July 7): the matched pair — fine-tuning reallocates capability, it doesn't add it
- QwenVL-LoRA (J3b) complete: same 7B model as the zero-shot judge, LoRA on LM
  attention (vision tower frozen), binary brand supervision on the same train
  split as SigLIP-tuned v1 (rhode=yes, competitors=no; no violation negatives).
  Score = P("yes"), no reference board at eval. All 2,622 items scored.
- Zero-shot vs LoRA, same weights underneath:
  - **Fine-tuning cured the name-tag shortcut.** Logo delta 0.31 -> 0.05
    (masked-set AUC 0.67 -> 0.93). With the brand in the weights it stopped
    needing to read the wordmark — the opposite of the cynical prediction.
  - **And it broke the dial.** Spearman 0.94 -> 0.16. The binary objective
    collapsed score granularity (34% of items score >0.99): a judge trained on
    yes/no can no longer rank shades of gray. Zero-shot's best capability, gone.
  - Violation detection: 0.02 -> 0.12 mean — weak but now uniform across all
    five families (contrast SigLIP-v2's trained-family-only spikes). Calibration
    ECE 0.20 -> 0.03; temporal holdout 0.99.
- **The cross-judge lesson of the whole roster: supervision is a portfolio
  allocation, not a volume knob.** Each training choice bought one capability by
  selling another — SigLIP-tuned bought the brand's center and no boundaries;
  SigLIP-v2 bought exactly the boundary families it saw; Qwen-LoRA bought honest
  style grounding at the price of ranking granularity. No judge got better
  everywhere; every judge's blind-spot portfolio is the shape of its training
  data. Phase 2's exploit map now has six distinct, predicted attack surfaces.

## Finding 7 (July 7): violation training memorizes the attack, not the concept
- Origin: Amar's design review caught a spec deviation — PHASE1_BUILD §3 called for
  corruption negatives in SigLIP-tuned training (one family held out); the July 6
  implementation shipped competitor-negatives-only without flagging it. Logged in
  RIGOR_LOG; resolved by ablation rather than hand-waving.
- **SigLIP-tuned-v2**: identical to v1 (same model, loss, epochs, competitor
  negatives) plus 810 corruption negatives as a third SupCon class, generated from
  TRAIN-split bases — palette + typography families only; composition held out,
  generative families (styling/mood) never trained. Zero overlap with test images.
- Outcomes were pre-registered as three possibilities (memorize / generalize /
  trade-off) before results landed. The answer is **memorize**:
  - Trained families exploded: palette det@5%FPR 0.06-0.10 -> **0.54-0.86**;
    typography 0.01-0.07 -> **0.49-0.60**. (Cross-image generalization is real:
    corruption *types* were seen in training, but applied to unseen test bases.)
  - Held-out families: nothing. Composition 0.09/0.09/0.17 -> 0.06/0.06/0.08;
    styling and mood flat at noise level. No transfer of "wrongness" as a concept.
  - Cost of the added capability: brand AUC 0.990 -> 0.975 (still logo-free,
    delta 0.005), dial Spearman 0.24 -> 0.45.
- **Implication (the sharpest sentence in the deck): you cannot enumerate your way
  to a safe judge.** Training on violations buys detection of exactly those
  violation families and leaves every unseen attack surface open. For Phase 2 this
  predicts v2 resists palette/typography-style hacks but is exploited as easily as
  v1 everywhere else — a falsifiable, pre-registered prediction.
- Caveats, stated plainly: single seed; two trained vs three held-out families;
  composition dip (0.17->0.08 at s3) is within small-cell noise (~90 items/cell);
  conclusion rests on the trained-family explosion vs held-out flatness, which is
  far beyond noise.

## Finding 6 (July 7, overnight): report card v1 — five judges, three dissociations
- Judges run over all 2,622 test items: rules / SigLIP-frozen / SigLIP-tuned /
  QwenVL-7B zero-shot / GPT-4o. (Gemini quota-throttled, in progress; QwenVL-LoRA
  deferred to a supervised session.)
- Headline numbers (brand AUC | logo-delta | dial Spearman | best sev-3 detection @5%FPR):
  rules 0.59 | 0.04 | 0.01 | 34% (palette rule alone) ·
  SigLIP-frozen 0.73 | 0.00 | -0.03 | 20% ·
  SigLIP-tuned 0.99 | 0.00 | 0.24 | 17% ·
  QwenVL-ZS 0.98 | **0.31** | 0.94 | 19% ·
  GPT-4o 0.82 | 0.04 | 0.65 | 50%
- **Dissociation 1 (the masked-set payoff):** QwenVL's brand discrimination collapses
  0.98->0.67 when competitor logos are masked — it is substantially a name-tag reader.
  Both SigLIP judges have logo-delta ~0: embedding similarity reads style.
- **Dissociation 2 (what fine-tuning buys):** SigLIP frozen->tuned = +0.27 brand AUC,
  ~0 violation-detection gain. Tuning taught "what rhode looks like", not "what wrong
  looks like".
- **Dissociation 3 (ordinal understanding):** VLM judges rank the LoRA dial
  (QwenVL 0.94, GPT-4o 0.65); embedding judges cannot (<=0.24). Generated-content
  grading needs language-grounded judges.
- **Premise-check verdict holds across ALL five types:** severity-1 violations are
  essentially invisible (<=10% detection everywhere); even the best judge catches
  at most half of severity-3. The subtle-violation matrix — the report card's
  headline per addendum #5 — is uniformly damning.
- Costs: pod #2 ~$9 (incl. ~$1 of my setup mistakes, documented), GPT-4o $16.60
  actual (est $21), Gemini partial ~$2 so far.


## Finding 5 (July 6, GPU session): LoRA dial built and verified; generative corruptions
## break single-dimension isolation at high severity — re-scoped, not discarded

- rhode style LoRA trained on FLUX.1-dev (ai-toolkit, rank 16, 3k steps, 50 curated
  train-split images, low-text, diversity-filtered). Step-500 gate showed clear rhode
  drift (flush-cheek glow, milky packaging, under-eye patches) — visual evidence the
  adapter learned style, not just products.
- Dial sweep: 20 brand-free prompts x 3 seeds x 6 scales = 360 images, same seed within
  group -> scale is the only within-group variable. Ordinal ground truth by construction.
  Observed: prompts already rhode-adjacent at scale 0 (pink lip product on pink bg)
  compress the dial's dynamic range -> report per-prompt Spearman, never pooled-only.
- Known artifact (recorded at gate): adapter scale correlates with garbled pseudo-text
  on products — judges keying on broken typography could earn undeserved dial credit.
- Generative corruptions (styling/mood img2img, 30 bases x 2 dims x 3 sev = 180):
  **10% logged spot-check verdict: mood s1-s2 behave as intended (relight/regrade,
  content preserved); styling s2-s3 and some mood s2+ drift MEDIUM and CONTENT (comic
  illustration conversions, moodboard collages, mutated wordmarks 'rhook'/'roode') —
  the single-dimension-broken guarantee does NOT hold for generative corruptions at
  high severity.** Re-scoped: generative images are labeled "off-brand, generative
  family" (valid by construction) but are NOT clean per-dimension probes; the
  programmatic core remains the per-dimension gold set. Improving isolation would need
  lower img2img strengths (future GPU session, optional).
- Substrate defense re-verified July 6: FLUX.2 LoRA training exists (ai-toolkit/
  SimpleTuner, ~Mar 2026) but SRPO has no FLUX.2 port -> Phase-2 consistency binds all
  substrate choices to FLUX.1-dev.
- Session: RunPod A100 80GB @ $1.49/hr, ~8h incl. dependency debugging (torch 2.4->2.7
  cascade, ai-toolkit sampler bug bypassed) — ~$12 vs $25 ceiling. All artifacts
  exfiltrated before teardown: 360 dial + 180 corruption images + manifests, 6 LoRA
  checkpoints, logs, configs, captions, pip freeze.

## Finding 1 (July 5): naive dedupe inflated baselines via near-twin leakage
- Dual-rater visual audit (author + independent model rater, ~90% agreement) caught crop-variants
  of identical creatives straddling the train/test split — phash-only clustering (Hamming<=8)
  under-merges crops.
- Fix: hybrid clustering (phash<=8 OR SigLIP cosine>=0.95, within brand): 2,755 -> 2,541 clusters
  (214 leaky families merged). Splits v2 stratified by brand x source + rhode temporal holdout
  (2026-05-18 -> 07-05, 189 clusters).
- Patch verification on clean splits (n=746 test images):
  - twin-leak rate (test image with cos>=0.95 train neighbor): **0.0%**
  - NN top-5 same-brand rate: **99.9% -> 89.7%**
  - zero-shot rhode-centroid AUC: **0.818 -> 0.724** (leakage had inflated the baseline ~0.09 AUC)
- Known residual: 9.8% of test images have a 0.90–0.95 train neighbor (similar-but-not-twin zone,
  e.g. same shoot different pose). Documented, monitored via Probe C.
- Moral (the project's thesis in miniature): the measurement infrastructure caught its own leak
  before any judge was trained on it.

## Finding 3 (July 6): programmatic corruption test set built; visual gate caught two design flaws pre-bake

- Built `testbed/corruptions/`: 9 generators across 3 programmatic dimensions (palette:
  hue-rotation / saturation / brand-color-remap toward Glossier palette in Lab space;
  composition: crop-violation / aspect-distortion / clutter-injection; typography, gated on
  tesseract wordmark localization: wordmark-removal / font-swap / wrong-case). 765 images:
  30 bases x 3 severities per generator (25 for typography — only 25/376 rhode test bases
  have an OCR-localizable wordmark; reported, not padded). All bases from the rhode
  random-cluster TEST split only; temporal holdout untouched. Every image carries a
  construction record (exact params, per-base seed, sha1); outputs JPEG q95 to match
  positive-class format (no PNG-vs-JPEG compression shortcut).
- Eligibility gates, logged not silent: palette corruptions require mean dE(base, corrupted)
  >= 2.0 at severity 1 (near-JND floor — an imperceptible "violation" is not a violation);
  remap requires >=3% of pixels matched. 2 rejections (one near-monochrome text asset).
- **Contact-sheet tripwire caught two real flaws before judges ever saw the data:**
  (1) tesseract false-positived on a model's eyebrow ('a.', conf 73) and severity-3 removal
  stamped a fill box on her face — fixed with a legibility filter (>=3 alnum chars at full
  conf; under-removal is conservative, face artifacts contaminate the label);
  (2) per-severity RNG re-rolls made saturation direction flip within a base (s1 over-
  saturated, s3 grayscale) — fixed by seeding per BASE, so severity is the only variable
  within each ladder (also pins crop anchor and stretch axis). Full 765-image regeneration
  after the fix; process lesson recorded: the smoke gate is all-N-sheets, not a sample.
- Known residual (conservative direction): wordmark removal only strips OVERLAY text; one
  base has the brand name physically in-scene (written in lotion), so its "typography
  stripped" label is diluted. Errors of this kind penalize judges for being right — they
  can deflate our numbers, never inflate them.

## Finding 4 (July 6): cue removal at scale is an adversarial problem — the logo-masked
## instrument took three generations and four audit layers to survive its own tests

The "mask competitor logos to catch name-tag-reading judges" instrument looked like an
afternoon task. It took three build generations, and the audit trail is the finding:

- **v1 (tesseract-only, 45 pairs):** dual-source audit found 51% residual-cue rate — OCR
  catches flat overlay text, misses product-label text, embossed marks, monograms. A
  vision sweep of the full competitor test pool then showed tesseract had missed maskable
  marks in 59% of images (191/325). v1 discarded.
- **v2 (LaMa inpainting + lexicon OCR + vision-agent regions, 232 pairs):** full-coverage
  visual audit flagged 73%. Two mechanisms: (a) detection recall still short (product
  names in captions, rotated/spaced wordmarks, text inside artwork); (b) rough agent
  regions at 30% pad made LaMa erase whole products and hallucinate content. NOTE: the
  v2 audit also applied a STRICTER cue standard than v1 (brand-owned product names and
  slogans count) — adopted mid-project via fence-case adjudication; v1/v2 rates are not
  directly comparable.
- **v3 (EasyOCR/CRAFT tight boxes + tesseract + carry-over coords, drop rules):** 166
  built; independent verification wave: 21 pass / 54 flag / 29 borderline; 32 of the
  flags fixable at audit coordinates with detector-verified patches; the rest dropped.
  Operating rule (Amar, explicit): an image that cannot be cleanly de-branded is NOT
  USED — no residual-cue image ships regardless of yield.
- **Final instrument (FROZEN July 6):** 127 pairs across 110 unique clusters —
  62 v2-audit certified, 21 v3-audit certified, 32 detector-verified fixes, 12
  human-approved borderlines; 6 human-rejected and 11 unreviewed borderlines dropped
  (unreviewed ≠ approved). Plus 116 natively-clean negatives. ~106 total dropped with
  logged reasons. Scope note: this instrument is a SUPPORTING row in the Phase-1 report
  card (logo-dependence delta), not load-bearing — the same question is probed by
  wordmark-removal corruptions and occlusion maps; Phases 2-3 do not use it.
- **Standing caveats, stated:** (1) masking completeness is verified by detection +
  triple-redundant audit, never assumed — and the measured logo-dependence delta remains
  a LOWER bound; (2) inpainting artifacts bias the delta conservative (they make masked
  images easier to reject, shrinking the apparent logo dependence); (3) the logo-masked
  probe is one of three instruments on this question (rhode wordmark-removal corruptions
  and occlusion maps are the others) — no conclusion rests on it alone.

## Finding 2 (July 5): corpus certification complete; rhode drifts fast
- Probe C (null control, 321 rhode images in one 14-day window, random A/B, image-split):
  **50.6% held-out ≈ chance** — dedup repair holds; probe methodology produces null when null is true.
- Probe A (temporal, rhode oldest-40% vs newest-40% clusters, cluster-split): **77.5% held-out** —
  rhode's creative identity measurably drifted within ~a year (evolution + any time-correlated
  artifacts, entangled by design; forensic source-probe apportions later). Confirms the
  temporal-holdout design matters: judges that memorize products will be punished.
- Dual-rater audit reconciliation (author vs independent model rater, 30 panels): ~90% agreement;
  both found the same mis-merged cluster and the same cross-split twins. Agreement certifies the
  model rater for future solo audit passes.
