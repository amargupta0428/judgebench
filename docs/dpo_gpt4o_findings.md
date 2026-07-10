# DPO preference attack via GPT-4o preferences (July 9)

**Question.** DPO (SDXL Diffusion-DPO-LoRA) was trained using GPT-4o's own
preferences as the reward — pairs harvested from the BoN pool where GPT-4o's
highest-rated candidate is the *chosen* image. This is the twin of the
SigLIP-DPO arm (which drove a hack-gap of **0.11** on the trainable SigLIP-tuned
judge). Does a frozen frontier API judge — GPT-4o — also get gamed by preference
training, or does it resist? GPT-4o already **resisted BoN selection** cleanly
(peak-decline **0.00**). Preference training is a strictly stronger attack
surface than selection: the generator learns the judge's likes from labeled
pairs instead of merely sampling for them — and, crucially, it needs only the
judge's *labels*, not access to its weights.

**Setup.** SDXL Diffusion-DPO-LoRA, 750 steps, checkpoints at 250/500/750.
Eval matched to the SigLIP-DPO / SRPO arms: 40 brand prompts x8 seeds + 10
non-brand control prompts x4, over base + ck250/500/750 (360 images/stage,
1,440 total), matched seeds 70000+c. Independent panel = the four SigLIP judges
scored over the same images; attacked judge = GPT-4o (`overall_consistency/10`,
1,440/1,440 scored, 0 errors).

## The result: GPT-4o is mildly gamed by preference training

Brand-fidelity score, base -> ck250 -> ck500 -> ck750:

| judge | base | ck250 | ck500 | ck750 | delta base->ck750 |
|---|---|---|---|---|---|
| **GPT-4o (ATTACKED)** | 0.765 | 0.790 | 0.813 | **0.848** | **+0.083** |
| siglip_frozen (indep) | 0.586 | 0.605 | 0.636 | 0.659 | +0.073 |
| siglip_tuned (indep) | 0.367 | 0.360 | 0.353 | 0.350 | -0.017 |
| siglip_tuned_v2 (indep) | 0.339 | 0.315 | 0.328 | 0.324 | -0.015 |
| siglip_tuned_v3 (indep) | 0.177 | 0.161 | 0.140 | 0.130 | -0.047 |

- **Attacked-judge brand delta: +0.083** (monotone: 0.765 -> 0.790 -> 0.813 ->
  0.848 — climbs at every checkpoint).
- **Independent-mean brand delta: -0.001** (flat; the frozen SigLIP drifts up,
  the tuned SigLIPs drift down, they cancel).
- **Hack-gap = +0.083 - (-0.001) = 0.085.**
- Control leakage on the attacked judge: **+0.058** (GPT-4o's control score also
  rose, 0.258 -> 0.315), so a meaningful share of the brand climb is
  general perceived-quality gain, not brand-specific. The hack-gap nets out the
  independent panel but not this within-judge control rise; both are reported.

## Verdict

**GPT-4o is *mildly gamed* by preference training — it does NOT resist DPO the
way it resisted BoN.** The attacked judge climbs monotonically (+0.085 hack-gap)
while an independent SigLIP panel stays flat: the reward-hack signature. The
magnitude lands just below the trainable SigLIP-DPO arm (**0.085 vs 0.11**),
which is the striking part — a *frozen frontier API judge with inaccessible
weights is nearly as gameable under preference training as a small reward model
you can fine-tune.* This is exactly what the attack surface predicts: DPO never
touches the judge's parameters, it only consumes the judge's *labels*, so an API
judge's weight-inaccessibility — which blunts BoN (0.00) and SRPO gradients
(structurally n/a) — buys it almost nothing here.

The three attacks now separate cleanly by how much judge access the optimizer
needs, and GPT-4o's row tells the security story in one line:

| attack | optimizer needs | GPT-4o hack-gap | SigLIP-tuned hack-gap |
|---|---|---|---|
| BoN selection | judge scores at sample time | 0.00 (resisted) | 0.13 |
| **DPO preference** | **judge labels only (offline)** | **0.085 (gamed)** | 0.11 |
| SRPO gradient | differentiable judge weights | n/a (no gradients) | 0.45 |

**Headline:** GPT-4o is robust to *selection* pressure but only *partially*
robust to *preference* pressure. The clean "frontier API judge, robust to
everything" story its BoN result alone suggests is wrong — preference training,
which needs nothing but the judge's outputs, drifts it almost as much as a
trainable reward model. A caveat worth stating: with control leakage +0.058,
part of GPT-4o's rise is genuine quality it now scores higher, so this is a
*soft* hack, not the hard reward-collapse SRPO produced on SigLIP.

## Reproduction / artifacts

- Eval images (1,440): `eval/dpo_gpt4o/images/{base,checkpoint-250,checkpoint-500,checkpoint-750}/` (gitignored)
- LoRA checkpoints (3): `eval/dpo_gpt4o/checkpoints/checkpoint-{250,500,750}/` (gitignored)
- Eval index: `eval/dpo_gpt4o/dpo_gpt4o_eval_index.jsonl`
- SigLIP panel scores: `eval/dpo_gpt4o/siglip_scores.json`
- GPT-4o judge cache (1,440, 0 errors): `judges/cache/j2_gpt4o_dpogpt4oeval/`
- Leaderboard cell: `eval/results/dpo_gpt4o.json`, `eval/results/leaderboard.json`
