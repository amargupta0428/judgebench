# SRPO arm — launch plan (Phase 2b, gradient pressure)

Approved July 8, ~$175 envelope incl. 4xH100 escalation. Timeboxed,
allowed-to-fail (spec). DPO leg is the independent fallback.

## Patch points in Tencent-Hunyuan/SRPO

1. `fastvideo/SRPO.py` — add `SigLIPBrand` to the reward registry
   (`supported_models`, the `args.reward_model` if-chain). Class ships in
   `siglip_reward.py`; copy weights to `./data/siglip_tuned`, calibration to
   `./data/j3_tuned_params.json`.
2. Force forward-only reward passes (inversion arm off): image-only reward has
   no pos/neg text axis. Locate `inversion` branching in the rollout loop
   (~line 618) and the flag's source in `train_one_step`; pin to the
   `inversion==1` path. Document as "Direct-Align core, plain reward
   maximization (ReFL loss, threshold 0.7)".
3. Training data: offline mode (README: <1,500 images suffice). Use our 40
   BoN prompts as the caption set; control-word machinery unused.
4. Their loss threshold 0.7 assumes reward in (0,1): our Platt-calibrated
   sigmoid matches. Check the reward-mean logs in the first 50 steps — if the
   base model already scores >0.7 mean (unlikely: BoN pool means will tell us
   tonight), raise the threshold to keep gradient signal.

## Hardware ladder + decision rules

- Start: 2xH100 SXM (~$6/hr), FSDP, VAE gradient checkpointing ON (their
  README's own memory advice), bf16, batch 1/GPU + accumulation.
- OOM after one mitigation cycle (checkpointing + batch floor + 8-bit Adam if
  trivially available) -> escalate 4xH100 (~$12/hr). Pre-authorized.
- No convergence signal (reward mean flat AND images unchanged by visual spot
  check) after ~2h of stepping -> kill, write up the attempt honestly.
- Hard budget stop: $175 total for the night incl. DPO; consult past that.

## Evaluation (after training)

Generate from the tuned model with the SAME 40 prompts x fixed seeds ->
score with the full held-out panel + instruments (same protocol as BoN
winners) -> compare against base-FLUX distribution. Hacked-vs-broken control:
non-brand prompts (dog, car, landscape) spot-checked — a judge-hacked model
still draws a normal dog; a collapsed one doesn't.

## Success criteria (pre-registered)

- Primary: judge score of tuned-model outputs rises materially vs base FLUX
  (optimization worked) while held-out panel / instruments stay flat or fall
  (Goodhart gap) — or genuinely rise (judge is robust; also a finding).
- The gap's per-dimension breakdown vs Phase 1's predicted blind spots
  (violation families, centroid-proximity) is the payoff comparison.
