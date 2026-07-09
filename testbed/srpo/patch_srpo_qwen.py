"""Pod-side: register QwenBrand in Tencent SRPO's reward registry.

Idempotent text patch of fastvideo/SRPO.py, mirroring patch_srpo.py (the
SigLIP arm). Safe to run on a tree already patched for SigLIPBrand — each
replacement anchors on the PickScore branch that survives prior patches.
Run from /workspace/SRPO. Expects ./data/qwen_reward.py + ./data/qwen_lora_adapter.
"""
from pathlib import Path

SRPO = Path("/workspace/SRPO/fastvideo/SRPO.py")
src = SRPO.read_text()

if "QwenBrand" in src:
    print("already patched")
    raise SystemExit

# registry line may or may not already carry SigLIPBrand
for probe in ('supported_models = ["HPS", "CLIP", "PickScore", "SigLIPBrand"]',
              'supported_models = ["HPS", "CLIP", "PickScore"]'):
    if probe in src:
        src = src.replace(
            probe,
            probe[:-1] + ', "QwenBrand"]\n'
            "    import sys as _sys; _sys.path.insert(0, './data')\n"
            "    from qwen_reward import QwenBrand", 1)
        break
else:
    raise SystemExit("registry anchor not found")

src = src.replace(
    '    elif args.reward_model == "PickScore":',
    '    elif args.reward_model == "QwenBrand":\n'
    '        print(f"Initializing {args.reward_model} reward model...")\n'
    '        reward_model = QwenBrand(\n'
    '            adapter_path="./data/qwen_lora_adapter", device=device)\n'
    '    elif args.reward_model == "PickScore":', 1)

# txt_ids shape fix (rediscovered July 9; the SigLIP-arm pod carried this as a
# hot edit that never made it into the recipe): our per-item text_ids are the
# documented (256,3) zeros, the collate batches to (1,256,3), and diffusers
# 0.32's FluxTransformer2DModel wants 2D (seq,3) — Tencent's
# `text_ids.repeat(encoder_hidden_states.shape[1],1)` errors on the 3D tensor.
n = src.count("txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1)")
src = src.replace(
    "txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1)",
    "txt_ids=text_ids.reshape(-1, text_ids.shape[-1])")
print(f"txt_ids shape fix applied at {n} sites")

# VAE gradient checkpointing (SRPO README's own memory advice; the stock
# script only enables tiling). The decode sits INSIDE the reward graph, so
# its activations (~GBs at 720px, all tiles retained until backward) are the
# other big lever. Checkpointing is exact; .train() is required for it to
# engage and is safe (no dropout / batch-dependent norms in the VAE).
vae_anchor = """    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype = torch.bfloat16,
    ).to(device)"""
assert vae_anchor in src, "VAE anchor not found"
src = src.replace(vae_anchor, vae_anchor +
                  "\n    vae.enable_gradient_checkpointing()\n"
                  "    vae.train()", 1)

SRPO.write_text(src)
print("patched: registry + import + QwenBrand branch + txt_ids fix + vae ckpt")
