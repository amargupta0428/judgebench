"""Pod-side: register SigLIPBrand in Tencent SRPO's reward registry.

Idempotent text patch of fastvideo/SRPO.py — inserts the import and an elif
branch. Run from /workspace/SRPO. Verifies by importing the patched module
path afterwards is left to the smoke run.
"""
from pathlib import Path

SRPO = Path("/workspace/SRPO/fastvideo/SRPO.py")
src = SRPO.read_text()

if "SigLIPBrand" in src:
    print("already patched")
    raise SystemExit

src = src.replace(
    "supported_models = [\"HPS\", \"CLIP\", \"PickScore\"]",
    "supported_models = [\"HPS\", \"CLIP\", \"PickScore\", \"SigLIPBrand\"]\n"
    "    import sys as _sys; _sys.path.insert(0, './data')\n"
    "    from siglip_reward import SigLIPBrand", 1)

src = src.replace(
    "    elif args.reward_model == \"PickScore\":",
    "    elif args.reward_model == \"SigLIPBrand\":\n"
    "        print(f\"Initializing {args.reward_model} reward model...\")\n"
    "        reward_model = SigLIPBrand(\n"
    "            model_path=\"./data/siglip_tuned\",\n"
    "            params_path=\"./data/j3_tuned_params.json\").to(device)\n"
    "    elif args.reward_model == \"PickScore\":", 1)

SRPO.write_text(src)
print("patched: registry + import + SigLIPBrand branch")
