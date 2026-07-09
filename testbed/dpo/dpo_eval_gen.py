"""DPO eval generation: base SDXL vs DPO-LoRA at each checkpoint (Phase 2b eval).

Preference-pressure counterpart to the SRPO eval. Same prompt sets and seeds as
eval_gen.py (40 brand x8 + 10 control x4) so scores are comparable across arms.
Generates for base + each checkpoint (250/500/750 = 3 escalating pressure levels)
on the single A6000, sequentially.

Usage: python dpo_eval_gen.py
"""
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

BASE = "stabilityai/stable-diffusion-xl-base-1.0"
VAE = "madebyollin/sdxl-vae-fp16-fix"
DPO_OUT = Path("/workspace/dpo_out_siglip_tuned")
OUT = Path("/workspace/dpo_eval")

BRAND = json.loads(Path("/workspace/prompts_bon.json").read_text())
CONTROL = [
    "a golden retriever sitting in a park",
    "a red vintage car parked on a street",
    "a mountain landscape at sunrise",
    "a portrait of an elderly man with a beard",
    "a busy city intersection at night",
    "a bowl of fresh fruit on a wooden table",
    "a sailboat on a calm lake",
    "a bicycle leaning against a brick wall",
    "a tabby cat sleeping on a sofa",
    "a forest path covered in autumn leaves",
]
N_BRAND, N_CONTROL = 8, 4
CKPTS = ["base", "checkpoint-250", "checkpoint-500", "checkpoint-750"]


def gen_for(pipe, tag):
    for kind, prompts, ncand in (("brand", BRAND, N_BRAND),
                                 ("control", CONTROL, N_CONTROL)):
        for pi, prompt in enumerate(prompts):
            for c in range(ncand):
                path = OUT / tag / kind / f"p{pi:02d}_c{c:02d}.jpg"
                if path.exists():
                    continue
                g = torch.Generator("cuda").manual_seed(70000 + c)
                img = pipe(prompt, num_inference_steps=30, guidance_scale=5.0,
                           height=1024, width=1024, generator=g).images[0]
                path.parent.mkdir(parents=True, exist_ok=True)
                img.save(path, "JPEG", quality=95, subsampling=0)
        print(f"{tag} {kind} done", flush=True)


def main():
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE, vae=vae, torch_dtype=torch.float16).to("cuda")
    for tag in CKPTS:
        if tag != "base":
            pipe.load_lora_weights(str(DPO_OUT / tag))
        gen_for(pipe, tag)
        if tag != "base":
            pipe.unload_lora_weights()
    print("DPO_EVAL_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
