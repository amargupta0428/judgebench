"""Paired eval generation: SRPO-tuned checkpoint vs base FLUX (Phase 2b eval).

Per model: 40 brand prompts x 8 candidates (seeds 70000+c) + 10 non-brand
control prompts x 4 candidates (hacked-vs-broken: a judge-hacked model still
draws a normal dog). Params match the BoN pool (28 steps, guidance 3.5,
1024px) so judge scores are comparable. Sharded by prompt across GPUs.

Usage (per GPU): CUDA_VISIBLE_DEVICES=k python eval_gen.py {base|tuned} k 4
"""
import json
import sys
from pathlib import Path

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel

MODEL, SHARD, NSHARDS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
FLUX = "/workspace/flux_dev"
CKPT = "/workspace/SRPO/output/siglip/checkpoint-200-0"
OUT = Path(f"/workspace/srpo_eval/{MODEL}")

BRAND = json.loads(Path("/workspace/SRPO/data/prompts_bon.json").read_text())
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


def jobs():
    out = [("brand", pi, p, c) for pi, p in enumerate(BRAND) for c in range(N_BRAND)]
    out += [("control", pi, p, c) for pi, p in enumerate(CONTROL) for c in range(N_CONTROL)]
    return [j for k, j in enumerate(out) if k % NSHARDS == SHARD]


def main():
    if MODEL == "tuned":
        transformer = FluxTransformer2DModel.from_pretrained(
            CKPT, torch_dtype=torch.bfloat16)
        pipe = FluxPipeline.from_pretrained(
            FLUX, transformer=transformer, torch_dtype=torch.bfloat16).to("cuda")
    else:
        pipe = FluxPipeline.from_pretrained(
            FLUX, torch_dtype=torch.bfloat16).to("cuda")
    todo = jobs()
    for n, (kind, pi, prompt, c) in enumerate(todo):
        path = OUT / kind / f"p{pi:02d}_c{c:02d}.jpg"
        if path.exists():
            continue
        g = torch.Generator("cuda").manual_seed(70000 + c)
        img = pipe(prompt, num_inference_steps=28, guidance_scale=3.5,
                   height=1024, width=1024, generator=g).images[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, "JPEG", quality=95, subsampling=0)
        if n % 10 == 0:
            print(f"shard{SHARD} {MODEL} {n}/{len(todo)}", flush=True)
    print(f"EVAL_GEN_DONE shard{SHARD} {MODEL}", flush=True)


if __name__ == "__main__":
    main()
