"""Pod-side: dial sweep + generative corruptions. FLUX.1-dev, A100 80GB.

Usage on pod:
    python pod_sweep.py dial      # 20 prompts x 3 seeds x 6 scales = 360 images
    python pod_sweep.py corrupt   # img2img styling/mood corruptions, 180 images

Everything writes construction-record manifests alongside images.
"""

import json
import sys
from pathlib import Path

import torch

PROMPTS = json.loads(Path("prompts.json").read_text())
LORA = "output/rhode_lora/rhode_lora.safetensors"   # ai-toolkit output path
SCALES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = [1001, 1002, 1003]
OUT = Path("sweep_out")


def dial():
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                        torch_dtype=torch.bfloat16).to("cuda")
    pipe.load_lora_weights(LORA, adapter_name="rhode")
    records = []
    OUT.mkdir(exist_ok=True)
    for pi, prompt in enumerate(PROMPTS["dial_sweep"]):
        for seed in SEEDS:
            for scale in SCALES:
                pipe.set_adapters(["rhode"], adapter_weights=[scale])
                g = torch.Generator("cuda").manual_seed(seed)
                img = pipe(prompt, num_inference_steps=28, guidance_scale=3.5,
                           height=1024, width=1024, generator=g).images[0]
                name = f"dial/p{pi:02d}_s{seed}_w{int(scale*100):03d}.jpg"
                (OUT / name).parent.mkdir(parents=True, exist_ok=True)
                img.save(OUT / name, "JPEG", quality=95, subsampling=0)
                records.append({"file": name, "prompt_idx": pi, "prompt": prompt,
                                "seed": seed, "adapter_scale": scale,
                                "label_ordinal": scale, "steps": 28, "guidance": 3.5})
        print(f"prompt {pi+1}/20 done")
    with (OUT / "manifest_dial.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def corrupt():
    from diffusers import FluxImg2ImgPipeline
    from PIL import Image
    pipe = FluxImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                               torch_dtype=torch.bfloat16).to("cuda")
    bases = json.loads(Path("corrupt_bases.json").read_text())  # 30 test-split images, shipped up
    records = []
    for dim in ("styling_corruption", "mood_corruption"):
        for sev in ("s1", "s2", "s3"):
            prompt = PROMPTS[dim][sev]
            strength = PROMPTS["img2img_strength"][sev]
            for b in bases:
                src = Image.open(f"bases/{b['file']}").convert("RGB")
                src = src.resize((1024, 1024)) if max(src.size) > 1024 else src
                g = torch.Generator("cuda").manual_seed(b["seed"])
                img = pipe(prompt=prompt, image=src, strength=strength,
                           num_inference_steps=28, guidance_scale=3.5,
                           generator=g).images[0]
                name = f"corrupt/{dim.split('_')[0]}/{sev}/{b['file']}"
                (OUT / name).parent.mkdir(parents=True, exist_ok=True)
                img.save(OUT / name, "JPEG", quality=95, subsampling=0)
                records.append({"file": name, "source_id": b["source_id"],
                                "dimension": dim.split("_")[0], "severity": int(sev[1]),
                                "prompt": prompt, "strength": strength,
                                "seed": b["seed"], "label": "off-brand",
                                "ground_truth_class": "generative"})
            print(f"{dim}/{sev} done")
    with (OUT / "manifest_corrupt.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    {"dial": dial, "corrupt": corrupt}[sys.argv[1]]()
