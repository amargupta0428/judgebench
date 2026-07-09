"""Pod-side: precompute FLUX text embeddings for the 40 BoN prompts (SRPO data).

Produces the LatentDataset layout under /workspace/SRPO/data/rl_embeddings/:
  prompt_embed/{i}.pt            T5 embeds (256, 4096)
  pooled_prompt_embeds/{i}.pt    CLIP pooled (768,)
  text_ids/{i}.pt                (256, 3) zeros (FLUX convention)
  videos2caption2.json           entries with caption/prompt fields

The 'prompt' field's "pos * neg" control-word format is filled with the plain
prompt on both sides — our image-only reward ignores text.
"""
import json
from pathlib import Path

import torch

FLUX = "/workspace/flux_dev"
OUT = Path("/workspace/SRPO/data/rl_embeddings")
PROMPTS = json.loads(Path("/workspace/SRPO/data/prompts_bon.json").read_text())


def main():
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(
        FLUX, transformer=None, vae=None, torch_dtype=torch.bfloat16).to("cuda")
    for d in ("prompt_embed", "pooled_prompt_embeds", "text_ids"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    anno = []
    for i, p in enumerate(PROMPTS):
        with torch.no_grad():
            embeds, pooled, text_ids = pipe.encode_prompt(
                prompt=p, prompt_2=p, max_sequence_length=256)
        torch.save(embeds[0].float().cpu(), OUT / "prompt_embed" / f"{i}.pt")
        torch.save(pooled[0].float().cpu(),
                   OUT / "pooled_prompt_embeds" / f"{i}.pt")
        torch.save(text_ids.float().cpu(), OUT / "text_ids" / f"{i}.pt")
        anno.append({"prompt_embed_path": f"{i}.pt", "text_ids": f"{i}.pt",
                     "pooled_prompt_embeds_path": f"{i}.pt",
                     "caption": p, "prompt": f"{p} * {p}"})
        print(f"{i+1}/{len(PROMPTS)}", flush=True)
    (OUT / "videos2caption2.json").write_text(json.dumps(anno, indent=1))
    print("PREP_PROMPTS_DONE")


if __name__ == "__main__":
    main()
