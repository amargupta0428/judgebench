"""BoN candidate-pool generation on FLUX.1-dev (Phase 2a).

Design (spec 2a): broad arm = 40 prompts x 64 candidates (every judge scores
this pool, incl. GPT-4o); deep arm = first 8 prompts extended to 512 candidates
(cheap local judges only — the large-N region where a Gao-style gold DECLINE
would show; ReflectionFlow saw only a plateau). Candidate index k always maps
to seed 50000+k, so the deep arm continues the broad arm without regeneration,
and every image is reproducible from its manifest line. Base FLUX.1-dev, no
brand LoRA: all brand pressure comes from judge selection, nothing else.
The pool doubles as the DPO pair source (spec 2b).

Resumable: existing files are skipped. Usage: python pod_bon.py smoke|run
"""
import json
import sys
import time
from pathlib import Path

import torch
from diffusers import FluxPipeline

HERE = Path(__file__).resolve().parent
PROMPTS = json.loads((HERE / "prompts_bon.json").read_text())
N_BROAD, DEEP_PROMPTS, N_DEEP = 64, 8, 512
STEPS, GUIDANCE, SIZE = 28, 3.5, 1024
OUT = Path("/workspace/bon_out")


def jobs(smoke=False):
    if smoke:
        return [(pi, c) for pi in range(5) for c in range(4)]
    out = [(pi, c) for pi in range(len(PROMPTS)) for c in range(N_BROAD)]
    out += [(pi, c) for pi in range(DEEP_PROMPTS) for c in range(N_BROAD, N_DEEP)]
    return out


BATCH = 4  # images per pipeline call (same prompt, per-image seeded generators)


def main(mode):
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                        torch_dtype=torch.bfloat16).to("cuda")
    todo = [j for j in jobs(mode == "smoke")
            if not (OUT / f"p{j[0]:02d}/c{j[1]:04d}.jpg").exists()]
    OUT.mkdir(parents=True, exist_ok=True)
    mf = (BON_MANIFEST := OUT / "manifest_bon.jsonl").open("a")
    t0, done, total = time.time(), 0, len(todo)
    # same-prompt batches, ordered broad-tier-first across ALL prompts (c<64
    # for p00..p39, then the deep extension) so the broad arm finishes early
    # and API-judge scoring can start while the deep arm generates. Candidate
    # seed stays 50000+c regardless of batch shape.
    batches = []
    for tier in (lambda c: c < N_BROAD, lambda c: c >= N_BROAD):
        by_prompt = {}
        for pi, c in todo:
            if tier(c):
                by_prompt.setdefault(pi, []).append(c)
        for pi, cands in by_prompt.items():
            for j in range(0, len(cands), BATCH):
                batches.append((pi, cands[j:j + BATCH]))
    for pi, chunk in batches:
        if True:
            gens = [torch.Generator("cuda").manual_seed(50000 + c) for c in chunk]
            imgs = pipe(PROMPTS[pi], num_inference_steps=STEPS,
                        guidance_scale=GUIDANCE, height=SIZE, width=SIZE,
                        generator=gens, num_images_per_prompt=len(chunk)).images
            for c, img in zip(chunk, imgs):
                path = OUT / f"p{pi:02d}/c{c:04d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                img.save(path, "JPEG", quality=95, subsampling=0)
                mf.write(json.dumps(
                    {"file": f"p{pi:02d}/c{c:04d}.jpg", "prompt_idx": pi,
                     "prompt": PROMPTS[pi], "candidate": c, "seed": 50000 + c,
                     "steps": STEPS, "guidance": GUIDANCE, "size": SIZE,
                     "model": "black-forest-labs/FLUX.1-dev"}) + "\n")
            mf.flush()
            done += len(chunk)
            if (done // BATCH) % 5 == 0:
                r = (time.time() - t0) / done
                print(f"{done}/{total} {r:.1f}s/img "
                      f"eta {(total - done) * r / 3600:.1f}h "
                      f"est ${(total - done) * r / 3600 * 2.99:.0f} remaining",
                      flush=True)
    print(f"BON_GEN_DONE rate={(time.time()-t0)/max(done,1):.1f}s/img new={done}",
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
