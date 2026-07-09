"""Embed the corpus with SigLIP (google/siglip-so400m-patch14-384).

Produces data/features/embeddings_<tag>.npz containing:
  files: list of "<brand>/<fname>" keys
  vecs:  float32 [N, D] L2-normalized image embeddings
Runs on MPS (Apple Silicon) if available. Usage:
  python judges/embed.py <images_dir> <tag>
"""
import json, os, sys
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL = "google/siglip-so400m-patch14-384"
BATCH = 16

def main(img_root, tag):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()

    files = []
    for brand in sorted(os.listdir(img_root)):
        bdir = os.path.join(img_root, brand)
        if not os.path.isdir(bdir): continue
        for f in sorted(os.listdir(bdir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append(f"{brand}/{f}")
    print("images:", len(files))

    vecs = []
    with torch.no_grad():
        for i in range(0, len(files), BATCH):
            batch = files[i:i+BATCH]
            imgs = [Image.open(os.path.join(img_root, f)).convert("RGB") for f in batch]
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            out = model.get_image_features(**inputs)
            feats = out if torch.is_tensor(out) else out.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
            vecs.append(feats.cpu().float().numpy())
            if (i // BATCH) % 10 == 0:
                print(f"  {i+len(batch)}/{len(files)}", flush=True)
    vecs = np.concatenate(vecs, axis=0)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "features", f"embeddings_{tag}.npz")
    np.savez_compressed(out, files=np.array(files), vecs=vecs)
    print("saved:", out, vecs.shape)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
