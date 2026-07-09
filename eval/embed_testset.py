"""Embed every test-set index item with SigLIP through the uniform loader."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from transformers import AutoImageProcessor, AutoModel

MODEL = "google/siglip-so400m-patch14-384"
BATCH = 12

def load_base(path):
    img = Image.open(path).convert("RGB")
    if max(img.size) > 1024:
        s = 1024 / max(img.size)
        img = img.resize((round(img.width*s), round(img.height*s)), Image.LANCZOS)
    return img

items = [json.loads(l) for l in (REPO/"eval/testset_index.jsonl").open()]
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device, "items:", len(items))
processor = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device).eval()
vecs, ids = [], []
with torch.no_grad():
    for i in range(0, len(items), BATCH):
        batch = items[i:i+BATCH]
        imgs = [load_base(b["path"]) for b in batch]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        out = model.get_image_features(**inputs)
        feats = out if torch.is_tensor(out) else out.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vecs.append(feats.cpu().float().numpy())
        ids += [b["item_id"] for b in batch]
        if (i//BATCH) % 20 == 0:
            print(f"{i+len(batch)}/{len(items)}", flush=True)
np.savez_compressed(REPO/"data/features/embeddings_testset.npz",
                    files=np.array(ids), vecs=np.concatenate(vecs))
print("saved", len(ids))
