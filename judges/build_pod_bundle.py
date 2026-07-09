"""Assemble the pod job bundle for the July 7 session (SigLIP-v2 + Qwen LoRA).

Produces <out>/job/ with:
  train_manifest_v2.json   SupCon classes: rhode=1, competitor=0, corrupted-rhode=2
  qwen_lora_manifest.json  rhode->"yes", glossier/ilia->"no" (train split only;
                           NO corruption negatives -- matched pair with SigLIP-v1)
  embed_corpus.json        train+val images, all brands (local centroid/Platt fit)
  embed_testset.json       the full 2,622-item test set
  images...                copied under repo-relative paths

Usage: .venv/bin/python judges/build_pod_bundle.py <out_dir>
"""
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) / "job"
CORR = REPO / "data" / "testset" / "train_neg_corruptions"

splits = json.loads((REPO / "data/features/splits_v2.json").read_text())
index = splits["_image_index"]


def img_path(image_id):
    src, rel = image_id.split(":", 1)
    folder = "images" if src == "fb" else "ig_images"
    return f"data/scrape/raw/{folder}/{rel}"


def copy(rel):
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(REPO / rel, dst)
    return rel


train_v2, qwen_manifest, corpus = [], [], []
for image_id, split in sorted(index.items()):
    brand = image_id.split(":")[1].split("/")[0]
    if split in ("train", "val"):
        corpus.append({"id": image_id, "path": copy(img_path(image_id))})
    if split == "train":
        label = 1 if brand == "rhode" else 0
        train_v2.append({"path": img_path(image_id), "label": label})
        qwen_manifest.append({"path": img_path(image_id),
                              "answer": "yes" if brand == "rhode" else "no"})

n_corr = 0
for mf in sorted(CORR.glob("manifest_*.jsonl")):
    for line in mf.open():
        rec = json.loads(line)
        rel = f"data/testset/train_neg_corruptions/{rec['out_file']}"
        copy(rel)
        train_v2.append({"path": rel, "label": 2})
        n_corr += 1

testset = []
for line in (REPO / "eval/testset_index.jsonl").open():
    it = json.loads(line)
    rel = str(Path(it["path"]).resolve().relative_to(REPO))
    testset.append({"id": it["item_id"], "path": copy(rel)})

(OUT / "train_manifest_v2.json").write_text(json.dumps(train_v2))
(OUT / "qwen_lora_manifest.json").write_text(json.dumps(qwen_manifest))
(OUT / "embed_corpus.json").write_text(json.dumps(corpus))
(OUT / "embed_testset.json").write_text(json.dumps(testset))
shutil.copy2(REPO / "judges/pod_judges.py", OUT / "pod_judges.py")

from collections import Counter
c = Counter(x["label"] for x in train_v2)
print(f"train_v2: {dict(c)} (2=corruption negs, n={n_corr})")
print(f"qwen_lora: {len(qwen_manifest)} | corpus: {len(corpus)} | testset: {len(testset)}")
print(f"bundle at {OUT}")
