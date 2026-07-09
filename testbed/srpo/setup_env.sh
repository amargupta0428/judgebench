#!/bin/bash
# SRPO pod environment — the exact recipe distilled from the July 8 debug
# gauntlet. Run on a fresh runpod pytorch-2.8 image AFTER /workspace/SRPO
# exists. Each fix is load-bearing; do not reorder.
set -e

# 1. The image ships a NIGHTLY dev torch whose ABI matches no release wheel.
pip install -q torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. flash-attn: only the version-matched GitHub release wheel links cleanly.
pip install -q --force-reinstall --no-deps \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

# 3. SRPO deps + the pins that keep Qwen/SigLIP loaders happy.
pip install -q -e /workspace/SRPO || true
pip install -q "transformers==4.57.1" diffusers accelerate sentencepiece protobuf hpsv2
pip install -q -U bitsandbytes   # old bnb imports removed triton.ops

# 4. hpsv2 packaging bugs: stray turtle import + unshipped BPE vocab.
HPS=/usr/local/lib/python3.11/dist-packages/hpsv2/src/open_clip
sed -i "/from turtle import forward/d" $HPS/factory.py
[ -s $HPS/bpe_simple_vocab_16e6.txt.gz ] || \
  curl -sL https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz \
    -o $HPS/bpe_simple_vocab_16e6.txt.gz

python - << 'EOF'
import torch
from flash_attn import flash_attn_varlen_qkvpacked_func
from hpsv2.src.open_clip import create_model_and_transforms
import diffusers, transformers
print("ENV_READY:", torch.__version__)
EOF
