#!/bin/bash
# SRPO with QwenVL-7B LoRA brand reward — 4xH100 (7B reward per rank; 2x is
# hopeless: SigLIP 400M already OOM'd 2x). Deltas vs launch_srpo.sh: reward
# QwenBrand, output dir, and checkpointing_steps=200 (Tencent's trainer saves
# ONCE at checkpointing_steps then only renders visualizations — the SigLIP
# arm's artifact is checkpoint-200-0; we match it exactly for comparability).
cd /workspace/SRPO
mkdir -p images output/qwen
# First 4x smoke OOM'd by 2 MiB with 610 MiB fragmented-reserved; this
# reclaims it (and is what the OOM message itself recommends).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node ${NPROC:-4} \
    fastvideo/SRPO.py \
    --seed 42 \
    --pretrained_model_name_or_path /workspace/flux_dev \
    --vae_model_path /workspace/flux_dev \
    --cache_dir data/.cache \
    --data_json_path data/rl_embeddings/videos2caption2.json \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 2 \
    --max_train_steps ${STEPS:-210} \
    --learning_rate 5e-6 \
    --mixed_precision bf16 \
    --checkpointing_steps ${CKPT_STEPS:-200} \
    --allow_tf32 \
    --train_guidence 1 \
    --output_dir ./output/qwen/ \
    --h 720 \
    --w 720 \
    --t 1 \
    --sampling_steps 25 \
    --image_p 'srpohps' \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 0.1 \
    --weight_decay 0.0001 \
    --shift 3 \
    --ignore_last \
    --discount_inv 0.3 0.01 \
    --discount_pos 0.1 0.25 \
    --timestep_length 1000 \
    --groundtruth_ratio 0.9 \
    --reward_model 'QwenBrand'
