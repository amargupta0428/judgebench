#!/bin/bash
# SRPO with SigLIP brand reward — single node, 2xH100 (escalate to 4x on OOM).
# Modeled on scripts/finetune/SRPO_training_hpsv2.sh; deltas: our model path,
# our embeddings, --reward_model SigLIPBrand, bounded max_train_steps.
cd /workspace/SRPO
mkdir -p images output/siglip

torchrun --standalone --nproc_per_node ${NPROC:-2} \
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
    --max_train_steps ${STEPS:-600} \
    --learning_rate 5e-6 \
    --mixed_precision bf16 \
    --checkpointing_steps 50 \
    --allow_tf32 \
    --train_guidence 1 \
    --output_dir ./output/siglip/ \
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
    --reward_model 'SigLIPBrand'
