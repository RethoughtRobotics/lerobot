#!/bin/bash
# ACT training script for andnetdeboer/rethoughtai dataset collection.
# Usage: bash tasks/rethoughtai/train_act.sh [DATASET_REPO_ID] [RUN_NAME]
#   DATASET_REPO_ID: HuggingFace dataset repo (default: andnetdeboer/rethoughtai)
#   RUN_NAME:        output subdirectory name (default: act_run1)
#
# Before running:
#   source .venv/bin/activate   (or: uv run bash tasks/rethoughtai/train_act.sh ...)
#   export HF_TOKEN=hf_...      (if dataset is private)

DATASET_REPO_ID="${1:-andnetdeboer/rethoughtai}"
RUN_NAME="${2:-act_run1}"
OUTPUT_DIR="outputs/${RUN_NAME}"

accelerate launch \
  --num_processes=1 \
  --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.n_obs_steps=1 \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.dim_model=512 \
  --policy.n_heads=8 \
  --policy.dim_feedforward=3200 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=1 \
  --policy.use_vae=true \
  --policy.latent_dim=32 \
  --policy.kl_weight=10.0 \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"MEAN_STD"}' \
  --policy.push_to_hub=true \
  --policy.repo_id="andnetdeboer/act-rethoughtai-${RUN_NAME}" \
  --wandb.enable=true \
  --wandb.project=rethoughtai-act \
  --wandb.entity=deboerandnet-northwestern-university \
  --output_dir="${OUTPUT_DIR}" \
  --steps=100000 \
  --batch_size=8 \
  --num_workers=4 \
  --log_freq=200 \
  --save_checkpoint=true \
  --save_freq=10000
