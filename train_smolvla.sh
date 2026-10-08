#!/bin/bash
# SmolVLA fine-tuning script for andnetdeboer/rethoughtai dataset collection.
# Usage: bash train_smolvla.sh [DATASET_REPO_ID] [RUN_NAME]
#   DATASET_REPO_ID: HuggingFace dataset repo (default: andnetdeboer/rethoughtai)
#   RUN_NAME:        output subdirectory name (default: smolvla_run1)
#
# Initialises from the official lerobot/smolvla_base checkpoint.
# Before running:
#   source .venv/bin/activate   (or: uv run bash train_smolvla.sh ...)
#   export HF_TOKEN=hf_...      (if dataset or base model is private)

DATASET_REPO_ID="${1:-andnetdeboer/rethoughtai}"
RUN_NAME="${2:-smolvla_run1}"
OUTPUT_DIR="outputs/${RUN_NAME}"

# Init from the public SmolVLA base checkpoint (expert + VLM weights).
PRETRAINED_CHECKPOINT="lerobot/smolvla_base"

accelerate launch \
  --num_processes=1 \
  --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --policy="${PRETRAINED_CHECKPOINT}" \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.n_obs_steps=1 \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.resize_imgs_with_padding='[512,512]' \
  --policy.max_state_dim=32 \
  --policy.max_action_dim=32 \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=30000 \
  --policy.gradient_checkpointing=true \
  --policy.push_to_hub=true \
  --policy.repo_id="andnetdeboer/smolvla-rethoughtai-${RUN_NAME}" \
  --wandb.enable=true \
  --wandb.project=rethoughtai-smolvla \
  --wandb.entity=deboerandnet-northwestern-university \
  --output_dir="${OUTPUT_DIR}" \
  --steps=30000 \
  --batch_size=8 \
  --num_workers=4 \
  --log_freq=100 \
  --save_checkpoint=true \
  --save_freq=5000
