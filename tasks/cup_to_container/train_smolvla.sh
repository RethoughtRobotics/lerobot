#!/bin/bash
# SmolVLA on the cup-to-container task, in a tmux session on GPU 1.
# Usage: bash tasks/cup_to_container/train_smolvla.sh [DATASET_REPO_ID] [RUN_NAME] [GPU]
#
# Default dataset is the full-resolution CupToContainer: SmolVLA resizes every image to 512x512
# internally, so 224x224 sources would only lose detail. It is GPU-bound, not data-bound.

set -euo pipefail

DATASET_REPO_ID="${1:-andnetdeboer/CupToContainer}"
RUN_NAME="${2:-smolvla_cup_run1}"
GPU="${3:-1}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="${REPO_DIR}/outputs/${RUN_NAME}"

"${REPO_DIR}/tools/tmux_train.sh" "smolvla-${RUN_NAME//_/-}" "${GPU}" "${OUTPUT_DIR}/train.log" -- \
    python tasks/cup_to_container/train.py \
    --policy_type smolvla \
    --dataset_repo_id "${DATASET_REPO_ID}" \
    --output_dir "${OUTPUT_DIR}" \
    --steps 30000 \
    --batch_size 64 \
    --save_freq 5000 \
    --log_freq 100 \
    --num_workers 8 \
    --hub_repo_id "andnetdeboer/smolvla_${DATASET_REPO_ID##*/}" \
    --wandb_project rethoughtai-smolvla \
    --wandb_entity deboerandnet-northwestern-university
