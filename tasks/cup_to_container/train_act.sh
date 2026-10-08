#!/bin/bash
# ACT on the cup-to-container task, in tmux session act-bluecup on GPU 0.
# Usage: bash tasks/cup_to_container/train_act.sh [DATASET_REPO_ID] [RUN_NAME] [GPU]
#
# Default dataset is BlueCupToContainer (224x224). ACT looks at the images directly, so pass
# andnetdeboer/CupToContainer to train at full resolution instead.

set -euo pipefail

DATASET_REPO_ID="${1:-andnetdeboer/BlueCupToContainer}"
RUN_NAME="${2:-act_bluecup_run1}"
GPU="${3:-0}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="${REPO_DIR}/outputs/${RUN_NAME}"

"${REPO_DIR}/tools/tmux_train.sh" "act-${RUN_NAME//_/-}" "${GPU}" "${OUTPUT_DIR}/train.log" -- \
    python tasks/cup_to_container/train.py \
    --policy_type act \
    --dataset_repo_id "${DATASET_REPO_ID}" \
    --output_dir "${OUTPUT_DIR}" \
    --steps 100000 \
    --batch_size 8 \
    --save_freq 10000 \
    --log_freq 100 \
    --num_workers 8 \
    --hub_repo_id "andnetdeboer/act_${DATASET_REPO_ID##*/}" \
    --wandb_project rethoughtai-act \
    --wandb_entity deboerandnet-northwestern-university
