#!/bin/bash
# Launch SmolVLA training on BlueCupToContainer in a persistent tmux window.
# Usage: bash train_smolvla_cup.sh [DATASET_REPO_ID] [RUN_NAME]
#
# Attach later:  tmux attach -t smolvla-bluecup
# Watch log:     tail -f outputs/smolvla_bluecup_run1/train.log

set -euo pipefail

DATASET_REPO_ID="${1:-andnetdeboer/BlueCupToContainer}"
RUN_NAME="${2:-smolvla_bluecup_run1}"
SESSION="smolvla-bluecup"
LEROBOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${LEROBOT_DIR}/outputs/${RUN_NAME}"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p "${OUTPUT_DIR}"

# Kill stale session if it exists
tmux kill-session -t "${SESSION}" 2>/dev/null || true

tmux new-session -d -s "${SESSION}" -x 220 -y 50
tmux set-option -t "${SESSION}" remain-on-exit on

tmux send-keys -t "${SESSION}" "cd ${LEROBOT_DIR}" Enter
tmux send-keys -t "${SESSION}" "source .venv/bin/activate" Enter
tmux send-keys -t "${SESSION}" "echo 'SmolVLA training starting — logging to ${LOG_FILE}'" Enter

tmux send-keys -t "${SESSION}" "CUDA_VISIBLE_DEVICES=1 python cup_to_container/train.py \
  --policy_type smolvla \
  --dataset_repo_id '${DATASET_REPO_ID}' \
  --output_dir '${OUTPUT_DIR}' \
  --steps 30000 \
  --batch_size 64 \
  --save_freq 5000 \
  --log_freq 100 \
  --num_workers 8 \
  --hub_repo_id 'andnetdeboer/smolvla_BlueCupToContainer' \
  --wandb_project 'rethoughtai-smolvla' \
  --wandb_entity 'deboerandnet-northwestern-university' \
  2>&1 | tee '${LOG_FILE}'; echo '--- PROCESS EXITED ---'" Enter

echo "SmolVLA training launched in tmux session '${SESSION}'"
echo "  Attach:   tmux attach -t ${SESSION}"
echo "  Log:      tail -f ${LOG_FILE}"
