#!/bin/bash
# Launch ACT training on CupToContainer in a persistent tmux window.
# Usage: bash train_act_cup.sh [DATASET_DIR] [RUN_NAME]
#
# Attach later:  tmux attach -t act-cup
# Watch log:     tail -f outputs/act_cup_run1/train.log

set -euo pipefail

DATASET_DIR="${1:-${HOME}/.cache/huggingface/lerobot/andnetdeboer/CupToContainer}"
RUN_NAME="${2:-act_cup_run1}"
SESSION="act-cup"
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
tmux send-keys -t "${SESSION}" "echo 'ACT training starting — logging to ${LOG_FILE}'" Enter

tmux send-keys -t "${SESSION}" "CUDA_VISIBLE_DEVICES=0 python cup_to_container/train.py \
  --policy_type act \
  --dataset_dir '${DATASET_DIR}' \
  --output_dir '${OUTPUT_DIR}' \
  --steps 100000 \
  --batch_size 8 \
  --save_freq 10000 \
  --log_freq 100 \
  --num_workers 4 \
  --hub_repo_id 'andnetdeboer/act_CupToContainer' \
  --wandb_project 'rethoughtai-act' \
  --wandb_entity 'deboerandnet-northwestern-university' \
  2>&1 | tee '${LOG_FILE}'; echo '--- PROCESS EXITED ---'" Enter

echo "ACT training launched in tmux session '${SESSION}'"
echo "  Attach:   tmux attach -t ${SESSION}"
echo "  Log:      tail -f ${LOG_FILE}"
