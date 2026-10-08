#!/bin/bash
# Run any training command in a detached tmux session, pinned to a GPU, with the output teed to a log.
#
# Usage:
#   tools/tmux_train.sh <session> <gpu> <log_file> -- <command...>
#
# Examples:
#   tools/tmux_train.sh act-mytask 0 outputs/act_mytask/train.log -- \
#       lerobot-train --policy.type=act --dataset.repo_id=me/MyTask --output_dir=outputs/act_mytask
#   tools/tmux_train.sh act-bluecup 0 outputs/act_bluecup_run1/train.log -- \
#       python tasks/cup_to_container/train.py --policy_type act --output_dir outputs/act_bluecup_run1
#
# Refuses to replace an existing session, so a live run is never killed by accident.

set -euo pipefail

if [ $# -lt 5 ] || [ "$4" != "--" ]; then
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

SESSION="$1"
GPU="$2"
LOG_FILE="$3"
shift 4
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session '${SESSION}' already exists; attach with 'tmux attach -t ${SESSION}' or pick another name." >&2
    exit 1
fi

mkdir -p "$(dirname "${LOG_FILE}")"
CMD="$(printf '%q ' "$@")"

tmux new-session -d -s "${SESSION}" -x 220 -y 50 -c "${REPO_DIR}"
tmux set-option -t "${SESSION}" remain-on-exit on
tmux send-keys -t "${SESSION}" "source .venv/bin/activate" Enter
tmux send-keys -t "${SESSION}" "CUDA_VISIBLE_DEVICES=${GPU} ${CMD} 2>&1 | tee '${LOG_FILE}'; echo \"--- PROCESS\" \"EXITED (\$(date +%T)) ---\"" Enter

echo "Launched in tmux session '${SESSION}' on GPU ${GPU}"
echo "  Attach:  tmux attach -t ${SESSION}"
echo "  Log:     tail -f ${LOG_FILE}"
