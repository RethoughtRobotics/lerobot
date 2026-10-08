#!/bin/bash
# Polls the HuggingFace collection until andnetdeboer/CupToContainer is ready,
# then downloads the dataset and launches both ACT and SmolVLA training runs.
#
# Prerequisites:
#   source .venv/bin/activate
#   export HF_TOKEN=hf_...      (read access to private dataset)
#
# The script checks every 5 minutes. It considers the dataset "ready" when it
# can download meta/info.json and the total_episodes field is >= MIN_EPISODES.

set -euo pipefail

REPO_ID="andnetdeboer/CupToContainer"
REPO_TYPE="dataset"
MIN_EPISODES=51
POLL_INTERVAL=300   # seconds between checks
DATASET_DIR="${HOME}/.cache/huggingface/lerobot/andnetdeboer/CupToContainer"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LEROBOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

HF_TOKEN="${HF_TOKEN:-}"

# ── helpers ─────────────────────────────────────────────────────────────────

check_dataset_ready() {
    python3 - <<EOF
import sys
try:
    from huggingface_hub import hf_hub_download
    import json
    info_path = hf_hub_download(
        repo_id="${REPO_ID}",
        filename="meta/info.json",
        repo_type="${REPO_TYPE}",
        token="${HF_TOKEN}" or None,
    )
    with open(info_path) as f:
        info = json.load(f)
    total = info.get("total_episodes", 0)
    print(f"Found {total} episodes", flush=True)
    sys.exit(0 if total >= ${MIN_EPISODES} else 1)
except Exception as e:
    print(f"Not ready: {e}", flush=True)
    sys.exit(1)
EOF
}

download_dataset() {
    echo "[monitor] Downloading dataset with huggingface-cli..."
    huggingface-cli download "${REPO_ID}" \
        --repo-type "${REPO_TYPE}" \
        --local-dir "${DATASET_DIR}" \
        ${HF_TOKEN:+--token "${HF_TOKEN}"}
}

# ── poll until ready ─────────────────────────────────────────────────────────

echo "[monitor] Watching for ${REPO_ID} (>= ${MIN_EPISODES} episodes)..."
while true; do
    if check_dataset_ready; then
        echo "[monitor] Dataset is ready."
        break
    fi
    echo "[monitor] Not ready yet. Retrying in ${POLL_INTERVAL}s..."
    sleep "${POLL_INTERVAL}"
done

# ── download ─────────────────────────────────────────────────────────────────

download_dataset

# ── validate transforms ───────────────────────────────────────────────────────

echo "[monitor] Validating TCP-relative transforms..."
cd "${LEROBOT_DIR}"
python cup_to_container/validate_transforms.py --dataset_dir "${DATASET_DIR}"

# ── compute normalization stats (one-time) ────────────────────────────────────

STATS_FILE="${SCRIPT_DIR}/stats.json"
if [[ ! -f "${STATS_FILE}" ]]; then
    echo "[monitor] Computing normalization stats from training split..."
    python cup_to_container/compute_stats.py \
        --dataset_dir "${DATASET_DIR}" \
        --output_path "${STATS_FILE}"
else
    echo "[monitor] Using cached stats: ${STATS_FILE}"
fi

# ── launch ACT (foreground) ───────────────────────────────────────────────────

echo "[monitor] Launching ACT training (tmux session: act-cup)..."
bash "${LEROBOT_DIR}/train_act_cup.sh" "${DATASET_DIR}"

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [[ "${NUM_GPUS}" -ge 2 ]]; then
    echo "[monitor] Two GPUs found — launching SmolVLA on GPU 1 (tmux session: smolvla-cup)."
    CUDA_VISIBLE_DEVICES=1 bash "${LEROBOT_DIR}/train_smolvla_cup.sh" "${DATASET_DIR}"
else
    echo "[monitor] One GPU — launching SmolVLA in its own tmux session (will queue behind ACT)."
    bash "${LEROBOT_DIR}/train_smolvla_cup.sh" "${DATASET_DIR}"
fi

echo "[monitor] Both sessions launched."
echo "  tmux attach -t act-cup"
echo "  tmux attach -t smolvla-cup"
