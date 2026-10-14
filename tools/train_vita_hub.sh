#!/bin/bash
# Train VITA with lerobot-train on any LeRobot Hub dataset (any state/action layout), for tools/pipeline.py.
#
# Usage:
#   tools/train_vita_hub.sh <hub_dataset_id> <output_dir> <model_repo_id> [extra lerobot-train args...]
#
# The dataset id is resolved with tools/dataset_ready.py --resolve_only (tolerates case / '_' vs '-').
# If <output_dir> exists (a retry): resume from its last checkpoint, or move it aside if it has none,
# since lerobot-train refuses an existing --output_dir.
# Validation loss: lerobot-train defaults hold out 10% of episodes (dataset.eval_split=0.1) and
# log eval loss every 2000 steps.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DS="$1"; OUT="$2"; MODEL="$3"; shift 3
cd "$REPO"
DS="$(.venv/bin/python tools/dataset_ready.py "$DS" --resolve_only)"
NAME="$(basename "$OUT")"

if [ -f "$OUT/checkpoints/last/pretrained_model/train_config.json" ]; then
    echo "[train_vita_hub] resuming from $OUT/checkpoints/last"
    exec .venv/bin/lerobot-train --resume=true \
        --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" "$@"
fi
[ -e "$OUT" ] && mv "$OUT" "$OUT.failed_$(date +%Y%m%d_%H%M%S)"

echo "[train_vita_hub] dataset=$DS output=$OUT model=$MODEL"
exec .venv/bin/lerobot-train \
    --dataset.repo_id="$DS" \
    --dataset.video_backend=torchcodec \
    --policy.type=vita \
    --policy.horizon=32 \
    --policy.n_action_steps=16 \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.private=true \
    --policy.repo_id="$MODEL" \
    --ema.enable=true \
    --seed=42 \
    --batch_size=128 \
    --steps=100000 \
    --save_freq=10000 \
    --log_freq=100 \
    --num_workers=8 \
    --env_eval_freq=0 \
    --output_dir="$OUT" \
    --job_name="$NAME" \
    --wandb.enable=true \
    --wandb.project=rethoughtai-vita \
    --wandb.entity=deboerandnet-northwestern-university \
    --wandb.disable_artifact=true \
    "$@"
