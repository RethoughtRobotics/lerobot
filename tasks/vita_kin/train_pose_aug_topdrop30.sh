#!/bin/bash
# VITA on BlueCupRedContainer_kin_pose (action = measured TCP pose, no lead) with image augmentation
# always on and the top camera (third_eye) dropped for 30% of samples; the wrist camera is never dropped.
# 30k steps (eval loss plateaued around 15-25k steps in the 100k runs). Runs in tmux on <gpu>.
#
# Usage: tasks/vita_kin/train_pose_aug_topdrop30.sh [gpu] [output_dir]
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU="${1:-0}"
OUT="${2:-outputs/vita-aug_BlueCupRedContainer_kin_pose_topdrop30}"
tools/tmux_train.sh vita-pose-aug "$GPU" "$(dirname "$OUT")/logs/$(basename "$OUT")_train.log" -- \
    tools/train_vita_hub.sh andnetdeboer/BlueCupRedContainer_kin_pose "$OUT" \
    andnetdeboer/vita_BlueCupRedContainer_kin_pose_aug_topdrop30 \
    --batch_size=128 --steps=30000 --save_freq=5000 --num_workers=16 \
    --dataset.image_transforms.enable=true \
    '--policy.camera_dropout={"observation.images.third_eye": 0.3}'
