"""
Transform validation check (spec §6).

For several frames, applies tcp_relative then apply_actions and verifies
round-trip error is < 1e-5. Also prints per-frame sample to catch sign/order bugs.

Usage:
    python validate_transforms.py \\
        --dataset_dir ~/.cache/huggingface/lerobot/andnetdeboer/CupToContainer
"""
import argparse
from pathlib import Path
import sys

import numpy as np


from lerobot.datasets import LeRobotDataset

from transforms import apply_actions_np, tcp_relative_np
from wrapped_dataset import TRAIN_EPISODES


def validate(dataset_dir: Path, n_frames: int = 20) -> None:
    dataset = LeRobotDataset(
        repo_id="andnetdeboer/CupToContainer",
        root=dataset_dir,
        delta_timestamps={"action": [k / 25.0 for k in range(100)]},
        episodes=TRAIN_EPISODES,
        video_backend="pyav",
    )

    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_frames, len(dataset)), replace=False)

    max_err = 0.0
    for idx in indices:
        sample = dataset[int(idx)]
        state = sample["observation.state"].numpy()   # (15,)
        action = sample["action"].numpy()             # (100, 8)

        targets = tcp_relative_np(action, state)      # (100, 7)
        reconstructed = apply_actions_np(targets, state[:3], state[3:7])  # (100, 8)

        err = np.abs(reconstructed - action).max()
        max_err = max(max_err, err)
        status = "OK" if err < 1e-5 else "FAIL"
        print(f"frame {idx:5d}  max_err={err:.2e}  [{status}]")

    print(f"\nOverall max round-trip error: {max_err:.2e}")
    if max_err < 1e-5:
        print("PASS — transform is consistent.")
    else:
        print("FAIL — check quaternion order (must be xyzw) and rotation side.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--n_frames", type=int, default=20)
    args = parser.parse_args()
    validate(args.dataset_dir, args.n_frames)
