"""
Compute normalization stats (mean/std) for the transformed CupToContainer data.

Stats are computed only over the training split (episodes not in HELD_OUT_EPISODES).
Outputs a stats.json file compatible with lerobot's LeRobotDatasetMetadata.

Usage:
    uv run python cup_to_container/compute_stats.py \\
        --dataset_repo_id andnetdeboer/BlueCupToContainer \\
        --output_path cup_to_container/stats_BlueCupToContainer.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add src to path for lerobot import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.datasets import LeRobotDataset

from transforms import state_to_17d_np, tcp_relative_np
from wrapped_dataset import HELD_OUT_EPISODES, TRAIN_EPISODES


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def compute_stats(repo_id: str, dataset_dir: Path | None, output_path: Path,
                  video_backend: str = "torchcodec", chunk_size: int = 100) -> dict:
    """Stats over the training split, read from the parquet columns only.

    Like ABC's states_actions.bin reader, this never decodes video: each frame's action
    chunk is rebuilt from the episode's action column, clamping past the episode end to its
    last frame, which matches LeRobotDataset's padding for delta_timestamps.
    """
    print(f"Loading {repo_id} (root={dataset_dir})")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_dir,
        episodes=TRAIN_EPISODES,
        video_backend=video_backend,
    )
    print(f"Training episodes: {len(TRAIN_EPISODES)} / {dataset.meta.total_episodes} total")
    print(f"Training frames: {len(dataset)}")

    cols = dataset.hf_dataset.with_format("numpy")
    states = np.stack(cols["observation.state"]).astype(np.float64)   # (N, 15)
    actions = np.stack(cols["action"]).astype(np.float64)             # (N, 8)
    episodes = np.asarray(cols["episode_index"]).reshape(-1)

    state_sum = np.zeros(17, dtype=np.float64)
    state_sum2 = np.zeros(17, dtype=np.float64)
    action_sum = np.zeros(7, dtype=np.float64)
    action_sum2 = np.zeros(7, dtype=np.float64)
    n_state = 0
    n_action = 0  # each frame contributes chunk_size action steps

    offsets = np.arange(chunk_size)
    for ep in tqdm(np.unique(episodes), desc="Computing stats"):
        rows = np.flatnonzero(episodes == ep)
        ep_states, ep_actions = states[rows], actions[rows]
        last = len(rows) - 1
        for t in range(len(rows)):
            state_raw = ep_states[t]
            action_raw = ep_actions[np.minimum(t + offsets, last)]   # (chunk_size, 8)

            s17 = state_to_17d_np(state_raw)                 # (17,)
            a7 = tcp_relative_np(action_raw, state_raw)      # (chunk_size, 7)

            state_sum += s17
            state_sum2 += s17.astype(np.float64) ** 2
            n_state += 1

            action_sum += a7.sum(axis=0)
            action_sum2 += (a7.astype(np.float64) ** 2).sum(axis=0)
            n_action += chunk_size

    state_mean = state_sum / n_state
    state_std = np.sqrt(np.maximum(state_sum2 / n_state - state_mean ** 2, 1e-8))
    action_mean = action_sum / n_action
    action_std = np.sqrt(np.maximum(action_sum2 / n_action - action_mean ** 2, 1e-8))

    # lerobot stats format: nested dict {key: {mean: [...], std: [...]}}
    stats = {
        "observation.state": {
            "mean": state_mean.tolist(),
            "std": state_std.tolist(),
            "min": None,
            "max": None,
        },
        "action": {
            "mean": action_mean.tolist(),
            "std": action_std.tolist(),
            "min": None,
            "max": None,
        },
    }

    # Preserve image stats from the original dataset (IDENTITY normalization uses them but
    # they're passed through unchanged — include them for completeness)
    for key, val in dataset.meta.stats.items():
        if key not in stats:
            stats[key] = val
    stats = _to_jsonable(stats)

    # Write to a temp file then rename, so a crash never leaves a truncated stats.json
    # that load_or_compute_stats would later try to load.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(stats, f, indent=2)
    tmp_path.replace(output_path)
    print(f"Stats written to {output_path}")
    print(f"  state mean  = {state_mean}")
    print(f"  state std   = {state_std}")
    print(f"  action mean = {action_mean}")
    print(f"  action std  = {action_std}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_repo_id", default="andnetdeboer/BlueCupToContainer")
    parser.add_argument("--dataset_dir", type=Path, default=None)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()
    compute_stats(args.dataset_repo_id, args.dataset_dir, args.output_path)
