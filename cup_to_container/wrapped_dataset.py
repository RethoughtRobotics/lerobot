"""
Dataset wrapper that applies CupToContainer transforms on the fly.

Wraps a LeRobotDataset and:
  - transforms observation.state  (15-D) → (17-D)
  - transforms action             (K, 8) → (K, 7)  TCP-relative
  - exposes a patched meta.features with the new shapes
  - exposes patched meta.stats (passed in at construction)
"""
import copy

import torch

from transforms import state_to_17d_torch, tcp_relative_torch

# Validation split: hold these 5 episodes out of 51 (0-indexed)
HELD_OUT_EPISODES = [4, 14, 24, 34, 44]
TRAIN_EPISODES = [i for i in range(51) if i not in HELD_OUT_EPISODES]


class _PatchedMeta:
    """Proxy for LeRobotDatasetMetadata with overridden features and stats."""

    def __init__(self, original_meta, features_patch: dict, stats: dict):
        self._meta = original_meta
        self._features = features_patch
        self.stats = stats

    @property
    def features(self):
        return self._features

    def __getattr__(self, name):
        return getattr(self._meta, name)


def _patch_features(original_features: dict) -> dict:
    """Return a copy of the dataset feature dict with updated shapes for our transformed keys."""
    features = copy.deepcopy(original_features)
    # observation.state: 15 → 17
    if "observation.state" in features:
        features["observation.state"]["shape"] = [17]
        # Drop 'names' so lerobot doesn't try to use the old 15-item list
        features["observation.state"].pop("names", None)
    # action: 8 → 7  (per-step shape; lerobot uses this for the action feature)
    if "action" in features:
        features["action"]["shape"] = [7]
        features["action"].pop("names", None)
    return features


class CupToContainerDataset(torch.utils.data.Dataset):
    """
    Wraps a LeRobotDataset of CupToContainer recordings.

    The underlying dataset must have been created with:
        delta_timestamps = {"action": [k / 25 for k in range(100)]}

    Transforms applied per __getitem__:
      1. state 15-D → 17-D (quaternion → 6-D rotation)
      2. action (100, 8) absolute → (100, 7) TCP-relative

    The wrapper does NOT pad action_is_pad; the training loss is applied to all
    100 steps including the repeated last-frame hold (see spec §3).
    """

    def __init__(self, base_dataset, stats: dict):
        self._dataset = base_dataset
        patched_features = _patch_features(dict(base_dataset.meta.features))
        self.meta = _PatchedMeta(base_dataset.meta, patched_features, stats)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self._dataset[idx]

        state_raw = sample["observation.state"]   # (15,) float32 tensor
        action_raw = sample["action"]              # (100, 8) float32 tensor

        sample["observation.state"] = state_to_17d_torch(state_raw)    # (17,)
        sample["action"] = tcp_relative_torch(action_raw, state_raw)   # (100, 7)

        return sample
