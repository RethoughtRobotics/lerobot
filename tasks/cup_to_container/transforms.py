"""
TCP-relative action transform and 17-D state encoding for CupToContainer.

State layout in the dataset (15-D):
  [0:3]   tcp.xyz        base-frame position
  [3:7]   qx qy qz qw   base-frame TCP orientation (xyzw)
  [7]     gripper        measured, 0=open, 1=closed
  [8:15]  left_s0..w2   joint angles

Action layout in the dataset (8-D, absolute):
  [0:3]   goal.xyz       base-frame goal position
  [3:7]   qx qy qz qw   base-frame goal orientation (xyzw)
  [7]     trigger        gripper command, 0=open, 1=closed

After transform:
  State  (17-D): [tcp.xyz (3), R_tcp first two cols (6), gripper (1), joints (7)]
  Action (7-D):  [dx dy dz (3), rotvec rx ry rz (3), trigger (1)]
    where  dp = goal.xyz - tcp.xyz
           r  s.t.  R_goal = exp(r) * R_tcp
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation


# ── numpy versions (used during offline stats computation) ──────────────────

def state_to_17d_np(state: np.ndarray) -> np.ndarray:
    """(15,) → (17,) replacing quaternion with 6-D rotation (first two columns)."""
    R = Rotation.from_quat(state[3:7])  # xyzw
    mat = R.as_matrix()                 # (3, 3)
    rot6 = np.concatenate([mat[:, 0], mat[:, 1]])  # col0 then col1, (6,)
    return np.concatenate([state[:3], rot6, state[7:8], state[8:15]]).astype(np.float32)


def tcp_relative_np(action_chunk: np.ndarray, state: np.ndarray) -> np.ndarray:
    """(K, 8) absolute goals + (15,) observation state → (K, 7) TCP-relative targets."""
    R_tcp = Rotation.from_quat(state[3:7])  # xyzw
    dp = action_chunk[:, :3] - state[:3]
    dr = (Rotation.from_quat(action_chunk[:, 3:7]) * R_tcp.inv()).as_rotvec()
    return np.column_stack([dp, dr, action_chunk[:, 7:8]]).astype(np.float32)


def apply_actions_np(targets: np.ndarray, tcp_xyz: np.ndarray, tcp_quat: np.ndarray) -> np.ndarray:
    """Inverse of tcp_relative_np. (K, 7) targets → (K, 8) absolute goals.

    Used for the transform validation check in §6 of the training spec.
    """
    R_tcp = Rotation.from_quat(tcp_quat)  # xyzw
    goal_xyz = targets[:, :3] + tcp_xyz
    R_goal = Rotation.from_rotvec(targets[:, 3:6]) * R_tcp
    goal_quat = R_goal.as_quat()  # xyzw
    return np.column_stack([goal_xyz, goal_quat, targets[:, 6:7]]).astype(np.float32)


# ── torch versions (used in dataset __getitem__ during training) ─────────────

def state_to_17d_torch(state: torch.Tensor) -> torch.Tensor:
    """(15,) float32 → (17,) float32 — quaternion → 6-D rotation."""
    state_np = state.numpy()
    return torch.from_numpy(state_to_17d_np(state_np))


def state_history_to_17d_torch(states: torch.Tensor) -> torch.Tensor:
    """(15,) → (17,), or (S, 15) observation history → (S, 17)."""
    if states.ndim == 1:
        return state_to_17d_torch(states)
    return torch.stack([state_to_17d_torch(s) for s in states])


def tcp_relative_torch(action_chunk: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """(K, 8) absolute goals + (15,) state → (K, 7) TCP-relative targets."""
    action_np = action_chunk.numpy()
    state_np = state.numpy()
    return torch.from_numpy(tcp_relative_np(action_np, state_np))
