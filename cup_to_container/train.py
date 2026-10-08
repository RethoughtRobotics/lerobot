"""
Training script for CupToContainer — ACT and SmolVLA.

Implements the spec in the CupToContainer training spec document:
  - Custom TCP-relative action transform  (absolute 8-D → relative 7-D)
  - 17-D state encoding  (quaternion → 6-D rotation)
  - Episode-based train/val split (hold out episodes 4,14,24,34,44)
  - Normalization stats computed from training split only

Usage (via shell wrappers train_act_cup.sh / train_smolvla_cup.sh):
    uv run python cup_to_container/train.py \\
        --policy_type act \\
        --dataset_repo_id andnetdeboer/BlueCupToContainer \\
        --output_dir outputs/act_cup_run1 \\
        --steps 100000 --batch_size 8 --save_freq 10000
"""
import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

# Add src to path (works when run from lerobot root dir)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.random_utils import set_seed

from compute_stats import compute_stats
from wrapped_dataset import CupToContainerDataset, TRAIN_EPISODES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

def _json_stats_to_tensors(stats: dict, device: torch.device) -> dict:
    """Convert {key: {mean: list, std: list}} -> {key: {mean: Tensor, std: Tensor}}."""
    out = {}
    for key, s in stats.items():
        if s.get("mean") is None:
            continue
        out[key] = {
            "mean": torch.tensor(s["mean"], dtype=torch.float32, device=device),
            "std": torch.tensor(s["std"], dtype=torch.float32, device=device),
        }
    return out


def build_policy_config(policy_type: str) -> ACTConfig | SmolVLAConfig:
    if policy_type == "act":
        return ACTConfig(
            input_features=None,
            output_features=None,
            n_obs_steps=1,
            chunk_size=100,
            n_action_steps=1,           # runtime ensembles with coeff=0.01
            vision_backbone="resnet18",
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
            dim_model=512,
            n_heads=8,
            dim_feedforward=3200,
            n_encoder_layers=4,
            n_decoder_layers=1,
            use_vae=True,
            latent_dim=32,
            kl_weight=10.0,
            normalization_mapping={
                "VISUAL": "MEAN_STD",
                "STATE": "MEAN_STD",
                "ACTION": "MEAN_STD",
            },
        )
    elif policy_type == "smolvla":
        return SmolVLAConfig(
            input_features=None,
            output_features=None,
            n_obs_steps=1,
            chunk_size=50,
            n_action_steps=50,
            vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            load_vlm_weights=True,
            freeze_vision_encoder=True,
            train_expert_only=True,
            train_state_proj=True,
            resize_imgs_with_padding=(512, 512),
            max_state_dim=32,
            max_action_dim=32,
            normalization_mapping={
                "VISUAL": "IDENTITY",
                "STATE": "MEAN_STD",
                "ACTION": "MEAN_STD",
            },
            optimizer_lr=1e-4,
            scheduler_warmup_steps=1000,
            scheduler_decay_steps=30000,
        )
    else:
        raise ValueError(f"Unknown policy_type: {policy_type}")


def load_or_compute_stats(stats_path: Path, repo_id: str, dataset_dir: Path | None, video_backend: str) -> dict:
    if stats_path.exists():
        logger.info(f"Loading cached stats from {stats_path}")
        with open(stats_path) as f:
            return json.load(f)
    logger.info("No cached stats found — computing from training split (this takes a few minutes)...")
    return compute_stats(repo_id, dataset_dir, stats_path, video_backend)


def build_optimizer(policy_type: str, policy: torch.nn.Module, steps: int):
    if policy_type == "act":
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=1e-5,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=1e-4,
        )
        return optimizer, None
    else:
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=1e-4,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=1e-10,
        )
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=2.5e-6)
        return optimizer, scheduler


def save_ckpt(policy, optimizer, scheduler, step: int, output_dir: Path, stats: dict) -> None:
    ckpt_dir = output_dir / "checkpoints" / f"{step:06d}"
    model_dir = ckpt_dir / "pretrained_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    policy.save_pretrained(model_dir)

    # Save stats alongside the model so inference can reconstruct normalization
    with open(model_dir / "normalization_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Save optimizer state
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")

    torch.save({"step": step}, ckpt_dir / "training_step.pt")
    logger.info(f"Checkpoint saved → {ckpt_dir}")


def train(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── stats ─────────────────────────────────────────────────────────────────
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    stats_path = Path(__file__).parent / f"stats_{args.dataset_repo_id.split('/')[-1]}.json"
    stats_json = load_or_compute_stats(stats_path, args.dataset_repo_id, dataset_dir, args.video_backend)
    tensor_stats = _json_stats_to_tensors(stats_json, device)

    # ── dataset ───────────────────────────────────────────────────────────────
    chunk_size = 100 if args.policy_type == "act" else 50
    logger.info(f"Loading {args.dataset_repo_id} (root={dataset_dir}) "
                f"(training episodes: {TRAIN_EPISODES[:5]}...)")
    base_train = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=dataset_dir,
        delta_timestamps={"action": [k / 25.0 for k in range(chunk_size)]},
        episodes=TRAIN_EPISODES,
        video_backend=args.video_backend,
        # Ship uint8 frames from DataLoader workers (4x less IPC than float32); converted on GPU below.
        return_uint8=True,
    )
    train_dataset = CupToContainerDataset(base_train, stats_json)
    logger.info(f"Training frames: {len(train_dataset)}")

    # ── policy ────────────────────────────────────────────────────────────────
    policy_cfg = build_policy_config(args.policy_type)
    logger.info(f"Creating {args.policy_type} policy")
    policy = make_policy(cfg=policy_cfg, ds_meta=train_dataset.meta)
    policy = policy.to(device)

    n_learn = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    logger.info(f"Params: {n_learn:,} learnable / {n_total:,} total")

    # ── preprocessor (normalization + device transfer) ────────────────────────
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        dataset_stats=tensor_stats,
    )

    # ── optimizer ─────────────────────────────────────────────────────────────
    optimizer, scheduler = build_optimizer(args.policy_type, policy, args.steps)

    # ── sampler & dataloader ──────────────────────────────────────────────────
    ep_meta = base_train.meta.episodes
    sampler = EpisodeAwareSampler(
        ep_meta["dataset_from_index"],
        ep_meta["dataset_to_index"],
        episode_indices_to_use=TRAIN_EPISODES,
        drop_n_last_frames=getattr(policy_cfg, "drop_n_last_frames", 0),
        shuffle=True,
        seed=42,
        # Episode-filtered datasets are indexed relatively (0..len-1), not by absolute frame index.
        absolute_to_relative_idx=base_train.absolute_to_relative_idx,
    )
    collate_fn = lerobot_collate_fn if train_dataset.meta.has_language_columns else None
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=collate_fn,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = bool(args.wandb_project)
    # `model` runs the forward pass; `policy` stays uncompiled for save_pretrained / push_to_hub.
    model = policy
    if args.compile:
        # ABC compiles its whole policy for throughput.
        logger.info("torch.compile enabled (first steps will be slow while it compiles)")
        model = torch.compile(policy)

    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=Path(args.output_dir).name,
            config=vars(args),
        )

    logger.info(f"Training: {args.steps} steps  batch={args.batch_size}  "
                f"save_freq={args.save_freq}  device={device}")

    # ── training loop ─────────────────────────────────────────────────────────
    policy.train()
    dl_iter = iter(dataloader)
    recent_losses = []
    t_start = time.perf_counter()

    for step in range(1, args.steps + 1):
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch = next(dl_iter)

        # uint8 images → float32 [0, 1], converted on the GPU
        for k in list(batch.keys()):
            if "image" in k and batch[k].dtype == torch.uint8:
                batch[k] = batch[k].to(device, non_blocking=True).to(torch.float32) / 255.0

        batch = preprocessor(batch)

        optimizer.zero_grad(set_to_none=True)
        # bf16 autocast needs no loss scaling (unlike fp16), so there is no GradScaler.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, output_dict = model(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        recent_losses.append(loss.item())

        if step % args.log_freq == 0:
            avg_loss = sum(recent_losses) / len(recent_losses)
            recent_losses.clear()
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.perf_counter() - t_start
            sps = step / elapsed
            logger.info(f"step {step:6d}/{args.steps}  loss={avg_loss:.4f}  "
                        f"lr={lr:.2e}  {sps:.1f} steps/s  "
                        f"eta={((args.steps - step) / sps / 3600):.1f}h")
            if use_wandb:
                wandb.log({"train/loss": avg_loss, "train/lr": lr,
                           "train/steps_per_sec": sps}, step=step)

        if step % args.save_freq == 0 or step == args.steps:
            save_ckpt(policy, optimizer, scheduler, step, output_dir, stats_json)

    # ── push final checkpoint ─────────────────────────────────────────────────
    if args.hub_repo_id:
        final_ckpt = output_dir / "checkpoints" / f"{args.steps:06d}" / "pretrained_model"
        logger.info(f"Pushing final checkpoint to {args.hub_repo_id}")
        policy.push_to_hub(
            args.hub_repo_id,
            private=True,
            local_dir=str(final_ckpt),
        )

    if use_wandb:
        wandb.finish()
    logger.info("Training complete.")


def parse_args():
    p = argparse.ArgumentParser(description="Train ACT or SmolVLA on CupToContainer")
    p.add_argument("--policy_type", choices=["act", "smolvla"], required=True)
    p.add_argument("--dataset_repo_id", default="andnetdeboer/BlueCupToContainer")
    p.add_argument("--dataset_dir", default=None,
                   help="Local dataset root; defaults to the HF cache (downloads if missing)")
    p.add_argument("--video_backend", choices=["torchcodec", "pyav"], default="torchcodec")
    p.add_argument("--compile", action="store_true", help="torch.compile the policy (ABC-style)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--save_freq", type=int, default=10_000)
    p.add_argument("--log_freq", type=int, default=200)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--hub_repo_id", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
