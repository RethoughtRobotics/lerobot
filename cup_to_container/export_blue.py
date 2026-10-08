"""
Re-export CupToContainer as BlueCupToContainer with ABC-style (amazon-far/abc) video.

Changes vs the source dataset (everything else — state, action, tasks, episodes — is copied as-is):
  - Every camera is letterboxed to 224x224 with ABC's `resize_with_pad` (aspect-preserving,
    bilinear + antialias, zero padding), instead of 640x400 / 640x480.
  - AV1 (libsvtav1), GOP 2, CRF 30, with SVT-AV1 fast-decode tuning.
  - Small video/data file targets so files hold roughly one episode each, keeping per-sample
    container opens cheap (ABC stores one mp4 per episode).

The result is still a LeRobot v3 dataset, so lerobot tooling and the HF viewer keep working.

Usage:
    uv run python cup_to_container/export_blue.py \\
        --src_root ~/.cache/huggingface/lerobot/andnetdeboer/CupToContainer \\
        [--push]
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import DEFAULT_FEATURES, HF_LEROBOT_HOME

IMAGE_SIZE = 224


def resize_with_pad(img_hwc: torch.Tensor, target_h: int = IMAGE_SIZE, target_w: int = IMAGE_SIZE) -> torch.Tensor:
    """ABC's letterbox resize (abc_minimal/preprocess.py), on float HWC images."""
    h, w, _ = img_hwc.shape
    if (h, w) == (target_h, target_w):
        return img_hwc
    ratio = max(w / target_w, h / target_h)
    new_h = max(1, round(h / ratio))
    new_w = max(1, round(w / ratio))
    resized = F.interpolate(
        img_hwc.permute(2, 0, 1).unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    pad_h0 = (target_h - new_h) // 2
    pad_h1 = target_h - new_h - pad_h0
    pad_w0 = (target_w - new_w) // 2
    pad_w1 = target_w - new_w - pad_w0
    padded = F.pad(resized, (pad_w0, pad_w1, pad_h0, pad_h1), value=0)
    return padded.permute(1, 2, 0)


def to_224_uint8(img_chw_uint8: torch.Tensor) -> torch.Tensor:
    x = img_chw_uint8.permute(1, 2, 0).float()
    return resize_with_pad(x).round().clamp(0, 255).to(torch.uint8)


def build_features(src_features: dict) -> dict:
    features = {}
    for key, ft in src_features.items():
        if key in DEFAULT_FEATURES:
            continue
        ft = {k: v for k, v in ft.items() if k != "info"}
        if ft["dtype"] == "video":
            ft["shape"] = [IMAGE_SIZE, IMAGE_SIZE, 3]
        features[key] = ft
    return features


def export(src_repo_id: str, src_root: Path, dst_repo_id: str, dst_root: Path,
           max_episodes: int | None = None, threads: int = 8) -> LeRobotDataset:
    src = LeRobotDataset(src_repo_id, root=src_root, video_backend="torchcodec", return_uint8=True)
    video_keys = src.meta.video_keys
    print(f"Source: {src.meta.total_episodes} episodes, {src.meta.total_frames} frames, cameras={video_keys}")

    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst = LeRobotDataset.create(
        repo_id=dst_repo_id,
        fps=src.meta.fps,
        features=build_features(src.meta.features),
        root=dst_root,
        robot_type=src.meta.robot_type,
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="libsvtav1", g=2, crf=30, fast_decode=1),
        # ~1 episode per file: a 224px episode is well under 1 MB of video.
        video_files_size_in_mb=1,
        data_files_size_in_mb=1,
        encoder_threads=threads,
    )

    keys = [k for k in dst.meta.features if k not in DEFAULT_FEATURES]
    episodes = src.meta.episodes
    n_episodes = min(src.meta.total_episodes, max_episodes or src.meta.total_episodes)
    for ep in tqdm(range(n_episodes), desc="Episodes"):
        start, end = int(episodes["dataset_from_index"][ep]), int(episodes["dataset_to_index"][ep])
        for idx in range(start, end):
            item = src[idx]
            frame = {"task": item["task"]}
            for k in keys:
                frame[k] = to_224_uint8(item[k]).numpy() if k in video_keys else item[k].numpy()
            dst.add_frame(frame)
        dst.save_episode()
    dst.finalize()

    # Keep provenance files from the source meta (e.g. rethoughtai_sources.json).
    for extra in (src_root / "meta").glob("*.json"):
        if not (dst_root / "meta" / extra.name).exists():
            shutil.copy2(extra, dst_root / "meta" / extra.name)
    return dst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_repo_id", default="andnetdeboer/CupToContainer")
    p.add_argument("--src_root", type=Path,
                   default=HF_LEROBOT_HOME / "andnetdeboer/CupToContainer")
    p.add_argument("--dst_repo_id", default="andnetdeboer/BlueCupToContainer")
    p.add_argument("--dst_root", type=Path, default=None)
    p.add_argument("--max_episodes", type=int, default=None, help="Export only the first N (for testing)")
    p.add_argument("--threads", type=int, default=8, help="CPU threads for resizing and encoding")
    p.add_argument("--push", action="store_true", help="Push to the HF Hub (public)")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    dst_root = args.dst_root or HF_LEROBOT_HOME / args.dst_repo_id
    dst = export(args.src_repo_id, args.src_root.expanduser(), args.dst_repo_id, dst_root,
                 args.max_episodes, args.threads)
    print(f"Wrote {dst.meta.total_episodes} episodes / {dst.meta.total_frames} frames to {dst_root}")
    if args.push:
        dst.push_to_hub(private=False, tags=["LeRobot", "abc-style-224"])
        print(f"Pushed https://huggingface.co/datasets/{args.dst_repo_id}")


if __name__ == "__main__":
    main()
