"""
Re-encode any LeRobot dataset's videos for fast training (ideas from amazon-far/abc).

Copies every episode, frame and non-video feature unchanged, and rewrites each camera stream:
  - optional resize to a square `--image_size` (letterbox `pad`, ABC's default, or center `crop`);
    omit it to keep the native resolution
  - AV1 (libsvtav1) with a short GOP and SVT-AV1 fast-decode, so random access decodes ~1 frame
  - small video/data files (about one episode each) so per-sample container opens stay cheap

The output is still a LeRobot v3 dataset, so `lerobot-train`, the HF viewer and all lerobot
tooling keep working.

Examples:
    # 224x224 letterboxed (ABC default), local only
    uv run python tools/reencode_dataset.py \\
        --src_repo_id andnetdeboer/CupToContainer --dst_repo_id andnetdeboer/BlueCupToContainer \\
        --image_size 224

    # native resolution, small files + fast-decode only, then publish privately
    uv run python tools/reencode_dataset.py \\
        --src_repo_id me/MyTask --dst_repo_id me/MyTask_fast --push --private
"""
import argparse
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import DEFAULT_FEATURES, HF_LEROBOT_HOME


def resize_with_pad(img_hwc: torch.Tensor, size: int) -> torch.Tensor:
    """ABC's letterbox resize (abc_minimal/preprocess.py): keep the whole frame, pad with zeros."""
    h, w, _ = img_hwc.shape
    if (h, w) == (size, size):
        return img_hwc
    ratio = max(w / size, h / size)
    new_h, new_w = max(1, round(h / ratio)), max(1, round(w / ratio))
    resized = F.interpolate(
        img_hwc.permute(2, 0, 1).unsqueeze(0), size=(new_h, new_w),
        mode="bilinear", align_corners=False, antialias=True,
    ).squeeze(0)
    pad_h0, pad_w0 = (size - new_h) // 2, (size - new_w) // 2
    padded = F.pad(resized, (pad_w0, size - new_w - pad_w0, pad_h0, size - new_h - pad_h0), value=0)
    return padded.permute(1, 2, 0)


def resize_with_crop(img_hwc: torch.Tensor, size: int) -> torch.Tensor:
    """Scale the short side to `size`, then center-crop: no padding, but loses the frame edges."""
    h, w, _ = img_hwc.shape
    scale = size / min(h, w)
    new_h, new_w = max(size, round(h * scale)), max(size, round(w * scale))
    resized = F.interpolate(
        img_hwc.permute(2, 0, 1).unsqueeze(0), size=(new_h, new_w),
        mode="bilinear", align_corners=False, antialias=True,
    ).squeeze(0)
    top, left = (new_h - size) // 2, (new_w - size) // 2
    return resized[:, top : top + size, left : left + size].permute(1, 2, 0)


def convert_frame(img_chw_uint8: torch.Tensor, size: int | None, mode: str) -> torch.Tensor:
    img = img_chw_uint8.permute(1, 2, 0)
    if size is None:
        return img
    resize = resize_with_pad if mode == "pad" else resize_with_crop
    return resize(img.float(), size).round().clamp(0, 255).to(torch.uint8)


def build_features(src_features: dict, size: int | None) -> dict:
    features = {}
    for key, ft in src_features.items():
        if key in DEFAULT_FEATURES:
            continue
        ft = {k: v for k, v in ft.items() if k != "info"}
        if ft["dtype"] == "video" and size is not None:
            ft["shape"] = [size, size, ft["shape"][2]]
        features[key] = ft
    return features


def reencode(args) -> LeRobotDataset:
    src_root = args.src_root.expanduser() if args.src_root else None
    dst_root = (args.dst_root or HF_LEROBOT_HOME / args.dst_repo_id).expanduser()
    src = LeRobotDataset(args.src_repo_id, root=src_root, video_backend="torchcodec", return_uint8=True)
    video_keys = src.meta.video_keys
    print(f"Source {args.src_repo_id}: {src.meta.total_episodes} episodes, "
          f"{src.meta.total_frames} frames, cameras={video_keys}")

    if dst_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst_root} exists; pass --overwrite to replace it")
        shutil.rmtree(dst_root)
    dst = LeRobotDataset.create(
        repo_id=args.dst_repo_id,
        fps=src.meta.fps,
        features=build_features(src.meta.features, args.image_size),
        root=dst_root,
        robot_type=src.meta.robot_type,
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="libsvtav1", g=args.gop, crf=args.crf, fast_decode=1),
        video_files_size_in_mb=args.file_size_mb,
        data_files_size_in_mb=args.file_size_mb,
        encoder_threads=args.threads,
    )

    keys = [k for k in dst.meta.features if k not in DEFAULT_FEATURES]
    episodes = src.meta.episodes
    n_episodes = min(src.meta.total_episodes, args.max_episodes or src.meta.total_episodes)
    for ep in tqdm(range(n_episodes), desc="Episodes"):
        start, end = int(episodes["dataset_from_index"][ep]), int(episodes["dataset_to_index"][ep])
        for idx in range(start, end):
            item = src[idx]
            frame = {"task": item["task"]}
            for k in keys:
                if k in video_keys:
                    frame[k] = convert_frame(item[k], args.image_size, args.resize_mode).numpy()
                else:
                    frame[k] = item[k].numpy()
            dst.add_frame(frame)
        dst.save_episode()
    dst.finalize()

    # Keep extra provenance files from the source meta (anything lerobot didn't regenerate).
    src_meta = Path(src.root) / "meta"
    for extra in src_meta.glob("*.json"):
        if not (dst_root / "meta" / extra.name).exists():
            shutil.copy2(extra, dst_root / "meta" / extra.name)
    print(f"Wrote {dst.meta.total_episodes} episodes / {dst.meta.total_frames} frames to {dst_root}")
    return dst


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_repo_id", required=True)
    p.add_argument("--src_root", type=Path, default=None, help="Local source root (default: HF cache)")
    p.add_argument("--dst_repo_id", required=True)
    p.add_argument("--dst_root", type=Path, default=None, help="Output root (default: HF cache)")
    p.add_argument("--image_size", type=int, default=None,
                   help="Square output size, e.g. 224. Omit to keep native resolution.")
    p.add_argument("--resize_mode", choices=["pad", "crop"], default="pad",
                   help="pad = letterbox whole frame (ABC); crop = center crop, no padding")
    p.add_argument("--crf", type=int, default=30, help="AV1 quality; lower = better/larger")
    p.add_argument("--gop", type=int, default=2, help="Keyframe interval")
    p.add_argument("--file_size_mb", type=int, default=1,
                   help="Target video/data file size; small ≈ one episode per file")
    p.add_argument("--threads", type=int, default=8, help="CPU threads for resizing and encoding")
    p.add_argument("--max_episodes", type=int, default=None, help="Only the first N episodes (testing)")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    p.add_argument("--push", action="store_true", help="Push the result to the HF Hub")
    p.add_argument("--private", action="store_true", help="With --push: make the dataset private")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    dst = reencode(args)
    if args.push:
        tags = ["LeRobot"] + ([f"{args.image_size}px"] if args.image_size else [])
        dst.push_to_hub(private=args.private, tags=tags)
        print(f"Pushed https://huggingface.co/datasets/{args.dst_repo_id}")


if __name__ == "__main__":
    main()
