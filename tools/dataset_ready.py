"""
Exit 0 when a LeRobot v3 Hub dataset is fully uploaded; print its resolved repo id.

The name is matched case-insensitively and ignoring '_'/'-' within the owner's datasets, so
"andnetdeboer/SweepTrash_Kin" also finds "andnetdeboer/sweeptrash-kin".

"Fully uploaded", all checked at one commit so a push landing mid-check can't mix revisions:
  - meta/info.json, meta/stats.json, meta/tasks.parquet and meta/episodes/ are present
  - meta/episodes lists exactly episodes 0..total_episodes-1, covering total_frames frames
  - every data parquet and every video file those episodes point to is in the repo
  - no commits for --settle_minutes (guards against metadata that is itself still being rewritten)
Any error (network, half-written parquet) counts as "not ready"; the caller just polls again.

Usage:
    python tools/dataset_ready.py andnetdeboer/SweepTrash_Kin [--settle_minutes 5] [--resolve_only]
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def resolve(api: HfApi, repo_id: str) -> str | None:
    owner, name = repo_id.split("/", 1)
    for d in api.list_datasets(author=owner):
        if norm(d.id.split("/", 1)[1]) == norm(name):
            return d.id
    return None


def check(api: HfApi, rid: str, settle_minutes: float) -> str | None:
    """Return None if ready, else the reason it is not."""
    head = api.list_repo_commits(rid, repo_type="dataset")[0]
    rev = head.commit_id
    files = set(api.list_repo_files(rid, repo_type="dataset", revision=rev))
    missing = [f for f in ("meta/info.json", "meta/stats.json", "meta/tasks.parquet") if f not in files]
    ep_files = sorted(f for f in files if f.startswith("meta/episodes/") and f.endswith(".parquet"))
    if missing or not ep_files:
        return f"missing {missing or 'meta/episodes/'}"

    def get(f):
        return hf_hub_download(rid, f, repo_type="dataset", revision=rev)

    info = json.load(open(get("meta/info.json")))
    if info.get("codebase_version", "").split(".")[0] != "v3":
        return f"unsupported codebase_version {info.get('codebase_version')}"
    n_ep, n_frames = info.get("total_episodes", 0), info.get("total_frames", 0)
    if n_ep <= 0:
        return "total_episodes is 0"
    eps = pd.concat([pd.read_parquet(get(f)) for f in ep_files], ignore_index=True)
    if sorted(eps["episode_index"].tolist()) != list(range(n_ep)):
        return f"meta/episodes has {len(eps)} episodes, info.json says {n_ep}"
    if int((eps["dataset_to_index"] - eps["dataset_from_index"]).sum()) != n_frames:
        return f"episode lengths don't add up to total_frames={n_frames}"

    needed = set()
    video_keys = [k for k, ft in info["features"].items() if ft.get("dtype") == "video"]
    for _, e in eps.iterrows():
        needed.add(info["data_path"].format(chunk_index=int(e["data/chunk_index"]),
                                            file_index=int(e["data/file_index"])))
        for k in video_keys:
            needed.add(info["video_path"].format(video_key=k, chunk_index=int(e[f"videos/{k}/chunk_index"]),
                                                 file_index=int(e[f"videos/{k}/file_index"])))
    absent = sorted(needed - files)
    if absent:
        return f"{len(absent)}/{len(needed)} data/video files not uploaded yet (e.g. {absent[0]})"

    age_min = (datetime.now(timezone.utc) - head.created_at).total_seconds() / 60
    if age_min < settle_minutes:
        return f"complete, but last commit was {age_min:.0f} min ago (settling for {settle_minutes} min)"
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("repo_id")
    p.add_argument("--settle_minutes", type=float, default=5)
    p.add_argument("--resolve_only", action="store_true", help="only print the matching repo id")
    args = p.parse_args()

    api = HfApi()
    try:
        rid = resolve(api, args.repo_id)
        if rid is None:
            print(f"{args.repo_id}: not on the Hub yet", file=sys.stderr)
            return 1
        reason = None if args.resolve_only else check(api, rid, args.settle_minutes)
    except Exception as e:
        print(f"{args.repo_id}: check failed ({type(e).__name__}: {e})", file=sys.stderr)
        return 1
    if reason:
        print(f"{rid}: not ready: {reason}", file=sys.stderr)
        return 1
    print(rid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
