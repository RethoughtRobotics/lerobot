# rethought-lerobot

General-purpose fork of [andnet-deboer/lerobot](https://github.com/andnet-deboer/lerobot) (`main`)
for training policies on new robots and tasks. On top of upstream it adds:

- **AgileX PiPER support**: `src/lerobot/robots/piper/` (real arm + `piper_sim`), PiPER in replay and
  async inference, and MolmoAct2 in async inference. See `AGILEX_PIPER.md`.
- **Faster data loading.** These fixes apply to every dataset and policy, including plain
  `lerobot-train`.
- **`tools/`**: task-agnostic utilities (dataset re-encoding, a tmux launcher).
- **`tasks/`**: one folder per task, holding only what is specific to that task.

## Setup

```bash
uv sync --locked --extra cup --extra piper --extra molmoact2 --extra async --extra groot
source .venv/bin/activate
```

This installs torch 2.11 (cu128), which works with the 550 driver, and torchcodec 0.11. Add every new
dependency to `pyproject.toml`, as an extra if it is optional, then run `uv lock`. `uv sync` removes
anything not in the lockfile.

## Workflow for a new task

1. **Record** a LeRobot dataset as usual and push it to the Hub.
2. **(Optional) re-encode it for fast training.** See "Choosing an image size" below.
   ```bash
   python tools/reencode_dataset.py --src_repo_id me/MyTask --dst_repo_id me/MyTask_224 \
       --image_size 224 --push --private
   ```
   Leave out `--image_size` to keep native resolution and only get the small-file and fast-decode
   encoding.
3. **Train.** Use plain `lerobot-train` unless the task needs custom transforms:
   ```bash
   tools/tmux_train.sh act-mytask 0 outputs/act_mytask/train.log -- \
       lerobot-train --policy.type=act --dataset.repo_id=me/MyTask_224 \
       --output_dir=outputs/act_mytask --num_workers=8 --wandb.enable=true
   ```
   `tools/tmux_train.sh <session> <gpu> <log> -- <cmd>` pins the run to a GPU and tees its output
   to a log. It never replaces an existing session.
4. **Custom task logic** goes in `tasks/<task>/`, as in `tasks/cup_to_container/`. Use that
   pattern: a dataset wrapper plus a small training loop.

### Choosing an image size

| Policy | Recommendation | Why |
|---|---|---|
| ACT, Diffusion (ResNet backbones) | 224 is standard (ABC, ACT-style VLAs); try 320–384 for fine placement | The model sees exactly the stored pixels, so smaller images are much faster |
| SmolVLA | Keep native resolution | It resizes to 512×512 internally, so 224 sources only lose detail. It is GPU-bound anyway |
| pi0 / SigLIP-224 models | 224 | Matches the model input |

`--resize_mode pad` (the default, as in ABC) keeps the whole frame and pads it with black.
`--resize_mode crop` center-crops instead, which avoids padding but loses the edges of the frame.
Inference must apply the same resize to live camera frames.

## Unattended training queues

`tools/pipeline.py <pipeline.json>` runs a list of jobs in order and watches each one, so a queue
keeps going overnight without anyone (or any Claude session) watching:

- **Ordering:** `after` (previous job finished), optional `precheck` command polled until it
  passes (e.g. access to a gated model), and a GPU with `gpu_mem_mib` free.
- **Watchdogs:** process exit, `success_regex` in the log, a stalled log (`stall_minutes`), and
  NaN/inf loss. Stalled or NaN runs are killed.
- **Recovery:** after an out-of-memory error, retries with the next `retries` override (e.g. a
  smaller `--batch_size`). Any failure before the first checkpoint gets one more try.
- **After success:** runs `on_success` commands, usually
  `tools/upload_checkpoints.py <run_dir> <repo_id> --collection <slug>` (every checkpoint as a
  `step-XXXXXX` branch, final on `main`).
- **Alerts:** `<state_dir>/alerts.log`, and W&B alerts (`wandb_alert`) that W&B can email you.
- **Restartable:** state lives in `<state_dir>/state.json`. A cron job running the
  `cron_watchdog.sh` pattern restarts a dead supervisor, which skips finished jobs and re-attaches
  to running ones.

Example: `/data/users/kmy2091/rethought-lerobot-outputs/pipeline_2026-10-11/pipeline.json`
(VITA -> Multi-Task DiT -> GR00T).

## Tasks

| Folder | What it is |
|---|---|
| `tasks/cup_to_container/` | ACT and SmolVLA with TCP-relative actions and a 17-D state. Custom `train.py`, transforms, and stats computed from parquet only (~45 s). Launchers default ACT to `BlueCupToContainer` (224) and SmolVLA to `CupToContainer` (native) |
| `tasks/rethoughtai/` | `lerobot-train` launchers for `andnetdeboer/rethoughtai` |
| `tasks/testtube_molmoact2/` | MolmoAct2 fine-tune on `CaptainRapid/TestTubeInsertion` |

Datasets: `andnetdeboer/CupToContainer` (original, 640×400 and 640×480) and
`andnetdeboer/BlueCupToContainer` (re-encoded at 224×224 letterbox, ~130 samples/s decode in a
single process versus ~7–15 for the original).

## Performance notes (2026-10-07)

These are what made training 8–10× slower than it should have been. All are fixed here.

1. **torchcodec silently broken.** torch had been downgraded to 2.6 under a torchcodec built for
   2.10, so LeRobot fell back to PyAV. The locked venv fixes this.
2. **PyAV with AV1 (dav1d)** started a thread pool sized to all 128 cores on every frame open
   (~230 ms per frame). `video_utils.decode_video_frames_pyav` now sets `thread_count = 1`.
3. **float32 frames over DataLoader IPC.** Custom loops should load uint8 (`return_uint8=True`)
   and convert on the GPU. `lerobot-train` already does this. Fixing it took the cup task from 31
   to 70 samples/s under load.
4. **Large images and huge video files.** Use `tools/reencode_dataset.py` (ABC-style).

ABC ideas not adopted: stacking all cameras into one mp4 and storing state/action as a flat
binary file. Both would break LeRobot dataset compatibility.
