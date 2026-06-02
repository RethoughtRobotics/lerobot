# Agilex Piper — LeRobot Integration

Bimanual Agilex Piper setup (2 × 7-DOF arms, 14 DOF total) with three RealSense cameras.

---

## Hardware assumptions

| Component | Default |
|---|---|
| Left arm CAN | `can0` |
| Right arm CAN | `can1` |
| Cameras | 3 × Intel RealSense (`left_gripper`, `right_gripper`, `head`) |
| State / action dims | 14 (6 joints + 1 gripper per arm) |

---

## 1. Prerequisites

**On both machines:**
```bash
git clone https://github.com/<your-fork>/lerobot.git
cd lerobot
uv sync --locked --extra all
```

**Piper SDK** (robot PC only):
```bash
pip install piper_sdk
```

**CAN setup** (robot PC only — run once per boot):
```bash
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
```

---

## 2. Configure camera serial numbers

Find your RealSense serial numbers:
```bash
lerobot-find-cameras
```

Then edit `src/lerobot/robots/piper/config_piper.py` and replace the three placeholders:

```python
"left_gripper": RealSenseCameraConfig(serial_number_or_name="<LEFT_GRIPPER_SERIAL>", ...),
"right_gripper": RealSenseCameraConfig(serial_number_or_name="<RIGHT_GRIPPER_SERIAL>", ...),
"head":          RealSenseCameraConfig(serial_number_or_name="<HEAD_SERIAL>", ...),
```

This is a one-time step — you never need to pass camera args on the command line after this.

---

## 3. Option A — Single machine (robot PC has a GPU)

```bash
lerobot-rollout \
    --strategy.type=base \
    --robot.type=piper \
    --inference.type=rtc \
    --policy.path=outputs/testtube_run1/checkpoints/006000/pretrained_model \
    --task="test tube insertion" \
    --duration=30
```

---

## 4. Option B — Split: GPU server + robot PC

Use this when inference runs on a separate GPU server.

### 4.1 On the GPU server

```bash
uv pip install 'lerobot[async]
```

```bash
python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080 \
    --fps=30
```

### 4.2 On the robot PC

Make sure the checkpoint is accessible (copy it over or use a shared mount).

```bash
python src/lerobot/async_inference/robot_client.py \
    --robot.type=piper \
    --task="test tube insertion" \
    --server_address=<SERVER_IP>:8080 \
    --policy_type=molmoact2 \
    --pretrained_name_or_path=outputs/testtube_run1/checkpoints/006000/pretrained_model \
    --policy_device=cuda \
    --client_device=cpu \
    --actions_per_chunk=50
```

---

## 5. Training a new policy

Collect or use an existing dataset on HuggingFace Hub, then train:

```bash
uv run lerobot-train \
    --dataset.repo_id=<HF_USER>/<DATASET> \
    --policy.type=molmoact2 \
    --output_dir=outputs/train/<run_name>
```

Checkpoints are saved to `outputs/train/<run_name>/checkpoints/`.

---

## 6. Troubleshooting

**Arm fails to enable:**
Check CAN interfaces are up (`ip link show can0`) and the arm is powered on before running.

**Camera not found:**
Re-run `lerobot-find-cameras` and confirm serial numbers match `config_piper.py`.

**`piper_sdk` not found:**
Install it with `pip install piper_sdk` on the robot PC. It is not required on the GPU server.

**Wrong observation shape:**
The model expects `observation.state` of shape `[14]`. Confirm both arms are connected — a single arm gives `[7]` and will fail.
