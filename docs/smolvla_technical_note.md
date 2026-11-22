**English** | [Türkçe](smolvla_technical_note.tr.md)

# IFARLAB VLA — SmolVLA Architecture, Data Schema & GPU Benchmark Technical Note

| Field | Value |
| :--- | :--- |
| **Project** | IFARLAB UR10e + Linear Rail VLA Control Pipeline (Gazebo Harmonic) |
| **Primary / Baseline Policy** | `lerobot/smolvla_base` (SmolVLA) · Baseline: ACT |
| **Dataset ID** | `ifarlab/pickplace_vla_v1` (10 Hz) |
| **Pinned LeRobot Commit** | `b13b952556c597d1e0849cadb6b6e7496bc5a2f3` |

---

## 1. Software Environment & Version Pinning

Model training and policy serving run inside an isolated **Python 3.12 Conda** environment (`lerobot`) located in `~/vla_ws`, decoupled from the system **ROS 2 Humble (Python 3.10)** workspace (`~/colcon_ws`). During closed-loop execution, the ROS 2 bridge node communicates with the policy server over **ZeroMQ / HTTP**.

| Component | Configuration |
| :--- | :--- |
| **Python / Conda Env** | Python `3.12` (`conda env: lerobot`) |
| **LeRobot Installation** | `pip install -e ".[smolvla]"` (commit: `b13b952556c597d1e0849cadb6b6e7496bc5a2f3`) |
| **Video Codec** | `ffmpeg` (`conda-forge`, for `LeRobotDataset` MP4 support) |
| **Pretrained Backbone** | `lerobot/smolvla_base` (VLM: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`) |
| **Related Files** | [`docs/installation.md`](installation.md), [`scripts/test_smolvla.py`](../scripts/test_smolvla.py), [`env/lerobot_commit.txt`](../env/lerobot_commit.txt) |

> [!IMPORTANT]
> Dataset conversion, model training, policy serving, and offline/closed-loop evaluation must all use the exact same pinned `lerobot` commit (`b13b952556c597d1e0849cadb6b6e7496bc5a2f3`).

---

## 2. SmolVLA Data Schema & Preprocessing Contract

Inspection of `configuration_smolvla.py`, `processor_smolvla.py`, and `modeling_smolvla.py` confirms the exact alignment between `ifarlab/pickplace_vla_v1` and SmolVLA's internal processing pipeline:

| Feature / Column | Dataset Format | Internal SmolVLA Processing & Normalization |
| :--- | :--- | :--- |
| `observation.images.scene` | `(240, 320, 3)` RGB video, 10 fps → tensor `[B, 3, 240, 320]` | `VISUAL: IDENTITY` (no mean/std). `prepare_images()` converts `uint8` via `/ 255.0`, aspect-ratio-preserving pads to **`512×512`** (`pad_value=0`), and scales to **`[-1, 1]`** (`img * 2.0 - 1.0`). |
| `observation.state` | `float32 [8]`: `[rail_m, shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, vacuum_on]` | `STATE: MEAN_STD` (using `meta/stats.json`). Zero-padded by `prepare_state()` to `max_state_dim = 32`. |
| `action` | `float32 [8]`: `t+1` target positions for 7 joints (`m`, `rad`) + `vacuum_cmd` (`0/1`) | `ACTION: MEAN_STD`. Zero-padded by `prepare_action()` to `max_action_dim = 32`. Post-processor `unnormalize` restores physical units on inference. |
| `task` | `"kırmızı küpü al ve kutuya koy"` | `NewLineTaskProcessorStep` appends `\n`; tokenized with `SmolVLM2` tokenizer (`max_length = 48`). |
| `extra.joint_ref`, `extra.cube_pose` | Offline analysis only | Does not start with `observation.`; excluded from `input_features` and never fed to the model. |

### Key Architecture & Fine-Tuning Parameters
* **Observation / Action Horizon:** `n_obs_steps = 1`, `chunk_size = 50`, `n_action_steps = 50`. From a single observation, the model predicts a **5-second (50-step)** action chunk at 10 Hz. During closed-loop control, the bridge node extracts the first `8` dimensions (`[:, :, :8]`), executes the first **`~10` steps (1 s)**, and re-queries the policy server.
* **Single-GPU Fine-Tuning Strategy:** By default, `freeze_vision_encoder = True`, `train_expert_only = True`, and `train_state_proj = True`. The VLM backbone stays frozen while only the action expert and state projection are trained (`lr = 1e-4`, `warmup = 1000`, `num_steps = 10` flow matching).

---

## 3. Lab GPU Performance & Memory Benchmark

The metrics below are measured on the lab GPU machine using `python scripts/test_smolvla.py --test-train`.

* **GPU Model / Total VRAM:** `[PENDING — 20 GB Lab GPU]`

### 3.1. Inference Performance (`batch_size = 1`, `320×240` RGB + `[8]` state)

| Metric | Value |
| :--- | :--- |
| Latency — Mean / Min / Max (ms) | `[PENDING]` |
| VRAM — Current / Peak (GB) | `[PENDING]` |
| Suitability for 10 Hz Control (1 s chunk replanning) | `[PENDING]` |

### 3.2. Training (`forward + backward`) Batch Size & VRAM Limits

| `batch_size` | Status (`OK` / `OOM`) | Step Time (ms) | Peak VRAM (GB) |
| :---: | :---: | :---: | :---: |
| `1` | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| `4` | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| `8` | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| `16` | `[PENDING]` | `[PENDING]` | `[PENDING]` |

* **Recommended `batch_size` for Fine-Tuning:** `[PENDING]`

---

## 4. Integration & Deployment Principles

1. **Avoid Double Image Normalization:** `prepare_images()` already performs `512×512` padding and `[-1, 1]` scaling *inside* the model. The data collection node and ROS 2 bridge node must pass raw `320×240` RGB frames (`uint8 [0, 255]` or `float32 [0, 1]`) without external `[-1, 1]` scaling.
2. **Joint Order & Unit Parity:** The 8-dimensional state/action order (`[sim_ur10e_base_to_robot_mount (m), shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3 (rad), vacuum (0/1)]`) must remain strictly identical across data collection, `LeRobotDataset` conversion, and the bridge node.
3. **Normalization Statistics Persistence:** The policy server must load the matching `meta/stats.json` alongside each checkpoint so `make_smolvla_pre_post_processors` applies consistent `normalize` / `unnormalize` transforms.
4. **Safety Clipping (Bridge Node):** Predicted actions must be clipped to joint/velocity limits (linear rail: `0.05–1.95 m`, max `1.25 m/s`) before dispatching trajectories to the controller.
