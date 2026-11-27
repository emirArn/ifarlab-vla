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

### Key Architecture, Inference & Fine-Tuning Parameters
* **Observation / Action Horizon & Inference:** `n_obs_steps = 1`, `chunk_size = 50`, `n_action_steps = 50`, `num_steps = 10` (flow-matching denoising steps at inference time, which directly controls latency). From a single observation, the model predicts a **5-second (50-step)** action chunk at 10 Hz. Because `_get_action_chunk()` automatically unpads actions to `action_feature.shape[0]`, the fine-tuned model outputs `[B, 50, 8]` directly. To replan every ~1 s in closed loop, the policy server calls `predict_action_chunk()` directly and executes the first **`~10` steps (1 s)** (alternatively, if `select_action()` is used, `n_action_steps` must be set to `10`).
* **Single-GPU Fine-Tuning Strategy:** By default, `freeze_vision_encoder = True`, `train_expert_only = True`, and `train_state_proj = True`. The VLM backbone stays frozen while only the action expert and state projection are trained (`lr = 1e-4`, `warmup = 1000`).

---

## 3. Lab GPU Performance & Memory Benchmark

The metrics below were measured on the lab machine (`matisse`) using `python scripts/test_smolvla.py --test-train` (both with and without `--amp` for `bfloat16`).

* **GPU Model / Total VRAM:** `NVIDIA GeForce RTX 4060` (`7.62 GB` VRAM)
* **Parameters / Weight Dtype:** `99.9M` trainable / `450.0M` total · `torch.bfloat16`

### 3.1. Inference Performance (`batch_size = 1`, 1 camera `320×240` RGB)

> [!NOTE]
> Pre-fine-tuning benchmarks with `lerobot/smolvla_base` use the base checkpoint's own 6-dim `observation.state` and first camera key (`observation.images.camera1`). Because states are internally zero-padded to `max_state_dim = 32`, the latency/VRAM difference relative to the 8-dim target schema is negligible.

| Metric | Value |
| :--- | :--- |
| End-to-End Latency — Mean / Min / Max (ms) | `98.6 – 100.3 ms` (`98.0 ms` min / `102.5 ms` max) |
| Model + Postprocess Latency — Mean (ms) | `98.3 – 100.0 ms` |
| VRAM — Current / Peak (GB) | `0.86 GB` / `0.89 GB` |
| Suitability for 10 Hz Control (1 s chunk replanning) | **Suitable** (~100 ms per query vs. 1,000 ms replanning interval) |

### 3.2. Training (`forward + backward + AdamW`) Batch Size & VRAM Limits

* **Precision Mode:** Weight dtype `torch.bfloat16` · tested with `AMP: False` and `AMP (bfloat16): True` (average of 3 steps after 1 warmup step).

| `batch_size` | Status (`OK` / `OOM`) | Step Time — AMP Off (ms) | Peak VRAM — AMP Off (GB) | Step Time — AMP On (ms) | Peak VRAM — AMP On (GB) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | `OK` | `62.2` | `1.65` | `65.3` | `1.65` |
| `4` | `OK` | `96.5` | `1.79` | `96.7` | `1.75` |
| `8` | `OK` | `167.3` | `2.32` | `179.0` | `2.22` |
| `16` | `OK` | `320.9` | `3.36` | `357.3` | `3.17` |

* **Recommended `batch_size` for Fine-Tuning:** **`16`** (`AMP Off` is ~10% faster at `320.9 ms/step` with only `3.36 GB` peak VRAM, leaving plenty of headroom even on an 8 GB RTX 4060).
* **Estimated 20,000-Step Fine-Tuning Time:** ~1.8 hours pure compute at `batch_size = 16` (excluding dataloader/video decoding overhead).

---

## 4. Integration & Deployment Principles

1. **Avoid Double Image Normalization:** `prepare_images()` already performs `512×512` padding and `[-1, 1]` scaling *inside* the model. The data collection node and ROS 2 bridge node must pass raw `320×240` RGB frames (`uint8 [0, 255]` or `float32 [0, 1]`) without external `[-1, 1]` scaling.
2. **Joint Order & Unit Parity:** The 8-dimensional state/action order (`[sim_ur10e_base_to_robot_mount (m), shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3 (rad), vacuum (0/1)]`) must remain strictly identical across data collection, `LeRobotDataset` conversion, and the bridge node.
3. **Normalization Statistics Persistence:** The policy server must load the matching `meta/stats.json` alongside each checkpoint so `make_smolvla_pre_post_processors` applies consistent `normalize` / `unnormalize` transforms.
4. **Safety Clipping (Bridge Node):** Predicted actions must be clipped to joint/velocity limits (linear rail: `0.05–1.95 m`, max `1.25 m/s`) before dispatching trajectories to the controller.

---

## 5. Future Work Suggestion: Voice Command Interface (Whisper)

Once the core closed-loop pipeline is validated, a lightweight speech-to-text front-end (`faster-whisper`) can optionally be integrated as a standalone ROS 2 node publishing recognized instructions to a `/vla/task` topic, without modifying the training pipeline or the `lerobot` environment.

Preliminary standalone benchmarks (measured on an `NVIDIA GeForce RTX 3050 Ti`, `4 GB` VRAM, in an isolated Conda environment):

| Model (`faster-whisper`) | Device / Precision | Latency (~3 s audio) | VRAM | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `large-v3-turbo` | `cuda` / `int8_float16` | `~420 – 490 ms` | `1.13 GB` | High accuracy in both Turkish & English |

* **Assessment:** Combined inference memory for `SmolVLA` (`0.89 GB`) and `faster-whisper large-v3-turbo` (`1.13 GB`) is **`~2.02 GB`**. Whether to integrate voice commands will be evaluated based on actual GPU and system resource availability after the core SmolVLA closed-loop pipeline is validated.
