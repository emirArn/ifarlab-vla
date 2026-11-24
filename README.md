**English** | [Türkçe](README.tr.md)

# IFARLAB VLA — Vision-Language-Action Pipeline

End-to-end **Vision-Language-Action (VLA)** pipeline for closed-loop robotic manipulation on a **UR10e manipulator mounted on a linear rail** with a **VGC10 vacuum gripper** in **Gazebo Harmonic** (ROS 2 Humble), using **LeRobot** and **SmolVLA** (with **ACT** as a control baseline).

---

## Overview

| Component | Specification |
| :--- | :--- |
| **Simulator / Middleware** | Gazebo Harmonic · ROS 2 Humble |
| **Robot Platform** | UR10e (6-DoF) + Linear Rail (1-DoF) + VGC10 Vacuum Gripper |
| **Observation Space** | Fixed scene camera (`320×240` RGB, 10 Hz) + 7 joints + vacuum state (`float32 [8]`) + text instruction |
| **Action Space** | 7 target joint positions (`m`, `rad`) + vacuum command (`float32 [8]`), 10 Hz |
| **Primary / Baseline Policy** | `lerobot/smolvla_base` (SmolVLA) · ACT (baseline) |
| **Dataset ID** | `ifarlab/pickplace_vla_v1` |
| **Pinned LeRobot Commit** | [`b13b952556c597d1e0849cadb6b6e7496bc5a2f3`](env/lerobot_commit.txt) |

---

## Architecture

To avoid dependency conflicts between **ROS 2 Humble (Python 3.10)** and **LeRobot / PyTorch (Python 3.12)**, the workspace is split into two isolated environments communicating over **ZeroMQ / HTTP**:

```text
Gazebo Harmonic + ROS 2 Humble (Python 3.10)        LeRobot Conda Env (Python 3.12)
┌──────────────────────────────────────────┐        ┌──────────────────────────────┐
│  Scene Camera + /sim/joint_states        │        │  Policy Server               │
│  + /vacuum/attached                      │ ─────► │  SmolVLA / ACT               │
│             ROS 2 Bridge Node            │  ZMQ/  │  + meta/stats.json           │
│  (10 Hz, safety clipping, trajectory)    │ ◄───── │  (action chunk: [K, 8])      │
└──────────────────────────────────────────┘  HTTP  └──────────────────────────────┘
```

---

## Repository Structure

```text
ifarlab-vla/
├── README.md                           # Project overview (EN)
├── README.tr.md                        # Project overview (TR)
├── .gitignore
├── env/
│   ├── lerobot_commit.txt              # Pinned LeRobot git commit hash
│   └── requirements_lock.txt           # Reference pip freeze snapshot
├── docs/
│   ├── installation.md                 # Step-by-step setup guide (EN)
│   ├── installation.tr.md              # Step-by-step setup guide (TR)
│   ├── smolvla_technical_note.md       # Architecture, data schema & GPU benchmark note (EN)
│   └── smolvla_technical_note.tr.md    # Architecture, data schema & GPU benchmark note (TR)
└── scripts/
    └── test_smolvla.py                 # SmolVLA inference latency & training VRAM benchmark
```

---

## Quick Start

1. **Install the isolated Python 3.12 Conda environment and pinned LeRobot version:**
   See [`docs/installation.md`](docs/installation.md).

2. **Run the SmolVLA GPU benchmark (inference latency + VRAM):**
   ```bash
   conda activate lerobot
   python scripts/test_smolvla.py

   # Include training (forward + backward + AdamW) VRAM check for batch sizes 1, 4, 8, 16:
   python scripts/test_smolvla.py --test-train

   # Training VRAM check with bfloat16 mixed precision (--amp):
   python scripts/test_smolvla.py --test-train --amp
   ```

3. **Review the data schema and preprocessing contract:**
   See [`docs/smolvla_technical_note.md`](docs/smolvla_technical_note.md).
