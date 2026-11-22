**English** | [Türkçe](installation.tr.md)

# Installation Guide — LeRobot & SmolVLA

This guide provides step-by-step instructions to set up a reproducible **LeRobot** and **SmolVLA** development, training, and inference environment for **IFARLAB VLA**.

To avoid dependency conflicts with **ROS 2 Humble (Python 3.10)**, the VLA stack runs inside an isolated **Python 3.12 Conda** environment (`lerobot`) in `~/vla_ws`, separate from the ROS 2 workspace (`~/colcon_ws`).

---

## Environment Summary

| Component | Target Specification |
| :--- | :--- |
| **Operating System** | Ubuntu 22.04+ (Linux x86_64) |
| **Python / Conda Env** | Python `3.12` (`conda env: lerobot`) |
| **Pinned LeRobot Commit** | `b13b952556c597d1e0849cadb6b6e7496bc5a2f3` |
| **Video Codec Support** | `ffmpeg` (`conda-forge`) |
| **Base Pretrained Model** | `lerobot/smolvla_base` |
| **Workspace Directory** | `~/vla_ws` |

---

## Prerequisites

* NVIDIA GPU with an up-to-date driver (`nvidia-smi` should report your GPU and CUDA version)
* `git` and `wget` installed (`sudo apt update && sudo apt install -y git wget`)
* At least **15–20 GB** of free disk space

```bash
nvidia-smi
```

---

## 1. Install Miniconda

Check whether `conda` is already available:

```bash
which conda
```

If not found, install Miniconda and initialize your shell:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

> [!NOTE]
> Recent Miniconda releases require accepting the Anaconda channel Terms of Service (`ToS`) before creating environments. Run the following if you encounter `CondaToSNonInteractiveError`:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

---

## 2. Create the Python 3.12 Environment and Install `ffmpeg`

Create and activate the `lerobot` environment, then install `ffmpeg` for `LeRobotDataset` MP4 video decoding/encoding:

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
conda install ffmpeg -c conda-forge -y
```

> [!IMPORTANT]
> Make sure `(lerobot)` appears in your terminal prompt before running any subsequent `pip` or `python` commands.

---

## 3. Install LeRobot and SmolVLA (Pinned Commit)

Clone `lerobot` into `~/vla_ws` and check out the pinned commit so that dataset conversion, training, and policy serving share the exact same codebase:

```bash
mkdir -p ~/vla_ws && cd ~/vla_ws
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout b13b952556c597d1e0849cadb6b6e7496bc5a2f3
pip install -e ".[smolvla]"
```

---

## 4. Verify Installation

Run the following checks:

```bash
# 1. Verify PyTorch CUDA and SmolVLA import
python -c "import torch, lerobot; from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('SmolVLAPolicy OK')"

# 2. Verify FFmpeg
ffmpeg -version | head -n 1

# 3. Verify pinned commit hash
git -C ~/vla_ws/lerobot rev-parse HEAD
```

**Expected output:**
* `CUDA: True` along with your NVIDIA GPU name
* `SmolVLAPolicy OK`
* `ffmpeg version ...`
* Commit hash: `b13b952556c597d1e0849cadb6b6e7496bc5a2f3`

---

## 5. Run the GPU Benchmark

From this repository root:

```bash
conda activate lerobot

# Inference latency & VRAM benchmark
python scripts/test_smolvla.py

# Inference + training forward/backward VRAM benchmark (batch sizes 1, 4, 8, 16)
python scripts/test_smolvla.py --test-train
```

---

## Troubleshooting

| Issue | Resolution |
| :--- | :--- |
| `conda: command not found` | Open a new terminal or run `source ~/.bashrc`. |
| `CondaToSNonInteractiveError` | Run the two `conda tos accept ...` commands in Step 1. |
| `CUDA: False` | Check `nvidia-smi`; if a CPU PyTorch wheel was installed, reinstall CUDA PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` |
| `OutOfMemoryError (OOM)` | SmolVLA fine-tuning requires high VRAM (~20 GB recommended). Try smaller batches with `--batch-sizes 1 2 4`. |
