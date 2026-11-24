[English](installation.md) | **Türkçe**

# Kurulum Rehberi — LeRobot & SmolVLA

Bu rehber, **IFARLAB VLA** çalışmaları için **LeRobot** ve **SmolVLA** geliştirme, eğitim ve çıkarım (inference) ortamının sıfırdan kurulumunu adım adım açıklar.

Sistemdeki **ROS 2 Humble (Python 3.10)** paketleriyle bağımlılık çakışması yaşanmaması için ortam, ayrı bir **Python 3.12 Conda** ortamında (`lerobot`) ve ROS 2 çalışma alanından (`~/colcon_ws`) bağımsız bir `~/vla_ws` dizininde izole olarak kurulur.

---

## Sistem ve Sürüm Özeti

| Bileşen | Hedef Değer |
| :--- | :--- |
| **İşletim Sistemi** | Ubuntu 22.04+ (Linux x86_64) |
| **Python / Ortam** | Python `3.12` (Miniconda — `lerobot` ortamı) |
| **LeRobot Sürümü (Sabit Commit)** | `b13b952556c597d1e0849cadb6b6e7496bc5a2f3` |
| **Video Desteği** | `ffmpeg` (`conda-forge`) |
| **Temel Model** | `lerobot/smolvla_base` (SmolVLA) |
| **Çalışma Dizini** | `~/vla_ws` |

---

## Ön Koşullar

* NVIDIA GPU ve **CUDA 13** uyumlu güncel sürücü (`nvidia-smi` çıktısı görünmeli; kilit dosyadaki `torch==2.15.0` CUDA 13 çalışma zamanı kullanır)
* `git` ve `wget` kurulu (yoksa: `sudo apt update && sudo apt install -y git wget`)
* En az **15–20 GB** boş disk alanı

```bash
nvidia-smi
```

---

## 1. Miniconda Kurulumu

Sistemde `conda` kurulu olup olmadığını kontrol edin:

```bash
which conda
```

Çıktı boşsa Miniconda'yı kurun ve shell entegrasyonunu etkinleştirin:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

> [!NOTE]
> Yeni Conda sürümlerinde ilk kullanımda Anaconda kanal kullanım şartlarını (`ToS`) onaylamak gerekir. Aksi halde `CondaToSNonInteractiveError` hatası alınır:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

---

## 2. Python 3.12 Ortamı ve `ffmpeg` Kurulumu

`lerobot` adında izole bir **Python 3.12** ortamı oluşturun, aktif edin ve `LeRobotDataset` MP4 video desteği için `ffmpeg` kurun:

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
conda install ffmpeg -c conda-forge -y
```

Alternatif olarak doğrudan depo içindeki [`env/environment.yml`](../env/environment.yml) dosyasıyla tek komutta da oluşturabilirsiniz:

```bash
conda env create -f env/environment.yml
conda activate lerobot
```

> [!IMPORTANT]
> Bu adımdan sonra terminal satırının başında `(lerobot)` etiketi görünmelidir. Sonraki tüm `pip` ve `python` komutları bu ortam aktifken çalıştırılmalıdır.

---

## 3. LeRobot ve SmolVLA Kurulumu (Sabit Sürüm)

Veri dönüştürme, eğitim ve politika sunucusunun birebir aynı kod tabanıyla çalışması için `lerobot` deposu sabitlenen commit üzerinden kurulur:

```bash
mkdir -p ~/vla_ws && cd ~/vla_ws
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout b13b952556c597d1e0849cadb6b6e7496bc5a2f3
pip install -e ".[smolvla]"
```

---

## 4. Kurulum Doğrulama

Aşağıdaki kontrollerin tamamının sorunsuz geçmesi gerekir:

```bash
# 1. PyTorch CUDA ve SmolVLA import kontrolü
python -c "import torch, lerobot; from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('SmolVLAPolicy OK')"

# 2. FFmpeg kontrolü
ffmpeg -version | head -n 1

# 3. Sabitlenen commit doğrulaması
git -C ~/vla_ws/lerobot rev-parse HEAD
```

**Beklenen çıktılar:**
* `CUDA: True` ve makinedeki NVIDIA GPU adı
* `SmolVLAPolicy OK`
* `ffmpeg version ...`
* Commit hash: `b13b952556c597d1e0849cadb6b6e7496bc5a2f3`

---

## 5. GPU Performans ve Bellek Testi

Bu depo kök dizininden:

```bash
conda activate lerobot

# Çıkarım (inference gecikmesi + VRAM) testi — varsayılan: 1 sabit sahne kamerası (--num-cameras 1)
python scripts/test_smolvla.py

# Çıkarım + batch size (1, 4, 8, 16) eğitim (forward + backward + AdamW) VRAM testi
python scripts/test_smolvla.py --test-train

# bfloat16 mixed precision (--amp) ile eğitim VRAM testi
python scripts/test_smolvla.py --test-train --amp
```

---

## Sorun Giderme

| Sorun | Çözüm |
| :--- | :--- |
| `conda: command not found` | Terminali kapatıp yeniden açın veya `source ~/.bashrc` çalıştırın. |
| `CondaToSNonInteractiveError` | Adım 1'deki iki `conda tos accept ...` komutunu çalıştırın. |
| `CUDA: False` dönüyor | `nvidia-smi` ile sürücünün CUDA 13 desteklediğini doğrulayın; CPU wheel kurulmuşsa kilit sürümle yeniden kurun: `pip install --force-reinstall torch==2.15.0 torchvision==0.29.2` |
| `OutOfMemoryError (OOM)` | SmolVLA ince ayarı için yüksek VRAM gerekir (önerilen ~20 GB). `--amp` (`bfloat16`) ve/veya `--batch-sizes 1 2 4` ile daha küçük batch değerlerini deneyin. |
