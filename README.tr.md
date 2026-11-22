[English](README.md) | **Türkçe**

# IFARLAB VLA — Vision-Language-Action Kontrol Hattı

**Gazebo Harmonic** (ROS 2 Humble) ortamında **lineer eksen üzerindeki UR10e** robot kolu ve **VGC10 vakum tutucu** için **LeRobot** ve **SmolVLA** (kontrol modeli olarak **ACT**) tabanlı uçtan uca kapalı döngü **Vision-Language-Action (VLA)** hattı.

---

## Proje Özeti

| Bileşen | Özellik |
| :--- | :--- |
| **Simülatör / Ara Katman** | Gazebo Harmonic · ROS 2 Humble |
| **Robot Platformu** | UR10e (6-DoF) + Lineer Ray (1-DoF) + VGC10 Vakum Tutucu |
| **Gözlem Uzayı** | Sabit sahne kamerası (`320×240` RGB, 10 Hz) + 7 eklem + vakum durumu (`float32 [8]`) + metin talimatı |
| **Eylem Uzayı** | 7 eklem hedef pozisyonu (`m`, `rad`) + vakum komutu (`float32 [8]`), 10 Hz |
| **Ana / Kontrol Modeli** | `lerobot/smolvla_base` (SmolVLA) · ACT (kontrol) |
| **Veri Seti Kimliği** | `ifarlab/pickplace_vla_v1` |
| **Sabitlenen LeRobot Commit'i** | [`b13b952556c597d1e0849cadb6b6e7496bc5a2f3`](env/lerobot_commit.txt) |

---

## Mimari

**ROS 2 Humble (Python 3.10)** ile **LeRobot / PyTorch (Python 3.12)** arasındaki bağımlılık çakışmalarını önlemek için sistem, **ZeroMQ / HTTP** üzerinden haberleşen iki izole ortama ayrılmıştır:

```text
Gazebo Harmonic + ROS 2 Humble (Python 3.10)        LeRobot Conda Ortamı (Python 3.12)
┌──────────────────────────────────────────┐        ┌──────────────────────────────┐
│  Sahne Kamerası + /sim/joint_states      │        │  Politika Sunucusu           │
│  + /vacuum/attached                      │ ─────► │  SmolVLA / ACT               │
│            ROS 2 Köprü Düğümü            │  ZMQ/  │  + meta/stats.json           │
│  (10 Hz, güvenlik kırpma, yörünge)       │ ◄───── │  (eylem dizisi: [K, 8])      │
└──────────────────────────────────────────┘  HTTP  └──────────────────────────────┘
```

---

## Depo Yapısı

```text
ifarlab-vla/
├── README.md                           # Proje özeti (EN)
├── README.tr.md                        # Proje özeti (TR)
├── .gitignore
├── env/
│   ├── lerobot_commit.txt              # Sabitlenen LeRobot git commit hash'i
│   └── requirements_lock.txt           # Referans pip freeze listesi
├── docs/
│   ├── installation.md                 # Adım adım kurulum rehberi (EN)
│   ├── installation.tr.md              # Adım adım kurulum rehberi (TR)
│   ├── smolvla_technical_note.md       # Mimari, veri yapısı ve GPU teknik notu (EN)
│   └── smolvla_technical_note.tr.md    # Mimari, veri yapısı ve GPU teknik notu (TR)
└── scripts/
    └── test_smolvla.py                 # SmolVLA gecikme ve VRAM test betiği
```

---

## Hızlı Başlangıç

1. **İzole Python 3.12 Conda ortamını ve sabitlenen LeRobot sürümünü kurun:**
   Bkz. [`docs/installation.tr.md`](docs/installation.tr.md).

2. **SmolVLA GPU testini (çıkarım gecikmesi + VRAM) çalıştırın:**
   ```bash
   conda activate lerobot
   python scripts/test_smolvla.py

   # Batch size 1, 4, 8, 16 için eğitim VRAM testini de eklemek için:
   python scripts/test_smolvla.py --test-train
   ```

3. **Veri yapısı ve ön-işleme sözleşmesini inceleyin:**
   Bkz. [`docs/smolvla_technical_note.tr.md`](docs/smolvla_technical_note.tr.md).
