[English](smolvla_technical_note.md) | **Türkçe**

# IFARLAB VLA — SmolVLA Mimari, Veri Yapısı ve GPU Değerlendirme Teknik Notu

| Alan | Değer |
| :--- | :--- |
| **Proje** | IFARLAB UR10e + Lineer Eksen VLA Kontrol Hattı (Gazebo Harmonic) |
| **Hedef / Kontrol Modeli** | `lerobot/smolvla_base` (SmolVLA) · Kontrol: ACT |
| **Veri Seti Kimliği** | `ifarlab/pickplace_vla_v1` (10 Hz) |
| **Sabitlenen LeRobot Commit'i** | `b13b952556c597d1e0849cadb6b6e7496bc5a2f3` |

---

## 1. Yazılım Ortamı ve Sürüm Sabitleme

Model eğitimi ve politika sunucusu, sistemdeki **ROS 2 Humble (Python 3.10)** bağımlılıklarıyla çakışmayı önlemek amacıyla ayrı bir **Python 3.12 Conda** ortamında (`lerobot`) ve ROS 2 çalışma alanından (`~/colcon_ws`) izole bir `~/vla_ws` dizininde yapılandırılmıştır. Kapalı döngü çalıştırma aşamasında ROS 2 köprü düğümü ile politika sunucusu **ZeroMQ / HTTP** üzerinden haberleşecektir.

| Bileşen | Yapılandırma |
| :--- | :--- |
| **Python / Sanal Ortam** | Python `3.12` (`conda env: lerobot`) |
| **LeRobot Kurulumu** | `pip install -e ".[smolvla]"` (commit: `b13b952556c597d1e0849cadb6b6e7496bc5a2f3`) |
| **Video Kodlayıcı / Çözücü** | `ffmpeg` (`conda-forge`, `LeRobotDataset` MP4 desteği için) |
| **Ön-Eğitimli Omurga** | `lerobot/smolvla_base` (VLM: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`) |
| **İlgili Dosyalar** | [`docs/installation.tr.md`](installation.tr.md), [`scripts/test_smolvla.py`](../scripts/test_smolvla.py), [`env/lerobot_commit.txt`](../env/lerobot_commit.txt) |

> [!IMPORTANT]
> Veri dönüştürme, model eğitimi, politika sunucusu ve değerlendirme aşamalarının tamamında aynı sabit `lerobot` commit'i (`b13b952556c597d1e0849cadb6b6e7496bc5a2f3`) kullanılmalıdır.

---

## 2. SmolVLA Veri Yapısı ve Ön-İşleme Sözleşmesi

`lerobot` kaynak kodundaki `configuration_smolvla.py`, `processor_smolvla.py` ve `modeling_smolvla.py` modülleri incelenerek `ifarlab/pickplace_vla_v1` veri setinin SmolVLA girdi/çıktı katmanlarıyla eşleşmesi doğrulanmıştır:

| Sütun / Alan | Veri Seti Formatı | Model İçi İşlem ve Normalizasyon |
| :--- | :--- | :--- |
| `observation.images.scene` | `(240, 320, 3)` RGB video, 10 fps → tensör `[B, 3, 240, 320]` | `VISUAL: IDENTITY` (istatistiksel normalizasyon yok). `prepare_images()` içinde `uint8` ise `/ 255.0`, en-boy oranı korunarak **`512×512` pad** (`pad_value=0`) ve **`[-1, 1]`** (`img * 2.0 - 1.0`) dönüşümü uygulanır. |
| `observation.state` | `float32 [8]`: `[rail_m, shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, vacuum_on]` | `STATE: MEAN_STD` (`meta/stats.json` üzerinden). `prepare_state()` içinde `max_state_dim = 32` boyutuna sıfır ile doldurulur (`pad_vector`). |
| `action` | `float32 [8]`: `t+1` adımının 7 eklem hedef pozisyonu (`m`, `rad`) + `vacuum_cmd` (`0/1`) | `ACTION: MEAN_STD`. `prepare_action()` içinde `max_action_dim = 32` boyutuna pad'lenir. Çıkarım sonrasında post-processor `unnormalize` adımıyla gerçek fiziksel birimlere geri çevrilir. |
| `task` | `"kırmızı küpü al ve kutuya koy"` | `NewLineTaskProcessorStep` sonuna `\n` ekler; `SmolVLM2` tokenizer ile tokenize edilir (`max_length = 48`). |
| `extra.joint_ref`, `extra.cube_pose` | Yalnızca çevrimdışı analiz amaçlı | `observation.` ön eki taşımadığı için `input_features` içine dahil edilmez, modele girdi olarak aktarılmaz. |

### Temel Mimari ve İnce Ayar (Fine-Tuning) Parametreleri
* **Gözlem ve Eylem Penceresi:** `n_obs_steps = 1`, `chunk_size = 50`, `n_action_steps = 50`. Model tek anlık gözlemden 10 Hz'de **5 saniyelik (50 adım)** eylem dizisi üretir. Kapalı döngü yürütmede `predict_action_chunk()` çıktısının ilk `8` boyutu (`[:, :, :8]`) alınarak ilk **`~10` adım (1 s)** uygulanır ve ardından yeniden sorgulama yapılır.
* **Tek GPU İnce Ayar Stratejisi:** Varsayılan olarak `freeze_vision_encoder = True`, `train_expert_only = True` ve `train_state_proj = True` etkindir. Görme-dil omurgası dondurularak yalnızca aksiyon uzmanı ve durum projeksiyon katmanı eğitilir (`lr = 1e-4`, `warmup = 1000`, `num_steps = 10` flow matching).

---

## 3. Lab GPU Performans ve Bellek Değerlendirmesi

Aşağıdaki ölçümler lab GPU makinesinde `python scripts/test_smolvla.py --test-train` betiği çalıştırılarak elde edilecektir.

* **GPU Modeli / Toplam VRAM:** `[ÖLÇÜM BEKLENİYOR — 20 GB Lab GPU]`

### 3.1. Çıkarım (Inference) Performansı (`batch_size = 1`, `320×240` RGB + `[8]` state)

| Ölçüt | Değer |
| :--- | :--- |
| Ortalama / Min / Maks Gecikme (ms) | `[ÖLÇÜM BEKLENİYOR]` |
| Anlık / Tepe VRAM Kullanımı (GB) | `[ÖLÇÜM BEKLENİYOR]` |
| 10 Hz Kontrol Döngüsü (1 s chunk yenileme) Uygunluğu | `[ÖLÇÜM BEKLENİYOR]` |

### 3.2. Eğitim (`forward + backward`) Batch Size ve VRAM Sınırları

| `batch_size` | Durum (`OK` / `OOM`) | 1 Adım Süresi (ms) | Tepe VRAM (GB) |
| :---: | :---: | :---: | :---: |
| `1` | `[ÖLÇÜM BEKLENİYOR]` | `[ÖLÇÜM BEKLENİYOR]` | `[ÖLÇÜM BEKLENİYOR]` |
| `4` | `[ÖLÇÜM BEKLENİYOR]` | `[ÖLÇÜM BEKLENİYOR]` | `[ÖLÇÜM BEKLENİYOR]` |
| `8` | `[ÖLÇÜM BEKLENİYOR]` | `[ÖLÇÜM BEKLENİYOR]` | `[ÖLÇÜM BEKLENİYOR]` |
| `16` | `[ÖLÇÜM BEKLENİYOR]` | `[ÖLÇÜM BEKLENİYOR]` | `[ÖLÇÜM BEKLENİYOR]` |

* **Eğitim İçin Önerilen `batch_size`:** `[ÖLÇÜM BEKLENİYOR]`

---

## 4. Entegrasyon ve Uygulama İlkeleri

1. **Çifte Görüntü Ön-İşlemenin Önlenmesi:** `prepare_images()` fonksiyonu `512×512` padding ve `[-1, 1]` ölçeklemesini modelin *içinde* gerçekleştirmektedir. Veri toplama düğümü ve köprü düğümü, görüntüyü yalnızca `320×240` RGB (`uint8 [0, 255]` veya `float32 [0, 1]`) formatında iletmeli; dışarıda tekrar `[-1, 1]` ölçeklemesi yapmamalıdır.
2. **Eklem Sırası ve Birim Tutarlılığı:** 8 boyutlu durum/eylem vektöründeki sıra (`[sim_ur10e_base_to_robot_mount (m), shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3 (rad), vacuum (0/1)]`) veri toplama düğümü, `LeRobotDataset` dönüştürücüsü ve köprü düğümü arasında birebir korunmalıdır.
3. **Normalizasyon İstatistiklerinin Birlikte Yüklenmesi:** Politika sunucusu, eğitilmiş checkpoint ile birlikte ilgili `meta/stats.json` istatistiklerini yükleyerek `make_smolvla_pre_post_processors` üzerinden tutarlı `normalize` / `unnormalize` uygulamalıdır.
4. **Güvenlik Kırpması (Köprü Düğümü):** Politika sunucusundan dönen eylemler kontrolcüye gönderilmeden önce ray konum/hız sınırları (`0.05–1.95 m`, maks `1.25 m/s`) ve eklem limitleri ile kırpılmalıdır.
