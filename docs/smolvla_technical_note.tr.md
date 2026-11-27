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

### Temel Mimari, Çıkarım ve İnce Ayar (Fine-Tuning) Parametreleri
* **Gözlem, Eylem Penceresi ve Çıkarım:** `n_obs_steps = 1`, `chunk_size = 50`, `n_action_steps = 50`, `num_steps = 10` (çıkarımda flow-matching gürültü giderme adım sayısı; gecikmeyi doğrudan belirler). Model tek anlık gözlemden 10 Hz'de **5 saniyelik (50 adım)** eylem dizisi üretir. `_get_action_chunk()` padding'i `action_feature` boyutuna göre otomatik kırptığı için ince ayarlı modelin çıktısı doğrudan `[B, 50, 8]` şeklindedir. Kapalı döngüde her ~1 saniyede bir yeniden sorgulama yapılabilmesi için politika sunucusu doğrudan `predict_action_chunk()` çağırarak ilk **`~10` adımı (1 s)** uygular (alternatif olarak `select_action()` kullanılacaksa `n_action_steps = 10` ayarlanmalıdır).
* **Tek GPU İnce Ayar Stratejisi:** Varsayılan olarak `freeze_vision_encoder = True`, `train_expert_only = True` ve `train_state_proj = True` etkindir. Görme-dil omurgası dondurularak yalnızca aksiyon uzmanı ve durum projeksiyon katmanı eğitilir (`lr = 1e-4`, `warmup = 1000`).

---

## 3. Lab GPU Performans ve Bellek Değerlendirmesi

Aşağıdaki ölçümler lab makinesinde (`matisse`) `python scripts/test_smolvla.py --test-train` (AMP kapalı ve `--amp` açık) betiği çalıştırılarak elde edilmiştir.

* **GPU Modeli / Toplam VRAM:** `NVIDIA GeForce RTX 4060` (`7.62 GB` VRAM)
* **Parametre / Ağırlık Tipi:** `99.9M` eğitilen / `450.0M` toplam · `torch.bfloat16`

### 3.1. Çıkarım (Inference) Performansı (`batch_size = 1`, 1 kamera `320×240` RGB)

> [!NOTE]
> İnce ayar öncesi `lerobot/smolvla_base` ile yapılan bu benchmark, temel checkpoint'in kendi 6 boyutlu `observation.state` ve ilk kamera anahtarı (`observation.images.camera1`) ile ölçülür. Model içinde durum vektörü zaten `max_state_dim = 32` boyutuna pad'lendiği için 8 boyutlu hedef şemaya göre süre/VRAM farkı ihmal edilebilir düzeydedir.

| Ölçüt | Değer |
| :--- | :--- |
| Uçtan Uca Gecikme — Ortalama / Min / Maks (ms) | `98.6 – 100.3 ms` (`98.0 ms` min / `102.5 ms` maks) |
| Sadece Model + Postprocess Gecikmesi — Ortalama (ms) | `98.3 – 100.0 ms` |
| Anlık / Tepe VRAM Kullanımı (GB) | `0.86 GB` / `0.89 GB` |
| 10 Hz Kontrol Döngüsü (1 s chunk yenileme) Uygunluğu | **Uygun** (1.000 ms yenileme aralığında ~100 ms sorgu süresi) |

### 3.2. Eğitim (`forward + backward + AdamW`) Batch Size ve VRAM Sınırları

* **Precision Modu:** Ağırlık tipi `torch.bfloat16` · `AMP: False` ve `AMP (bfloat16): True` ile ölçülmüştür (1 warmup + 3 adım ortalaması).

| `batch_size` | Durum (`OK` / `OOM`) | 1 Adım — AMP Kapalı (ms) | Tepe VRAM — AMP Kapalı (GB) | 1 Adım — AMP Açık (ms) | Tepe VRAM — AMP Açık (GB) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | `OK` | `62.2` | `1.65` | `65.3` | `1.65` |
| `4` | `OK` | `96.5` | `1.79` | `96.7` | `1.75` |
| `8` | `OK` | `167.3` | `2.32` | `179.0` | `2.22` |
| `16` | `OK` | `320.9` | `3.36` | `357.3` | `3.17` |

* **Eğitim İçin Önerilen `batch_size`:** **`16`** (`AMP Kapalı` modda `320.9 ms/adım` ile ~%10 daha hızlı ve yalnızca `3.36 GB` tepe VRAM tüketiyor; 8 GB RTX 4060'ta bile bolca bellek payı bırakıyor).
* **Tahmini 20.000 Adım İnce Ayar Süresi:** `batch_size = 16` ile yalnızca hesaplama süresi yaklaşık **~1.8 saat** (veri yükleme/video çözme ek süresi hariç).

---

## 4. Entegrasyon ve Uygulama İlkeleri

1. **Çifte Görüntü Ön-İşlemenin Önlenmesi:** `prepare_images()` fonksiyonu `512×512` padding ve `[-1, 1]` ölçeklemesini modelin *içinde* gerçekleştirmektedir. Veri toplama düğümü ve köprü düğümü, görüntüyü yalnızca `320×240` RGB (`uint8 [0, 255]` veya `float32 [0, 1]`) formatında iletmeli; dışarıda tekrar `[-1, 1]` ölçeklemesi yapmamalıdır.
2. **Eklem Sırası ve Birim Tutarlılığı:** 8 boyutlu durum/eylem vektöründeki sıra (`[sim_ur10e_base_to_robot_mount (m), shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3 (rad), vacuum (0/1)]`) veri toplama düğümü, `LeRobotDataset` dönüştürücüsü ve köprü düğümü arasında birebir korunmalıdır.
3. **Normalizasyon İstatistiklerinin Birlikte Yüklenmesi:** Politika sunucusu, eğitilmiş checkpoint ile birlikte ilgili `meta/stats.json` istatistiklerini yükleyerek `make_smolvla_pre_post_processors` üzerinden tutarlı `normalize` / `unnormalize` uygulamalıdır.
4. **Güvenlik Kırpması (Köprü Düğümü):** Politika sunucusundan dönen eylemler kontrolcüye gönderilmeden önce ray konum/hız sınırları (`0.05–1.95 m`, maks `1.25 m/s`) ve eklem limitleri ile kırpılmalıdır.

---

## 5. Gelecek Çalışma Önerisi: Sesli Komut Arayüzü (Whisper)

Ana kapalı döngü kontrol hattı doğrulandıktan sonra, eğitim hattına veya sabitlenmiş `lerobot` ortamına dokunmadan, mikrofondan alınan sesli komutları `/vla/task` ROS 2 topic'ine metin olarak yayınlayan hafif bir `faster-whisper` düğümü isteğe bağlı bir genişletme olarak kullanılabilir.

İzole bir Conda ortamında yapılan ön test sonuçları (`NVIDIA GeForce RTX 3050 Ti`, `4 GB` VRAM):

| Model (`faster-whisper`) | Cihaz / Precision | Gecikme (~3 sn ses) | VRAM | Not |
| :--- | :---: | :---: | :---: | :--- |
| `large-v3-turbo` | `cuda` / `int8_float16` | `~420 – 490 ms` | `1.13 GB` | Hem Türkçe hem İngilizce komutlarda yüksek doğruluk |

* **Değerlendirme:** Çıkarım sırasında `SmolVLA` (`0.89 GB`) ve `faster-whisper large-v3-turbo` (`1.13 GB`) birlikte toplam **`~2.02 GB` VRAM** gerektirmektedir. Sesli komut desteğinin sisteme dahil edilip edilmeyeceği, ana SmolVLA kapalı döngü hattı test edilip doğrulandıktan sonra hedef makinedeki gerçek GPU durumuna göre değerlendirilecektir.
