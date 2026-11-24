#!/usr/bin/env python3
"""
IFARLAB VLA — SmolVLA Inference & VRAM / Batch Size Test Scripti (2. Hafta)

Proje Veri Yapısı Özeti (ifarlab/pickplace_vla_v1):
- observation.images.scene : (3, 240, 320) RGB, 10 fps (model içinde 512x512 pad + [-1, 1])
- observation.state        : float32 [8] -> [rail_m, shoulder_pan, shoulder_lift,
                             elbow, wrist_1, wrist_2, wrist_3, vacuum_on] (model içinde 32'ye pad)
- action                   : float32 [8] -> t+1 adımının 7 eklem pozisyonu + vacuum_cmd
- task                     : "kırmızı küpü al ve kutuya koy"
- chunk_size               : 50 adım (köprü düğümünde ilk ~10 adım = 1 s yürütülür)

Kullanım:
  python scripts/test_smolvla.py                # Sadece inference (gecikme + VRAM) testi (1 kamera)
  python scripts/test_smolvla.py --test-train   # Inference + batch_size (1, 4, 8, 16) eğitim VRAM testi
  python scripts/test_smolvla.py --test-train --amp  # bfloat16 mixed precision ile eğitim testi
"""

import argparse
import gc
import time

import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy

MODEL_ID = "lerobot/smolvla_base"
TASK_TEXT = "kırmızı küpü al ve kutuya koy"
IMG_H, IMG_W = 240, 320
STATE_DIM = 8  # ray + 6 eklem + vakum


def gpu_mem_gb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.memory_allocated() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    return allocated, peak


def select_camera_keys(model: SmolVLAPolicy, num_cameras: int) -> list[str]:
    """
    Modelin kamera anahtarlarından projemize uygun sayıda (varsayılan: 1 sabit sahne kamerası)
    anahtar seçer. num_cameras <= 0 ise checkpoint'teki tüm kamera anahtarlarını kullanır.
    """
    all_keys = list(model.config.image_features.keys()) or ["observation.images.scene"]
    if num_cameras <= 0 or num_cameras >= len(all_keys):
        return all_keys
    return all_keys[:num_cameras]


def build_dummy_frame(model: SmolVLAPolicy, img_keys: list[str]) -> dict:
    """
    SmolVLA preprocessor'ının beklediği tekil gözlem sözlüğünü projedeki 320x240 görüntü,
    durum vektörü ve Türkçe görev metniyle sentetik olarak oluşturur.
    """
    frame = {key: torch.rand(3, IMG_H, IMG_W, dtype=torch.float32) for key in img_keys}

    state_dim = STATE_DIM
    if model.config.robot_state_feature is not None:
        state_dim = model.config.robot_state_feature.shape[0]

    frame["observation.state"] = torch.zeros(state_dim, dtype=torch.float32)
    frame["task"] = TASK_TEXT
    return frame


def run_inference_benchmark(
    model: SmolVLAPolicy,
    preprocess,
    postprocess,
    device: torch.device,
    img_keys: list[str],
    runs: int = 10,
):
    print("\n=== 1. INFERENCE (ÇIKARIM) TESTİ ===")
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    raw_frame = build_dummy_frame(model, img_keys)

    # Warmup (ilk çalışma derleme/önbellek yüzünden yavaş olur)
    with torch.no_grad():
        batch = preprocess(raw_frame)
        _ = postprocess(model.predict_action_chunk(batch))
    if device.type == "cuda":
        torch.cuda.synchronize()

    e2e_times_ms = []
    model_times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            batch = preprocess(raw_frame)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_model = time.perf_counter()
            action_chunk = model.predict_action_chunk(batch)
            action_chunk = postprocess(action_chunk)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        e2e_times_ms.append((t_end - t0) * 1000.0)
        model_times_ms.append((t_end - t_model) * 1000.0)

    avg_e2e, min_e2e, max_e2e = sum(e2e_times_ms) / len(e2e_times_ms), min(e2e_times_ms), max(e2e_times_ms)
    avg_mod = sum(model_times_ms) / len(model_times_ms)
    alloc_gb, peak_gb = gpu_mem_gb()

    print(f"Kullanılan kamera sayısı ({len(img_keys)}) : {img_keys}")
    print(f"Çıktı action_chunk şekli         : {tuple(action_chunk.shape)}")
    print(f"Uçtan uca gecikme (ort/min/maks) : {avg_e2e:.1f} ms / {min_e2e:.1f} ms / {max_e2e:.1f} ms")
    print(f"Sadece model + postprocess (ort) : {avg_mod:.1f} ms")
    print(f"VRAM (anlık / tepe)              : {alloc_gb:.2f} GB / {peak_gb:.2f} GB")


def run_train_batch_benchmark(
    model: SmolVLAPolicy,
    preprocess,
    device: torch.device,
    img_keys: list[str],
    batch_sizes: list[int],
    use_amp: bool = False,
    train_steps: int = 3,
):
    print("\n=== 2. EĞİTİM (FORWARD + BACKWARD + ADAMW) BATCH SIZE / VRAM TESTİ ===")
    if device.type != "cuda":
        print("CUDA yok, eğitim VRAM testi atlandı.")
        return

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    param_dtype = next(model.parameters()).dtype
    print(
        f"Parametreler: {trainable_count / 1e6:.1f}M eğitilen / {total_params / 1e6:.1f}M toplam "
        f"| Ağırlık dtype: {param_dtype} | AMP (bfloat16): {use_amp} | 1 warmup + {train_steps} adım ort."
    )

    for bs in batch_sizes:
        optimizer = None
        batch = None
        loss = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model.train()
            optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

            # Preprocessor tekil örneklerde add_batch_dim yaptığı için batch'i tek tek hazırlayıp birleştiriyoruz
            items = [preprocess(build_dummy_frame(model, img_keys)) for _ in range(bs)]
            batch = {}
            for k in items[0].keys():
                if isinstance(items[0][k], torch.Tensor):
                    batch[k] = torch.cat([it[k] for it in items], dim=0).to(device)
                else:
                    batch[k] = items[0][k]
            del items  # GPU'da çift tensör kopyası kalmaması için temizle

            action_dim = model.config.action_feature.shape[0] if model.config.action_feature else STATE_DIM
            batch["action"] = torch.zeros(bs, model.config.chunk_size, action_dim, device=device, dtype=torch.float32)
            batch["action_is_pad"] = torch.zeros(bs, model.config.chunk_size, device=device, dtype=torch.bool)

            # 1 warmup adımı (ilk adım CUDA/cuDNN derleme maliyetini içerir)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, _ = model.forward(batch)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            step_times_ms = []
            for _ in range(train_steps):
                t0 = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss, _ = model.forward(batch)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                step_times_ms.append((time.perf_counter() - t0) * 1000.0)

            avg_step_ms = sum(step_times_ms) / len(step_times_ms)
            _, peak_gb = gpu_mem_gb()

            print(
                f"[OK]  batch_size={bs:<2d} | 1 adım (ort): {avg_step_ms:7.1f} ms | "
                f"Tepe VRAM: {peak_gb:.2f} GB"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] batch_size={bs:<2d} | Bellek yetersiz (Out of Memory)")
        except Exception as e:
            print(f"[HATA] batch_size={bs:<2d} | {type(e).__name__}: {e}")
        finally:
            model.zero_grad(set_to_none=True)
            del optimizer, batch, loss
            gc.collect()
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="IFARLAB SmolVLA GPU & Gecikme Testi")
    parser.add_argument("--runs", type=int, default=10, help="Inference tekrar sayısı")
    parser.add_argument("--test-train", action="store_true", help="Eğitim batch_size VRAM testini de çalıştır")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16], help="Denenecek batch size listesi")
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=1,
        help="Kullanılacak kamera sayısı (varsayılan: 1 — IFARLAB tek sabit sahne kamerası; 0 = tümü)",
    )
    parser.add_argument("--amp", action="store_true", help="Eğitim testinde bfloat16 mixed precision (autocast) kullan")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz     : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Toplam VRAM: {total_gb:.2f} GB")

    print(f"\nModel yükleniyor: {MODEL_ID} ...")
    model = SmolVLAPolicy.from_pretrained(MODEL_ID).to(device)
    preprocess, postprocess = make_pre_post_processors(
        model.config,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    img_keys = select_camera_keys(model, args.num_cameras)

    print("\n=== MODEL KONFİGÜRASYON ÖZETİ ===")
    print(f"chunk_size               : {model.config.chunk_size}")
    print(f"n_action_steps           : {model.config.n_action_steps}")
    print(f"max_state_dim / action   : {model.config.max_state_dim} / {model.config.max_action_dim}")
    print(f"resize_imgs_with_padding : {model.config.resize_imgs_with_padding}")
    print(f"image_features (model)   : {list(model.config.image_features.keys())}")
    print(f"image_features (test)    : {img_keys}")
    print(f"normalization_mapping    : {model.config.normalization_mapping}")

    run_inference_benchmark(model, preprocess, postprocess, device, img_keys, runs=args.runs)

    if args.test_train:
        run_train_batch_benchmark(model, preprocess, device, img_keys, args.batch_sizes, use_amp=args.amp)


if __name__ == "__main__":
    main()
