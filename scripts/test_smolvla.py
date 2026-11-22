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
  python test_smolvla.py                # Sadece inference (gecikme + VRAM) testi
  python test_smolvla.py --test-train   # Inference + batch_size (1, 4, 8, 16) eğitim VRAM testi
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


def build_dummy_frame(model: SmolVLAPolicy, batch_size: int = 1, include_action: bool = False) -> dict:
    """
    SmolVLA'nın beklediği girdi sözlüğünü (batch) bizim projedeki 320x240 görüntü,
    8 boyutlu durum ve Türkçe görev metniyle sentetik olarak oluşturur.
    """
    frame = {}

    # smolvla_base varsayılan kamera anahtarlarını otomatik doldur
    img_keys = list(model.config.image_features.keys()) or ["observation.images.scene"]
    for key in img_keys:
        if batch_size == 1 and not include_action:
            # preprocess add_batch_dim ekleyeceği için tekil örnekte (C, H, W)
            frame[key] = torch.rand(3, IMG_H, IMG_W, dtype=torch.float32)
        else:
            frame[key] = torch.rand(batch_size, 3, IMG_H, IMG_W, dtype=torch.float32)

    state_dim = STATE_DIM
    if model.config.robot_state_feature is not None:
        state_dim = model.config.robot_state_feature.shape[0]

    if batch_size == 1 and not include_action:
        frame["observation.state"] = torch.zeros(state_dim, dtype=torch.float32)
        frame["task"] = TASK_TEXT
    else:
        frame["observation.state"] = torch.zeros(batch_size, state_dim, dtype=torch.float32)
        frame["task"] = [TASK_TEXT] * batch_size

    if include_action:
        action_dim = STATE_DIM
        if model.config.action_feature is not None:
            action_dim = model.config.action_feature.shape[0]
        frame["action"] = torch.zeros(batch_size, model.config.chunk_size, action_dim, dtype=torch.float32)

    return frame


def run_inference_benchmark(model: SmolVLAPolicy, preprocess, postprocess, device: torch.device, runs: int = 10):
    print("\n=== 1. INFERENCE (ÇIKARIM) TESTİ ===")
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    raw_frame = build_dummy_frame(model, batch_size=1, include_action=False)
    batch = preprocess(raw_frame)

    # Warmup (ilk çalışma derleme/önbellek yüzünden yavaş olur)
    with torch.no_grad():
        _ = model.predict_action_chunk(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            action_chunk = model.predict_action_chunk(batch)
            action_chunk = postprocess(action_chunk)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    alloc_gb, peak_gb = gpu_mem_gb()

    print(f"Çıktı action_chunk şekli : {tuple(action_chunk.shape)}")
    print(f"Gecikme (ortalama / min / max): {avg_ms:.1f} ms / {min_ms:.1f} ms / {max_ms:.1f} ms")
    print(f"VRAM (anlık / tepe)           : {alloc_gb:.2f} GB / {peak_gb:.2f} GB")


def run_train_batch_benchmark(model: SmolVLAPolicy, preprocess, device: torch.device, batch_sizes: list[int]):
    print("\n=== 2. EĞİTİM (FORWARD + BACKWARD) BATCH SIZE / VRAM TESTİ ===")
    if device.type != "cuda":
        print("CUDA yok, eğitim VRAM testi atlandı.")
        return

    for bs in batch_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model.train()
            # Preprocessor tekil örneklerde add_batch_dim yaptığı için batch'i tek tek hazırlayıp birleştiriyoruz
            items = [preprocess(build_dummy_frame(model, batch_size=1, include_action=False)) for _ in range(bs)]
            batch = {}
            for k in items[0].keys():
                if isinstance(items[0][k], torch.Tensor):
                    batch[k] = torch.cat([it[k] for it in items], dim=0).to(device)
                else:
                    batch[k] = items[0][k]

            action_dim = model.config.action_feature.shape[0] if model.config.action_feature else STATE_DIM
            batch["action"] = torch.zeros(bs, model.config.chunk_size, action_dim, device=device, dtype=torch.float32)
            batch["action_is_pad"] = torch.zeros(bs, model.config.chunk_size, device=device, dtype=torch.bool)

            t0 = time.perf_counter()
            loss, _ = model.forward(batch)
            loss.backward()
            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t0) * 1000.0
            _, peak_gb = gpu_mem_gb()
            model.zero_grad(set_to_none=True)

            print(f"[OK]  batch_size={bs:<2d} | 1 adım: {step_ms:7.1f} ms | Tepe VRAM: {peak_gb:.2f} GB | loss={loss.item():.4f}")
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] batch_size={bs:<2d} | Bellek yetersiz (Out of Memory)")
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[HATA] batch_size={bs:<2d} | {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(description="IFARLAB SmolVLA GPU & Gecikme Testi")
    parser.add_argument("--runs", type=int, default=10, help="Inference tekrar sayısı")
    parser.add_argument("--test-train", action="store_true", help="Eğitim batch_size VRAM testini de çalıştır")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16], help="Denenecek batch size listesi")
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

    print("\n=== MODEL KONFİGÜRASYON ÖZETİ ===")
    print(f"chunk_size               : {model.config.chunk_size}")
    print(f"n_action_steps           : {model.config.n_action_steps}")
    print(f"max_state_dim / action   : {model.config.max_state_dim} / {model.config.max_action_dim}")
    print(f"resize_imgs_with_padding : {model.config.resize_imgs_with_padding}")
    print(f"image_features           : {list(model.config.image_features.keys())}")
    print(f"normalization_mapping    : {model.config.normalization_mapping}")

    run_inference_benchmark(model, preprocess, postprocess, device, runs=args.runs)

    if args.test_train:
        run_train_batch_benchmark(model, preprocess, device, args.batch_sizes)


if __name__ == "__main__":
    main()
