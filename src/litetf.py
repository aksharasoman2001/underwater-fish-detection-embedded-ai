"""
litetf.py — TFLite Inference Benchmarking Script for Raspberry Pi 5
DARCY: Design and Development of an Underwater Event Detection System
       Based on Embedded AI

Author  : Akshara Soman
Lab     : Lab-STICC (UMR CNRS 6285), ENIB Brest, France
Program : MSc Automotive Embedded Systems, ESIGELEC
Date    : 2025


Usage:
    python src/litetf.py --model best_int8.tflite --images data/test/images/
    python src/litetf.py --model best_int8.tflite --images data/test/images/ --runs 400
"""

import argparse
import time
import os
import glob
import numpy as np
import cv2
import psutil


# ─────────────────────────────────────────────
# LOAD TFLITE MODEL
# ─────────────────────────────────────────────

def load_interpreter(model_path):
    """Load TFLite model with XNNPACK delegate for ARM acceleration."""
    try:
        # TFLite Runtime (lightweight — recommended for Raspberry Pi)
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(
            model_path  = model_path,
            num_threads = 4,           # use all 4 Cortex-A76 cores
        )
        print("Using: tflite_runtime")
    except ImportError:
        # Fallback to full TensorFlow
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(
            model_path  = model_path,
            num_threads = 4,
        )
        print("Using: tensorflow")

    interpreter.allocate_tensors()
    return interpreter


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(image_path, imgsz):
    """Load and preprocess a single image for TFLite inference."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0      # normalise to [0, 1]
    img = np.expand_dims(img, axis=0)          # add batch dimension → (1, H, W, 3)
    return img


# ─────────────────────────────────────────────
# BENCHMARK
# ─────────────────────────────────────────────

def benchmark(model_path, images_dir, num_runs, imgsz, warmup_runs=10):

    print("\n" + "=" * 60)
    print("DARCY — Raspberry Pi 5 TFLite Benchmark")
    print("=" * 60)
    print(f"Model      : {model_path}")
    print(f"Images     : {images_dir}")
    print(f"Runs       : {num_runs}")
    print(f"Warmup     : {warmup_runs} frames")
    print(f"Image size : {imgsz}x{imgsz}")

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model size : {model_size_mb:.2f} MB")

    interpreter    = load_interpreter(model_path)
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input shape: {input_details[0]['shape']}")

    # ── Collect test images ──
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))

    if not image_files:
        print(f"\nNo images found in: {images_dir}")
        print("Falling back to random dummy input...")
        use_dummy = True
    else:
        use_dummy = False
        print(f"Test images: {len(image_files)} found")

    # ── Warm-up runs ──
    # Run model on dummy inputs to initialise the TFLite interpreter
    # and XNNPACK delegate before timing begins.
    print(f"\nWarming up ({warmup_runs} runs)...")
    dummy = np.random.rand(1, imgsz, imgsz, 3).astype(np.float32)
    for _ in range(warmup_runs):
        interpreter.set_tensor(input_details[0]['index'], dummy)
        interpreter.invoke()

    # ── Benchmark loop ──
    # Measure ONLY the interpreter.invoke() call — as described in report.
    # This isolates pure inference time from I/O and preprocessing overhead.
    print(f"Running benchmark ({num_runs} inferences)...")

    inference_times = []
    process    = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    for i in range(num_runs):
        if use_dummy:
            inp = np.random.rand(1, imgsz, imgsz, 3).astype(np.float32)
        else:
            img_path = image_files[i % len(image_files)]
            inp = preprocess(img_path, imgsz)
            if inp is None:
                continue

        interpreter.set_tensor(input_details[0]['index'], inp)

        # ── Time ONLY the invoke() call ──
        t_start = time.perf_counter()
        interpreter.invoke()
        t_end = time.perf_counter()

        _ = interpreter.get_tensor(output_details[0]['index'])

        inference_times.append((t_end - t_start) * 1000)   # convert to ms

    mem_after   = process.memory_info().rss / (1024 * 1024)
    cpu_percent = psutil.cpu_percent(interval=1)

    # ── Compute statistics ──
    times   = np.array(inference_times)
    avg_ms  = float(np.mean(times))
    min_ms  = float(np.min(times))
    max_ms  = float(np.max(times))
    std_ms  = float(np.std(times))
    avg_fps = 1000.0 / avg_ms

    # ── Print results ──
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Model size       : {model_size_mb:.2f} MB")
    print(f"  Frames tested    : {len(inference_times)}")
    print(f"  Avg inference    : {avg_ms:.1f} ms")
    print(f"  Min inference    : {min_ms:.1f} ms")
    print(f"  Max inference    : {max_ms:.1f} ms")
    print(f"  Std deviation    : {std_ms:.1f} ms")
    print(f"  FPS              : {avg_fps:.2f}")
    print(f"  CPU usage        : {cpu_percent:.1f}%")
    print(f"  Memory footprint : {mem_after - mem_before:.1f} MB")
    print("=" * 60)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark TFLite INT8 model on Raspberry Pi 5"
    )
    parser.add_argument("--model",  type=str, required=True,
                        help="Path to .tflite model file")
    parser.add_argument("--images", type=str, default="data/test/images",
                        help="Directory of test images (default: data/test/images)")
    parser.add_argument("--runs",   type=int, default=400,
                        help="Number of inference runs (default: 400)")
    parser.add_argument("--imgsz",  type=int, default=640,
                        help="Input image size (default: 640)")
    args = parser.parse_args()

    benchmark(args.model, args.images, args.runs, args.imgsz)
