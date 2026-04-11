"""
convert_tflite.py — Export YOLOv11 to TFLite INT8
DARCY: Design and Development of an Underwater Event Detection System
       Based on Embedded AI

Author  : Akshara Soman
Lab     : Lab-STICC (UMR CNRS 6285), ENIB Brest, France
Program : MSc Automotive Embedded Systems, ESIGELEC
Date    : 2025

Description:
    Converts a trained YOLOv11 PyTorch model (.pt) to a quantized
    TensorFlow Lite INT8 model (.tflite) for deployment on Raspberry Pi 5.

    Post-Training Quantization (INT8) reduces model size by ~50%
    and improves inference speed with minimal accuracy loss.

    Results from report:
        YOLOv11n: 5.21 MB (.pt)  →  2.63 MB (.tflite INT8)
        YOLOv11s: 18.28 MB (.pt) →  9.17 MB (.tflite INT8)

Usage:
    python src/convert_tflite.py --weights best.pt --imgsz 640
"""

import argparse
import os
from ultralytics import YOLO


def convert(weights, imgsz):
    print("\n" + "=" * 60)
    print("YOLOv11 → TFLite INT8 Conversion")
    print("=" * 60)
    print(f"Input weights : {weights}")
    print(f"Image size    : {imgsz}x{imgsz}")
    print(f"Quantization  : INT8 (post-training quantization)")

    if not os.path.exists(weights):
        raise FileNotFoundError(f"Weights file not found: {weights}")

    # Load trained YOLOv11 model
    model = YOLO(weights)

    # Export to TFLite INT8
    # batch=1        → single image inference (required for embedded)
    # dynamic=False  → static input shape for TFLite compatibility
    # int8=True      → full integer quantization (FP32 → INT8)
    # A representative dataset is used automatically by Ultralytics
    # to calibrate the INT8 quantization thresholds.
    model.export(
        format  = "tflite",
        batch   = 1,
        dynamic = False,
        int8    = True,
        imgsz   = imgsz,
    )

    # The exported file is saved in the same directory as the .pt file
    tflite_path = weights.replace(".pt", "_saved_model/best_int8.tflite")

    print(f"\nConversion complete!")
    print(f"TFLite model saved at: {tflite_path}")

    # Print size comparison
    if os.path.exists(tflite_path):
        pt_size     = os.path.getsize(weights) / (1024 * 1024)
        tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
        reduction   = (1 - tflite_size / pt_size) * 100

        print(f"\nModel size comparison:")
        print(f"  PyTorch (.pt)  : {pt_size:.2f} MB")
        print(f"  TFLite INT8    : {tflite_size:.2f} MB")
        print(f"  Size reduction : {reduction:.1f}%")

    print(f"\nNext step: copy {tflite_path} to Raspberry Pi 5")
    print(f"Then run : python src/detect_picam.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert trained YOLOv11 model to TFLite INT8 for Raspberry Pi 5"
    )
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained .pt weights (e.g. best.pt)")
    parser.add_argument("--imgsz",   type=int, default=640,
                        help="Input image size (default: 640)")
    args = parser.parse_args()

    convert(args.weights, args.imgsz)
