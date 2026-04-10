"""
train.py — Two-Phase YOLOv11 Training Pipeline
DARCY: Design and Development of an Underwater Event Detection System
       Based on Embedded AI

Author  : Akshara Soman
Lab     : Lab-STICC (UMR CNRS 6285), ENIB Brest, France
Program : MSc Automotive Embedded Systems, ESIGELEC
Date    : 2025

Description:
    Trains YOLOv11n on the Fish Recognition Ground-Truth (FRGT) dataset
    using a two-phase strategy:
      Phase 1 — Transfer learning with frozen backbone + progressive augmentation
      Phase 2 — Full fine-tuning with minimal augmentation

    This approach was designed specifically for underwater imagery challenges:
    low visibility, colour distortion, turbidity, and limited dataset size.

Usage:
    python src/train.py
    python src/train.py --model yolo11s.pt --device 0
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="DARCY — YOLOv11 Training Pipeline")
    parser.add_argument("--data",    type=str, default="/home/aksharams/DARCY/fish/fish_dataset.yaml",
                        help="Path to dataset .yaml file")
    parser.add_argument("--model",   type=str, default="yolo11n.pt",
                        help="Pretrained weights to start from (default: yolo11n.pt)")
    parser.add_argument("--device",  type=int, default=0,
                        help="GPU device ID (default: 0)")
    parser.add_argument("--imgsz",   type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--batch",   type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--project", type=str, default="fish_training",
                        help="Project folder for saving results")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Transfer Learning with Progressive Augmentation
# ─────────────────────────────────────────────────────────────────────────────
#
# Strategy:
#   - Start from ImageNet-pretrained YOLOv11n weights
#   - Freeze the first 5 backbone layers so low-level visual features
#     (edges, textures) are preserved while the detection head adapts
#     to underwater fish
#   - Apply a 3-stage progressive augmentation schedule:
#       Stage 1 (epochs   0–100): strong augmentation — forces the model
#                                  to learn under worst-case underwater conditions
#       Stage 2 (epochs 101–200): medium augmentation — stabilises training
#       Stage 3 (epochs 201–300): light augmentation  — refines predictions
#
# Rationale (from report):
#   Underwater images suffer from colour shifts (blue/green dominance),
#   turbidity, motion blur, and occlusion. Heavy augmentation in early epochs
#   mimics these real-world degradations, following the curriculum learning
#   principle: train hard first, then simplify.

def phase1(args):
    print("\n" + "=" * 65)
    print("  PHASE 1 — Transfer Learning + Progressive Augmentation")
    print("=" * 65)

    model = YOLO(args.model)

    # Progressive augmentation schedule
    # Each tuple: (stage_name, epochs, hsv_s, hsv_v, mosaic, mixup)
    stages = [
        ("Stage 1 — Strong  (epochs   0–100)", 100, 0.7,  0.4,  1.0, 0.1),
        ("Stage 2 — Medium  (epochs 101–200)", 200, 0.5,  0.3,  0.5, 0.05),
        ("Stage 3 — Light   (epochs 201–300)", 300, 0.25, 0.15, 0.1, 0.0),
    ]

    for stage_name, epoch_end, hsv_s, hsv_v, mosaic, mixup in stages:
        print(f"\n  {stage_name}")
        print(f"  hsv_s={hsv_s} | hsv_v={hsv_v} | mosaic={mosaic} | mixup={mixup}")

        model.train(
            data             = args.data,
            epochs           = epoch_end,
            imgsz            = args.imgsz,
            batch            = args.batch,
            device           = args.device,
            project          = args.project,
            name             = "phase1_transfer",
            exist_ok         = True,       # continue into same folder across stages

            # Frozen backbone — preserves generic visual features
            freeze           = 5,

            # Optimizer
            optimizer        = "SGD",
            lr0              = 0.01,       # initial learning rate
            lrf              = 0.01,       # final lr = lr0 * lrf
            momentum         = 0.937,
            weight_decay     = 0.0005,
            warmup_epochs    = 3.0,
            warmup_momentum  = 0.8,

            # Early stopping — prevent wasted compute on stalled models
            patience         = 50,

            # Progressive augmentation values for this stage
            hsv_h            = 0.015,      # hue — colour distortion underwater
            hsv_s            = hsv_s,      # saturation — turbidity simulation
            hsv_v            = hsv_v,      # brightness — low light simulation
            degrees          = 10.0,       # rotation
            translate        = 0.1,        # translation
            scale            = 0.5,        # scaling
            shear            = 2.0,        # shear distortion
            fliplr           = 0.5,        # horizontal flip (fish swim both ways)
            flipud           = 0.0,
            mosaic           = mosaic,     # mosaic — combines 4 images
            mixup            = mixup,      # MixUp — blends two images

            # Hardware
            workers          = 8,
            cache            = True,       # cache images in RAM for speed
            amp              = True,       # mixed precision (FP16)

            save             = True,
            save_period      = 50,
            plots            = True,
            verbose          = True,
        )

    best_p1 = Path(args.project) / "phase1_transfer" / "weights" / "best.pt"
    print(f"\n  Phase 1 complete.")
    print(f"  Best weights → {best_p1}")
    return str(best_p1)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Full Fine-Tuning
# ─────────────────────────────────────────────────────────────────────────────
#
# Strategy:
#   - Load best checkpoint from Phase 1
#   - Unfreeze ALL layers for end-to-end optimisation
#   - Lower learning rate (0.001 → 0.0001) for careful, stable convergence
#   - Minimal augmentation — let the model sharpen its predictions
#     without heavy distortions obscuring fine details
#
# Rationale (from report):
#   After Phase 1 has adapted the backbone to underwater fish features,
#   Phase 2 allows every layer to refine jointly. The reduced augmentation
#   and learning rate prevent overshooting the optimum found in Phase 1.

def phase2(args, phase1_weights):
    print("\n" + "=" * 65)
    print("  PHASE 2 — Full Fine-Tuning (All Layers Unfrozen)")
    print("=" * 65)
    print(f"  Loading Phase 1 weights: {phase1_weights}")

    model = YOLO(phase1_weights)

    model.train(
        data             = args.data,
        epochs           = 100,
        imgsz            = args.imgsz,
        batch            = args.batch,
        device           = args.device,
        project          = args.project,
        name             = "phase2_finetune",

        # All layers unfrozen
        freeze           = 0,

        # Optimizer — lower lr for precise refinement
        optimizer        = "SGD",
        lr0              = 0.001,
        lrf              = 0.01,
        momentum         = 0.937,
        weight_decay     = 0.0005,
        warmup_epochs    = 1.0,

        patience         = 50,

        # Minimal augmentation
        hsv_h            = 0.005,
        hsv_s            = 0.2,
        hsv_v            = 0.1,
        degrees          = 0.0,
        translate        = 0.05,
        scale            = 0.2,
        shear            = 0.0,
        fliplr           = 0.5,
        flipud           = 0.0,
        mosaic           = 0.0,
        mixup            = 0.0,

        workers          = 8,
        cache            = True,
        amp              = True,

        save             = True,
        save_period      = 20,
        plots            = True,
        verbose          = True,
    )

    best_p2 = Path(args.project) / "phase2_finetune" / "weights" / "best.pt"
    print(f"\n  Phase 2 complete.")
    print(f"  Final model → {best_p2}")
    return str(best_p2)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    args = get_args()

    print("\n" + "=" * 65)
    print("  DARCY — Underwater Fish Detection | Training Pipeline")
    print("=" * 65)
    print(f"  Base model  : {args.model}")
    print(f"  Dataset     : {args.data}")
    print(f"  Device      : GPU {args.device}")
    print(f"  Image size  : {args.imgsz}x{args.imgsz}")
    print(f"  Batch size  : {args.batch}")
    print(f"  Output dir  : {args.project}/")

    # Validate dataset file exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Dataset file not found: {args.data}\n"
            f"Please update --data to point to your fish_dataset.yaml"
        )

    # Run both phases
    phase1_weights = phase1(args)
    final_weights  = phase2(args, phase1_weights)

    print("\n" + "=" * 65)
    print("  Training complete!")
    print(f"  Final model : {final_weights}")
    print(f"  Next step   : python src/convert_tflite.py --weights {final_weights}")
    print("=" * 65)
