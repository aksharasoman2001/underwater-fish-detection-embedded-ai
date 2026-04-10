<div align="center">

# 🐟 DARCY — Underwater Fish Detection on Embedded AI

**Real-time fish detection using YOLOv11 + TFLite INT8, deployed entirely on a Raspberry Pi 5 — no GPU, no cloud.**

*MSc Internship · Lab-STICC (UMR CNRS 6285) · ENIB Brest, France · March–September 2025*

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-orange?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![TFLite INT8](https://img.shields.io/badge/TFLite-INT8-green?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org/lite)
[![Raspberry Pi 5](https://img.shields.io/badge/Raspberry_Pi_5-4GB-red?style=flat-square&logo=raspberry-pi&logoColor=white)](https://raspberrypi.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

[📄 Internship Report](docs/internship_report.pdf) · [📊 Presentation Slides](docs/internship_presentation.pptx) · [🔗 LinkedIn](https://www.linkedin.com/in/akshara-soman)

</div>

---

## 🎬 Live Demo

> **Drop your demo GIF here!**
> See [How to add your demo](#-how-to-add-your-demo-video) at the bottom of this file.

```
[ YOUR DEMO GIF GOES HERE ]
```

---

## 📌 Overview

This project designs and deploys a real-time **underwater fish detection system** using **embedded AI** — running entirely on a **Raspberry Pi 5** with no GPU or internet connection.

The system was built during a 6-month MSc internship at **Lab-STICC** (CNRS UMR 6285, ENIB Brest), within the **AI & Oceans** division (OSE team), under the EUR ISblue programme. It bridges the gap between state-of-the-art deep learning research and practical field-deployable hardware for **marine monitoring, biodiversity assessment, and pollution detection**.

**Key challenge:** Most underwater AI research runs on powerful GPU workstations. This project proves that real-time fish detection is achievable on a €70 single-board computer.

---

## 🎯 Key Results

| Model | Format | Size | Inference | FPS | mAP@0.5 | mAP@0.5-0.95 |
|---|---|---|---|---|---|---|
| YOLOv11n | INT8 TFLite | **2.63 MB** | 306 ms | **3.26** | 0.980 | 0.864 |
| YOLOv11s | INT8 TFLite | 9.17 MB | 837 ms | 1.19 | 0.979 | 0.865 |
| YOLOv11s + Aug | INT8 TFLite | 9.17 MB | 832 ms | 1.20 | 0.982 | **0.903** |
| Hybrid (n→s) | INT8 TFLite | 9.17 MB | 841 ms | 1.19 | **0.984** | 0.886 |

> ✅ **Best real-time option: YOLOv11n INT8 TFLite — 3.26 FPS at only 2.63 MB with mAP@0.5 = 0.980**

---

## 🖼️ Detection Samples

Sample bounding box outputs from live PiCam testing on Raspberry Pi 5:

> *(Add your detection screenshots here — drag images into `results/detection_samples/`)*

| Clear water | Low light | Turbid water |
|---|---|---|
| `[add image]` | `[add image]` | `[add image]` |

---

## 🧠 Methodology

### Dataset
- **Primary dataset**: Fish Recognition Ground-Truth (FRGT)
- Single-class detection problem (`fish`)
- Manual bounding box annotation with LabelImg (YOLO format)
- Augmentations: HSV shifts, mosaic, MixUp, flips, noise injection

### Two-Phase Training Strategy

```
Phase 1 — Transfer Learning          Phase 2 — Fine-Tuning
────────────────────────────         ──────────────────────────────
• Pretrained YOLOv11 weights         • Load best Phase 1 checkpoint
• Freeze backbone (5 layers)         • Unfreeze all layers
• Train detection head               • Minimal augmentation
• Progressive augmentation           • lr = 0.001
• lr = 0.01, 0–300 epochs            • 100+ additional epochs
```

### Model Optimization
- **Post-Training Quantization**: FP32 → INT8 using TFLite
- **XNNPACK delegate** for ARM Cortex-A76 CPU acceleration
- Model size reduced ~50% with minimal accuracy drop

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Model | YOLOv11n / YOLOv11s (Ultralytics) |
| Training | PyTorch 2.4.1 · NVIDIA A40 GPU (ENIB server) |
| Deployment | TensorFlow Lite INT8 · XNNPACK delegate |
| Hardware | Raspberry Pi 5 (4GB) · PiCam v3 |
| Libraries | OpenCV · NumPy · psutil |
| OS | Raspberry Pi OS 64-bit · Linux 6.6 · Python 3.11 |

---

## ⚙️ Hardware Setup

| Component | Specification |
|---|---|
| SoC | Broadcom BCM2712 |
| CPU | Quad-core ARM Cortex-A76 @ 2.4 GHz |
| RAM | 4 GB LPDDR4X |
| Camera | Raspberry Pi Camera v3 (PiCam) |
| OS | Raspberry Pi OS 64-bit, Linux 6.6 |
| Power | 27W USB-C |

---

## 📁 Project Structure

```
underwater-fish-detection-embedded-ai/
│
├── src/
│   ├── train.py               # YOLOv11 two-phase training pipeline
│   ├── detect_picam.py        # Live detection on Raspberry Pi + PiCam
│   ├── convert_tflite.py      # Export YOLOv11 → TFLite INT8
│   └── benchmark_rpi.py       # FPS, CPU, memory benchmarking
│
├── results/
│   ├── detection_samples/     # Bounding box outputs from PiCam
│   └── metrics/               # Loss curves, precision/recall plots
│
├── docs/
│   ├── internship_report.pdf  # Full MSc internship report
│   └── internship_presentation.pptx
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/underwater-fish-detection-embedded-ai.git
cd underwater-fish-detection-embedded-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (on GPU)

```bash
python src/train.py
```

### 4. Convert to TFLite INT8

```bash
python src/convert_tflite.py --weights best.pt --imgsz 640
```

### 5. Run live detection on Raspberry Pi 5

```bash
python src/detect_picam.py --model best_int8.tflite
```

---

## 📊 Training Curves

> *(Add your loss curves and precision/recall plots here — drag images into `results/metrics/`)*

---

## 🔭 Future Work

- [ ] Apply **pruning + quantization-aware training (QAT)** for further compression
- [ ] Knowledge distillation: Hybrid → Nano for better accuracy at same size
- [ ] Test on **Google Coral Edge TPU** and **NVIDIA Jetson Nano**
- [ ] Multimodal fusion: vision + underwater acoustics
- [ ] Real underwater field deployment (variable lighting, turbidity)

---

## 🏛️ Context

This work was conducted at:
- **Lab-STICC** (UMR CNRS 6285) — AI & Oceans Division, OSE Team, ENIB Brest
- Supported by **EUR ISblue** (Interdisciplinary Graduate School for the Blue Planet)
- Presented at the **ROGER** research day (Drones and Underwater Image Processing), Brest 2025

**Academic supervisors:** A. Benzinou, K. Nasreddine, H.N. Tran (Lab-STICC) · R. Khemmar (ESIGELEC)

---

## 👩‍💻 Author

**Akshara Soman**  
MSc Automotive Embedded Systems — ESIGELEC  
[LinkedIn](https://www.linkedin.com/in/akshara-soman)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
