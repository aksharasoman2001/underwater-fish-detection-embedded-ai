# 🐟 Underwater Fish Detection — Embedded AI on Raspberry Pi 5

> Real-time underwater event detection using **YOLOv11 + TFLite INT8**, deployed on a **Raspberry Pi 5**.  
> MSc Internship Project — Lab-STICC (UMR CNRS 6285), ENIB Brest, France · March–September 2025

---

## 📌 Overview

This project designs and deploys a real-time **underwater fish detection system** based on embedded AI. The system runs entirely on a **Raspberry Pi 5** (no GPU), making it suitable for integration into underwater drones (AUVs/ROVs) for marine monitoring, biodiversity assessment, and pollution detection.

The work was carried out as part of an MSc internship at **Lab-STICC**, within the **AI & Oceans** division (OSE team), under the EUR ISblue programme.

---

## 🎯 Key Results

| Model | Format | Size | Inference | FPS | mAP@0.5 | mAP@0.5-0.95 |
|---|---|---|---|---|---|---|
| YOLOv11n | INT8 TFLite | **2.63 MB** | 306 ms | **3.26** | 0.980 | 0.864 |
| YOLOv11s | INT8 TFLite | 9.17 MB | 837 ms | 1.19 | 0.979 | 0.865 |
| YOLOv11s + Aug | INT8 TFLite | 9.17 MB | 832 ms | 1.20 | 0.982 | **0.903** |
| Hybrid (n→s) | INT8 TFLite | 9.17 MB | 841 ms | 1.19 | **0.984** | 0.886 |

**✅ Best real-time option: YOLOv11n INT8 TFLite — 3.26 FPS at 2.63 MB with mAP@0.5 = 0.980**

---

## 🛠️ Tech Stack

- **Model**: YOLOv11n / YOLOv11s (Ultralytics)
- **Training**: PyTorch 2.4.1 · NVIDIA A40 GPU
- **Deployment**: TensorFlow Lite (INT8 quantization) · XNNPACK delegate
- **Hardware**: Raspberry Pi 5 (4GB) · PiCam v3
- **Libraries**: OpenCV · NumPy · psutil
- **OS**: Raspberry Pi OS 64-bit · Linux kernel 6.6 · Python 3.11

---

## 📁 Project Structure

```
underwater-fish-detection-embedded-ai/
│
├── src/
│   ├── train.py               # YOLOv11 training pipeline (two-phase)
│   ├── detect_picam.py        # Live detection on Raspberry Pi + PiCam
│   ├── convert_tflite.py      # Export YOLOv11 → TFLite INT8
│   └── benchmark_rpi.py       # Benchmarking script (FPS, CPU, memory)
│
├── results/
│   ├── detection_samples/     # Sample bounding box outputs from PiCam
│   └── metrics/               # Loss curves, precision/recall plots
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

## 🧠 Methodology

### Dataset
- **Primary dataset**: Fish Recognition Ground-Truth (FRGT)
- Single-class detection problem (`fish`)
- Manual bounding box annotation (YOLO format)
- Augmentations: HSV shifts, mosaic, MixUp, flips, noise injection

### Training Strategy (Two-Phase)
| Phase | Details |
|---|---|
| Phase 1 | Transfer learning — frozen backbone (5 layers), lr=0.01, 0–300 epochs, progressive augmentation |
| Phase 2 | Fine-tuning — all layers unfrozen, lr=0.001, 100+ epochs, minimal augmentation |

### Model Optimization
- **Post-Training Quantization**: FP32 → INT8 using TFLite
- **XNNPACK delegate** enabled for CPU acceleration on ARM Cortex-A76
- Model size reduced by ~50% with minimal accuracy loss

---

## 📊 Detection Samples

Sample outputs from live PiCam testing on Raspberry Pi 5:

> *(Add your bounding box screenshots here — drag and drop images into the `results/detection_samples/` folder)*

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

## 🔭 Future Work

- [ ] Apply **pruning + quantization-aware training (QAT)** for further compression
- [ ] Knowledge distillation: Hybrid → Nano
- [ ] Test on **Google Coral Edge TPU** and **NVIDIA Jetson Nano**
- [ ] Multimodal fusion: vision + underwater acoustics
- [ ] Real underwater field deployment (variable lighting, turbidity)

---

## 🏛️ Context

This work was conducted at:
- **Lab-STICC** (UMR CNRS 6285) — AI & Oceans Division, OSE Team
- **ENIB** — École Nationale d'Ingénieurs de Brest, France
- Supported by **EUR ISblue** (Interdisciplinary Graduate School for the Blue Planet)
- Academic supervisors: A. Benzinou, K. Nasreddine, H.N. Tran (Lab-STICC) · R. Khemmar (ESIGELEC)

---

## 👩‍💻 Author

**Akshara Soman**  
MSc Automotive Embedded Systems — ESIGELEC  


---

## 📄 License

This project is licensed under the MIT License.
