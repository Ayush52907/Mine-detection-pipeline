# Multi-Stage Mine Detection Pipeline 🛸

An end-to-end vision pipeline designed for autonomous hazard identification in aerial drone surveillance. This system bridges the gap between hardware constraints and high-fidelity AI, utilizing a **Two-Stage Verification** approach to maximize mission safety.

---

## 🏗️ Directory Structure

```text
├── data/
│   └── sample_images/  # Place target aerial footage here
├── models/             # Store .pt (YOLO) and .keras (ResNet) weights here
├── output/             
│   ├── verified_mines/ # Final confirmed threats
│   ├── verified_safe/  # Filtered "hard negatives" (rocks/shadows)
│   ├── yolo_crops/     # Raw localized detections
│   └── yolo_detections/# Annotated full-frame images
├── pipeline.py         # Main inference orchestration script
├── resnetv2_training.py# Model optimization and training script
└── requirements.txt    # Project dependencies
```

---

## 🧠 System Architecture

The pipeline processes high-resolution footage through two specialized neural networks:

1. **Stage 1: Localization (YOLOv8)**
   * Rapidly scans the environment to identify regions of interest.
   * Optimized for high-speed inference to maintain real-time flight telemetry.
   * Generates bounding box crops for deep analysis.

2. **Stage 2: Verification (ResNet50V2)**
   * Acts as the secondary "Safety Gate."
   * Specifically tuned for **High Recall** to ensure zero missed threats.
   * Uses deep feature extraction to differentiate between actual mines and environmental "noise" like complex shadows or dry vegetation.

---

## ⚙️ Model Optimization (High-Recall Strategy)

Standard classification models optimize for overall accuracy. In mine-clearance, the cost of a **False Negative** (missing a mine) is catastrophic. Our training logic (`resnetv2_training.py`) implements several safety-critical engineering choices:

* **Hardware Simulation (Drone Blur):** A custom `random_drone_blur` augmentation simulates the motion blur and resolution drops typical of low-altitude drone feeds.
* **Algorithmic Bias (Focal Loss):** We utilized **Binary Focal Crossentropy** ($\gamma=2.0$) to force the model to learn from "hard" samples that standard loss functions might ignore.
* **2.0x Safety Weighting:** We applied dynamic class weights with an explicit **2x multiplier** for the 'mine' class, biasing the gradient updates toward caution.
* **Two-Phase Fine-Tuning:**
   * **Phase 1:** Warmup of the dense head to protect pre-trained ImageNet features.
   * **Phase 2:** Deep fine-tuning of the top 50 layers of the ResNet backbone.
* **Threshold Calibration:** The deployment threshold is strictly tuned to **$\le$ 0.20** based on validation F1-score mapping, ensuring maximum sensitivity.

---

## 🚀 Getting Started

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Ayush52907/Mine-Detection-Vision-Pipeline.git
cd Mine-Detection-Vision-Pipeline

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Models
Place your `detection.pt` and `resnet_mine_model.keras` files into the `/models` directory.

### 3. Run Inference
```bash
python pipeline.py
```
The script will automatically clear previous runs and sort fresh results into the `/output` subdirectories for auditing.

---
*"Building autonomous systems that understand the physical world."* 🤖⚡

---
