# Multi-Stage Mine Detection Pipeline 🛸

An end-to-end vision pipeline designed for autonomous hazard identification in aerial drone surveillance. This system bridges the gap between hardware constraints and high-fidelity AI, utilizing a **Two-Stage Verification** approach to maximize mission safety.

<img width="1280" height="720" alt="det_stereo_0002_left1" src="https://github.com/user-attachments/assets/03d6c219-3448-44e0-bd50-c4e32d1d5e3e" />


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

## ⚙️ Model Optimization & Performance (The "Secret Sauce")

In mine detection, a **False Negative** (missing a mine) is a catastrophic failure. Standard models optimize for overall accuracy, which fails in safety-critical edge cases with heavy class imbalance. We engineered the ResNet50V2 model using a custom training loop (`training/train.py`) designed to force the network to prioritize high recall.

### 1. Hardware-Aware Data Augmentation
* **Dynamic Drone Blur:** Implemented a custom `random_drone_blur` function within the `tf.data` pipeline. This randomly downscales and upscales batches in real-time, simulating the motion blur and dynamic resolution drops experienced by a live drone camera feed over varied terrain.
* **Geospatial Augmentation:** Applied heavy random rotations, translations, and contrast shifts to mimic varying flight altitudes and sun angles.

### 2. Combating Class Imbalance
* **Binary Focal Crossentropy:** Replaced standard categorical loss with Focal Loss ($\gamma=2.0$) to heavily penalize the model for missing "hard negatives" (e.g., rocks or shadows that closely resemble mines) rather than easily classified safe terrain.
* **Programmatic Safety Weighting:** Calculated memory-safe dynamic class weights across the dataset, applying an explicit **2.0x Safety Multiplier** to the hazard class. This biases the gradient updates to favor caution.

### 3. Architectural Preservation (Two-Phase Training)
* **Phase 1 (Warmup):** Froze the ResNet backbone to train the newly initialized dense head with a high learning rate ($1e-3$). This prevents "gradient shock" from destroying the pre-trained ImageNet feature extractors.
* **Phase 2 (Deep Fine-Tuning):** Unfroze the top 50 layers of the backbone with a reduced learning rate ($1e-5$) to allow the model to learn terrain-specific geometric features.

### 4. Precision-Recall Threshold Calibration
* Instead of relying on a standard 0.5 confidence threshold, the script actively evaluates the validation set post-training. By mapping precision, recall, and F1-scores across discrete intervals, we identified that dropping the deployment threshold to **$\le$ 0.20** yields the maximum safety-critical recall required for mission deployment.

---

## 📊 Evaluation & Metrics

### Detection Performance
The model achieves a **0.9941 Recall** on the 'mine' class, ensuring that nearly all potential hazards are successfully localized and passed to the verification stage.

<img width="551" height="209" alt="image" src="https://github.com/user-attachments/assets/75bcc047-b972-4747-a92d-cad95d69daaa" />

### Confusion Matrix & Error Analysis
The secondary classifier effectively filters out "Safe" anomalies (rocks, shadows) while maintaining a near-zero false-negative rate in the "Mine" category.

<img width="705" height="636" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/0368f55f-903c-4e65-bc17-0e8dbc9b8bf3" />

### Training Dynamics (Phase 1 vs Phase 2)
The graphs below illustrate the shift from **Phase 1 (Warmup)** to **Phase 2 (Deep Fine-Tuning)**. Note the spike in validation recall and precision once the top 50 layers of the ResNet backbone were unfrozen, allowing the model to adapt to specific terrain geometry.

<img width="1455" height="451" alt="Graph" src="https://github.com/user-attachments/assets/c7d81e81-4535-41ee-90c7-268bb4e88202" />

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
