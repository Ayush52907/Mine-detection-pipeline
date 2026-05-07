# Multi-Stage Mine Detection Pipeline 🛸

An end-to-end vision pipeline designed for autonomous hazard identification in aerial drone surveillance.

This system bridges the gap between hardware constraints and high-fidelity AI, utilizing a two-stage verification approach to maximize mission safety and minimize false triggers in complex terrain.

---

## 🧠 System Architecture

The pipeline processes high-resolution aerial footage through two distinct neural networks:

### 1. Stage 1: Localization (YOLOv8)
* Scans the full-frame input feed for potential hazards.
* Outputs cropped bounding boxes of regions of interest.
* Optimized for high-speed edge inference to keep up with real-time flight data.

### 2. Stage 2: Verification (ResNet50V2)
* Acts as the secondary safety gate.
* Performs granular classification on the YOLO crops to verify if the anomaly is a threat.
* Optimized strictly for **High Recall** to filter out environmental noise (rocks, shadows, dry grass).

---

## ⚙️ Model Optimization & Performance (The "Secret Sauce")

In mine detection, a **False Negative** (missing a mine) is a catastrophic failure. Standard models optimize for overall accuracy, which fails in safety-critical edge cases. 

We engineered the ResNet50V2 model (`resnetv2_training.py`) using the following techniques to force the network to prioritize high recall:

* **Dataset Hardening:** Augmented the dataset with specifically chosen "hard negatives" (visual anomalies that look like mines) to reduce false positives without sacrificing recall.
* **Hardware Simulation (Random Blur):** Implemented a custom `random_drone_blur` function in the `tf.data` pipeline. This randomly downscales and upscales batches to simulate the motion blur and dynamic resolution drops experienced by a live drone camera feed.
* **Algorithmic Safety Nets:**
  * **Focal Loss:** Utilized `BinaryFocalCrossentropy` to heavily penalize the model for missing difficult, hard-to-see mines.
  * **Dynamic Class Weighting:** Calculated memory-safe dynamic class weights, applying an explicit **2.0x Safety Multiplier** to the 'mine' class to bias the network toward caution.
* **Two-Phase Fine-Tuning:** Executed a warmup phase on the dense head, followed by deep fine-tuning of the ResNet backbone.
* **Recall-Driven Callbacks:** Standard training monitors loss or accuracy. We explicitly monitored `val_recall` for both our `EarlyStopping` (patience=5) and `ReduceLROnPlateau` (factor=0.2) callbacks to ensure the model's architecture saved the weights with the highest safety rating.

---

## 🚀 How to Run the Pipeline

### 1. Environment Setup
Clone the repository and set up a virtual environment to ensure dependency isolation:
```bash
git clone [https://github.com/Ayush52907/Mine-Detection-Vision-Pipeline.git](https://github.com/Ayush52907/Mine-Detection-Vision-Pipeline.git)
cd Mine-Detection-Vision-Pipeline

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required libraries
pip install -r requirements.txt

```
### 2. Add Model Weights
*Note: Due to file size constraints, the trained model weights are stored locally.*
 * Place your trained YOLOv8 weights (e.g., detection.pt) directly into the models/ directory.
 * Place your high-recall ResNet classifier weights (e.g., resnet_mine_model.keras) into the models/ directory.
### 3. Execution
Ensure your target aerial imagery is placed in the data/sample_images/ directory, then execute the pipeline:
```bash
python pipeline.py

```
### 4. Results & Auditing
The pipeline utilizes a "clean slate" execution, automatically clearing old data and generating fresh results in the output/ directory:
 * output/yolo_detections/: Full aerial images with localized bounding boxes.
 * output/yolo_crops/: Raw extracted anomalies prior to verification.
 * output/verified_mines/: 🚨 **Critical Threats** - Crops verified as mines by the ResNet threshold.
 * output/verified_safe/: ✅ **Safe Terrain** - Hard negatives (rocks, shadows) filtered out by the secondary classifier.

*Built for the intersection of autonomous agents and physical hardware.* 🤖⚡
