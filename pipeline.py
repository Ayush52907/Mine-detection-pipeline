import tensorflow as tf
from ultralytics import YOLO
import cv2
import numpy as np
import os
import shutil

# ==========================================
# 1. PATHS & CONFIGURATION (GitHub Portable)
# ==========================================
# This gets the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# YOLO Paths
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'detection.pt')  # Replace detection.pt with your model's filename
INPUT_IMAGES_FOLDER = os.path.join(BASE_DIR, 'data', 'input_images')

# Output Paths (Created inside an /output folder)
YOLO_BBOX_FOLDER = os.path.join(BASE_DIR, 'output', 'yolo_detections')
YOLO_CROPS_FOLDER = os.path.join(BASE_DIR, 'output', 'yolo_crops')

# Classification Paths
CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'resnet_mine_model.keras') # Replace resnet_mine_model.keras with your model's filename
FINAL_MINE_FOLDER = os.path.join(BASE_DIR, 'output', 'verified_mines')
FINAL_SAFE_FOLDER = os.path.join(BASE_DIR, 'output', 'verified_safe')

# Thresholds
YOLO_CONFIDENCE = 0.2
CLASSIFIER_THRESHOLD = 0.5
# ==========================================

def setup_folders():
    """Clears existing results and recreates folders for a fresh run."""
    folders = [YOLO_BBOX_FOLDER, YOLO_CROPS_FOLDER, FINAL_MINE_FOLDER, FINAL_SAFE_FOLDER]
    
    print("--- PRE-RUN CLEANUP ---")
    for folder in folders:
        if os.path.exists(folder):
            try:
                print(f"Clearing: {os.path.basename(folder)}...")
                shutil.rmtree(folder)  # Deletes the folder and all its contents
            except Exception as e:
                print(f"Warning: Could not fully clear {folder}. Error: {e}")
        
        # Recreate the folder fresh
        os.makedirs(folder, exist_ok=True)
    print("Cleanup complete. Starting models with a clean slate.\n")

def run_yolo_stage():
    """Runs YOLOv8 to find objects and save cropped images."""
    print("\n--- STAGE 1: YOLOv8 OBJECT DETECTION ---")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = sorted([f for f in os.listdir(INPUT_IMAGES_FOLDER) if f.lower().endswith(valid_extensions)])
    
    crop_count = 0
    
    for filename in files:
        img_path = os.path.join(INPUT_IMAGES_FOLDER, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Run YOLO Inference
        results = yolo_model(img, conf=YOLO_CONFIDENCE, verbose=False)
        
        # Save the full image with bounding boxes drawn on it
        annotated_img = results[0].plot()
        cv2.imwrite(os.path.join(YOLO_BBOX_FOLDER, f"det_{filename}"), annotated_img)
        
        # Loop through every bounding box found in the image
        boxes = results[0].boxes
        for i, box in enumerate(boxes):
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop the image using Numpy slicing
            crop = img[y1:y2, x1:x2]
            
            # Save the crop
            crop_filename = f"crop_{i}_{filename}"
            cv2.imwrite(os.path.join(YOLO_CROPS_FOLDER, crop_filename), crop)
            crop_count += 1
            
    print(f"YOLO Stage Complete. Found and cropped {crop_count} potential targets.")
    return crop_count

def run_classifier_stage():
    """Runs ResNet/DenseNet to verify the YOLO crops."""
    print("\n--- STAGE 2: SECONDARY CLASSIFICATION ---")
    classifier_model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)
    
    files = sorted([f for f in os.listdir(YOLO_CROPS_FOLDER)])
    if not files:
        print("No crops found for classification.")
        return

    verified_mines = 0

    print(f"{'Crop Filename':<40} | {'Prediction':<15} | {'Confidence'}")
    print("-" * 85)

    for filename in files:
        img_path = os.path.join(YOLO_CROPS_FOLDER, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Preprocess for Keras
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (224, 224))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = classifier_model.predict(img_array, verbose=0)[0][0]
        
        # Sort based on Safety Threshold
        if prediction >= CLASSIFIER_THRESHOLD:
            confidence = prediction * 100
            verified_mines += 1
            color_tag = "!!! MINE !!!"
            shutil.copy(img_path, os.path.join(FINAL_MINE_FOLDER, filename))
        else:
            confidence = (1 - prediction) * 100
            color_tag = "Safe"
            shutil.copy(img_path, os.path.join(FINAL_SAFE_FOLDER, filename))
            
        print(f"{filename[:39]:<40} | {color_tag:<15} | {confidence:.2f}%")

    print("-" * 85)
    print(f"\n--- FINAL SYSTEM PIPELINE SUMMARY ---")
    print(f"Total Initial Detections (YOLO):  {len(files)}")
    print(f"Verified True Mines (Classifier): {verified_mines}")
    print(f"False Alarms Filtered Out:        {len(files) - verified_mines}")

if __name__ == "__main__":
    setup_folders()
    crops_generated = run_yolo_stage()
    
    if crops_generated > 0:
        run_classifier_stage()
