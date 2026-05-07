import os
import cv2

# The root folders
BASE_YOLO_DIR = '/content/Mines'
BASE_OUTPUT_DIR = '/content/Cropped_Dataset'

# Ensure 0='mine' and 1='not-mine' matches your data.yaml
CLASSES = {0: 'mine', 1: 'not-mine'}

def universal_cropper():
    # Loop through 'train', 'valid', and 'test'
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(BASE_YOLO_DIR, split, 'images')
        lbl_dir = os.path.join(BASE_YOLO_DIR, split, 'labels')

        # Skip if the folder doesn't exist (e.g., if 'test' is missing)
        if not os.path.exists(lbl_dir): continue

        print(f"Processing {split} set...")

        # Create output directories
        for class_name in CLASSES.values():
            os.makedirs(os.path.join(BASE_OUTPUT_DIR, split, class_name), exist_ok=True)

        for label_file in os.listdir(lbl_dir):
            if not label_file.endswith('.txt'): continue

            img_file = label_file.replace('.txt', '.jpg')
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path)
            if img is None: continue

            h, w, _ = img.shape

            with open(os.path.join(lbl_dir, label_file), 'r') as f:
                for i, line in enumerate(f.readlines()):
                    parts = line.strip().split()
                    if len(parts) < 5: continue

                    class_id = int(parts[0])
                    x_center, y_center, box_w, box_h = map(float, parts[1:5])

                    x1 = int((x_center - box_w/2) * w)
                    y1 = int((y_center - box_h/2) * h)
                    x2 = int((x_center + box_w/2) * w)
                    y2 = int((y_center + box_h/2) * h)

                    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    if crop.size == 0: continue

                    crop_resized = cv2.resize(crop, (224, 224))
                    save_path = os.path.join(BASE_OUTPUT_DIR, split, CLASSES[class_id], f"{label_file[:-4]}_c{i}.jpg")
                    cv2.imwrite(save_path, crop_resized)

    print("\nAll sets (train/valid/test) are now cropped and organized!")

universal_cropper()
