import os
import cv2
import numpy as np
from tqdm import tqdm

DATASET_PATH = "data/American/"
IMG_SIZE = 64
MAX_IMAGES_PER_CLASS = 200
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")
OUTPUT_FILE = "data/preprocessed_data.npz"

images = []
labels = []

print("Loading and preprocessing dataset...")

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_path):
        continue

    image_files = os.listdir(label_path)
    count = 0
    for img_file in tqdm(image_files, desc=f"Processing {label}", leave=False):
        if count >= MAX_IMAGES_PER_CLASS:
            break
        if not img_file.lower().endswith(VALID_EXTENSIONS):
            continue

        img_path = os.path.join(label_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        images.append(img_gray)
        labels.append(label)
        count += 1

images = np.array(images, dtype=np.float32) / 255.0  # normalize
labels = np.array(labels)

# Flatten images for KNN
X = images.reshape(len(images), -1)
y = labels

# Save preprocessed data
np.savez_compressed(OUTPUT_FILE, X=X, y=y)
print(f"Preprocessed data saved to {OUTPUT_FILE}")
