import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# Paths
DATASET_PATH = "data/American/"
OUTPUT_FILE = "data/landmark_data_right_hand.npz"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Lists to store landmarks and labels
X, y = [], []

print("Extracting landmarks...")

# Loop through dataset folders
for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_path):
        continue

    for img_name in tqdm(os.listdir(label_path), desc=f"{label}"):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Convert image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if not result.multi_hand_landmarks:
            continue

        # Make sure handedness info exists
        if not result.multi_handedness:
            continue

        handedness = result.multi_handedness[0].classification[0].label
        hand = result.multi_hand_landmarks[0]

        # Safety check
        if len(hand.landmark) != 21:
            continue

        landmarks = []

        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        # Normalize to wrist
        base_x, base_y, base_z = landmarks[0:3]
        for i in range(0, len(landmarks), 3):
            landmarks[i]   -= base_x
            landmarks[i+1] -= base_y
            landmarks[i+2] -= base_z

        # 👉 FORCE RIGHT HAND
        # If detected hand is LEFT, flip x-axis
        if handedness == "Left":
            for i in range(0, len(landmarks), 3):
                landmarks[i] *= -1

        # Scale normalization
        landmarks = np.array(landmarks)
        scale = np.linalg.norm(landmarks[3:6]) + 1e-6
        landmarks = landmarks / scale

        X.append(landmarks)
        y.append(label)

# Save as compressed NumPy file
np.savez_compressed(OUTPUT_FILE, X=np.array(X), y=np.array(y))
print(f"Saved landmark dataset → {OUTPUT_FILE}")
