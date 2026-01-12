# ·············································
# :           Importing required libraries    :
# ·············································
import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# ·············································
# :                   Defining paths          :
# ·············································
DATASET_PATH = "data/American/"
OUTPUT_FILE = "data/landmark_data_right_hand.npz"

# ·············································
# :          Initializing MediaPipe Hands     :
# ·············································
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Initializing containers for landmark data and labels
X, y = [], []

print("Extracting landmarks...")

# Iterating through dataset folders and reading labeled images
for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_path):
        continue

    for img_name in tqdm(os.listdir(label_path), desc=f"{label}"):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Converting image to RGB and detecting hand landmarks
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        # Skipping images without detected hands or handedness
        if not result.multi_hand_landmarks:
            continue
        if not result.multi_handedness:
            continue

        handedness = result.multi_handedness[0].classification[0].label
        hand = result.multi_hand_landmarks[0]

        # Ensuring complete landmark detection
        if len(hand.landmark) != 21:
            continue

        landmarks = []

        # Extracting 3D landmark coordinates
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        # Normalizing landmarks relative to the wrist position
        base_x, base_y, base_z = landmarks[0:3]
        for i in range(0, len(landmarks), 3):
            landmarks[i]   -= base_x
            landmarks[i+1] -= base_y
            landmarks[i+2] -= base_z

        # Converting all samples to right-hand orientation
        if handedness == "Left":
            for i in range(0, len(landmarks), 3):
                landmarks[i] *= -1

        # Applying scale normalization for consistency
        landmarks = np.array(landmarks)
        scale = np.linalg.norm(landmarks[3:6]) + 1e-6
        landmarks = landmarks / scale

        X.append(landmarks)
        y.append(label)


# ·············································
# :     Saving data as compressed NumPy file  :
# ·············································
np.savez_compressed(OUTPUT_FILE, X=np.array(X), y=np.array(y))
print(f"Saved landmark dataset → {OUTPUT_FILE}")
