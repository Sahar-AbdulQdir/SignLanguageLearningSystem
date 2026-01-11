# Import necessary libraries
import os
import json
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

# paths
EXISTING_DATA = "data/landmark_data_right_hand.npz"
CUSTOM_WORDS_DIR = "data/custom_words/"
COMBINED_OUTPUT = "data/combined_landmark_data.npz"
MODEL_OUTPUT = "models/knn_combined_model2.pkl"

# load existing dataset
print("Loading existing dataset...")
existing_data = np.load(EXISTING_DATA)
X_existing = existing_data["X"]
y_existing = existing_data["y"]
print(f"Existing data: {X_existing.shape[0]} samples, {len(np.unique(y_existing))} classes")

# load custom words data
X_custom, y_custom = [], []
print("\nLoading custom words data...")

# iterate through custom words directory
if os.path.exists(CUSTOM_WORDS_DIR):
    # load each custom word's landmark data
    for word_dir in os.listdir(CUSTOM_WORDS_DIR):
        word_path = os.path.join(CUSTOM_WORDS_DIR, word_dir)
        if os.path.isdir(word_path):
            for npz_file in os.listdir(word_path):
                if npz_file.endswith("_landmarks.npz"):
                    data = np.load(os.path.join(word_path, npz_file))
                    X_custom.append(data["X"])
                    y_custom.append(data["y"])

# combine custom words data
if X_custom:
    X_custom = np.vstack(X_custom)
    y_custom = np.concatenate(y_custom)
    print(f"Custom words data: {X_custom.shape[0]} samples, {len(np.unique(y_custom))} classes")
    X = np.vstack([X_existing, X_custom])
    y = np.concatenate([y_existing, y_custom])
else:
    print("No custom words data found.")
    X, y = X_existing, y_existing

# normalize landmarks
print("\nNormalizing landmarks...")
X_norm = np.copy(X)

for i in range(X.shape[0]):
    landmarks = X[i].reshape(-1, 3)
    wrist = landmarks[0]
    landmarks -= wrist
    # scale normalization
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    landmarks /= max_dist
    X_norm[i] = landmarks.flatten()

X = X_norm

# data augmentation for rare classes
print("\nAugmenting rare classes...")
min_samples = 50
unique_classes, counts = np.unique(y, return_counts=True)

X_aug = [x for x in X]
y_aug = [label for label in y]

# augment classes with fewer than min_samples 
for cls, count in zip(unique_classes, counts):
    if count < min_samples:
        idxs = np.where(y == cls)[0]
        needed = min_samples - count
        # simple augmentation: add Gaussian noise
        for _ in range(needed):
            sample = X[np.random.choice(idxs)]
            noise = np.random.normal(0, 0.02, size=sample.shape)
            X_aug.append(sample + noise)
            y_aug.append(cls)

X = np.array(X_aug)
y = np.array(y_aug)

# save combined dataset
np.savez_compressed(COMBINED_OUTPUT, X=X, y=y)
print(f"\nCombined dataset saved: {COMBINED_OUTPUT}")
print(f"Total samples after augmentation: {X.shape[0]}")
print(f"Total classes: {len(np.unique(y))}")

# training model
print("\nTraining new model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

model = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    metric="euclidean"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# save model
os.makedirs("models", exist_ok=True)
with open(MODEL_OUTPUT, "wb") as f:
    pickle.dump({
        "model": model,
        "X_train": X_train,
        "y_train": y_train,
        "all_classes": list(np.unique(y))
    }, f)

print(f"\nModel saved to: {MODEL_OUTPUT}")

# update UI configuration
all_classes = sorted(np.unique(y).tolist())
config = {
    "all_classes": all_classes,
    "letters_numbers": [c for c in all_classes if len(c) == 1],
    "words": [c for c in all_classes if len(c) > 1]
}

os.makedirs("config", exist_ok=True)
with open("config/signs_config.json2", "w") as f:
    json.dump(config, f, indent=2)

print("UI configuration updated: config/signs_config2.json")
