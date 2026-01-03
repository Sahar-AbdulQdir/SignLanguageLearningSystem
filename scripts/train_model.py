import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# =======================
# PATHS
# =======================
EXISTING_DATA = "data/landmark_data_right_hand.npz"
CUSTOM_WORDS_DIR = "data/custom_words/"
COMBINED_OUTPUT = "data/combined_landmark_data.npz"
MODEL_OUTPUT = "models/knn_combined_model.pkl"

# =======================
# LOAD AND COMBINE DATA
# =======================
def combine_datasets():
    # Load existing data (letters/numbers)
    print("Loading existing dataset...")
    existing_data = np.load(EXISTING_DATA)
    X_existing = existing_data["X"]
    y_existing = existing_data["y"]
    
    print(f"Existing data: {X_existing.shape[0]} samples, {len(np.unique(y_existing))} classes")
    
    # Load custom words data
    X_custom = []
    y_custom = []
    
    print("\nLoading custom words data...")
    for word_dir in os.listdir(CUSTOM_WORDS_DIR):
        word_path = os.path.join(CUSTOM_WORDS_DIR, word_dir)
        
        if os.path.isdir(word_path):
            npz_files = [f for f in os.listdir(word_path) if f.endswith("_landmarks.npz")]
            
            for npz_file in npz_files:
                data_path = os.path.join(word_path, npz_file)
                data = np.load(data_path)
                
                X_custom.append(data["X"])
                y_custom.append(data["y"])
    
    if X_custom:
        X_custom = np.vstack(X_custom)
        y_custom = np.concatenate(y_custom)
        print(f"Custom words data: {X_custom.shape[0]} samples, {len(np.unique(y_custom))} classes")
        
        # Combine datasets
        X_combined = np.vstack([X_existing, X_custom])
        y_combined = np.concatenate([y_existing, y_custom])
    else:
        X_combined = X_existing
        y_combined = y_existing
        print("No custom words data found.")
    
    # Save combined dataset
    np.savez_compressed(COMBINED_OUTPUT, X=X_combined, y=y_combined)
    print(f"\nCombined dataset saved: {COMBINED_OUTPUT}")
    print(f"Total samples: {X_combined.shape[0]}")
    print(f"Total classes: {len(np.unique(y_combined))}")
    
    return X_combined, y_combined

# =======================
# TRAIN NEW MODEL
# =======================
def train_combined_model():
    X, y = combine_datasets()
    
    print("\n" + "="*50)
    print("Training new model with combined data...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Train model
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="euclidean"
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump({
            "model": model,
            "X_train": X_train,
            "y_train": y_train,
            "all_classes": list(np.unique(y))
        }, f)
    
    print(f"\nModel saved to: {MODEL_OUTPUT}")
    print(f"Classes in model: {list(np.unique(y))}")
    
    return model

# =======================
# UPDATE UI CONFIG
# =======================
def update_ui_config():
    """Create a configuration file with all available signs"""
    data = np.load(COMBINED_OUTPUT)
    all_classes = sorted(np.unique(data["y"]).tolist())
    
    config = {
        "all_classes": all_classes,
        "letters_numbers": [c for c in all_classes if len(c) == 1],
        "words": [c for c in all_classes if len(c) > 1]
    }
    
    import json
    with open("config/signs_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nUI configuration updated: config/signs_config.json")
    print(f"Total signs available: {len(all_classes)}")
    print(f"Letters/Numbers: {len(config['letters_numbers'])}")
    print(f"Words: {len(config['words'])}")

if __name__ == "__main__":
    train_combined_model()
    update_ui_config()