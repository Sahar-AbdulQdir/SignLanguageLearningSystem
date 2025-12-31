import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

data = np.load("data/landmark_data_right_hand.npz")
X, y = data["X"], data["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    metric="euclidean"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

with open("models/knn_landmark_model_right.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "X": X_train,
        "y": y_train
    }, f)

print("Model and training data saved!")

