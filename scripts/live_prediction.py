import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load trained model
with open("models/knn_landmark_model_right.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        hand = result.multi_hand_landmarks[0]
        handedness = result.multi_handedness[0].classification[0].label

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
        if handedness == "Left":
            for i in range(0, len(landmarks), 3):
                landmarks[i] *= -1

        # Scale normalization
        landmarks = np.array(landmarks)
        scale = np.linalg.norm(landmarks[3:6]) + 1e-6
        landmarks = landmarks / scale

        X_input = landmarks.reshape(1, -1)
        pred = model.predict(X_input)[0]
        confidence = np.max(model.predict_proba(X_input)) * 100

        cv2.putText(
            frame,
            f"{pred} ({confidence:.1f}%)",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS
        )

    cv2.imshow("Sign Recognition (Landmarks)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
