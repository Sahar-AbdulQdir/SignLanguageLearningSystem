import streamlit as st
import cv2
import numpy as np
import pickle
import mediapipe as mp

MODEL_PATH = "models/sign_classifier.pkl"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False)
mp_draw = mp.solutions.drawing_utils

def learn_sign():
    st.header("📘 Learn Sign Language")

    target_sign = st.selectbox(
        "Choose a letter or number to learn:",
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    )

    run = st.checkbox("Start Camera")

    if not run:
        st.info("Turn on the camera to start learning")
        return

    cap = cv2.VideoCapture(0)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    frame_window = st.image([])

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        feedback = "No hand detected"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

                prediction = model.predict([landmarks])[0]
                confidence = np.max(model.predict_proba([landmarks])) * 100

                if prediction == target_sign:
                    feedback = f"✅ Correct! Confidence: {confidence:.2f}%"
                else:
                    feedback = f"❌ Try again. Model sees: {prediction}"

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, feedback, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        frame_window.image(frame, channels="BGR")

    cap.release()
