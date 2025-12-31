import streamlit as st
import cv2
import pickle
import mediapipe as mp
import numpy as np

MODEL_PATH = "models/sign_classifier.pkl"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)
mp_draw = mp.solutions.drawing_utils

def translate_sign():
    st.header("🔤 Translate Sign Language")

    option = st.radio("Choose input method:", ["Upload Image", "Use Camera"])

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload a sign image", type=["jpg", "png"])

        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y])

                    prediction = model.predict([landmarks])[0]
                    st.success(f"Predicted Sign: {prediction}")
            else:
                st.error("No hand detected")

    else:
        run = st.checkbox("Start Camera")
        cap = cv2.VideoCapture(0)
        frame_window = st.image([])

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y])

                    prediction = model.predict([landmarks])[0]
                    cv2.putText(frame, f"Prediction: {prediction}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            frame_window.image(frame, channels="BGR")

        cap.release()
