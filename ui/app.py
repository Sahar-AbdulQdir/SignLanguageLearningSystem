import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
import time

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Sign Language Learning System",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
with open("models/knn_landmark_model_right.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]

# -------------------------------
# MediaPipe Hands
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------------------------------
# Session State
# -------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "letter" not in st.session_state:
    st.session_state.letter = "A"

letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# -------------------------------
# Camera
# -------------------------------
cap = cv2.VideoCapture(0)

# -------------------------------
# Utility: Extract Landmarks
# -------------------------------
def extract_landmarks(hand_landmarks, handedness):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    base_x, base_y, base_z = landmarks[0:3]
    for i in range(0, len(landmarks), 3):
        landmarks[i] -= base_x
        landmarks[i+1] -= base_y
        landmarks[i+2] -= base_z

    if handedness == "Left":
        for i in range(0, len(landmarks), 3):
            landmarks[i] *= -1

    landmarks = np.array(landmarks)
    landmarks /= (np.linalg.norm(landmarks[3:6]) + 1e-6)
    return landmarks.reshape(1, -1)

# -------------------------------
# Keyboard Listener
# -------------------------------
key = st.text_input(
    "Keyboard Control (H=Home, L=Learn, T=Translate, N=Next Letter, Q=Quit)",
    max_chars=1
).upper()

if key == "H":
    st.session_state.page = "home"
elif key == "L":
    st.session_state.page = "learn"
elif key == "T":
    st.session_state.page = "translate"
elif key == "N":
    idx = letters.index(st.session_state.letter)
    st.session_state.letter = letters[(idx + 1) % len(letters)]

# -------------------------------
# HOME PAGE
# -------------------------------
if st.session_state.page == "home":
    st.title("🤟 Sign Language Learning System")
    st.subheader("Student-Level Machine Learning Project")
    st.markdown("""
    ### Modes
    - **Learn Mode**: Practice letters and receive confidence feedback
    - **Translate Mode**: Show signs and get predicted letters
    """)
    st.info("Press **L** to Learn or **T** to Translate")

# -------------------------------
# LEARN MODE
# -------------------------------
elif st.session_state.page == "learn":
    st.title("📘 Learn Mode")
    st.subheader(f"Target Letter: **{st.session_state.letter}**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Reference Sign")
        st.image(
            f"Images/sign_images/{st.session_state.letter}.jpg",
            width=300,
            caption="Imitate this sign"
        )

    with col2:
        st.markdown("### Camera Feedback")
        frame_placeholder = st.empty()
        text_placeholder = st.empty()

        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks and res.multi_handedness:
                hand = res.multi_hand_landmarks[0]
                handedness = res.multi_handedness[0].classification[0].label

                X = extract_landmarks(hand, handedness)
                pred = model.predict(X)[0]
                confidence = np.max(model.predict_proba(X)) * 100

                probs = model.predict_proba(X)[0]
                target_idx = list(model.classes_).index(st.session_state.letter)
                similarity = probs[target_idx] * 100

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

                frame_placeholder.image(frame, channels="BGR")

                status = "✅ Correct" if pred == st.session_state.letter else "❌ Incorrect"

                text_placeholder.markdown(f"""
                **Predicted:** {pred}  
                **Match:** {status}  
                **Similarity:** {similarity:.1f}%  
                **Confidence:** {confidence:.1f}%  
                """)

            else:
                frame_placeholder.image(frame, channels="BGR")
                text_placeholder.info("Show your hand to the camera")

    st.warning("Press **N** for next letter | **H** to go Home")

# -------------------------------
# TRANSLATE MODE
# -------------------------------
elif st.session_state.page == "translate":
    st.title("🔤 Translate Mode")
    st.subheader("Show a sign to the camera")

    frame_placeholder = st.empty()
    text_placeholder = st.empty()

    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks and res.multi_handedness:
            hand = res.multi_hand_landmarks[0]
            handedness = res.multi_handedness[0].classification[0].label

            X = extract_landmarks(hand, handedness)
            pred = model.predict(X)[0]
            confidence = np.max(model.predict_proba(X)) * 100

            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            frame_placeholder.image(frame, channels="BGR")
            text_placeholder.success(
                f"Detected: **{pred}** | Confidence: **{confidence:.1f}%**"
            )
        else:
            frame_placeholder.image(frame, channels="BGR")
            text_placeholder.info("Waiting for hand sign...")

    st.warning("Press **H** to return Home")

# -------------------------------
# Cleanup
# -------------------------------
if key == "Q":
    cap.release()
    st.stop()
