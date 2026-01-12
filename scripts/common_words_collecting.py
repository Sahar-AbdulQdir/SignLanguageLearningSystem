# ·············································
# :      Importing required libraries          :
# ·············································
import os
import cv2
import numpy as np
import mediapipe as mp

# ·············································
# :    Defining configuration settings        :
# ·············································

CUSTOM_WORDS = [
    "Hello", "Mother", "Where", "Stop",
    "Calm Down"]

SAMPLES_PER_WORD = 50
OUTPUT_DIR = "data/custom_words/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ·············································
# :      Initializing MediaPipe Hands         :
# ·············································
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# ·············································
# :      Collecting static sign data          :
# ·············································
def collect_word_data(word):
    # Creating directories and initializing camera and storage for samples
    print(f"\nCollecting static sign for: {word}")
    word_dir = os.path.join(OUTPUT_DIR, word)
    os.makedirs(word_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    collected = []

    # Capturing frames until the required number of samples is reached
    while len(collected) < SAMPLES_PER_WORD:
        ret, frame = cap.read()
        if not ret:
            break

        # Flipping frame, processing hand landmarks, and preparing display output
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        display = frame.copy()
        cv2.putText(display, f"{word}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(display, f"{len(collected)}/{SAMPLES_PER_WORD}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(display, "SPACE = capture | Q = quit", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Detecting hand landmarks and drawing them on the frame
        if result.multi_hand_landmarks and result.multi_handedness:
            hand = result.multi_hand_landmarks[0]
            handedness = result.multi_handedness[0].classification[0].label

            mp.solutions.drawing_utils.draw_landmarks(
                display, hand, mp_hands.HAND_CONNECTIONS
            )

        cv2.imshow(f"Collecting {word}", display)
        key = cv2.waitKey(1) & 0xFF

        # Capturing landmarks, normalizing them, and saving samples on key press
        if key == ord(' ') and result.multi_hand_landmarks:
            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Normalizing landmarks relative to the wrist
            base_x, base_y, base_z = landmarks[0:3]
            for i in range(0, 63, 3):
                landmarks[i]   -= base_x
                landmarks[i+1] -= base_y
                landmarks[i+2] -= base_z

            # Converting all samples to a consistent right-hand orientation
            if handedness == "Left":
                for i in range(0, 63, 3):
                    landmarks[i] *= -1

            # Applying scale normalization for consistency across samples
            landmarks = np.array(landmarks)
            scale = np.linalg.norm(landmarks[3:6]) + 1e-6
            landmarks = landmarks / scale

            collected.append(landmarks)

            cv2.imwrite(
                os.path.join(word_dir, f"{len(collected):03d}.jpg"),
                frame
            )

            print(f"Captured {len(collected)}/{SAMPLES_PER_WORD}")

        elif key == ord('q'):
            break

    # Releasing resources and saving collected landmark data
    cap.release()
    cv2.destroyAllWindows()

    if collected:
        np.savez_compressed(
            os.path.join(word_dir, f"{word}_landmarks.npz"),
            X=np.array(collected),
            y=np.array([word] * len(collected))
        )
        print(f"Saved {len(collected)} samples for {word}")


# ·············································
# :      Starting the data collection         :
# ·············································
if __name__ == "__main__":
    for word in CUSTOM_WORDS:
        collect_word_data(word)

    print("Done")
