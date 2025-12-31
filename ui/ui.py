import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
from tkinter import Tk, filedialog

# -------------------------------
# Load trained model
# -------------------------------
MODEL_PATH = "models/knn_landmark_model_right.pkl"
with open("models/knn_landmark_model_right.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
X_train = data["X"]
y_train = data["y"]

# -------------------------------
# Backgrounds
# -------------------------------
bg_home = np.full((720, 1280, 3), (200, 200, 200), dtype=np.uint8)
bg_learn = np.full((720, 1280, 3), (180, 220, 255), dtype=np.uint8)
bg_translate = np.full((720, 1280, 3), (220, 255, 180), dtype=np.uint8)

# Draw home buttons
cv2.rectangle(bg_home, (100, 200), (400, 300), (0, 255, 0), -1)
cv2.putText(bg_home, "LEARN", (180, 260),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

cv2.rectangle(bg_home, (500, 200), (800, 300), (255, 0, 0), -1)
cv2.putText(bg_home, "TRANSLATE", (520, 260),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

# -------------------------------
# MediaPipe Hands setup
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------------------------------
# Globals
# -------------------------------
page = "home"
letter = "A"
confidence = 0
similarity = 0
prediction_made = False

letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
letter_buttons = {}

start_x, start_y = 50, 50
btn_w, btn_h = 80, 80
gap = 20

for idx, ltr in enumerate(letters):
    x = start_x + (idx % 7) * (btn_w + gap)
    y = start_y + (idx // 7) * (btn_h + gap)
    letter_buttons[ltr] = (x, y, x + btn_w, y + btn_h)

uploaded_images = []  # For Translate page

# -------------------------------
# Mouse callback
# -------------------------------
def mouse_click(event, x, y, flags, param):
    global page, letter, uploaded_images, prediction_made

    if event == cv2.EVENT_LBUTTONDOWN:

        if page == "home":
            if 100 <= x <= 400 and 200 <= y <= 300:
                page = "learn"
                letter = "A"
                prediction_made = False
            elif 500 <= x <= 800 and 200 <= y <= 300:
                page = "translate"

        elif page == "learn":
            for ltr, (x1, y1, x2, y2) in letter_buttons.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    letter = ltr
                    prediction_made = False   # reset when letter changes

            if 50 <= x <= 200 and 620 <= y <= 700:
                page = "home"
                prediction_made = False

            if 850 <= x <= 1000 and 620 <= y <= 700:
                current_index = letters.index(letter)
                letter = letters[(current_index + 1) % len(letters)]
                prediction_made = False

        elif page == "translate":
            # Back button
            if 50 <= x <= 200 and 500 <= y <= 600:
                page = "home"

            # Upload button
            if 500 <= x <= 700 and 500 <= y <= 600:
                Tk().withdraw()
                file_paths = filedialog.askopenfilenames(
                    title="Select Images",
                    filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
                )
                uploaded_images = list(file_paths)

cv2.namedWindow("Sign Language App")
cv2.setMouseCallback("Sign Language App", mouse_click)

# -------------------------------
# Video capture
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("WARNING: Cannot open camera. Learn/Translate camera features will be disabled.")

# -------------------------------
# Main loop
# -------------------------------
while True:
    ret, cam_frame = False, None

    if cap.isOpened():
        ret, cam_frame = cap.read()
        if ret:
            cam_frame = cv2.flip(cam_frame, 1)

    # -------------------------------
    # HOME PAGE
    # -------------------------------
    if page == "home":
        frame = bg_home.copy()
        
        # Add title
        cv2.putText(frame, "SIGN LANGUAGE LEARNING APP", (350, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.putText(frame, "Choose a mode to begin", (450, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

    # -------------------------------
    # LEARN PAGE
    # -------------------------------
    elif page == "learn":
        frame = bg_learn.copy()
        prediction_made = False
        
        # Title
        cv2.putText(frame, "LEARN MODE - Select a letter and imitate it", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Draw letter buttons with highlight for selected
        for ltr, (x1, y1, x2, y2) in letter_buttons.items():
            # Highlight selected letter
            if ltr == letter:
                cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 255, 0), 3)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), -1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 200), -1)
            
            cv2.putText(frame, ltr, (x1 + 20, y1 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # LEFT: Reference image section
        cv2.rectangle(frame, (95, 95), (405, 405), (0, 0, 0), 2)  # Border
        cv2.putText(frame, f"REFERENCE: {letter}", (100, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Load and display reference sign image
        sign_path = f"Images/sign_images/{letter}.jpg"
        if os.path.exists(sign_path):
            sign_img = cv2.imread(sign_path)
            if sign_img is not None:
                sign_img = cv2.resize(sign_img, (300, 300))
                frame[100:400, 100:400] = sign_img
                
                # Add "Imitate this sign" text
                cv2.putText(frame, "Imitate this sign", (150, 420),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 2)
            else:
                cv2.putText(frame, "Image load error!", (150, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"{letter}.jpg not found", (120, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Place images in sign_images/", (80, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Process camera feed for prediction
        if ret and cam_frame is not None:
            rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            if result.multi_hand_landmarks and result.multi_handedness:
                hand = result.multi_hand_landmarks[0]
                handedness = result.multi_handedness[0].classification[0].label
                
                if len(hand.landmark) == 21:
                    landmarks = []
                    for lm in hand.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    # Normalize landmarks
                    base_x, base_y, base_z = landmarks[0:3]
                    for i in range(0, len(landmarks), 3):
                        landmarks[i] -= base_x
                        landmarks[i + 1] -= base_y
                        landmarks[i + 2] -= base_z
                    
                    if handedness == "Left":
                        for i in range(0, len(landmarks), 3):
                            landmarks[i] *= -1
                    
                    landmarks = np.array(landmarks)
                    scale = np.linalg.norm(landmarks[3:6]) + 1e-6
                    landmarks = landmarks / scale
                    
                    X_input = landmarks.reshape(1, -1)
                    pred = model.predict(X_input)[0]
                    confidence = np.max(model.predict_proba(X_input)) * 100
                    prediction_made = True
                    
                    # Calculate similarity to selected letter
                    if letter in model.classes_:
                        probas = model.predict_proba(X_input)[0]
                        letter_index = list(model.classes_).index(letter)
                        similarity = probas[letter_index] * 100
                    
                    # Draw hand landmarks on camera feed
                    mp.solutions.drawing_utils.draw_landmarks(
                        cam_frame, hand, mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Display camera feed
                    cam_display = cv2.resize(cam_frame, (300, 300))
                    frame[420:720, 100:400] = cam_display
                    
                    # Add camera feed border and label
                    cv2.rectangle(frame, (95, 415), (405, 725), (0, 0, 0), 2)
                    cv2.putText(frame, "YOUR CAMERA", (100, 410),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.putText(frame, "Show your hand here", (140, 450),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            else:
                # No hand detected - show placeholder
                cv2.rectangle(frame, (95, 415), (405, 725), (200, 200, 200), -1)
                cv2.rectangle(frame, (95, 415), (405, 725), (0, 0, 0), 2)
                cv2.putText(frame, "YOUR CAMERA", (100, 410),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame, "Show your hand to begin", (140, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                cv2.putText(frame, "Make sure hand is visible", (130, 540),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        else:
            # Camera not available
            cv2.rectangle(frame, (95, 415), (405, 725), (200, 200, 200), -1)
            cv2.rectangle(frame, (95, 415), (405, 725), (0, 0, 0), 2)
            cv2.putText(frame, "CAMERA UNAVAILABLE", (110, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # RIGHT: Feedback panel
        cv2.rectangle(frame, (450, 100), (800, 400), (250, 250, 250), -1)
        cv2.rectangle(frame, (450, 100), (800, 400), (0, 0, 0), 2)
        cv2.putText(frame, "PERFORMANCE FEEDBACK", (460, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        if prediction_made:
            # Get similarity to selected letter
            if letter in model.classes_:
                y_pos = 170
                lines = []
                colors = []
                
                # Line 1: Selected letter
                lines.append(f"Selected Letter: {letter}")
                colors.append((0, 0, 255))
                
                # Line 2: Predicted letter
                lines.append(f"Predicted: {pred}")
                colors.append((0, 200, 0) if pred == letter else (0, 0, 255))
                
                # Line 3: Match status
                match_status = "CORRECT MATCH ✓" if pred == letter else "NOT MATCHING ✗"
                lines.append(match_status)
                colors.append((0, 200, 0) if pred == letter else (0, 0, 255))
                
                # Line 4: Similarity score
                lines.append(f"Similarity Score: {similarity:.1f}%")
                if similarity > 80:
                    colors.append((0, 200, 0))
                elif similarity > 60:
                    colors.append((0, 150, 255))
                else:
                    colors.append((0, 0, 255))
                
                # Line 5: Overall confidence
                lines.append(f"Overall Confidence: {confidence:.1f}%")
                colors.append((100, 100, 100))
                
                # Display all lines
                for i, (line, color) in enumerate(zip(lines, colors)):
                    cv2.putText(frame, line, (470, y_pos + i * 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Progress bar for similarity
                bar_y = 330
                cv2.rectangle(frame, (470, bar_y), (780, bar_y + 20), (200, 200, 200), -1)
                bar_width = int(similarity * 3.1)  # Scale to ~310 pixels for 100%
                
                # Color based on similarity
                if similarity > 80:
                    bar_color = (0, 255, 0)
                    feedback = "Excellent! Perfect match!"
                elif similarity > 60:
                    bar_color = (0, 200, 200)
                    feedback = "Good! Keep practicing"
                elif similarity > 40:
                    bar_color = (255, 200, 0)
                    feedback = "Getting closer..."
                else:
                    bar_color = (255, 100, 0)
                    feedback = "Try to match the reference better"
                
                cv2.rectangle(frame, (470, bar_y), (470 + bar_width, bar_y + 20), bar_color, -1)
                
                # Add percentage text on bar
                cv2.putText(frame, f"{similarity:.0f}%", 
                            (470 + bar_width - 40 if bar_width > 40 else 470, bar_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Feedback message
                cv2.putText(frame, feedback, (470, 370),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)
                
        else:
            # No prediction yet
            cv2.putText(frame, "Waiting for hand detection...", (470, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            cv2.putText(frame, "Show your hand to the camera", (470, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            cv2.putText(frame, "and try to match the reference", (470, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            cv2.putText(frame, "sign on the left.", (470, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        
        # Back button
        cv2.rectangle(frame, (50, 620), (200, 700), (0, 0, 255), -1)
        cv2.putText(frame, "BACK", (70, 680),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Next button
        cv2.rectangle(frame, (850, 620), (1000, 700), (0, 255, 0), -1)
        cv2.putText(frame, "NEXT", (870, 680),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, "Try next letter", (840, 720),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1)

    # -------------------------------
    # TRANSLATE PAGE
    # -------------------------------
    elif page == "translate":
        frame = bg_translate.copy()
        
        # Title
        cv2.putText(frame, "TRANSLATE MODE", (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, "Upload images or use camera to translate signs", (350, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        
        # Upload button
        cv2.rectangle(frame, (500, 500), (700, 600), (0, 100, 200), -1)
        cv2.putText(frame, "UPLOAD", (510, 560),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Select sign images", (500, 620),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Display uploaded images
        y_offset = 50
        for idx, path in enumerate(uploaded_images[:6]):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (150, 150))
                x_offset = 50 + (idx % 3) * 160
                y_offset = 100 + (idx // 3) * 170
                frame[y_offset:y_offset + 150, x_offset:x_offset + 150] = img
                
                # Predict and display result on image
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                hands_img = mp.solutions.hands.Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=0.7
                )
                res = hands_img.process(img_rgb)
                
                if res.multi_hand_landmarks:
                    hand = res.multi_hand_landmarks[0]
                    landmarks = []
                    
                    for lm in hand.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    base_x, base_y, base_z = landmarks[0:3]
                    for i in range(0, len(landmarks), 3):
                        landmarks[i] -= base_x
                        landmarks[i + 1] -= base_y
                        landmarks[i + 2] -= base_z
                    
                    landmarks = np.array(landmarks)
                    scale = np.linalg.norm(landmarks[3:6]) + 1e-6
                    landmarks = landmarks / scale
                    
                    X_input = landmarks.reshape(1, -1)
                    pred = model.predict(X_input)[0]
                    conf = np.max(model.predict_proba(X_input)) * 100
                    
                    # Draw prediction on image
                    cv2.putText(frame, f"{pred} ({conf:.0f}%)",
                                (x_offset, y_offset + 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Camera feed for real-time translation
        if ret and cam_frame is not None:
            # Display camera feed
            cam_display = cv2.resize(cam_frame, (300, 300))
            frame[100:400, 600:900] = cam_display
            
            cv2.rectangle(frame, (595, 95), (905, 405), (0, 0, 0), 2)
            cv2.putText(frame, "LIVE CAMERA TRANSLATION", (600, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Process for translation
            rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            if result.multi_hand_landmarks and result.multi_handedness:
                hand = result.multi_hand_landmarks[0]
                handedness = result.multi_handedness[0].classification[0].label
                
                if len(hand.landmark) == 21:
                    landmarks = []
                    for lm in hand.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    base_x, base_y, base_z = landmarks[0:3]
                    for i in range(0, len(landmarks), 3):
                        landmarks[i] -= base_x
                        landmarks[i + 1] -= base_y
                        landmarks[i + 2] -= base_z
                    
                    if handedness == "Left":
                        for i in range(0, len(landmarks), 3):
                            landmarks[i] *= -1
                    
                    landmarks = np.array(landmarks)
                    scale = np.linalg.norm(landmarks[3:6]) + 1e-6
                    landmarks = landmarks / scale
                    
                    X_input = landmarks.reshape(1, -1)
                    pred = model.predict(X_input)[0]
                    conf = np.max(model.predict_proba(X_input)) * 100
                    
                    # Draw landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        cam_display, hand, mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Display prediction
                    cv2.putText(frame, f"Detected: {pred}", (600, 430),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.putText(frame, f"Confidence: {conf:.1f}%", (600, 470),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            else:
                cv2.putText(frame, "Show sign to camera", (620, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        
        # Back button
        cv2.rectangle(frame, (50, 500), (200, 600), (0, 0, 255), -1)
        cv2.putText(frame, "BACK", (70, 560),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Instructions:", (50, 650),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, "1. Upload images or use camera", (50, 680),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(frame, "2. System will detect and translate signs", (50, 700),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # -------------------------------
    # Show frame
    # -------------------------------
    cv2.imshow("Sign Language App", frame)
    
    # Display FPS (optional)
    if page != "home":
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
        cv2.putText(frame, f"FPS: {int(fps)}", (1150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()