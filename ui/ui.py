import tkinter as tk
import customtkinter as ct
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tkinter import PhotoImage, filedialog, Button
import json

# =======================
# LOAD MODEL
# =======================
try:
    # Load the combined model
    with open("models/knn_combined_model.pkl", "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    all_classes = data["all_classes"]  # This now includes words
    
    # Load configuration
    with open("config/signs_config.json", "r") as f:
        config = json.load(f)
    
    print(f"Model loaded successfully. Classes: {all_classes}")
    print(f"Available signs: {len(all_classes)} total")
    print(f"Words: {config['words']}")
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to original classes plus some common words
    all_classes = ["A", "B", "C", "1", "2", "3", "HELLO", "THANK YOU", "YES", "NO"]
    config = {
        "letters_numbers": ["A", "B", "C", "1", "2", "3"],
        "words": ["HELLO", "THANK YOU", "YES", "NO"]
    }

# =======================
# MEDIAPIPE SETUP
# =======================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,  # Lowered for better detection
    min_tracking_confidence=0.5
)

# =======================
# HELPERS
# =======================
def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        print("No hand detected!")
        return None

    hand = result.multi_hand_landmarks[0]
    
    if not result.multi_handedness:
        print("No handedness info!")
        return None
        
    handedness = result.multi_handedness[0].classification[0].label
    print(f"Detected hand: {handedness}")

    landmarks = []
    for lm in hand.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    # Normalize to wrist
    base_x, base_y, base_z = landmarks[0:3]
    for i in range(0, len(landmarks), 3):
        landmarks[i]   -= base_x
        landmarks[i+1] -= base_y
        landmarks[i+2] -= base_z

    # FORCE RIGHT HAND (as in training)
    if handedness == "Left":
        print("Left hand detected, flipping to right...")
        for i in range(0, len(landmarks), 3):
            landmarks[i] *= -1

    landmarks = np.array(landmarks)
    
    # Scale normalization (using distance between wrist and index finger MCP)
    if len(landmarks) >= 6:  # Make sure we have enough landmarks
        scale = np.linalg.norm(landmarks[3:6]) + 1e-6
        landmarks = landmarks / scale
    else:
        print("Not enough landmarks for scaling!")
        return None
        
    return landmarks

def predict_landmarks(landmarks):
    try:
        X = landmarks.reshape(1, -1)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        conf = np.max(proba) * 100
        
        # Get probability for each class
        prob_dict = {}
        for i, class_name in enumerate(all_classes):
            prob_dict[class_name] = proba[i] * 100
        
        return pred, conf, prob_dict
    except Exception as e:
        print(f"Prediction error: {e}")
        # Return dummy values for testing
        dummy_probs = {cls: 0 for cls in all_classes}
        dummy_probs["A"] = 100  # Default to A for testing
        return "A", 100, dummy_probs

# =======================
# APP SETUP
# =======================
root = ct.CTk()
root.geometry("1200x700")
root.title("Sign Language App")

# =======================
# FRAMES
# =======================
Start_frame = ct.CTkFrame(root)
Home_frame = ct.CTkFrame(root)
Learn_frame = ct.CTkFrame(root)
Translate_frame = ct.CTkFrame(root)
Upload_frame = ct.CTkFrame(root)

for frame in (Start_frame, Home_frame, Learn_frame, Translate_frame, Upload_frame):
    frame.place(relwidth=1, relheight=1)

# =======================
# BACKGROUND HANDLER
# =======================
def set_bg(frame, path):
    try:
        img = Image.open(path)
        bg = tk.Label(frame)
        bg.place(relwidth=1, relheight=1)

        def resize(e):
            resized = img.resize((e.width, e.height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            bg.configure(image=photo)
            bg.image = photo

        frame.bind("<Configure>", resize)
    except:
        # If background image fails, just set solid color
        frame.configure(bg="#2b2b2b")

set_bg(Start_frame, "Images/Backgrounds/Start_page.png")
set_bg(Home_frame, "Images/Backgrounds/Home_page.png")
set_bg(Learn_frame, "Images/Backgrounds/Learn_page.png")
set_bg(Translate_frame, "Images/Backgrounds/Translate_page.png")
set_bg(Upload_frame, "Images/Backgrounds/Upload_page.png")

# =======================
# FRAME SWITCHING
# =======================
def switch_frame(frame):
    # Stop camera if running
    if hasattr(root, 'camera_active') and root.camera_active:
        stop_camera()
    frame.tkraise()

# =======================
# CAMERA FUNCTIONS
# =======================
def start_camera_in_frame(frame_widget, show_prediction=False, target_sign=None):
    """Start camera and display it inside the given widget"""
    if hasattr(root, 'camera_active') and root.camera_active:
        stop_camera()
    
    root.camera_active = True
    root.camera_widget = frame_widget
    root.show_prediction = show_prediction
    root.target_sign = target_sign
    
    # Create video label if it doesn't exist
    if not hasattr(frame_widget, 'video_label'):
        frame_widget.video_label = tk.Label(frame_widget, bg="#fbf4e4")
        frame_widget.video_label.place(x=1100, y=200, width=500, height=500)
    
    # Create feedback label for Learn mode
    if target_sign and not hasattr(frame_widget, 'feedback_label'):
        frame_widget.feedback_label = ct.CTkLabel(
            frame_widget,
            text="",
            font=("Arial", 16),
            text_color="black",
            bg_color="#fbf4e4"
        )
        frame_widget.feedback_label.place(x=710, y=500, width=300, height=50)
    
    # Create top predictions frame
    # feedback for Learn mode 
    if target_sign and not hasattr(frame_widget, 'top_predictions_frame'):
        frame_widget.top_predictions_frame = ct.CTkFrame(frame_widget,
                                                        width=340,
                                                        height=60,
                                                        border_color="#000000",
                                                        fg_color="#fbf4e4")
        frame_widget.top_predictions_frame.place(x=735, y=505)
        
        # Title
        ct.CTkLabel(
            frame_widget.top_predictions_frame,
            text="Top Predictions:",
            font=("Arial", 14, "bold")
        ).place(x=100, y=500)
        
        # Create prediction labels
        frame_widget.prediction_labels = []
        for i in range(3):
            label = ct.CTkLabel(
                frame_widget.top_predictions_frame,
                text="",
                font=("Arial", 12)
            )
            label.place(x=1, y=15)
            frame_widget.prediction_labels.append(label)
    
    # Debug label
    if target_sign and not hasattr(frame_widget, 'debug_label'):
        frame_widget.debug_label = ct.CTkLabel(
            frame_widget,
            text="",
            font=("Arial", 10),
            text_color="black"
        )
        frame_widget.debug_label.place(x=650, y=100, width=500, height=30)
    
    # Start video capture
    root.cap = cv2.VideoCapture(0)
    if not root.cap.isOpened():
        print("Error: Could not open camera")
        if hasattr(frame_widget, 'debug_label'):
            frame_widget.debug_label.configure(text="Camera error!")
        return
    
    update_camera_frame()

def update_camera_frame():
    if not hasattr(root, 'camera_active') or not root.camera_active:
        return
    
    ret, frame = root.cap.read()
    if not ret:
        print("Failed to grab frame")
        return
    
    frame = cv2.flip(frame, 1)
    
    # Draw hand landmarks for visual feedback
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    hand_detected = False
    if result.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
    
    # Process for prediction if needed
    if root.show_prediction and hand_detected:
        landmarks = extract_landmarks(frame)
        
        # Update debug info
        if hasattr(root.camera_widget, 'debug_label'):
            debug_text = f"Hand detected: {hand_detected}"
            if landmarks is not None:
                debug_text += f" | Landmarks shape: {landmarks.shape}"
            root.camera_widget.debug_label.configure(text=debug_text)
        
        if landmarks is not None:
            try:
                pred, conf, prob_dict = predict_landmarks(landmarks)
                
                # Check if we're in learn mode with target sign
                if root.target_sign:
                    target_prob = prob_dict.get(root.target_sign, 0)
                    
                    # Debug: Print all probabilities
                    print(f"\nCurrent probabilities:")
                    for sign, prob in prob_dict.items():
                        print(f"  {sign}: {prob:.1f}%")
                    
                    # Update feedback based on closeness
                    if target_prob >= 80:
                        feedback_text = f"Excellent! ({target_prob:.1f}% match)"
                        feedback_color = (0, 255, 0)  # Green
                        bg_color = "#00AA00"
                    elif target_prob >= 60:
                        feedback_text = f"Good! ({target_prob:.1f}% match)"
                        feedback_color = (0, 200, 100)  # Yellow-green
                        bg_color = "#44AA00"
                    elif target_prob >= 40:
                        feedback_text = f"Getting there ({target_prob:.1f}% match)"
                        feedback_color = (0, 150, 255)  # Orange
                        bg_color = "#FF8800"
                    else:
                        feedback_text = f"Keep trying ({target_prob:.1f}% match)"
                        feedback_color = (0, 0, 255)  # Red
                        bg_color = "#FF0000"
                    
                    # Update feedback label
                    if hasattr(root.camera_widget, 'feedback_label'):
                        root.camera_widget.feedback_label.configure(
                            text=feedback_text,
                            text_color=bg_color
                        )
                    
                    # Update top predictions display
                    if hasattr(root.camera_widget, 'prediction_labels'):
                        # Sort probabilities in descending order
                        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:1]
                        
                        for i, (sign, prob) in enumerate(sorted_probs):
                            # Highlight target sign
                            if sign == root.target_sign:
                                label_text = f"✓ {sign}: {prob:.1f}%"
                                text_color = "#00FF00"
                            else:
                                label_text = f"  {sign}: {prob:.1f}%"
                                text_color = "white"
                            
                            root.camera_widget.prediction_labels[i].configure(
                                text=label_text,
                                text_color="black",
                                fg_color="#fbf4e4",          # CustomTkinter background
                                font=("Arial", 18),# Bigger text
                                )

                    # Display on video frame
                    cv2.putText(frame, f"Target: {root.target_sign}", (30, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, feedback_color, 2)
                    cv2.putText(frame, f"Match: {target_prob:.1f}%", (30, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2)
                    cv2.putText(frame, f"Predicted: {pred}", (30, 130),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    # Live translation mode - simple display
                    cv2.putText(frame, f"{pred} ({conf:.1f}%)", (30, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    
            except Exception as e:
                print(f"Error in prediction loop: {e}")
                cv2.putText(frame, "Prediction error", (30, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        # Show detection status
        if not hand_detected:
            cv2.putText(frame, "Show your hand to camera", (30, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    # Convert to RGB and then to ImageTk
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img = img.resize((500, 400), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(image=img)
    
    # Update label
    root.camera_widget.video_label.configure(image=photo)
    root.camera_widget.video_label.image = photo
    
    # Schedule next update
    root.camera_widget.after(10, update_camera_frame)

def stop_camera():
    if hasattr(root, 'camera_active') and root.camera_active:
        root.camera_active = False
        if hasattr(root, 'cap'):
            root.cap.release()
        if hasattr(root.camera_widget, 'video_label'):
            root.camera_widget.video_label.configure(image='')
        # Clear feedback labels
        if hasattr(root.camera_widget, 'feedback_label'):
            root.camera_widget.feedback_label.configure(text="")
        if hasattr(root.camera_widget, 'prediction_labels'):
            for label in root.camera_widget.prediction_labels:
                label.configure(text="")

# =======================
# START PAGE
# =======================
# Start_img = PhotoImage(file="Images/Buttons/HomeBtn.png")
start_btn = ct.CTkButton(
    Start_frame,
    text="Start",
    font=("Arial", 30, "bold"),
    fg_color="#b7cbe9",  
    text_color="black",
    width=150,
    height=70,
    background_corner_colors=("#fbf4e4", "#fbf4e4", "#fbf4e4", "#fbf4e4"),
    corner_radius=20,
    command=lambda: switch_frame(Home_frame)
)
start_btn.place(x=510, y=550)

# =======================
# HOME PAGE
# =======================
back_btn_home = ct.CTkButton(
    Home_frame,
    text="◀",
    font=("Arial", 20),
    fg_color="#272727",
    text_color="white",
    corner_radius=12,
    background_corner_colors=("#fbf4e4", "#fbf4e4", "#fbf4e4", "#fbf4e4"),
    command=lambda: switch_frame(Home_frame),
    width=50,
    height=50
)
back_btn_home.place(x=55, y=55)

# button_img = PhotoImage(file="Images/Buttons/startB.png")
# Create button with image
learn_btn = ct.CTkButton(
    Home_frame,
    text="Learn Signs",
    font=("Arial", 20),
    width=180,
    height=80,
    fg_color="#ffc1f0",  # 
    text_color="#272727",
    corner_radius=20,
    background_corner_colors=("#fbf4e4", "#fbf4e4", "#fbf4e4", "#fbf4e4"),
    command=lambda: switch_frame(Learn_frame)
)
learn_btn.place(x=290, y=435)

translate_btn = ct.CTkButton(
    Home_frame,
    text="Live translation",
    font=("Arial", 20),
    width=180,
    height=80,
    fg_color="#ffc1f0",  # 
    text_color="#272727",
    background_corner_colors=("#fbf4e4", "#fbf4e4", "#fbf4e4", "#fbf4e4"),
    corner_radius=20,
    command=lambda: switch_frame(Translate_frame)
)
translate_btn.place(x=730, y=435)

Upload_frame_btn = ct.CTkButton(
    Home_frame,
    text="Translate Image",
    font=("Arial", 20),
    width=180,
    height=80,
    fg_color="#ffc1f0",  # 
    background_corner_colors=("#fbf4e4", "#fbf4e4", "#fbf4e4", "#fbf4e4"),
    text_color="#272727",
    corner_radius=20,
    command=lambda: switch_frame(Upload_frame)
)
Upload_frame_btn.place(x=525, y=538)

# =======================
# LEARN UI
# =======================
back_btn_learn = ct.CTkButton(
    Learn_frame,
    text="◀",
    font=("Arial", 20),
    fg_color="#272727",
    text_color="white",
    corner_radius=12,
    background_corner_colors=("#fbf4e4", "#fbf4e4", "#fbf4e4", "#fbf4e4"),
    command=lambda: switch_frame(Home_frame),
    width=50,
    height=50
)
back_btn_learn.place(x=55, y=55)

selected_sign = tk.StringVar(value="A")
learn_values = config.get("letters_numbers", []) + config.get("words", [])

sign_menu = ct.CTkOptionMenu(
    Learn_frame,
    values=learn_values,
    variable=selected_sign,
    width=320,
    height=40,
    corner_radius=15,
    fg_color="#91b6ff",
    button_color="#91b6ff",
    dropdown_fg_color="#91b6ff",
    bg_color="#fbf4e4",
    dropdown_text_color="white"
)
sign_menu.place(x=140, y=110)

sign_img_label = ct.CTkLabel(
    Learn_frame,
    text="",
    width=260,
    height=200,
    fg_color="#fbf4e4",
    corner_radius=12
)
sign_img_label.place(x=155, y=165)

def update_sign_image(*_):
    sign_name = selected_sign.get()
    
    # Determine folder and extension
    if sign_name in config.get("letters_numbers", []):
        img_folder = "images/sign_images/"
        img_ext = ".jpg"
    else:
        img_folder = "images/words_images/"
        img_ext = ".png"
    
    try:
        img = Image.open(f"{img_folder}{sign_name}{img_ext}").resize((400, 400))
        photo = ImageTk.PhotoImage(img)
        sign_img_label.configure(image=photo)
        sign_img_label.image = photo
    except:
        # Placeholder if image not found
        placeholder = Image.new('RGB', (2, 2), color='#fbf4e4')
        photo = ImageTk.PhotoImage(placeholder)
        sign_img_label.configure(image=photo)
        sign_img_label.image = photo

selected_sign.trace_add("write", update_sign_image)
update_sign_image()

# Tips label
tips_label = ct.CTkLabel(
    Learn_frame,
    text="Tips:",
    font=("Arial", 22),
    fg_color="#fbf4e4",
    text_color="#000000"
)
tips_label.place(x=135, y=450)

tips_info1 = ct.CTkLabel(
    Learn_frame,
    text="Use your RIGHT hand",
    font=("Arial", 18),
    fg_color="#fbf4e4",
    justify="left",
    text_color="#000000"
)
tips_info1.place(x=145, y=495)

tips_info2 = ct.CTkLabel(
    Learn_frame,
    text="Make sure hand is clearly visible",
    font=("Arial", 18),
    fg_color="#fbf4e4",
    justify="left",
    text_color="#000000"
)
tips_info2.place(x=145, y=529)

tips_info3 = ct.CTkLabel(
    Learn_frame,
    text="Good lighting helps",
    font=("Arial", 18),
    fg_color="#fbf4e4",
    justify="left",
    text_color="#000000"
)
tips_info3.place(x=145, y=560)

Camera_on_button_img = PhotoImage(file="Images/Buttons/camera_on.png")

start_practice_btn = Button(
    Learn_frame,
    image=Camera_on_button_img,
    width=70,
    height=60,
    bg="#fbf4e4",  # 
    bd=0,  # remove border
    highlightthickness=0,
    command=lambda: start_camera_in_frame(Learn_frame, show_prediction=True, target_sign=selected_sign.get())
)
start_practice_btn.place(x=970, y=430)

Camera_off_button_img = PhotoImage(file="Images/Buttons/camera_off.png")

stop_camera_btn_learn = Button(
    Learn_frame,
    image=Camera_off_button_img,
    width=90,
    height=65,
    bg="#fbf4e4",  # 
    bd=0,  # remove border
    highlightthickness=0,
    command=stop_camera
)
stop_camera_btn_learn.place(x=960, y=525)

# =======================
# Upload UI
# =======================
back_btn_translate = ct.CTkButton(
    Upload_frame,
    text="◀",
    font=("Arial", 20),
    fg_color="#272727",
    text_color="white",
    corner_radius=12,
    background_corner_colors=("#fbf4e4", "#fbf4e4", "#fbf4e4", "#fbf4e4"),
    command=lambda: switch_frame(Home_frame),
    width=50,
    height=50
)
back_btn_translate.place(x=55, y=55)

uploaded_img_label = ct.CTkLabel(
    Upload_frame,
    text="",
    width=330,
    height=330,
    fg_color="#fbf4e4",
)
uploaded_img_label.place(x=180, y=175)

result_label = ct.CTkLabel(
    Upload_frame,
    text="",
    font=("Arial", 18),
    text_color="black",
    fg_color="#fbf4e4",
    justify="left"
)
result_label.place(x=630, y=370)

def translate_image():
    path = filedialog.askopenfilename()
    if not path:
        return
    
    # Display the uploaded image
    try:
        img = Image.open(path)
        img = img.resize((430, 430),Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        uploaded_img_label.configure(image=photo)
        uploaded_img_label.image = photo
    except:
        result_label.configure(text="Error loading image")
        return
    
    # Process for translation
    cv_img = cv2.imread(path)
    landmarks = extract_landmarks(cv_img)

    if landmarks is None:
        result_label.configure(text="No hand detected")
        return

    pred, conf, prob_dict = predict_landmarks(landmarks)
    
    # Show top 3 predictions
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    pred_text = f"Top Prediction: {pred} ({conf:.1f}%)\n\nOther possibilities:\n"
    for i, (sign, prob) in enumerate(sorted_probs[1:], 1):
        pred_text += f"{i}. {sign}: {prob:.1f}%\n"
    
    result_label.configure(text=pred_text)

# upload_btn_img = PhotoImage(file="Images/Buttons/startBtns.png")
upload_btn = ct.CTkButton(
    Upload_frame,
    text="UPLOAD IMAGE",
    font=("Arial", 16),
    text_color="black",
    fg_color="#b9ca84",
    background_corner_colors=("#fbf4e4", "#fbf4e4", "#fbf4e4", "#fbf4e4"),
    corner_radius=20,
    width=150,
    height=60,
    command=translate_image
)
upload_btn.place(x=155, y=535)

# =======================
# TRANSLATE UI
# =======================
back_btn_translate = ct.CTkButton(
    Translate_frame,
    text="◀",
    font=("Arial", 20),
    fg_color="#272727",
    text_color="white",
    corner_radius=12,
    background_corner_colors=("#fbf4e4", "#fbf4e4", "#fbf4e4", "#fbf4e4"),
    command=lambda: switch_frame(Home_frame),
    width=50,
    height=50
)
back_btn_translate.place(x=55, y=55)


# Load the original image
original_img = Image.open("Images/Buttons/camera_on2.png")

# Resize the image to your desired size (300x140)
resized_img = original_img.resize((100, 120), Image.Resampling.LANCZOS)
start_camera_img = ImageTk.PhotoImage(resized_img)

# Create the button
start_live_btn = Button(
    Translate_frame,
    image=start_camera_img,
    bg="#fbf4e4",
    bd=0,
    highlightthickness=0,
    command=lambda: start_camera_in_frame(
        Translate_frame,
        show_prediction=True,
        target_sign=None
    )
)
start_live_btn.place(x=890, y=200)
# Load original image
original_stop_img = Image.open("Images/Buttons/camera_off2.png")

# Resize image to desired size, e.g., 250x250
resized_stop_img = original_stop_img.resize((100, 120), Image.Resampling.LANCZOS)
stop_camera_img = ImageTk.PhotoImage(resized_stop_img)

# Create the button
stop_camera_btn_translate = Button(
    Translate_frame,
    image=stop_camera_img,
    bg="#fbf4e4",
    bd=0,
    highlightthickness=0,
    command=stop_camera
)
stop_camera_btn_translate.place(x=890, y=360)
# =======================
# START APP
# =======================
switch_frame(Upload_frame)

# Clean up on close
def on_closing():
    if hasattr(root, 'camera_active') and root.camera_active:
        stop_camera()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()