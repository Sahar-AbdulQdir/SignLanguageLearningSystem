import tkinter as tk
import customtkinter as ct
from PIL import Image, ImageTk

ct.set_appearance_mode("dark")

# -----------------------
# App setup
# -----------------------
root = ct.CTk()
root.geometry("1000x600")
root.title("Sign Language App")

# -----------------------
# Helper functions
# -----------------------
def hide_buttons(button_list):
    for btn in button_list:
        btn.place_forget()

def show_buttons(button_list):
    # Show buttons for the current frame
    for btn in button_list:
        btn.place(btn.saved_position)

def switch_frame(frame):
    # Hide all buttons first
    hide_buttons(buttons_start)
    hide_buttons(buttons_home)
    hide_buttons(buttons_learn)
    hide_buttons(buttons_translate)

    # Bring selected frame to front
    frame.tkraise()
    
    # Show buttons for the current frame
    if frame == Start_frame:
        show_buttons(buttons_start)
    elif frame == Home_frame:
        show_buttons(buttons_home)
    elif frame == Learn_frame:
        show_buttons(buttons_learn)
    elif frame == Translate_frame:
        show_buttons(buttons_translate)

# -----------------------
# Frames
# -----------------------
Start_frame = ct.CTkFrame(root)
Home_frame = ct.CTkFrame(root)
Learn_frame = ct.CTkFrame(root)
Translate_frame = ct.CTkFrame(root)

for frame in (Start_frame, Home_frame, Learn_frame, Translate_frame):
    frame.place(relwidth=1, relheight=1)

# -----------------------
# Backgrounds
# -----------------------
def set_bg(frame, path):
    img = Image.open(path)
    bg_label = tk.Label(frame)
    bg_label.place(relwidth=1, relheight=1)

    def resize_bg(event):
        resized = img.resize((event.width, event.height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        bg_label.config(image=photo)
        bg_label.image = photo

    frame.bind("<Configure>", resize_bg)

set_bg(Start_frame, "Images/Backgrounds/Start_page.png")
set_bg(Home_frame, "Images/Backgrounds/Home_page.png")
set_bg(Learn_frame, "Images/Backgrounds/Learn_page.png")
set_bg(Translate_frame, "Images/Backgrounds/Translate_page.png")

# -----------------------
# START PAGE
# -----------------------
start_btn_img = tk.PhotoImage(file="Images/Buttons/start_btn.png")
start_btn = tk.Button(
    Start_frame,
    image=start_btn_img,
    borderwidth=0,
    command=lambda: switch_frame(Home_frame)
)
start_btn.place(x=400, y=420)
start_btn.saved_position = {"x": 400, "y": 420}

buttons_start = [start_btn]

# -----------------------
# HOME PAGE
# -----------------------
home_back_img = tk.PhotoImage(file="Images/Buttons/back_btn.png")
home_back_btn = tk.Button(
    Home_frame,
    image=home_back_img,
    borderwidth=0,
    command=lambda: switch_frame(Start_frame)
)
home_back_btn.place(x=20, y=20)
home_back_btn.saved_position = {"x": 20, "y": 20}

learn_btn_img = tk.PhotoImage(file="Images/Buttons/learn_btn.png")
learn_btn = tk.Button(
    Home_frame,
    image=learn_btn_img,
    borderwidth=0,
    command=lambda: switch_frame(Learn_frame)
)
learn_btn.place(x=300, y=250)
learn_btn.saved_position = {"x": 300, "y": 250}

translate_btn_img = tk.PhotoImage(file="Images/Buttons/translate_btn.png")
translate_btn = tk.Button(
    Home_frame,
    image=translate_btn_img,
    borderwidth=0,
    command=lambda: switch_frame(Translate_frame)
)
translate_btn.place(x=550, y=250)
translate_btn.saved_position = {"x": 550, "y": 250}

buttons_home = [home_back_btn, learn_btn, translate_btn]

# -----------------------
# LEARN UI
# -----------------------
learn_back_img = tk.PhotoImage(file="Images/Buttons/back_btn.png")
learn_back_btn = tk.Button(
    Learn_frame,
    image=learn_back_img,
    borderwidth=0,
    command=lambda: switch_frame(Home_frame)
)
learn_back_btn.place(x=20, y=20)
learn_back_btn.saved_position = {"x": 20, "y": 20}

buttons_learn = [learn_back_btn]

# -----------------------
# TRANSLATE UI
# -----------------------
translate_back_img = tk.PhotoImage(file="Images/Buttons/back_btn.png")
translate_back_btn = tk.Button(
    Translate_frame,
    image=translate_back_img,
    borderwidth=0,
    command=lambda: switch_frame(Home_frame)
)
translate_back_btn.place(x=20, y=20)
translate_back_btn.saved_position = {"x": 20, "y": 20}

translate_action_img = tk.PhotoImage(file="Images/Buttons/upload_btn.png")
translate_action_btn = tk.Button(
    Translate_frame,
    image=translate_action_img,
    borderwidth=0,
    command=lambda: print("Translate logic here")
)
translate_action_btn.place(x=400, y=450)
translate_action_btn.saved_position = {"x": 400, "y": 450}

buttons_translate = [translate_back_btn, translate_action_btn]

# -----------------------
# Start app - Show only Start frame buttons initially
# -----------------------
switch_frame(Start_frame)
root.mainloop()