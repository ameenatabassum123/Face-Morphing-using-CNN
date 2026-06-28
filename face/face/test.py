import tkinter as tk
from tkinter import filedialog, Label, Button, Entry, Frame, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ======================
# SETTINGS
# ======================

MODEL_PATH = "face_morphing_cnn.h5"
IMG_SIZE = (150, 150)

CLASS_MAP = {
    0: "MORPHED FACE",
    1: "REAL FACE"
}

ADMIN_USER = "admin"
ADMIN_PASS = "admin"

# ======================
# LOAD MODEL
# ======================

model = load_model(MODEL_PATH)
print("Model loaded")

# ======================
# ROOT WINDOW
# ======================

root = tk.Tk()
root.title("Face Morph Detection")
root.geometry("500x550")


# ======================
# MAIN APP FRAME
# ======================

class AppFrame(Frame):
    def __init__(self, master):
        super().__init__(master)

        self.panel = Label(self)
        self.panel.pack(pady=20)

        self.result_label = Label(self, text="", font=("Arial", 16, "bold"))
        self.result_label.pack(pady=10)

        self.score_label = Label(self, text="", font=("Arial", 12))
        self.score_label.pack()

        self.btn = Button(self, text="Select Image", command=self.load_image, font=("Arial", 14))
        self.btn.pack(pady=20)

    # -------- Prediction --------
    def predict_image(self, path):
        img = image.load_img(path, target_size=IMG_SIZE)
        arr = image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        pred = model.predict(arr)[0][0]
        label_index = int(pred > 0.5)

        self.result_label.config(text=f"Prediction: {CLASS_MAP[label_index]}")
        

    # -------- Load Image --------
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.bmp")]
        )
        if not path:
            return

        img = Image.open(path)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)

        self.panel.config(image=img_tk)
        self.panel.image = img_tk

        self.predict_image(path)


# ======================
# LOGIN FRAME
# ======================

class LoginFrame(Frame):
    def __init__(self, master):
        super().__init__(master)

        Label(self, text="Admin Login", font=("Arial", 20, "bold")).pack(pady=30)

        Label(self, text="Username").pack()
        self.user_entry = Entry(self, font=("Arial", 12))
        self.user_entry.pack(pady=5)

        Label(self, text="Password").pack()
        self.pass_entry = Entry(self, show="*", font=("Arial", 12))
        self.pass_entry.pack(pady=5)

        Button(self, text="Login", font=("Arial", 14), command=self.check_login).pack(pady=20)

    def check_login(self):
        user = self.user_entry.get()
        pw = self.pass_entry.get()

        if user == ADMIN_USER and pw == ADMIN_PASS:
            self.pack_forget()
            app_frame.pack(fill="both", expand=True)
        else:
            messagebox.showerror("Login Failed", "Invalid username or password")


# ======================
# INIT FRAMES
# ======================

login_frame = LoginFrame(root)
app_frame = AppFrame(root)

login_frame.pack(fill="both", expand=True)

# ======================
# RUN
# ======================

root.mainloop()
