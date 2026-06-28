import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# GLOBAL VARIABLES

DATASET_PATH = ""

TRAINED_MODEL_PATH = "present_trained.h5"
PRETRAINED_MODEL_PATH = "face_morphing_cnn.h5"

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 5

model = None
text_area = None



# MAIN UI (COLORFUL DASHBOARD)

def main_ui():
    global text_area

    root = tk.Tk()
    root.title("Face Morphing Detection System")
    root.geometry("1000x600")
    root.configure(bg="#ecf0f1")

    # ---------- HEADER ----------
    header = tk.Frame(root, bg="#2c3e50", height=80)
    header.pack(fill=tk.X)

    tk.Label(
        header,
        text="🧠 Face Morphing Detection Dashboard",
        font=("Segoe UI", 22, "bold"),
        bg="#2c3e50",
        fg="white"
    ).pack(pady=18)

    # ---------- CONTENT ----------
    content = tk.Frame(root, bg="#ecf0f1")
    content.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    # ---------- LEFT PANEL ----------
    left = tk.Frame(content, bg="#ecf0f1")
    left.pack(side=tk.LEFT, fill=tk.Y, padx=10)

    def make_btn(text, color, cmd):
        return tk.Button(
            left,
            text=text,
            command=cmd,
            bg=color,
            fg="white",
            font=("Segoe UI", 13, "bold"),
            width=22,
            height=2,
            relief="flat",
            cursor="hand2"
        )

    make_btn("Upload Dataset", "#2980b9", upload_dataset).pack(pady=8)
    make_btn("Preprocess & Graph", "#8e44ad", preprocess).pack(pady=8)
    make_btn("Train CNN Model", "#27ae60", train_model).pack(pady=8)
    make_btn("Predict Image", "#e67e22", predict_image).pack(pady=8)
    make_btn("Logout", "#c0392b", root.destroy).pack(pady=20)

    # ---------- RIGHT PANEL ----------
    right = tk.Frame(content, bg="white", bd=2, relief="groove")
    right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

    tk.Label(
        right,
        text="📄 System Logs",
        font=("Segoe UI", 14, "bold"),
        bg="white"
    ).pack(anchor="w", padx=10, pady=5)

    text_area = tk.Text(
        right,
        height=18,
        font=("Consolas", 11),
        bg="#fdfefe",
        fg="#2c3e50"
    )
    text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    root.mainloop()



# UPLOAD DATASET

def upload_dataset():
    global DATASET_PATH
    DATASET_PATH = filedialog.askdirectory(initialdir=".")

    if DATASET_PATH:
        total = sum(len(files) for _, _, files in os.walk(DATASET_PATH))
        text_area.insert(tk.END, f"\n Dataset Loaded\nPath: {DATASET_PATH}\n")
        text_area.insert(tk.END, f"Total Images: {total}\n")



# PREPROCESS + GRAPH

def preprocess():
    if not DATASET_PATH:
        messagebox.showwarning("Warning", "Please upload dataset first")
        return

    class_folders = [
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ]

    class_counts = {}
    for cls in class_folders:
        cls_path = os.path.join(DATASET_PATH, cls)
        class_counts[cls] = len([
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg", ".tif"))
        ])

    plt.bar(class_counts.keys(), class_counts.values(), color="#3498db")
    plt.title("Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.show()

    text_area.insert(tk.END, "✅ Preprocessing Completed\n")
    for cls, cnt in class_counts.items():
        text_area.insert(tk.END, f"{cls}: {cnt} images\n")



# TRAIN CNN

def train_model():
    global model

    if not DATASET_PATH:
        messagebox.showwarning("Warning", "Please upload dataset first")
        return

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(train_data, epochs=EPOCHS, validation_data=val_data)
    model.save(TRAINED_MODEL_PATH)

    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.legend()
    plt.title("Training Accuracy")
    plt.show()

    text_area.insert(tk.END, f"🧪 Model trained & saved as {TRAINED_MODEL_PATH}\n")



# PREDICTION

def predict_image():
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        messagebox.showerror("Error", "Pretrained model not found")
        return

    img_path = filedialog.askopenfilename(initialdir='test_images',
        filetypes=[("Image Files", "*.jpg *.png *.jpeg *.tif")]
    )
    if not img_path:
        return

    model = load_model(PRETRAINED_MODEL_PATH)

    img = load_img(img_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)[0][0]

    result = "🛑 MORPHED FACE" if pred < 0.5 else "✅ REAL FACE"

    popup = tk.Toplevel()
    popup.title("Prediction Result")
    popup.configure(bg="white")

    show_img = Image.open(img_path).resize((260, 260))
    show_img = ImageTk.PhotoImage(show_img)

    tk.Label(popup, image=show_img, bg="white").pack(pady=10)
    popup.image = show_img

    tk.Label(
        popup,
        text=f"{result}\nConfidence: {pred:.2f}",
        font=("Segoe UI", 16, "bold"),
        bg="white"
    ).pack(pady=10)



# LOGIN FUNCTION

def login():
    if username.get() == "admin" and password.get() == "admin":
        login_win.destroy()
        main_ui()
    else:
        messagebox.showerror("Login Failed", "Invalid credentials")



# LOGIN WINDOW

login_win = tk.Tk()
login_win.title("Login")
login_win.geometry("420x300")
login_win.configure(bg="#2c3e50")

tk.Label(
    login_win,
    text="🔐 Admin Login",
    font=("Segoe UI", 20, "bold"),
    bg="#2c3e50",
    fg="white"
).pack(pady=30)

username = tk.StringVar()
password = tk.StringVar()

tk.Entry(login_win, textvariable=username, font=("Segoe UI", 12)).pack(pady=5)
tk.Entry(login_win, textvariable=password, show="*", font=("Segoe UI", 12)).pack(pady=5)

tk.Button(
    login_win,
    text="LOGIN",
    command=login,
    bg="#27ae60",
    fg="white",
    font=("Segoe UI", 13, "bold"),
    width=15
).pack(pady=25)

login_win.mainloop()
