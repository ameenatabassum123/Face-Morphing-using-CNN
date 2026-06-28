# Face Morphing Detection using CNN

This repository contains a **Deep Learning-based Face Morphing Detection System** featuring a user-friendly graphical dashboard built using Python, TensorFlow/Keras, and Tkinter. The system trains a Convolutional Neural Network (CNN) to distinguish between **Real (Normal)** and **Morphed** face images.

---

## 🚀 Key Features

*   **GUI Dashboard:** An interactive desktop application to manage dataset loading, data visualization, training, and predicting.
*   **CNN Model Training:** Train a custom Convolutional Neural Network from scratch with dynamic plotting of training accuracy/loss.
*   **Real-time Predictions:** Load a pre-trained model and predict whether a selected image is a **Real Face** or a **Morphed Face** with confidence scores.
*   **Visual Logs:** Real-time logging of actions, dataset size, and preprocessing steps inside the Tkinter dashboard.

---

## 📂 Project Structure

```text
face-morphing-project/
│
├── .gitignore                      # Git configuration to ignore system and IDE files
├── README.md                       # Project documentation (this file)
└── face/
    └── face/
        ├── app.py                  # Main GUI dashboard application (Upload, Preprocess, Train, Predict)
        ├── test.py                 # Alternate/simplified GUI application for predictions
        ├── train_face_morphing_cnn.py # Script for standalone training & testing
        ├── req.txt                 # Python package dependencies
        │
        ├── face_morphing_cnn.h5    # Pre-trained CNN weights file
        ├── face_morphing_cnn_new.h5# New model weights saved after running train_face_morphing_cnn.py
        ├── present_trained.h5      # Model weights saved after training via the GUI dashboard
        │
        ├── dataset2/               # Dataset directory used for training
        │   ├── morphing/           # Subdirectory containing morphed face images
        │   └── normal/             # Subdirectory containing normal (real) face images
        │
        ├── test_images/            # Folder containing sample images for testing predictions
        └── Figure_1.png            # Sample training performance visualization
```

---

## 🧠 CNN Architecture

The model uses a sequential CNN structure optimized for binary classification (Real vs. Morphed):

1.  **Conv2D Layer (32 filters, 3x3 kernel, ReLU)** + **MaxPooling2D (2x2)**
2.  **Conv2D Layer (64 filters, 3x3 kernel, ReLU)** + **MaxPooling2D (2x2)**
3.  **Conv2D Layer (128 filters, 3x3 kernel, ReLU)** + **MaxPooling2D (2x2)**
4.  **Flatten Layer** to convert 2D feature maps to 1D vector
5.  **Dense Layer (128 neurons, ReLU)** with **Dropout (0.5)** to prevent overfitting
6.  **Dense Output Layer (1 neuron, Sigmoid activation)** for binary classification

---

## 🛠️ Setup & Installation

### Prerequisites
*   **Python:** version `3.11.2` (recommended)
*   **Git** (for version control)

### Step 1: Clone the repository
```bash
git clone https://github.com/ameenatabassum123/Face-Morphing-using-CNN.git
cd Face-Morphing-using-CNN/face/face
```

### Step 2: Install dependencies
Install the required packages using the requirements file:
```bash
pip install -r req.txt
```
*Note: Make sure your tensorflow and keras match version `2.15.0` as defined in `req.txt`.*

---

## 💻 How to Run the Applications

### 1. Main Dashboard (`app.py`)
Run the main graphical interface:
```bash
python app.py
```
*   **Login Credentials:**
    *   **Username:** `admin`
    *   **Password:** `admin`
*   **Features:**
    *   **Upload Dataset:** Select your training dataset folder (e.g., `dataset2`).
    *   **Preprocess & Graph:** Visualize the class distribution of real vs. morphed faces.
    *   **Train CNN Model:** Train the network for 5 epochs and view accuracy plots. Saved to `present_trained.h5`.
    *   **Predict Image:** Select an image from `test_images/` to verify if it is morphed or real.

### 2. Alternative Predictor (`test.py`)
A lightweight tool dedicated to loading the `face_morphing_cnn.h5` model and running predictions:
```bash
python test.py
```
*   **Login Credentials:** Same as above (`admin` / `admin`).
*   Click **Select Image** to load a test file and view the predicted label immediately on the screen.

### 3. Standalone Training Script (`train_face_morphing_cnn.py`)
Run this script to train the model directly via the console without the GUI:
```bash
python train_face_morphing_cnn.py
```
*   Trains on the dataset specified in `DATASET_PATH` for 15 epochs.
*   Saves the final model to `face_morphing_cnn_new.h5`.
*   Plots training vs. validation accuracy and loss graphs using Matplotlib.
