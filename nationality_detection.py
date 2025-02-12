import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# Load trained model
model = load_model("nationality_detection_model.h5")

# Label mapping
nationalities = ["Indian", "American", "African", "Other"]

# Function to preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict nationality
def predict_nationality(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    nationality = nationalities[np.argmax(prediction)]
    return nationality

# GUI setup
root = tk.Tk()
root.title("Nationality Detection")
root.geometry("500x500")

label = Label(root, text="Upload an Image", font=("Arial", 14))
label.pack(pady=20)

# Function to handle image upload
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        nationality = predict_nationality(file_path)
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        result_label.config(text=f"Predicted Nationality: {nationality}")

upload_button = Button(root, text="Upload Image", command=upload_image)
upload_button.pack()

image_label = Label(root)
image_label.pack()

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack()

root.mainloop()
