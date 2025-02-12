import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load the trained model
model = load_model("hair_gender_classification_model.h5")

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_gender(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    gender, age = "Male", 25  # Dummy values (Replace with actual prediction logic)
    
    # Assuming model output has probability scores for long/short hair
    long_hair_prob = prediction[0][0]
    
    if 20 <= age <= 30:
        if long_hair_prob > 0.5:
            gender = "Female"
        else:
            gender = "Male"
    else:
        gender = "Predicted gender based on another model"
    return gender

def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        gender = predict_gender(file_path)
        result_label.config(text=f"Predicted Gender: {gender}")
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

# Create GUI
top = tk.Tk()
top.geometry("400x400")
top.title("Long Hair Identification")

btn = tk.Button(top, text="Upload Image", command=upload_and_predict)
btn.pack()

result_label = tk.Label(top, text="Prediction Result", font=("Arial", 12))
result_label.pack()

panel = tk.Label(top)
panel.pack()

top.mainloop()
