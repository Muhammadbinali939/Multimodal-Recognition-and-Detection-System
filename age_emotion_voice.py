import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import soundfile as sf
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from keras.models import load_model
import python_speech_features as psf
import joblib

# Load pre-trained models
age_model = load_model('age_detection_model.h5')  # Train an age detection model beforehand
emotion_model = load_model('emotion_detection_model.h5')  # Train an emotion classification model
scaler = joblib.load('scaler.pkl')  # StandardScaler for feature normalization

# Function to extract MFCC features
def extract_features(audio_file):
    signal, sample_rate = librosa.load(audio_file, sr=22050)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

# Function to predict age
def predict_age(audio_file):
    features = extract_features(audio_file)
    features = scaler.transform([features])  # Normalize
    age_prediction = age_model.predict(features)
    predicted_age = int(age_prediction[0][0])  # Assuming regression model output
    return predicted_age

# Function to predict emotion
def predict_emotion(audio_file):
    features = extract_features(audio_file)
    features = scaler.transform([features])
    emotion_prediction = emotion_model.predict(features)
    emotion_labels = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprised']  # Example emotions
    predicted_emotion = emotion_labels[np.argmax(emotion_prediction)]
    return predicted_emotion

# Function to check gender (Assume a gender classification model is available)
def check_gender(audio_file):
    return 'male'  # Placeholder, implement a gender classification model

# GUI Functionality
def process_audio():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
    if not file_path:
        return
    
    gender = check_gender(file_path)
    if gender != 'male':
        messagebox.showerror("Invalid Input", "Upload male voice")
        return
    
    predicted_age = predict_age(file_path)
    if predicted_age > 60:
        predicted_emotion = predict_emotion(file_path)
        result_label.config(text=f"Age: {predicted_age}\nSenior Citizen\nEmotion: {predicted_emotion}")
    else:
        result_label.config(text=f"Age: {predicted_age}")

# GUI Implementation
root = tk.Tk()
root.title("Age and Emotion Detection from Voice")
root.geometry("400x300")

title_label = tk.Label(root, text="Upload an Audio File", font=("Arial", 14))
title_label.pack(pady=10)

upload_button = tk.Button(root, text="Select Audio File", command=process_audio, font=("Arial", 12))
upload_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
