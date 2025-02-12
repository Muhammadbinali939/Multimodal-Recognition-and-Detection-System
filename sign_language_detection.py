import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
import time

# Load pre-trained model for sign language recognition
model = load_model("sign_language_model.h5")

# Define labels
labels = {0: "Hello", 1: "Thank You", 2: "Yes", 3: "No", 4: "Help"}

# Function to check if the current time is within the allowed period
def is_within_time():
    current_time = datetime.now().time()
    start_time = datetime.strptime("18:00:00", "%H:%M:%S").time()
    end_time = datetime.strptime("22:00:00", "%H:%M:%S").time()
    return start_time <= current_time <= end_time

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame if within allowed time
    if is_within_time():
        # Preprocess the frame
        img = cv2.resize(frame, (128, 128))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = model.predict(img)
        predicted_label = labels[np.argmax(prediction)]
        
        # Display result
        cv2.putText(frame, f"Prediction: {predicted_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv2.putText(frame, "Model inactive. Allowed Time: 6 PM - 10 PM", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Sign Language Detection", frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
