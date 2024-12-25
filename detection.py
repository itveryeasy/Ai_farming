import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained model
try:
    model = load_model('inception.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Preprocess function to resize the image
def preprocess_image(img):
    resized_img = cv2.resize(img, (224, 224))  # Resize the image to match the model's input shape
    normalized_img = resized_img / 255.0       # Normalize pixel values to [0, 1]
    expanded_img = np.expand_dims(normalized_img, axis=0)  # Add batch dimension
    return expanded_img

while True:
    # Read a frame from the webcam
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame from webcam.")
        break

    # Preprocess the image
    preprocessed_img = preprocess_image(img)

    # Perform inference
    try:
        results = model.predict(preprocessed_img, verbose=0)
    except Exception as e:
        print(f"Error during inference: {e}")
        break

    # Process the results (Assuming the model outputs class probabilities)
    predicted_class = np.argmax(results, axis=1)[0]
    confidence = np.max(results) * 100

    # Display the prediction on the frame
    label = f"Class: {predicted_class}, Confidence: {confidence:.2f}%"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame with prediction
    cv2.imshow('Webcam Inference', img)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

# README
# This script captures video from the webcam and performs real-time inference using a pre-trained TensorFlow model.
# Features:
# - Resizes webcam frames to match the model's input requirements.
# - Normalizes the image data for consistent inference results.
# - Displays real-time predictions including the predicted class and confidence score.
# Requirements:
# - OpenCV
# - TensorFlow/Keras
# Usage:
# 1. Ensure the model file 'inception.h5' is in the same directory as this script.
# 2. Run the script and allow access to the webcam.
# 3. Quit by pressing the 'q' key.
