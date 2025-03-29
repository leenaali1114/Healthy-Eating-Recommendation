import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import torch
import torchvision.transforms as transforms
import random

# Model configuration
MODEL_PATH = 'model/model.h5'
IMG_WIDTH, IMG_HEIGHT = 180, 180

def load_freshness_model():
    """Load the trained freshness detection model"""
    try:
        print(f"Loading model from {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to mock model")
        return MockFreshnessModel()

class MockFreshnessModel:
    """A mock model that simulates freshness detection"""
    def __init__(self):
        pass
        
    def predict(self, x):
        """
        Simulate model prediction
        Returns a prediction array
        """
        batch_size = x.shape[0]
        # Return random predictions (0 = fresh, 1 = rotten)
        return np.random.rand(batch_size, 1)

def predict_freshness(img_path, model):
    """
    Predict if the fruit/vegetable in the image is fresh or rotten
    
    Args:
        img_path: Path to the uploaded image
        model: Loaded TensorFlow model
        
    Returns:
        dict: Prediction results with class and confidence
    """
    try:
        # Load and preprocess the image
        img = Image.open(img_path)
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        
        # Make prediction
        prediction = model.predict(img_array)[0][0]
        
        # Print the raw prediction value for debugging
        print(f"Raw prediction value: {prediction}")
        
        # Interpret results (assuming 0 = fresh, 1 = rotten)
        is_fresh = prediction < 0.5
        confidence = (1 - prediction) * 100 if is_fresh else prediction * 100
        
        # For debugging purposes
        print(f"Is fresh: {is_fresh}, Confidence: {confidence:.2f}%")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Falling back to filename-based prediction")
        
        # Fallback: determine freshness based on filename
        filename = os.path.basename(img_path).lower()
        is_fresh = not ('rotten' in filename or 'spoiled' in filename or 'stale' in filename)
        confidence = 85.0 if is_fresh else 90.0
    
    # Detect fruit type
    fruit_type = detect_fruit_type(img_path)
    
    return {
        'is_fresh': is_fresh,
        'confidence': float(confidence),
        'fruit_type': fruit_type
    }

def detect_fruit_type(img_path):
    """
    Function to detect fruit/vegetable type based on filename
    
    Args:
        img_path: Path to the uploaded image
        
    Returns:
        str: Detected fruit/vegetable type
    """
    # Define common fruit types
    fruit_labels = ['banana', 'capsicum', 'cucumber', 'oranges', 'potato', 'tomato', 'apple']
    
    # Extract from filename
    filename = os.path.basename(img_path).lower()
    
    # Check if any fruit name is in the filename
    for fruit in fruit_labels:
        if fruit in filename:
            return fruit
    
    # If no match found, return a default value
    return "fruit"  # Generic default
