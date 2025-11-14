# app/ml_model/model_handler.py

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# --- 1. Model Configuration ---
# Define all model-specific settings here.
# This makes it easy to add/remove models.
#
# !! IMPORTANT !!
# Update the 'img_height', 'img_width', etc., for your 'dual_path'
# model if they are different from the original.
#
MODEL_CONFIGS = {
    "original": {
        "filename": "my_smoke_detector_model.keras",
        "img_height": 100,
        "img_width": 100,
        "norm_factor": 255.0,
        "threshold": 0.5
    },
    "dual_path": {
        "filename": "my_DUAL_PATH_model.keras",
        "img_height": 100,  # <-- Change if different
        "img_width": 100,   # <-- Change if different
        "norm_factor": 255.0, # <-- Change if different
        "threshold": 0.5      # <-- Change if different
    }
}

# Global dictionary to hold the loaded model objects
models = {}

def load_models():
    """
    Loads ALL Keras models defined in MODEL_CONFIGS
    into the global 'models' dictionary.
    Called once at server startup.
    """
    global models
    
    # Get the directory where this script is located
    model_dir = os.path.dirname(__file__)

    for model_name, config in MODEL_CONFIGS.items():
        model_path = os.path.join(model_dir, config["filename"])

        if not os.path.exists(model_path):
            print(f"Error: Model file for '{model_name}' not found at {model_path}")
            continue

        try:
            print(f"Loading model: '{model_name}'...")
            models[model_name] = tf.keras.models.load_model(model_path)
            print(f"Model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")

def preprocess_image(image_bytes: bytes, img_width: int, img_height: int, norm_factor: float) -> np.ndarray:
    """
    Preprocesses the input image bytes to match a model's
    expected input format.
    """
    # Open the image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB (handles images with alpha channels, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Resize to the target dimensions
    image = image.resize((img_width, img_height))
    
    # Convert image to a numpy array
    image_array = np.array(image)
    
    # Normalize the image
    image_array = image_array / norm_factor
    
    # Expand dimensions to create a "batch" of 1
    # Model expects (1, height, width, channels)
    image_batch = np.expand_dims(image_array, axis=0)
    
    return image_batch

def process_image_prediction(image_bytes: bytes, model_name: str) -> dict:
    """
    Runs the full prediction pipeline for a *specific model*.
    
    1. Gets the correct model and config
    2. Preprocesses the image
    3. Runs inference
    4. Post-processes the result
    """
    global models
    
    # 1. Check if model and config exist
    if model_name not in models:
        raise RuntimeError(f"Model '{model_name}' is not loaded.")
    if model_name not in MODEL_CONFIGS:
        raise RuntimeError(f"Model config for '{model_name}' not found.")

    model = models[model_name]
    config = MODEL_CONFIGS[model_name]

    # 2. Preprocess the image using its specific config
    preprocessed_batch = preprocess_image(
        image_bytes,
        config["img_width"],
        config["img_height"],
        config["norm_factor"]
    )
    
    # 3. Run prediction
    prediction = model.predict(preprocessed_batch)
    
    # Get the raw sigmoid output (confidence score)
    confidence = prediction[0][0]
    
    # 4. Post-process the result
    if confidence > config["threshold"]:
        label = "smoke"
    else:
        label = "no_smoke"
        
    return {
        "label": label,
        "confidence": float(confidence),
        "model_used": model_name
    }