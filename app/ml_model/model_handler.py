import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# --- CRITICAL: MUST MATCH YOUR TRAINING ---
# Change these values to match the exact input size
# and normalization you used in Google Colab.
IMG_HEIGHT = 100
IMG_WIDTH = 100
NORMALIZATION_FACTOR = 255.0  # (e.g., 255.0 or 127.5)
PREDICTION_THRESHOLD = 0.5    # (0.5 for binary classification)
# -------------------------------------------

# Global variable to hold the loaded model
model = None
MODEL_FILENAME = 'my_smoke_detector_model.keras'

def load_model():
    """
    Loads the Keras model into the global 'model' variable.
    This function is called once at server startup.
    """
    global model
    
    # Construct the full path to the model file
    # This assumes the script is run from the project root (smoke-detector-api/)
    model_path = os.path.join(os.getcwd(), 'app', 'ml_model', MODEL_FILENAME)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocesses the input image bytes to match the model's
    expected input format.
    """
    # Open the image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB (handles images with alpha channels, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Resize to the target dimensions
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Convert image to a numpy array
    image_array = np.array(image)
    
    # Normalize the image (e.g., divide by 255.0)
    image_array = image_array / NORMALIZATION_FACTOR
    
    # Expand dimensions to create a "batch" of 1
    # Model expects (1, height, width, channels)
    image_batch = np.expand_dims(image_array, axis=0)
    
    return image_batch

def process_image_prediction(image_bytes: bytes) -> dict:
    """
    Runs the full prediction pipeline:
    1. Preprocesses the image
    2. Runs inference
    3. Post-processes the result
    """
    global model
    if model is None:
        raise RuntimeError("Model is not loaded. Please restart the server.")

    # 1. Preprocess the image
    preprocessed_batch = preprocess_image(image_bytes)
    
    # 2. Run prediction
    prediction = model.predict(preprocessed_batch)
    
    # Get the raw sigmoid output (confidence score)
    # This assumes a single output neuron for binary classification
    confidence = prediction[0][0]
    
    # 3. Post-process the result
    if confidence > PREDICTION_THRESHOLD:
        label = "smoke"
    else:
        label = "no_smoke"
        
    return {
        "label": label,
        "confidence": float(confidence)
    }