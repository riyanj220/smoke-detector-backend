import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.ml_model import model_handler
from contextlib import asynccontextmanager
import io

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    print("Server is starting up...")
    model_handler.load_model()
    print("Model loaded. Server is ready.")
    try:
        yield
    finally:
        # shutdown cleanup (if needed)
        print("Server is shutting down...")

# Initialize the FastAPI app with the lifespan manager
app = FastAPI(
    title="Smoke Detector API",
    description="An API to classify thermal images as 'smoke' or 'no_smoke'.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/", tags=["Health Check"])
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Welcome to the Smoke Detector API! Visit /docs for documentation."}

@app.post("/original-predict/", tags=["Prediction"])
async def predict_smoke(file: UploadFile = File(...)):
    """
    Receives a thermal image and predicts if it contains smoke.
    
    - **file**: The thermal image file to be analyzed.
    """
    # Ensure the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")
    
    try:
        # Read the image file as bytes
        image_bytes = await file.read()
        
        # Process the image and get a prediction
        prediction_result = model_handler.process_image_prediction(image_bytes)
        
        # Return the result as a JSON response
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": prediction_result["label"],
            "confidence": f"{prediction_result['confidence']:.4f}"
        })

    except Exception as e:
        # Handle any errors during prediction
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# This part is for running the app directly with `python app/main.py`
# but it's better to use `uvicorn app.main:app --reload` from the root.
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)