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
    # Call the new plural function
    model_handler.load_models() 
    print("All models loaded. Server is ready.")
    try:
        yield
    finally:
        # shutdown cleanup (if needed)
        print("Server is shutting down...")

# Initialize the FastAPI app with the lifespan manager
app = FastAPI(
    title="Smoke Detector API",
    description="An API to classify thermal images using multiple models.",
    version="2.0.0",  # Updated version
    lifespan=lifespan,
)

@app.get("/", tags=["Health Check"])
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Welcome to the Smoke Detector API! Visit /docs for documentation."}

@app.post("/predict/original/", tags=["Prediction"])
async def predict_original_model(file: UploadFile = File(...)):
    """
    Receives a thermal image and predicts smoke using the **Original Model**.
    
    - **file**: The thermal image file to be analyzed.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")
    
    try:
        image_bytes = await file.read()
        
        # Call the handler and specify "original" model
        prediction_result = model_handler.process_image_prediction(
            image_bytes, 
            model_name="original"
        )
        
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": prediction_result["label"],
            "confidence": f"{prediction_result['confidence']:.4f}",
            "model_used": prediction_result["model_used"]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/predict/dual-path/", tags=["Prediction"])
async def predict_dual_path_model(file: UploadFile = File(...)):
    """
    Receives a thermal image and predicts smoke using the **Dual Path Model**.
    
    - **file**: The thermal image file to be analyzed.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")
    
    try:
        image_bytes = await file.read()
        
        # Call the handler and specify "dual_path" model
        prediction_result = model_handler.process_image_prediction(
            image_bytes, 
            model_name="dual_path"
        )
        
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": prediction_result["label"],
            "confidence": f"{prediction_result['confidence']:.4f}",
            "model_used": prediction_result["model_used"]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# This part is for running the app directly
if __name__ == "__main__":
    # Note: Correct way to run from project root is:
    # uvicorn app.main:app --reload
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)