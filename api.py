"""
Cardiac Scan Analysis API
FastAPI backend for serving model predictions
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from predict import CardiacPredictor
from typing import Optional
import uvicorn
from pydantic import BaseModel
import os

# Initialize predictor
predictor = CardiacPredictor()

# Define response model
class PredictionResult(BaseModel):
    class_name: str
    confidence: float
    explanation: str
    gradcam: Optional[str] = None

# Initialize FastAPI
app = FastAPI(
    title="Cardiac Scan Analysis API",
    description="API for detecting cardiac abnormalities from medical scans",
    version="1.0.0"
)

# CORS configuration (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    """Service health check endpoint"""
    return {
        "status": "active",
        "model": "CardiacResNet",
        "ready": os.path.exists("best_model.pth")
    }

@app.post("/predict", response_model=PredictionResult)
async def predict_scan(
    scan: UploadFile = File(..., description="Cardiac scan image (CT/MRI)"),
    include_gradcam: bool = False
):
    """
    Analyze cardiac scan and return prediction
    
    Parameters:
    - scan: Image file (JPEG/PNG)
    - include_gradcam: Whether to return Grad-CAM visualization
    
    Returns:
    - Prediction result with explanation
    """
    try:
        # Verify image file
        if not scan.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image (JPEG/PNG)")
        
        # Read image data
        image_data = await scan.read()
        
        # Make prediction
        result = predictor.predict(image_data)
        
        # Optionally remove gradcam to reduce payload
        if not include_gradcam:
            result.pop('gradcam')
        
        return result
    
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, log_level="info", reload=True)