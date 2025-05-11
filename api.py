from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from predict import CardiacPredictor
from typing import Optional
import uvicorn
from pydantic import BaseModel
import os

predictor = CardiacPredictor()

class PredictionResult(BaseModel):
    class_name: str
    confidence: float
    explanation: str
    gradcam: Optional[str] = None

app = FastAPI(
    title="Cardiac Scan Analysis API",
    description="API for detecting cardiac abnormalities from medical scans",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
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
    try:
        if not scan.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image (JPEG/PNG)")
        image_data = await scan.read()
        result = predictor.predict(image_data)
        if not include_gradcam and 'gradcam' in result:
            result.pop('gradcam')
        return PredictionResult(**result)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, log_level="info")
