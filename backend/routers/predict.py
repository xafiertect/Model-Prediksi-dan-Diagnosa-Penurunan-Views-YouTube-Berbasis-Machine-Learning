import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any

from schemas.prediction import PredictionInput, PredictionOutput

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)

MODEL_PATH = os.getenv("MODEL_PATH", "./models")
MODEL_FILE = os.path.join(MODEL_PATH, "rf_classifier.pkl")
SCALER_FILE = os.path.join(MODEL_PATH, "scaler.pkl")

# Load models safely at startup if possible, or lazy load
# For robust web servers, lazy loading or checking existence is better if models update
def load_ml_assets():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        raise FileNotFoundError("Model or Scaler file not found. Please train and save models first.")
    
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, scaler

@router.post("/", response_model=PredictionOutput, responses={503: {"description": "Model not ready"}})
async def predict_performance(input_data: PredictionInput):
    try:
        model, scaler = load_ml_assets()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
         raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading models: {str(e)}"
        )
    
    try:
        # Prepare the features DataFrame exactly as trained
        features = pd.DataFrame([{
            'views': input_data.views,
            'ctr': input_data.ctr,
            'impressions': input_data.impressions,
            'avg_view_duration': input_data.avg_view_duration,
            'engagement_rate': input_data.engagement_rate
        }])

        # Note: In a real scenario, the scaler might expect more features depending on notebook 07.
        # This handles the exact 5 features given in the prompt.
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction_num = model.predict(features_scaled)[0]
        # In sklearn, predict_proba returns array of shape (n_samples, n_classes)
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(max(probabilities))
        
        # Map number back to label (Assuming 0=Declining, 1=Normal, 2=Viral based on Notebook 07)
        label_map = {0: "Declining", 1: "Normal", 2: "Viral"}
        status_label = label_map.get(int(prediction_num), "Normal")

        # Mock predicted views logic (could be another model, e.g., xgb_regressor)
        # We will use a simple heuristic if we don't have a separate regressor loaded
        predicted_views = int(input_data.views * (1.1 + (input_data.ctr / 100)))

        # Simple recommendation engine based on status
        recommendation = "Keep up the good work!"
        if status_label == "Declining":
            recommendation = "Consider improving thumbnail design and hook within the first 10 seconds to boost CTR and retention."
        elif status_label == "Viral":
            recommendation = "Excellent performance! Engage with comments and consider a follow-up video on the same topic."

        return PredictionOutput(
            status=status_label,
            confidence=confidence,
            predicted_views=predicted_views,
            recommendation=recommendation
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )
