from pydantic import BaseModel, Field, field_validator
from typing import Literal
from datetime import datetime

class PredictionInput(BaseModel):
    views: int = Field(ge=0, description="Total views")
    ctr: float = Field(ge=0, le=100, description="Click-through rate (percentage)")
    impressions: int = Field(ge=0, description="Total impressions")
    avg_view_duration: float = Field(ge=0, description="Average view duration in seconds")
    engagement_rate: float = Field(ge=0, le=100, description="Engagement rate (percentage)")

    @field_validator('ctr', 'engagement_rate')
    @classmethod
    def check_percentage_range(cls, v, info):
        if not (0 <= v <= 100):
            raise ValueError(f"{info.field_name} must be between 0 and 100")
        return v

class PredictionOutput(BaseModel):
    status: Literal["Viral", "Normal", "Declining"]
    confidence: float = Field(ge=0, le=1, description="Confidence probability of the prediction")
    predicted_views: int = Field(description="Predicted future views")
    recommendation: str = Field(description="Actionable recommendation based on prediction")

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
