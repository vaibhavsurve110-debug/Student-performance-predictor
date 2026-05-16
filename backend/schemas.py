"""
schemas.py — Pydantic models for FastAPI request/response validation.
Covers all 32 input features of the UCI Student Performance dataset.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class StudentInput(BaseModel):
    # Demographic
    school: str = Field(default="GP", description="School: 'GP' or 'MS'")
    sex: str = Field(default="M", description="Sex: 'M' or 'F'")
    age: int = Field(default=17, ge=15, le=22, description="Age (15-22)")
    address: str = Field(default="U", description="Address: 'U' (urban) or 'R' (rural)")
    famsize: str = Field(default="GT3", description="Family size: 'LE3' or 'GT3'")
    Pstatus: str = Field(default="T", description="Parent cohabitation: 'T' (together) or 'A' (apart)")

    # Family background
    Medu: int = Field(default=2, ge=0, le=4, description="Mother education (0-4)")
    Fedu: int = Field(default=2, ge=0, le=4, description="Father education (0-4)")
    Mjob: str = Field(default="other", description="Mother's job")
    Fjob: str = Field(default="other", description="Father's job")
    reason: str = Field(default="course", description="Reason for school choice")
    guardian: str = Field(default="mother", description="Guardian: 'mother', 'father', 'other'")

    # Study habits
    traveltime: int = Field(default=1, ge=1, le=4, description="Travel time to school (1-4)")
    studytime: int = Field(default=2, ge=1, le=4, description="Weekly study time (1-4)")
    failures: int = Field(default=0, ge=0, le=3, description="Number of past failures (0-3)")

    # Support
    schoolsup: str = Field(default="no", description="School extra support: 'yes'/'no'")
    famsup: str = Field(default="yes", description="Family educational support: 'yes'/'no'")
    paid: str = Field(default="no", description="Extra paid classes: 'yes'/'no'")
    activities: str = Field(default="no", description="Extra-curricular activities: 'yes'/'no'")
    nursery: str = Field(default="yes", description="Attended nursery: 'yes'/'no'")
    higher: str = Field(default="yes", description="Wants higher education: 'yes'/'no'")
    internet: str = Field(default="yes", description="Internet access at home: 'yes'/'no'")
    romantic: str = Field(default="no", description="In romantic relationship: 'yes'/'no'")

    # Behavioural
    famrel: int = Field(default=4, ge=1, le=5, description="Family relationship quality (1-5)")
    freetime: int = Field(default=3, ge=1, le=5, description="Free time after school (1-5)")
    goout: int = Field(default=3, ge=1, le=5, description="Going out with friends (1-5)")
    Dalc: int = Field(default=1, ge=1, le=5, description="Workday alcohol consumption (1-5)")
    Walc: int = Field(default=1, ge=1, le=5, description="Weekend alcohol consumption (1-5)")
    health: int = Field(default=3, ge=1, le=5, description="Current health status (1-5)")
    absences: int = Field(default=4, ge=0, le=93, description="Number of school absences")

    # Prior grades (most predictive features)
    G1: int = Field(default=10, ge=0, le=20, description="First period grade (0-20)")
    G2: int = Field(default=10, ge=0, le=20, description="Second period grade (0-20)")


class PredictionResponse(BaseModel):
    student_id: Optional[str] = None
    predicted_class: int
    predicted_label: str
    confidence: float
    probabilities: dict
    top_features: List[dict]
    shap_explanation: Optional[dict] = None
    ai_counselor_plan: Optional[str] = None
    timestamp: str


class ModelMetrics(BaseModel):
    name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    notes: str


class FeatureImportance(BaseModel):
    feature: str
    importance: float
    rank: int


class HistoryRecord(BaseModel):
    id: str
    input: dict
    result: PredictionResponse
    timestamp: str
