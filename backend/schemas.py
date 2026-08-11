from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

# User schemas
class UserBase(BaseModel):
    email: str
    full_name: str
    role: str = Field(..., description="User role: patient, doctor, or admin")

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

# Symptom and clinical data schemas
class SymptomData(BaseModel):
    # Demographic
    age: int = Field(..., ge=18, le=100, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="0: Female, 1: Male")

    # Symptoms (1-10 scale for severity)
    fatigue_level: int = Field(1, ge=1, le=10, description="Fatigue severity (1-10)")
    alcohol_consumption: float = Field(0, ge=0, le=50, description="Alcohol consumption (drinks/week)")

    # Physical symptoms
    weight_loss_kg: float = Field(0, ge=0, le=20, description="Weight loss in kg")
    abdominal_swelling: bool = Field(False, description="Abdominal swelling present")
    appetite_loss: bool = Field(False, description="Loss of appetite")
    jaundice: bool = Field(False, description="Jaundice present")
    fever: bool = Field(False, description="Fever present")

    # Clinical examination (0-3 scale)
    ascites: Union[int, bool] = Field(False, description="Ascites present (Y/N) or severity 0-3")
    hepatomegaly: Union[int, bool] = Field(False, description="Hepatomegaly present (Y/N) or severity 0-3")
    spiders: Union[int, bool] = Field(False, description="Spider angiomas present (Y/N) or severity 0-3")
    edema: int = Field(0, ge=0, le=2, description="Edema: 0=None(N), 1=Controlled(S), 2=Overt(Y)")

    # Laboratory tests (ranges match Mayo Clinic PBC trial dataset)
    bilirubin: float = Field(1.4, ge=0.1, le=28.0, description="Bilirubin (mg/dL)")
    cholesterol: float = Field(310, ge=50, le=1800, description="Cholesterol (mg/dL)")
    albumin: float = Field(3.5, ge=1.0, le=5.0, description="Albumin (g/dL)")
    copper: float = Field(73, ge=4, le=590, description="Copper (μg/dL)")
    alk_phos: float = Field(1259, ge=35, le=14000, description="Alkaline phosphatase (IU/L)")
    ast: float = Field(115, ge=5, le=460, description="AST/SGOT (IU/L)")
    triglycerides: float = Field(108, ge=20, le=600, description="Triglycerides (mg/dL)")
    platelets: float = Field(251, ge=20, le=600, description="Platelets (×10³/µL) — enter e.g. 251 for 251,000/µL")
    prothrombin: float = Field(10.6, ge=8.0, le=18.0, description="Prothrombin time (seconds)")

# Prediction result schemas
class PredictionResult(BaseModel):
    stage: str = Field(..., description="Predicted cirrhosis stage (0-4)")
    stage_description: str = Field(..., description="Human-readable stage description")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0-1)")
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for each stage")
    recommendations: List[str] = Field(default_factory=list, description="Treatment recommendations")
    follow_up_required: bool = Field(False, description="Whether follow-up is required")
    specialist_referral: bool = Field(False, description="Whether specialist referral needed")

# Explanation schemas
class SHAPExplanation(BaseModel):
    feature_importance: Dict[str, float] = Field(..., description="SHAP feature importance scores")
    base_value: float = Field(..., description="SHAP base value")
    feature_names: List[str] = Field(..., description="Feature names")
    shap_values: List[float] = Field(..., description="SHAP values for features")

class GradCAMExplanation(BaseModel):
    heatmap: str = Field(..., description="Base64 encoded heatmap image")
    overlay: str = Field(..., description="Base64 encoded overlay image")
    target_regions: List[Dict[str, Any]] = Field(..., description="Identified regions of interest")
    confidence_map: Dict[str, float] = Field(..., description="Confidence scores for regions")

class ExplanationResult(BaseModel):
    shap_explanation: Optional[SHAPExplanation] = None
    gradcam_explanation: Optional[GradCAMExplanation] = None
    natural_language_explanation: str = Field(..., description="Human-readable explanation")
    key_factors: List[str] = Field(..., description="Key factors influencing prediction")
    confidence_interpretation: str = Field(..., description="Confidence score interpretation")

# Diagnosis schemas
class DiagnosisRequest(BaseModel):
    symptoms: SymptomData
    include_image_analysis: bool = Field(False, description="Whether to include image analysis")

class DiagnosisResponse(BaseModel):
    diagnosis_id: int
    prediction: PredictionResult
    explanations: ExplanationResult
    report_url: str = Field(..., description="URL to download PDF report")
    processing_time_seconds: float = Field(0.0, description="Time taken for diagnosis")
    completed: bool = Field(False, description="Whether the conversation is complete")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    session_type: str = Field("medical_qa", description="Session type: symptom_collection, medical_qa, education")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ChatResponse(BaseModel):
    session_id: str
    message: str
    next_question: Optional[str] = None
    completed: bool = False
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ChatMessage(BaseModel):
    type: str = Field(..., description="Message type: user or bot")
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ChatHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage]

# Report schemas
class ReportData(BaseModel):
    patient_name: str
    patient_id: str
    diagnosis_date: datetime
    symptoms: SymptomData
    prediction: PredictionResult
    explanations: ExplanationResult
    recommendations: List[str]
    follow_up_schedule: Optional[str] = None

# Analytics schemas
class DiagnosisStats(BaseModel):
    total_diagnoses: int
    stage_distribution: Dict[str, int]
    average_confidence: float
    risk_level_distribution: Dict[str, int]
    time_period: str

class UserStats(BaseModel):
    total_users: int
    active_users: int
    diagnoses_per_user: float
    user_role_distribution: Dict[str, int]

# API response schemas
class APIResponse(BaseModel):
    success: bool = True
    message: Optional[str] = None
    data: Optional[Any] = None
    errors: Optional[List[str]] = None

class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    page_size: int
    pages: int

# Error schemas
class ErrorDetail(BaseModel):
    field: str
    message: str

class ValidationErrorResponse(BaseModel):
    error: str = "Validation Error"
    details: List[ErrorDetail]

# Health check schema
class HealthStatus(BaseModel):
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, str] = Field(default_factory=dict, description="Status of dependent services")

# Model management schemas
class ModelInfo(BaseModel):
    name: str
    version: str
    type: str
    framework: str
    accuracy: Optional[float] = None
    is_active: bool
    created_at: datetime

class ModelUpdateRequest(BaseModel):
    model_name: str
    version: str
    is_active: bool = True

# Audit log schemas
class AuditLogEntry(BaseModel):
    user_id: Optional[int]
    action: str
    resource: str
    ip_address: str
    user_agent: Optional[str]
    success: bool
    timestamp: datetime
    details: Optional[Dict[str, Any]]
