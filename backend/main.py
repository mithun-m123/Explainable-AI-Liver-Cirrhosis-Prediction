from dotenv import load_dotenv
load_dotenv()

import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import uvicorn
import os
import json
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
import cv2
import numpy as np
from PIL import Image
import io
import base64
import joblib
import requests
import time
import sys
from pathlib import Path

# Add ai_pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'ai_pipeline'))

# Optional ML/DL imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torchvision.transforms as transforms
    from torchvision import models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Import local modules
from database import get_db, engine, SessionLocal, User, DiagnosisRecord, ChatSession, ChatMessage
from schemas import (
    UserCreate, UserLogin, DiagnosisRequest, DiagnosisResponse,
    ChatRequest, ChatResponse, PredictionResult, SymptomData
)
from auth import get_current_user, create_access_token, verify_token
from auth import require_role
from models_handler import HybridModelHandler
from xai_logic import XAIExplainer
from chatbot_api import MedicalChatbot
from pdf_gen import MedicalReportGenerator
from lab_ocr import extract_lab_values

# Import AI engines
try:
    from integration import create_ai_system
    AI_SYSTEM_AVAILABLE = True
except ImportError:
    AI_SYSTEM_AVAILABLE = False
    logger.warning("AI System integration not available")

try:
    from chatbot import create_chatbot
    RAG_CHATBOT_AVAILABLE = True
except ImportError:
    RAG_CHATBOT_AVAILABLE = False
    logger.warning("RAG Chatbot not available")


# Initialize FastAPI app
app = FastAPI(
    title="Liver Cirrhosis Detection API",
    description="AI-powered non-invasive liver cirrhosis detection system",
    version="1.0.0"
)

# CORS middleware (allow local dev frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
model_handler = HybridModelHandler()
xai_explainer = XAIExplainer()
chatbot_api = MedicalChatbot()
pdf_generator = MedicalReportGenerator()

# Initialize AI System Integration
ai_system = None
rag_chatbot = None

if AI_SYSTEM_AVAILABLE:
    try:
        ai_system = create_ai_system()
        logger.info("✓ AI System Integration loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load AI System: {e}")

if RAG_CHATBOT_AVAILABLE:
    try:
        rag_chatbot = create_chatbot(provider='local')  # Use local by default
        logger.info("✓ RAG Chatbot loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load RAG Chatbot: {e}")

# Security
security = HTTPBearer()

# Create database tables
from database import Base
Base.metadata.create_all(bind=engine)

@app.get("/")
async def root():
    return {"message": "Liver Cirrhosis Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}


@app.get("/patients")
async def list_patients(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Return a list of patient users (doctor/admin only)"""
    if current_user.role not in ["doctor", "admin"]:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    patients = db.query(User).filter(User.role == "patient").all()
    result = []
    for p in patients:
        result.append({
            "id": p.id,
            "patient_uid": p.patient_uid or f"PAT-{p.id:05d}",
            "email": p.email,
            "full_name": p.full_name,
            "is_active": p.is_active,
            "created_at": p.created_at.isoformat(),
        })
    return {"total": len(result), "patients": result}


@app.post("/patients/add")
async def add_patient_by_doctor(
    patient_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Doctor/admin adds a new patient directly."""
    if current_user.role not in ["doctor", "admin"]:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    full_name = patient_data.get("full_name", "").strip()
    email     = patient_data.get("email", "").strip()
    password  = patient_data.get("password", "liver1234")

    if not full_name:
        raise HTTPException(status_code=422, detail="Patient name is required")
    if not email:
        raise HTTPException(status_code=422, detail="Email is required")

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
    hashed = pwd_context.hash(password)

    last_user = db.query(User).order_by(User.id.desc()).first()
    next_id   = (last_user.id + 1) if last_user else 1
    uid       = f"PAT-{next_id:05d}"

    new_patient = User(
        email=email,
        hashed_password=hashed,
        full_name=full_name,
        role="patient",
        patient_uid=uid,
    )
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)

    return {
        "success": True,
        "patient": {
            "id": new_patient.id,
            "patient_uid": new_patient.patient_uid,
            "full_name": new_patient.full_name,
            "email": new_patient.email,
            "is_active": new_patient.is_active,
            "created_at": new_patient.created_at.isoformat(),
        }
    }


@app.post("/patients/{patient_id}/deactivate")
async def deactivate_patient(patient_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Mark a patient user as inactive (doctor/admin only)"""
    # Only allow doctors or admins
    if current_user.role not in ["doctor", "admin"]:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    # Prevent deactivating yourself
    if current_user.id == patient_id:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")

    patient = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    try:
        patient.is_active = False
        db.add(patient)
        db.commit()
        db.refresh(patient)
        return {"success": True, "patient_id": patient.id, "is_active": patient.is_active}
    except Exception as e:
        logger.error(f"Failed to deactivate patient {patient_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to deactivate patient")

@app.post("/auth/register", response_model=Dict[str, Any])
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        # Check if user exists
        db_user = db.query(User).filter(User.email == user.email).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Hash password
        pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
        hashed_password = pwd_context.hash(user.password)

        # Generate unique patient UID  e.g. PAT-00042
        last_user = db.query(User).order_by(User.id.desc()).first()
        next_id = (last_user.id + 1) if last_user else 1
        patient_uid = f"PAT-{next_id:05d}"

        # Create user
        db_user = User(
            email=user.email,
            hashed_password=hashed_password,
            full_name=user.full_name,
            role=user.role,
            patient_uid=patient_uid,
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)

        # Create access token
        access_token = create_access_token(data={"sub": db_user.email})

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": db_user.id,
                "patient_uid": db_user.patient_uid,
                "email": db_user.email,
                "full_name": db_user.full_name,
                "role": db_user.role
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/auth/login", response_model=Dict[str, Any])
async def login_user(user: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user"""
    try:
        # Find user
        db_user = db.query(User).filter(User.email == user.email).first()
        if not db_user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Verify password
        pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
        if not pwd_context.verify(user.password, db_user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Create access token
        access_token = create_access_token(data={"sub": db_user.email})

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": db_user.id,
                "email": db_user.email,
                "full_name": db_user.full_name,
                "role": db_user.role
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.get("/auth/me", response_model=Dict[str, Any])
async def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "is_active": current_user.is_active,
    }

@app.post("/predict", response_model=DiagnosisResponse)
async def predict_cirrhosis(
    background_tasks: BackgroundTasks,
    symptoms: str = Form(...),
    image: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Predict liver cirrhosis from symptoms and/or imaging"""
    started_at = time.perf_counter()
    try:
        # Parse symptoms
        symptom_payload = json.loads(symptoms)
        symptom_data = SymptomData(**symptom_payload)

        # Process image if provided
        image_data = None
        if image:
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            image_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Make prediction
        prediction_result = await model_handler.predict(
            symptoms=symptom_data,
            image=image_data
        )

        # Generate explanations
        explanations = await xai_explainer.explain_prediction(
            prediction_result=prediction_result,
            symptom_data=symptom_data,
            image_data=image_data
        )

        # Save diagnosis record
        symptom_dict = symptom_data.model_dump() if hasattr(symptom_data, "model_dump") else symptom_data.dict()
        prediction_dict = prediction_result.model_dump() if hasattr(prediction_result, "model_dump") else prediction_result.dict()
        explanation_dict = explanations.model_dump() if hasattr(explanations, "model_dump") else explanations.dict()
        diagnosis_record = DiagnosisRecord(
            user_id=current_user.id,
            symptoms=symptom_dict,
            prediction_result=prediction_dict,
            explanations=explanation_dict,
            confidence_score=prediction_result.confidence,
            risk_level=prediction_result.risk_level,
            recommended_actions=prediction_result.recommendations
        )

        if image_data is not None:
            # Convert image to base64 for storage
            _, buffer = cv2.imencode('.jpg', image_data)
            diagnosis_record.image_data = base64.b64encode(buffer).decode()

        db.add(diagnosis_record)
        db.commit()
        db.refresh(diagnosis_record)

        # Generate PDF report in background
        background_tasks.add_task(
            pdf_generator.generate_report,
            diagnosis_record.id,
            current_user.id
        )

        return DiagnosisResponse(
            diagnosis_id=diagnosis_record.id,
            prediction=prediction_result,
            explanations=explanations,
            report_url=f"/reports/{diagnosis_record.id}",
            processing_time_seconds=round(time.perf_counter() - started_at, 3)
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/diagnosis/predict", response_model=DiagnosisResponse)
async def diagnosis_predict_alias(
    background_tasks: BackgroundTasks,
    symptoms: str = Form(...),
    image: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return await predict_cirrhosis(background_tasks, symptoms, image, current_user, db)

@app.get("/history", response_model=List[Dict[str, Any]])
async def get_diagnosis_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 10
):
    """Get user's diagnosis history"""
    try:
        diagnoses = db.query(DiagnosisRecord).filter(
            DiagnosisRecord.user_id == current_user.id
        ).order_by(DiagnosisRecord.created_at.desc()).offset(skip).limit(limit).all()

        result = []
        for diagnosis in diagnoses:
            result.append({
                "id": diagnosis.id,
                "created_at": diagnosis.created_at,
                "prediction_result": diagnosis.prediction_result,
                "confidence_score": diagnosis.confidence_score,
                "risk_level": diagnosis.risk_level,
                "report_url": f"/reports/{diagnosis.id}.pdf"
            })

        return result
    except Exception as e:
        logger.error(f"History retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")

@app.get("/diagnosis/results/{diagnosis_id}", response_model=Dict[str, Any])
async def get_diagnosis_result(
    diagnosis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    diagnosis = db.query(DiagnosisRecord).filter(
        DiagnosisRecord.id == diagnosis_id,
        DiagnosisRecord.user_id == current_user.id
    ).first()
    if not diagnosis:
        raise HTTPException(status_code=404, detail="Diagnosis not found")
    return _frontend_prediction_payload(diagnosis)

def _frontend_prediction_payload(diagnosis: DiagnosisRecord) -> Dict[str, Any]:
    prediction = diagnosis.prediction_result or {}
    probabilities = prediction.get("probabilities", {})
    stage = prediction.get("stage", "stage_0")
    stage_probability = probabilities.get(stage, prediction.get("confidence", 0.0))
    stage_number = stage.split("_")[-1] if "_" in stage else stage
    return {
        "id": diagnosis.id,
        "diagnosis_id": diagnosis.id,
        "disease_stage": prediction.get("stage_description", f"Stage {stage_number}"),
        "risk_level": prediction.get("risk_level", diagnosis.risk_level),
        "cirrhosis_risk": round(sum(prob for key, prob in probabilities.items() if key in ["stage_2", "stage_3", "stage_4"]) * 100, 1),
        "confidence_score": prediction.get("confidence", diagnosis.confidence_score),
        "stage_probability": stage_probability,
        "recommendation": " ".join(prediction.get("recommendations", [])[:2]),
        "recommendations": prediction.get("recommendations", []),
        "probabilities": probabilities,
        "prediction": prediction,
        "explanations": diagnosis.explanations,
    }

@app.post("/chat/start", response_model=ChatResponse)
async def start_chat_session(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start a new chat session"""
    try:
        # Create chat session
        chat_session = ChatSession(
            user_id=current_user.id,
            session_type=request.session_type,
            context=json.dumps(request.context or {})
        )
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)

        # Get initial response from chatbot
        response = await chatbot_api.start_conversation(
            session_id=str(chat_session.id),
            user_info={
                "name": current_user.full_name,
                "medical_history": request.context.get("medical_history", [])
            }
        )

        # Save initial message
        chat_message = ChatMessage(
            session_id=chat_session.id,
            message_type="bot",
            content=response["message"],
            message_metadata=response.get("metadata", {})
        )
        db.add(chat_message)
        db.commit()

        return ChatResponse(
            session_id=str(chat_session.id),
            message=response["message"],
            next_question=response.get("next_question"),
            completed=False
        )
    except Exception as e:
        logger.error(f"Chat start error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start chat")

from pydantic import BaseModel

class ChatMessagePayload(BaseModel):
    session_id: str
    message: str

@app.post("/chat/message", response_model=ChatResponse)
async def send_chat_message(
    payload: ChatMessagePayload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send message to chat session"""
    session_id = payload.session_id
    message = payload.message
    try:
        # Get chat session
        chat_session = db.query(ChatSession).filter(
            ChatSession.id == int(session_id),
            ChatSession.user_id == current_user.id
        ).first()
        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        # Save user message
        user_message = ChatMessage(
            session_id=chat_session.id,
            message_type="user",
            content=message
        )
        db.add(user_message)
        db.commit()

        # Get chatbot response
        response = await chatbot_api.process_message(
            session_id=session_id,
            message=message,
            context=json.loads(chat_session.context)
        )

        # Save bot response
        bot_message = ChatMessage(
            session_id=chat_session.id,
            message_type="bot",
            content=response["message"],
            message_metadata=response.get("metadata", {})
        )
        db.add(bot_message)
        db.commit()

        return ChatResponse(
            session_id=session_id,
            message=response["message"],
            next_question=response.get("next_question"),
            completed=response.get("completed", False)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat message error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process message")

@app.get("/chat/history/{session_id}")
async def get_chat_history(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get chat session history"""
    try:
        # Verify session ownership
        chat_session = db.query(ChatSession).filter(
            ChatSession.id == int(session_id),
            ChatSession.user_id == current_user.id
        ).first()
        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        # Get messages
        messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == int(session_id)
        ).order_by(ChatMessage.created_at).all()

        return {
            "session_id": session_id,
            "messages": [
                {
                    "type": msg.message_type,
                    "content": msg.content,
                    "timestamp": msg.created_at,
                    "metadata": msg.message_metadata or {}
                }
                for msg in messages
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat history error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

@app.get("/reports/{diagnosis_id}")
async def get_report(
    diagnosis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get diagnosis report PDF — generates on demand if not cached."""
    try:
        # Verify diagnosis ownership
        diagnosis = db.query(DiagnosisRecord).filter(
            DiagnosisRecord.id == diagnosis_id,
            DiagnosisRecord.user_id == current_user.id
        ).first()
        if not diagnosis:
            raise HTTPException(status_code=404, detail="Diagnosis report not found")

        # Look for any existing report for this diagnosis (filename includes timestamp)
        import glob
        reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        existing = glob.glob(os.path.join(reports_dir, f"liver_report_{diagnosis_id}_*.pdf"))
        if existing:
            report_path = existing[-1]   # use most recent
        else:
            # Generate PDF now
            generated_path = await pdf_generator.generate_report(diagnosis_id, current_user.id)
            # generated_path may be relative — make absolute
            if not os.path.isabs(generated_path):
                report_path = os.path.join(os.path.dirname(__file__), generated_path)
            else:
                report_path = generated_path

        if not os.path.exists(report_path):
            raise HTTPException(status_code=500, detail="Report file could not be generated")

        from fastapi.responses import FileResponse
        return FileResponse(
            report_path,
            media_type='application/pdf',
            filename=f"liver_diagnosis_{diagnosis_id}.pdf"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve report: {str(e)}")


# ======================== ANALYTICS ENDPOINTS ========================

@app.get("/analytics/user-stats")
async def get_user_stats(
    user_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get statistics for a user (or all users if role is doctor/admin and user_id is not specified)"""
    target_user_id = user_id
    if target_user_id is None:
        if current_user.role not in ["doctor", "admin"]:
            target_user_id = current_user.id
    else:
        if current_user.id != target_user_id and current_user.role not in ["doctor", "admin"]:
            raise HTTPException(status_code=403, detail="Not enough permissions")

    query = db.query(DiagnosisRecord)
    if target_user_id is not None:
        query = query.filter(DiagnosisRecord.user_id == target_user_id)

    records = query.all()
    total_diagnoses = len(records)
    
    if total_diagnoses == 0:
        return {
            "total_diagnoses": 0,
            "average_confidence": 0.0,
            "risk_counts": {"low": 0, "medium": 0, "high": 0, "critical": 0},
            "last_diagnosis_date": None
        }

    confidences = [r.confidence_score for r in records]
    avg_confidence = sum(confidences) / len(confidences)
    
    risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for r in records:
        r_lvl = (r.risk_level or "low").lower()
        if r_lvl in risk_counts:
            risk_counts[r_lvl] += 1
        else:
            risk_counts["low"] += 1

    last_record = max(records, key=lambda r: r.created_at)

    return {
        "total_diagnoses": total_diagnoses,
        "average_confidence": round(avg_confidence, 3),
        "risk_counts": risk_counts,
        "last_diagnosis_date": last_record.created_at.isoformat()
    }


@app.get("/analytics/trends")
async def get_diagnosis_trends(
    user_id: Optional[int] = None,
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get diagnosis trends grouped by date"""
    target_user_id = user_id
    if target_user_id is None:
        if current_user.role not in ["doctor", "admin"]:
            target_user_id = current_user.id
    else:
        if current_user.id != target_user_id and current_user.role not in ["doctor", "admin"]:
            raise HTTPException(status_code=403, detail="Not enough permissions")

    cutoff_date = datetime.utcnow() - timedelta(days=days)
    query = db.query(DiagnosisRecord).filter(DiagnosisRecord.created_at >= cutoff_date)
    if target_user_id is not None:
        query = query.filter(DiagnosisRecord.user_id == target_user_id)

    records = query.order_by(DiagnosisRecord.created_at.asc()).all()

    # Group by date (YYYY-MM-DD)
    trends = {}
    for r in records:
        date_str = r.created_at.strftime("%Y-%m-%d")
        if date_str not in trends:
            trends[date_str] = {
                "date": date_str,
                "count": 0,
                "confidences": [],
                "risk_scores": []
            }
        
        trends[date_str]["count"] += 1
        trends[date_str]["confidences"].append(r.confidence_score)
        
        r_lvl = (r.risk_level or "low").lower()
        risk_score = 1
        if r_lvl == "medium":
            risk_score = 2
        elif r_lvl == "high":
            risk_score = 3
        elif r_lvl == "critical":
            risk_score = 4
        trends[date_str]["risk_scores"].append(risk_score)

    trend_list = []
    for date_str, data in sorted(trends.items()):
        avg_conf = sum(data["confidences"]) / len(data["confidences"])
        avg_risk = sum(data["risk_scores"]) / len(data["risk_scores"])
        trend_list.append({
            "date": date_str,
            "count": data["count"],
            "average_confidence": round(avg_conf, 3),
            "average_risk_score": round(avg_risk, 2)
        })

    return trend_list


@app.get("/analytics/risk-distribution")
async def get_risk_distribution(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get overall risk distribution across all patients (doctor/admin only)"""
    if current_user.role not in ["doctor", "admin"]:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    records = db.query(DiagnosisRecord).all()
    risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for r in records:
        r_lvl = (r.risk_level or "low").lower()
        if r_lvl in risk_counts:
            risk_counts[r_lvl] += 1
        else:
            risk_counts["low"] += 1

    result = [
        {"name": "Low Risk", "value": risk_counts["low"], "color": "#10B981"},
        {"name": "Medium Risk", "value": risk_counts["medium"], "color": "#F59E0B"},
        {"name": "High Risk", "value": risk_counts["high"], "color": "#EF4444"},
        {"name": "Critical Risk", "value": risk_counts["critical"], "color": "#7F1D1D"}
    ]
    return result


# ======================== LAB REPORT OCR ENDPOINT ========================

@app.post("/labs/extract-from-image")
async def extract_labs_from_image(
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a lab report image (JPG/PNG) and receive extracted lab values.

    Returns a JSON object with recognised field names matching SymptomData,
    plus metadata:
      _ocr_available : bool  – whether Tesseract OCR was used
      _error         : str   – present only if extraction failed
    """
    allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
    if image.content_type and image.content_type not in allowed_types:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{image.content_type}'. Upload a JPG or PNG image."
        )

    try:
        image_bytes = await image.read()
        if len(image_bytes) > 15 * 1024 * 1024:   # 15 MB guard
            raise HTTPException(status_code=413, detail="Image too large (max 15 MB).")

        result = extract_lab_values(image_bytes)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Lab OCR endpoint error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to process image.")


# ======================== NEW AI SYSTEM ENDPOINTS ========================

@app.get("/ai/health")
async def ai_health():
    """Check AI system health status"""
    if not ai_system:
        raise HTTPException(status_code=503, detail="AI System not initialized")
    
    health_status = ai_system.health_check()
    return {
        "status": "operational" if health_status['overall_status'] == 'operational' else 'degraded',
        "components": health_status,
        "timestamp": datetime.utcnow()
    }


@app.post("/ai/predict/symptoms")
async def predict_from_symptoms(
    symptoms: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Predict liver cirrhosis from symptoms only (ML model).
    
    Request body: {
        "age": 45,
        "alcohol_consumption": 20,
        "bilirubin": 1.5,
        ...
    }
    """
    if not ai_system or not ai_system.ml_engine:
        raise HTTPException(status_code=503, detail="ML Engine not available")
    
    try:
        prediction = ai_system.predict_from_symptoms(symptoms)
        
        if 'error' in prediction:
            raise HTTPException(status_code=400, detail=prediction['error'])
        
        return {
            "prediction": prediction,
            "timestamp": datetime.utcnow(),
            "source": "ml_engine"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Symptom prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/ai/predict/image")
async def predict_from_image(
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Predict liver cirrhosis from medical image (DL model).
    Accepts: JPEG, PNG, BMP, TIFF
    """
    if not ai_system or not ai_system.dl_engine:
        raise HTTPException(status_code=503, detail="DL Engine not available")
    
    try:
        contents = await image.read()
        prediction = ai_system.predict_from_image(contents)
        
        if 'error' in prediction:
            raise HTTPException(status_code=400, detail=prediction['error'])
        
        return {
            "prediction": prediction,
            "image_name": image.filename,
            "timestamp": datetime.utcnow(),
            "source": "dl_engine"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image prediction error: {e}")
        raise HTTPException(status_code=500, detail="Image prediction failed")


@app.post("/ai/predict/hybrid")
async def predict_hybrid(
    symptoms: Optional[Dict[str, Any]] = None,
    image: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Hybrid prediction combining both symptoms (ML) and image (DL).
    Returns fused prediction with XAI explanations.
    """
    if not ai_system:
        raise HTTPException(status_code=503, detail="AI System not initialized")
    
    if not symptoms and not image:
        raise HTTPException(status_code=400, detail="Provide either symptoms or image")
    
    try:
        image_data = None
        if image:
            image_data = await image.read()
        
        result = ai_system.predict_hybrid(symptoms=symptoms, image_data=image_data)
        
        # Save diagnosis record
        diagnosis_record = DiagnosisRecord(
            user_id=current_user.id,
            symptoms=symptoms or {},
            prediction_result=result['prediction'],
            explanations=result.get('xai_report', {}),
            confidence_score=result['prediction'].get('confidence', 0),
            risk_level=result['prediction'].get('risk_level', 'unknown'),
            recommended_actions=result['prediction'].get('recommendations', [])
        )
        
        db.add(diagnosis_record)
        db.commit()
        db.refresh(diagnosis_record)
        
        return {
            "diagnosis_id": diagnosis_record.id,
            "prediction": result['prediction'],
            "xai_report": result['xai_report'],
            "ml_prediction": result.get('ml_prediction'),
            "dl_prediction": result.get('dl_prediction'),
            "timestamp": datetime.utcnow(),
            "report_url": f"/reports/{diagnosis_record.id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid prediction error: {e}")
        raise HTTPException(status_code=500, detail="Hybrid prediction failed")


@app.get("/ai/diagnosis/{diagnosis_id}/xai")
async def get_xai_report(
    diagnosis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed XAI (Explainable AI) report for a diagnosis.
    Includes SHAP values for symptoms and Grad-CAM for images.
    """
    try:
        diagnosis = db.query(DiagnosisRecord).filter(
            DiagnosisRecord.id == diagnosis_id,
            DiagnosisRecord.user_id == current_user.id
        ).first()
        
        if not diagnosis:
            raise HTTPException(status_code=404, detail="Diagnosis not found")
        
        return {
            "diagnosis_id": diagnosis_id,
            "xai_report": diagnosis.explanations,
            "created_at": diagnosis.created_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"XAI report error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve XAI report")


@app.post("/ai/chat/medical")
async def medical_chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Chat with medical AI assistant about liver health.
    Uses RAG (Retrieval-Augmented Generation) for medical knowledge.
    """
    if not rag_chatbot:
        raise HTTPException(status_code=503, detail="Medical chatbot not available")
    
    try:
        response = rag_chatbot.chat(
            user_message=request.message,
            user_id=str(current_user.id)
        )
        
        # Save to database
        chat_session = db.query(ChatSession).filter(
            ChatSession.user_id == current_user.id
        ).first()
        
        if not chat_session:
            chat_session = ChatSession(
                user_id=current_user.id,
                context=json.dumps({})
            )
            db.add(chat_session)
            db.commit()
        
        # Save messages
        user_msg = ChatMessage(
            session_id=chat_session.id,
            message_type="user",
            content=request.message
        )
        db.add(user_msg)
        
        bot_msg = ChatMessage(
            session_id=chat_session.id,
            message_type="bot",
            content=response['response'],
            message_metadata={
                'sources': response.get('sources', []),
                'confidence': response.get('confidence', 'unknown')
            }
        )
        db.add(bot_msg)
        db.commit()
        
        return {
            "session_id": chat_session.id,
            "message": response['response'],
            "sources": response.get('sources', []),
            "confidence": response.get('confidence', 'unknown'),
            "timestamp": datetime.utcnow()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Medical chat error: {e}")
        raise HTTPException(status_code=500, detail="Chat failed")


@app.get("/ai/diagnosis-history")
async def get_diagnosis_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 20
):
    """Get user's diagnosis history"""
    try:
        diagnoses = db.query(DiagnosisRecord).filter(
            DiagnosisRecord.user_id == current_user.id
        ).order_by(
            DiagnosisRecord.created_at.desc()
        ).limit(limit).all()
        
        return {
            "total": len(diagnoses),
            "diagnoses": [
                {
                    "id": d.id,
                    "stage": d.prediction_result.get('stage', 'unknown') if isinstance(d.prediction_result, dict) else d.prediction_result,
                    "confidence": d.confidence_score,
                    "risk_level": d.risk_level,
                    "created_at": d.created_at
                }
                for d in diagnoses
            ]
        }
    except Exception as e:
        logger.error(f"Diagnosis history error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")


# ======================== ERROR HANDLERS ========================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": True, "message": exc.detail, "status_code": exc.status_code},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": True, "message": "Internal server error", "status_code": 500},
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
