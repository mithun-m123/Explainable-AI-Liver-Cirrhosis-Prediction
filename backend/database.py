from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./liver_cirrhosis.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all models
Base = declarative_base()

class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    patient_uid = Column(String(20), unique=True, index=True, nullable=True)  # e.g. PAT-00042
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="patient")  # patient, doctor, admin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    diagnosis_records = relationship("DiagnosisRecord", back_populates="user")
    chat_sessions = relationship("ChatSession", back_populates="user")

class DiagnosisRecord(Base):
    """Medical diagnosis records"""
    __tablename__ = "diagnosis_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Input data
    symptoms = Column(JSON, nullable=False)  # Symptom questionnaire responses
    image_data = Column(Text)  # Base64 encoded image data

    # Prediction results
    prediction_result = Column(JSON, nullable=False)  # Full prediction object
    explanations = Column(JSON)  # XAI explanations
    confidence_score = Column(Float, nullable=False)
    risk_level = Column(String(50), nullable=False)  # low, medium, high, critical
    recommended_actions = Column(JSON)  # Treatment recommendations

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="diagnosis_records")

class ChatSession(Base):
    """Chatbot conversation sessions"""
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_type = Column(String(50), nullable=False)  # symptom_collection, medical_qa, education
    context = Column(JSON, default=dict)  # Session context data
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    """Individual chat messages"""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    message_type = Column(String(20), nullable=False)  # user, bot
    content = Column(Text, nullable=False)
    message_metadata = Column(JSON, default=dict)  # Additional message metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")

class ModelVersion(Base):
    """Track different model versions and performance"""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # ml, dl, ensemble
    framework = Column(String(50), nullable=False)  # sklearn, pytorch, tensorflow

    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)

    # Training metadata
    training_data_size = Column(Integer)
    training_duration = Column(Float)  # hours
    hyperparameters = Column(JSON)

    # File paths
    model_path = Column(String(500))
    scaler_path = Column(String(500))

    # Status
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    """Audit trail for HIPAA compliance"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String(100), nullable=False)  # login, predict, view_report, etc.
    resource = Column(String(255))  # What was accessed
    ip_address = Column(String(45))  # IPv4/IPv6
    user_agent = Column(Text)
    success = Column(Boolean, default=True)
    details = Column(JSON)  # Additional context
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User")

# Create all tables
def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

# Dependency to get database session
def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Database initialization
def init_db():
    """Initialize database with sample data"""
    create_tables()

    # Create default admin user if not exists
    db = SessionLocal()
    try:
        admin_user = db.query(User).filter(User.email == "admin@liverai.com").first()
        if not admin_user:
            from passlib.context import CryptContext
            pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

            admin = User(
                email="admin@liverai.com",
                hashed_password=pwd_context.hash("admin123"),
                full_name="System Administrator",
                role="admin"
            )
            db.add(admin)
            db.commit()
            print("Default admin user created: admin@liverai.com / admin123")
    except Exception as e:
        print(f"Error creating admin user: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
