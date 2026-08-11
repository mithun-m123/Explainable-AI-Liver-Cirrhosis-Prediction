# Liver Cirrhosis Detection System

An explainable AI-powered web application for non-invasive liver cirrhosis staging. It combines a clinical symptom ensemble (Random Forest, Gradient Boosting, XGBoost), an optional deep learning imaging pipeline (EfficientNet), and a RAG-based medical chatbot into a single full-stack application.

---

## Features

- **Multi-model ensemble prediction** — RF + GradBoost + XGBoost soft-voted on 15 Mayo Clinic PBC features
- **Medical image analysis** — HOG + texture + LBP feature classifier with optional CNN (EfficientNet)
- **Explainability** — SHAP feature importance + Grad-CAM heatmaps
- **Lab report OCR** — Auto-extract values from uploaded lab report images (Tesseract)
- **Medical chatbot** — Rule-based fallback; upgrades to Groq/OpenAI/LangChain RAG when configured
- **PDF reports** — Auto-generated per-diagnosis reports
- **Role-based access** — `patient`, `doctor`, `admin` roles with JWT auth
- **Analytics dashboard** — Risk distribution, diagnosis trends, per-user stats
- **React frontend** — Responsive SPA with Recharts visualisations

---

## Project Structure

```
liver-cirrhosis-prediction/
│
├── backend/                    # FastAPI application
│   ├── main.py                 # App entry point — all API routes
│   ├── database.py             # SQLAlchemy models + session management
│   ├── auth.py                 # JWT authentication, RBAC helpers
│   ├── schemas.py              # Pydantic request/response models
│   ├── models_handler.py       # HybridModelHandler — ensemble inference
│   ├── xai_logic.py            # SHAP + Grad-CAM explainability
│   ├── chatbot_api.py          # MedicalChatbot (NLP pipeline)
│   ├── lab_ocr.py              # Tesseract OCR for lab reports
│   ├── pdf_gen.py              # fpdf2 PDF report generator
│   ├── preprocessing.py        # Data + image preprocessing utilities
│   ├── models/                 # Trained model binaries (.pkl, .h5)
│   └── reports/                # Generated PDF reports (gitignored)
│
├── ai_pipeline/                # Advanced ML/DL training pipeline
│   ├── integration.py          # AISystemIntegration coordinator
│   ├── ml_engine.py            # XGBoost training engine
│   ├── dl_engine.py            # EfficientNet training engine
│   ├── fusion_engine.py        # Prediction fusion (4 strategies)
│   ├── xai_engine.py           # SHAP + Grad-CAM for pipeline
│   ├── chatbot.py              # LangChain RAG chatbot
│   ├── preprocessing.py        # Pipeline-specific preprocessing
│   ├── train_ml.py             # ML training entry point
│   └── train_dl.py             # DL training entry point
│
├── frontend/                   # React 19 SPA
│   ├── src/
│   │   ├── App.js              # Router + auth state
│   │   ├── api.js              # Axios API client
│   │   ├── index.js            # React entry point
│   │   ├── components/         # Navbar, Sidebar
│   │   ├── pages/              # HomePage, DiagnosisPage, ResultsPage,
│   │   │                       # HistoryPage, ReportsPage, DashboardPage,
│   │   │                       # ChatbotPage, PatientsPage
│   │   └── styles/
│   └── public/
│
├── datasets/                   # Training datasets (CSV)
├── scripts/                    # Utility scripts (test, seed, report gen)
├── docs/                       # Architecture and implementation docs
│
├── .env.example                # Environment variable template
├── .gitignore
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Multi-container Docker setup
├── Dockerfile                  # Backend container
├── nginx.conf                  # Reverse proxy config
├── run.bat                     # One-click launcher (Windows)
└── run.sh                      # One-click launcher (Linux / macOS)
```

---

## Quick Start

### Prerequisites

| Tool | Version |
|---|---|
| Python | 3.10 or higher |
| Node.js | 18 or higher |
| npm | 9 or higher |

### Option A — One-Click Launch

**Windows**
```bat
run.bat
```

**Linux / macOS**
```bash
chmod +x run.sh
./run.sh
```

The scripts will automatically create a virtual environment, install all dependencies, copy `.env.example → .env`, and start both servers.

---

### Option B — Manual Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-username/liver-cirrhosis-prediction.git
cd liver-cirrhosis-prediction
```

**2. Configure environment variables**
```bash
cp .env.example .env
# Open .env and set JWT_SECRET_KEY and any optional API keys
```

**3. Set up the backend**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

**4. Start the backend**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**5. Set up and start the frontend**
```bash
cd frontend
npm install
npm start
```

---

## Access

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Interactive API Docs (Swagger) | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |

### Default Admin Account

```
Email:    admin@liverai.com
Password: admin123
```

> Change this immediately in any non-local environment.

---

## API Overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/register` | Register new user |
| `POST` | `/auth/login` | Authenticate, receive JWT |
| `GET` | `/auth/me` | Get current user profile |
| `POST` | `/predict` | Diagnose from symptoms + optional image |
| `GET` | `/history` | User's diagnosis history |
| `GET` | `/diagnosis/results/{id}` | Single diagnosis detail |
| `GET` | `/reports/{id}` | Download PDF report |
| `POST` | `/chat/start` | Open chatbot session |
| `POST` | `/chat/message` | Send message to chatbot |
| `POST` | `/labs/extract-from-image` | OCR lab values from image |
| `GET` | `/analytics/user-stats` | Stats for user or all patients |
| `GET` | `/analytics/trends` | Diagnosis trends over time |
| `GET` | `/analytics/risk-distribution` | Risk distribution (admin/doctor) |
| `GET` | `/patients` | List patients (doctor/admin) |
| `POST` | `/patients/add` | Add patient (doctor/admin) |
| `GET` | `/health` | Health check |

Full specification: [`docs/API_SPECIFICATIONS.md`](docs/API_SPECIFICATIONS.md)

---

## Optional Features

### Enable XGBoost
```bash
pip install xgboost>=2.0.3
```

### Enable SHAP Explainability
```bash
pip install shap>=0.43.0
```

### Enable Deep Learning (EfficientNet)
```bash
pip install tensorflow>=2.15.0
```

### Enable AI Chatbot (Groq — free tier)
1. Get a free API key at https://console.groq.com
2. Set `GROQ_API_KEY=your-key` in `.env`

### Enable Lab Report OCR
1. Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
2. `pip install pytesseract>=0.3.10`

---

## Docker

```bash
docker-compose up --build
```

Services started: `backend` (port 8000), `frontend` (port 3000), `nginx` (port 80).

---

## Model Training

To retrain the ML ensemble on the provided dataset:
```bash
cd ai_pipeline
python train_ml.py
```

To retrain the EfficientNet imaging model:
```bash
cd ai_pipeline
python train_dl.py
```

Trained models are saved to `backend/models/`.

---

## Tech Stack

**Backend**
- FastAPI · SQLAlchemy · SQLite / PostgreSQL · JWT (python-jose) · fpdf2

**AI / ML**
- scikit-learn · XGBoost · TensorFlow / Keras · OpenCV · SHAP · LangChain

**Frontend**
- React 19 · React Router 7 · Axios · Recharts · Lucide React

**Infrastructure**
- Docker · Nginx · GitHub Actions (CI/CD ready)

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## Disclaimer

This system is designed for **research and educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical decisions.
