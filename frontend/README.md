# LiverAI optimized frontend (FastAPI-compatible)

This is a replacement `frontend` folder for the `Explainable-AI-Liver-Cirrhosis-Prediction` project. It does not replace or modify the Python backend or the trained models.

## What is corrected

- Adds the missing `public/manifest.json` referenced by `public/index.html`, and adds a matching SVG application icon.
- Uses `POST /auth/login`, `POST /auth/register`, and `GET /auth/me` with the backend's `access_token` and `user` response.
- Sends diagnosis requests to `POST /predict` as `multipart/form-data` with `symptoms` (JSON string), `assessment_mode` (`symptoms_only`, `symptoms_labs`, or `full`), and optional `image`.
- Uses the backend's exact symptom/lab keys, including `alk_phos`, `ast`, `platelets`, and `prothrombin`.
- Uses `POST /labs/extract-from-image`, `GET /history`, `GET /diagnosis/results/{id}`, and `GET /reports/{id}`.
- Uses existing analytics, patient-management, and chat endpoints instead of the old unsupported endpoints.
- Preserves authentication across reloads and handles expired authentication tokens.
- Provides API error messages in the UI and responsive layout.

## Requirements

- Node.js and npm installed.
- Python FastAPI backend running on `http://127.0.0.1:8000` (or configure another URL).
- Required model weights and datasets installed where the backend expects them. These are not bundled in this frontend archive.

## Install and run on Windows (PowerShell)

1. Make a backup of your current `frontend` folder.
2. Extract this archive and replace the existing project `frontend` folder with the `frontend` folder from this archive.
3. Open a terminal:

   ```powershell
   cd frontend
   Copy-Item .env.example .env
   npm install
   npm start
   ```

4. Open `http://localhost:3000`.
5. In another terminal, start FastAPI from the `backend` directory using the repository's established environment, e.g. `python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload`.

If your backend URL differs, edit `frontend/.env` and set `REACT_APP_API_URL`, then restart `npm start`.

## Endpoint map used by this UI

| Feature | Backend endpoint |
|---|---|
| Login | `POST /auth/login` |
| Registration | `POST /auth/register` |
| Current user | `GET /auth/me` |
| Diagnosis | `POST /predict` |
| Lab OCR | `POST /labs/extract-from-image` |
| History | `GET /history` |
| Diagnosis result | `GET /diagnosis/results/{diagnosis_id}` |
| PDF report | `GET /reports/{diagnosis_id}` |
| Analytics | `GET /analytics/user-stats`, `GET /analytics/trends`, `GET /analytics/risk-distribution` |
| Patient management | `GET /patients`, `POST /patients/add`, `POST /patients/{id}/deactivate` |
| Chat | `POST /chat/start`, `POST /chat/message` |

## Important limitations

- This archive contains the frontend source only, not a complete copy of the Python backend, database, AI models, or datasets.
- Repository model files/datasets are separate dependencies; a frontend build can succeed even when inference fails due to absent weights.
- The frontend's clinical validation checks input ranges, but does not validate the medical correctness of a prediction.
- The app is a software/research demonstration and does not replace a qualified clinician's assessment.
