"""
Hybrid AI model handler — uses trained ML models from cirrhosis.csv dataset.

Model loading order:
  1. Ensemble of XGBoost + GradientBoosting + RandomForest (all three trained on dataset)
  2. Falls back to single best model (symptom_model.pkl)
  3. Ultimate fallback: rule-based scoring (if no trained models found)
"""

import logging
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

from schemas import PredictionResult, SymptomData

# ── Feature order must match training ────────────────────────────────────────
FEATURE_NAMES = [
    "Age", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema",
    "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos",
    "SGOT", "Tryglicerides", "Platelets", "Prothrombin",
]

# Categorical encodings used during training (LabelEncoder alphabetical order)
CAT_ENCODING = {
    "Sex":         {"F": 0, "M": 1},
    "Ascites":     {"N": 0, "Y": 1},
    "Hepatomegaly":{"N": 0, "Y": 1},
    "Spiders":     {"N": 0, "Y": 1},
    "Edema":       {"N": 0, "S": 1, "Y": 2},  # S = mild/on diuretics
}

STAGE_DESCRIPTIONS = {
    0: "Stage 1 - Mild Fibrosis",
    1: "Stage 2 - Moderate Fibrosis",
    2: "Stage 3 - Severe Fibrosis",
    3: "Stage 4 - Cirrhosis",
}

RISK_LEVELS = {0: "low", 1: "medium", 2: "high", 3: "critical"}

MODELS_DIR = Path(__file__).parent / "models"


def _load_model(filename: str):
    path = MODELS_DIR / filename
    if path.exists() and JOBLIB_OK:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return joblib.load(path)
        except Exception as e:
            logger.warning("Could not load %s: %s", filename, e)
    return None


def _load_metadata() -> dict:
    path = MODELS_DIR / "model_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


class HybridModelHandler:
    """Runs ensemble of trained ML models, falls back to rule-based scoring."""

    def __init__(self):
        self.scaler   = _load_model("symptom_scaler.pkl")
        self.xgb      = _load_model("xgb_model.pkl")
        self.rf       = _load_model("rf_model.pkl")
        self.gb       = _load_model("gb_model.pkl")
        self.best     = _load_model("symptom_model.pkl")   # best single model
        self.metadata = _load_metadata()

        loaded = [n for n, m in [("XGBoost", self.xgb), ("RandomForest", self.rf),
                                  ("GradBoost", self.gb), ("BestSingle", self.best)]
                  if m is not None]
        if loaded:
            logger.info("Trained models loaded: %s", ", ".join(loaded))
        else:
            logger.warning("No trained models found — using rule-based fallback.")

    # ── Public API ────────────────────────────────────────────────────────────

    async def predict(
        self, symptoms: SymptomData, image: Optional[np.ndarray] = None
    ) -> PredictionResult:

        features = self._symptom_to_features(symptoms)

        if self.scaler is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features

        # Try ensemble prediction
        ensemble_proba = self._ensemble_predict(features_scaled)

        if ensemble_proba is not None:
            predicted_class = int(np.argmax(ensemble_proba))
            confidence      = float(ensemble_proba[predicted_class])
        else:
            # Pure rule-based fallback
            rule_score      = self._rule_score(symptoms)
            predicted_class = int(np.clip(round(rule_score * 3), 0, 3))
            confidence      = float(np.clip(0.55 + abs(rule_score - 0.5) * 0.4, 0.50, 0.85))
            ensemble_proba  = self._stage_probs_from_score(rule_score)

        # Optionally blend image signal
        if image is not None and CV2_AVAILABLE:
            img_score  = self._image_score(image)
            # Shift probabilities slightly toward image signal
            img_proba  = self._stage_probs_from_score(img_score)
            blend      = 0.75  # 75% model, 25% image
            ensemble_proba = blend * ensemble_proba + (1 - blend) * img_proba
            ensemble_proba = ensemble_proba / ensemble_proba.sum()
            predicted_class = int(np.argmax(ensemble_proba))
            confidence      = float(ensemble_proba[predicted_class])

        probabilities   = {f"stage_{i}": round(float(p), 4) for i, p in enumerate(ensemble_proba)}
        recommendations = self._recommendations(predicted_class, confidence)

        return PredictionResult(
            stage=f"stage_{predicted_class}",
            stage_description=STAGE_DESCRIPTIONS[predicted_class],
            confidence=round(confidence, 4),
            risk_level=RISK_LEVELS[predicted_class],
            probabilities=probabilities,
            recommendations=recommendations,
            follow_up_required=predicted_class >= 1,
            specialist_referral=predicted_class >= 2,
        )

    # ── Feature engineering ────────────────────────────────────────────────

    def _symptom_to_features(self, s: SymptomData) -> np.ndarray:
        """
        Map SymptomData → 15-feature vector matching training order.

        Dataset column mapping:
          Age        → age (days in dataset, but we pass years — convert)
          Sex        → sex (0=F, 1=M)
          Ascites    → ascites (bool → Y/N → encoded 0/1)
          Hepatomegaly → hepatomegaly (bool → Y/N → encoded 0/1)
          Spiders    → spiders (bool → Y/N → encoded 0/1)
          Edema      → edema (0=N, 1=S, 2=Y → encoded 0/1/2)
          Bilirubin  → bilirubin
          Cholesterol → cholesterol
          Albumin    → albumin
          Copper     → copper
          Alk_Phos   → alk_phos
          SGOT       → ast  (same test, different name)
          Tryglicerides → triglycerides
          Platelets  → platelets (frontend sends ×10³/µL, dataset also ×10³/µL)
          Prothrombin → prothrombin
        """
        # Age: dataset stores in days; we receive years — convert back
        age_days = float(s.age) * 365.25

        # Categorical encodings (LabelEncoder alphabetical order from training)
        # Ascites: N=0, Y=1
        ascites_enc = 1.0 if (s.ascites if isinstance(s.ascites, bool) else s.ascites >= 1) else 0.0
        # Hepatomegaly: N=0, Y=1
        hepato_enc = 1.0 if (s.hepatomegaly if isinstance(s.hepatomegaly, bool) else s.hepatomegaly >= 1) else 0.0
        # Spiders: N=0, Y=1
        spiders_enc = 1.0 if (s.spiders if isinstance(s.spiders, bool) else s.spiders >= 1) else 0.0
        # Edema: N=0, S=1, Y=2
        if isinstance(s.edema, bool):
            edema_enc = 2.0 if s.edema else 0.0
        else:
            edema_enc = float(min(int(s.edema), 2))

        vec = [
            age_days,                   # Age (days)
            float(s.sex),               # Sex 0=F 1=M
            ascites_enc,                # Ascites
            hepato_enc,                 # Hepatomegaly
            spiders_enc,                # Spiders
            edema_enc,                  # Edema
            float(s.bilirubin),         # Bilirubin mg/dL
            float(s.cholesterol),       # Cholesterol mg/dL
            float(s.albumin),           # Albumin g/dL
            float(s.copper),            # Copper µg/dL
            float(s.alk_phos),          # Alk_Phos U/L
            float(s.ast),               # SGOT / AST U/L
            float(s.triglycerides),     # Tryglicerides mg/dL
            float(s.platelets),         # Platelets ×10³/µL (same unit as dataset)
            float(s.prothrombin),       # Prothrombin sec
        ]
        return np.array([vec], dtype=np.float32)

    # ── Ensemble logic ──────────────────────────────────────────────────────

    def _ensemble_predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Soft-vote across available trained models."""
        probas = []
        weights = []

        for model, weight in [(self.xgb, 0.40), (self.gb, 0.40), (self.rf, 0.20)]:
            if model is None:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = model.predict_proba(X)[0]
                if len(p) == 4:
                    probas.append(p)
                    weights.append(weight)
            except Exception as e:
                logger.debug("Model predict error: %s", e)

        # Fall back to single best model
        if not probas and self.best is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = self.best.predict_proba(X)[0]
                if len(p) == 4:
                    return np.array(p)
            except Exception as e:
                logger.debug("Best model predict error: %s", e)

        if not probas:
            return None

        total_w = sum(weights)
        blended = sum(w * p for w, p in zip(weights, probas)) / total_w
        return blended / blended.sum()

    # ── Rule-based fallback ────────────────────────────────────────────────

    def _rule_score(self, s: SymptomData) -> float:
        score = 0.0
        score += min(s.age / 100, 1) * 0.08
        score += min(s.fatigue_level / 10, 1) * 0.07
        score += min(s.alcohol_consumption / 35, 1) * 0.10
        score += min(s.weight_loss_kg / 15, 1) * 0.07
        score += 0.07 if s.abdominal_swelling else 0
        score += 0.05 if s.appetite_loss else 0
        score += 0.10 if s.jaundice else 0
        score += min(s.ascites / 3, 1) * 0.09
        score += min(s.hepatomegaly / 3, 1) * 0.06
        score += min(s.edema / 3, 1) * 0.06
        score += np.clip((s.bilirubin - 1.2) / 8, 0, 1) * 0.09
        score += np.clip((3.5 - s.albumin) / 2.5, 0, 1) * 0.08
        score += np.clip((s.ast - 45) / 250, 0, 1) * 0.07
        score += np.clip((150000 - s.platelets) / 100000, 0, 1) * 0.06
        score += np.clip((s.prothrombin - 12) / 6, 0, 1) * 0.05
        return float(np.clip(score, 0, 1))

    def _stage_probs_from_score(self, score: float) -> np.ndarray:
        cls = int(np.clip(round(score * 3), 0, 3))
        conf = float(np.clip(0.55 + abs(score - 0.5) * 0.4, 0.50, 0.85))
        probs = np.ones(4) * ((1 - conf) / 3)
        probs[cls] = conf
        return probs / probs.sum()

    # ── Image scoring ─────────────────────────────────────────────────────

    def _image_score(self, image: np.ndarray) -> float:
        """Use trained image classifier; fall back to texture heuristic."""
        img_model  = _load_model("image_model.pkl")
        img_scaler = _load_model("image_scaler.pkl")
        img_le     = _load_model("image_label_encoder.pkl")

        if img_model is not None and img_scaler is not None and img_le is not None:
            try:
                feat   = self._extract_image_features(image)
                feat_s = img_scaler.transform([feat])
                proba  = img_model.predict_proba(feat_s)[0]
                classes = list(img_le.classes_)
                risk_map = {"Normal": 0.10, "Benign": 0.50, "Malignant": 0.95}
                score = sum(proba[i] * risk_map.get(c, 0.5)
                            for i, c in enumerate(classes))
                logger.info("Image model score: %.3f  proba: %s",
                            score, dict(zip(classes, [round(float(p), 3) for p in proba])))
                return float(np.clip(score, 0, 1))
            except Exception as e:
                logger.warning("Image model inference failed: %s", e)

        if not CV2_AVAILABLE:
            return 0.5
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            gray = cv2.resize(gray, (256, 256))
            texture  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            contrast = float(gray.std())
            return float(np.clip(
                0.55 * np.clip(texture / 1600, 0, 1) +
                0.45 * np.clip(contrast / 90, 0, 1), 0, 1
            ))
        except Exception as e:
            logger.warning("Image scoring fallback error: %s", e)
            return 0.5

    def _extract_image_features(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG + texture + color histogram + LBP (matches train_image_model.py)."""
        IMG_SIZE = (128, 128)
        if len(image.shape) == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, IMG_SIZE)
        gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        resized = cv2.resize(gray, (64, 64))
        gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        hog_feats, cell = [], 8
        for i in range(0, resized.shape[0], cell):
            for j in range(0, resized.shape[1], cell):
                cm = mag[i:i+cell, j:j+cell]
                ca = ang[i:i+cell, j:j+cell] % 180
                hist = np.zeros(9)
                for b in range(9):
                    mask = (ca >= b*20) & (ca < (b+1)*20)
                    hist[b] = cm[mask].sum()
                if hist.sum() > 0:
                    hist /= hist.sum()
                hog_feats.extend(hist)
        hog = np.array(hog_feats, dtype=np.float32)

        f = gray.astype(np.float32)
        mean = f.mean(); std = f.std()
        skew = float(np.mean(((f - mean) / (std + 1e-8)) ** 3))
        kurt = float(np.mean(((f - mean) / (std + 1e-8)) ** 4))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        dx = np.diff(f, axis=1); dy = np.diff(f, axis=0)
        contrast_v = float(np.mean(dx**2) + np.mean(dy**2))
        tex = np.array([mean, std, skew, kurt, lap_var, contrast_v], dtype=np.float32)

        chist = []
        for ch in range(3):
            h2, _ = np.histogram(img_rgb[:, :, ch], bins=32, range=(0, 256))
            chist.extend(h2.astype(np.float32) / (h2.sum() + 1e-8))
        chist = np.array(chist)

        gray_s = cv2.resize(gray, (32, 32))
        lbp_h, _ = np.histogram(gray_s.ravel(), bins=64, range=(0, 256))
        lbp = lbp_h.astype(np.float32) / (lbp_h.sum() + 1e-8)

        return np.concatenate([hog, tex, chist, lbp])


    # ── Recommendations ───────────────────────────────────────────────────

    def _recommendations(self, stage: int, confidence: float) -> List[str]:
        recs = {
            0: [
                "Consult a hepatologist for baseline evaluation and monitoring.",
                "Avoid alcohol and hepatotoxic medications.",
                "Monitor liver function tests every 6–12 months.",
            ],
            1: [
                "Schedule specialist consultation for fibrosis assessment.",
                "Follow a liver-friendly diet with reduced salt and alcohol.",
                "Repeat imaging and clinical monitoring as advised.",
            ],
            2: [
                "Urgent hepatology review is strongly recommended.",
                "Screen for cirrhosis complications and portal hypertension.",
                "Discuss intensive monitoring and long-term care planning.",
            ],
            3: [
                "Immediate specialist management is required.",
                "Evaluate for advanced cirrhosis complications and transplant pathway.",
                "Seek emergency care for bleeding, confusion, fever, or severe swelling.",
            ],
        }
        result = list(recs.get(stage, recs[0]))
        if confidence < 0.65:
            result.append(
                "Prediction confidence is moderate — confirm with clinical tests and specialist review."
            )
        result.append("This AI output supports clinical screening only and is not a final medical diagnosis.")
        return result
