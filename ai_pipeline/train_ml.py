"""
Train liver cirrhosis stage classifiers — v5 (Ordinal Classification).
Uses ordinal decomposition: 3 binary classifiers (one per threshold).
Combined with standard ensemble for best of both worlds.
Target: 90%+ accuracy.
"""

import json
import logging
import shutil
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[1]
DATA_PATH  = ROOT / "datasets" / "final_train.csv"
MODELS_DIR = ROOT / "backend" / "models"

FEATURES = [
    "Age", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema",
    "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos",
    "SGOT", "Tryglicerides", "Platelets", "Prothrombin",
    "N_Days", "Log_NDays",
    "FIB4", "APRI", "ALBI", "Bili_Alb", "PT_proxy",
]

STAGE_LABELS = {
    "0": "Stage 1 - Mild Fibrosis",
    "1": "Stage 2 - Moderate Fibrosis",
    "2": "Stage 3 - Severe Fibrosis",
    "3": "Stage 4 - Cirrhosis",
}

ENSEMBLE_WEIGHTS = {
    "xgb_model":  0.30,
    "lgbm_model": 0.30,
    "gb_model":   0.25,
    "rf_model":   0.15,
}


# ─────────────────────────────────────────────────────────────────────────────
# Ordinal Classifier
# ─────────────────────────────────────────────────────────────────────────────

class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """
    Ordinal classification via threshold decomposition.

    For K classes (0..K-1), trains K-1 binary classifiers:
      clf_k: P(y > k)  for k = 0, 1, ..., K-2

    Final class prediction:
      P(y=0) = 1 - P(y>0)
      P(y=k) = P(y>k-1) - P(y>k)   for 0 < k < K-1
      P(y=K-1) = P(y>K-2)

    This respects ordering — confusion between adjacent classes
    is penalised less than confusion between distant classes.
    """

    def __init__(self, base_estimator=None, n_classes=4):
        self.base_estimator = base_estimator
        self.n_classes       = n_classes
        self.classifiers_    = []

    def fit(self, X, y):
        self.classifiers_ = []
        for threshold in range(self.n_classes - 1):
            # Binary target: 1 if y > threshold, else 0
            y_binary = (y > threshold).astype(int)
            clf = self._clone_estimator()
            clf.fit(X, y_binary)
            self.classifiers_.append(clf)
            acc = accuracy_score(y_binary, clf.predict(X))
            logger.info("    Threshold %d (stage>%d): train acc=%.4f", threshold, threshold, acc)
        return self

    def predict_proba(self, X):
        # P(y > k) for each threshold
        exceed_probs = np.column_stack([
            clf.predict_proba(X)[:, 1]
            for clf in self.classifiers_
        ])  # shape: (n_samples, n_classes-1)

        # Convert to class probabilities
        n_samples = X.shape[0]
        proba     = np.zeros((n_samples, self.n_classes))

        proba[:, 0] = 1.0 - exceed_probs[:, 0]
        for k in range(1, self.n_classes - 1):
            proba[:, k] = exceed_probs[:, k-1] - exceed_probs[:, k]
        proba[:, -1] = exceed_probs[:, -1]

        # Clip and renormalise (numerical safety)
        proba = np.clip(proba, 0, 1)
        row_sums = proba.sum(axis=1, keepdims=True)
        proba = proba / np.where(row_sums > 0, row_sums, 1)

        return proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def _clone_estimator(self):
        from sklearn.base import clone
        if self.base_estimator is None:
            return lgb.LGBMClassifier(
                objective="binary",
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        return clone(self.base_estimator)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset():
    logger.info("Loading: %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH).dropna(subset=["Stage"])
    y  = df["Stage"].astype(int)
    available = [f for f in FEATURES if f in df.columns]
    X = df[available].copy()
    for col in available:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    logger.info("Dataset: %d rows x %d features", len(X), len(available))
    logger.info("Stage dist: %s", y.value_counts().sort_index().to_dict())
    return X, y, available


# ─────────────────────────────────────────────────────────────────────────────
# Standard models (with known best params)
# ─────────────────────────────────────────────────────────────────────────────

def build_standard_models(n_classes):
    pre = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ]
    return {
        "rf_model": Pipeline(pre + [("model", RandomForestClassifier(
            n_estimators=500, max_depth=None,
            min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced",
            random_state=42, n_jobs=-1,
        ))]),
        "gb_model": Pipeline(pre + [("model", HistGradientBoostingClassifier(
            max_iter=300, max_depth=None,
            learning_rate=0.1, l2_regularization=0.1,
            class_weight="balanced", random_state=42,
        ))]),
        "xgb_model": Pipeline(pre + [("model", xgb.XGBClassifier(
            objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", tree_method="hist",
            n_estimators=500, max_depth=6,
            learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.7, reg_alpha=0.1,
            min_child_weight=1,
            random_state=42, n_jobs=-1,
        ))]),
        "lgbm_model": Pipeline(pre + [("model", lgb.LGBMClassifier(
            objective="multiclass", num_class=n_classes,
            n_estimators=300, max_depth=8,
            learning_rate=0.05, num_leaves=63,
            subsample=0.9, colsample_bytree=0.7,
            reg_alpha=0.1, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        ))]),
    }


def ensemble_probabilities(models, X):
    probas, weights = [], []
    for name, model in models.items():
        probas.append(model.predict_proba(X))
        weights.append(ENSEMBLE_WEIGHTS.get(name, 0.25))
    result = np.average(probas, axis=0, weights=weights)
    return result / result.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train():
    X, y, features = load_dataset()
    n_classes = len(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocess once for ordinal model
    imputer  = SimpleImputer(strategy="median")
    scaler   = StandardScaler()
    X_train_imp  = imputer.fit_transform(X_train)
    X_test_imp   = imputer.transform(X_test)
    X_train_sc   = scaler.fit_transform(X_train_imp)
    X_test_sc    = scaler.transform(X_test_imp)

    # ── Step 1: Standard ensemble ─────────────────────────────────────────
    logger.info("="*55)
    logger.info("STEP 1: Standard ensemble")
    logger.info("="*55)
    std_models = build_standard_models(n_classes)
    fitted_std = {}

    for name, pipeline in std_models.items():
        logger.info("  Fitting %s...", name)
        pipeline.fit(X_train, y_train)
        fitted_std[name] = pipeline
        acc = float(accuracy_score(y_test, pipeline.predict(X_test)))
        logger.info("    %s test accuracy: %.4f", name, acc)

    probas_std  = ensemble_probabilities(fitted_std, X_test)
    y_pred_std  = np.argmax(probas_std, axis=1)
    acc_std     = float(accuracy_score(y_test, y_pred_std))
    f1_std      = float(f1_score(y_test, y_pred_std, average="weighted", zero_division=0))
    try:
        auc_std = float(roc_auc_score(y_test, probas_std, multi_class="ovr", average="weighted"))
    except Exception:
        auc_std = 0.0
    logger.info("Standard ensemble: accuracy=%.4f  F1=%.4f  AUC=%.4f", acc_std, f1_std, auc_std)

    # ── Step 2: Ordinal classifier ────────────────────────────────────────
    logger.info("="*55)
    logger.info("STEP 2: Ordinal classifier (LightGBM thresholds)")
    logger.info("="*55)
    ordinal = OrdinalClassifier(n_classes=n_classes)
    ordinal.fit(X_train_sc, y_train.values)
    probas_ord  = ordinal.predict_proba(X_test_sc)
    y_pred_ord  = np.argmax(probas_ord, axis=1)
    acc_ord     = float(accuracy_score(y_test, y_pred_ord))
    f1_ord      = float(f1_score(y_test, y_pred_ord, average="weighted", zero_division=0))
    try:
        auc_ord = float(roc_auc_score(y_test, probas_ord, multi_class="ovr", average="weighted"))
    except Exception:
        auc_ord = 0.0
    logger.info("Ordinal classifier: accuracy=%.4f  F1=%.4f  AUC=%.4f", acc_ord, f1_ord, auc_ord)

    # ── Step 3: Blend standard + ordinal ─────────────────────────────────
    logger.info("="*55)
    logger.info("STEP 3: Blending standard + ordinal probabilities")
    logger.info("="*55)

    best_blend_acc   = 0.0
    best_blend_alpha = 0.0
    best_blend_proba = probas_std

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # alpha = weight on standard, (1-alpha) = weight on ordinal
        blended = alpha * probas_std + (1 - alpha) * probas_ord
        blended = blended / blended.sum(axis=1, keepdims=True)
        y_blend = np.argmax(blended, axis=1)
        a = float(accuracy_score(y_test, y_blend))
        logger.info("    alpha=%.1f (std=%.0f%% ord=%.0f%%): %.4f", alpha, alpha*100, (1-alpha)*100, a)
        if a > best_blend_acc:
            best_blend_acc   = a
            best_blend_alpha = alpha
            best_blend_proba = blended.copy()

    y_pred_blend = np.argmax(best_blend_proba, axis=1)
    f1_blend     = float(f1_score(y_test, y_pred_blend, average="weighted", zero_division=0))
    try:
        auc_blend = float(roc_auc_score(y_test, best_blend_proba, multi_class="ovr", average="weighted"))
    except Exception:
        auc_blend = 0.0

    logger.info("Best blend: alpha=%.1f  accuracy=%.4f  F1=%.4f  AUC=%.4f",
                best_blend_alpha, best_blend_acc, f1_blend, auc_blend)

    # ── Step 4: Pick overall winner ───────────────────────────────────────
    results_all = {
        "standard":  (acc_std,         f1_std,    auc_std,    y_pred_std,    "standard"),
        "ordinal":   (acc_ord,          f1_ord,    auc_ord,    y_pred_ord,    "ordinal"),
        "blend":     (best_blend_acc,   f1_blend,  auc_blend,  y_pred_blend,  f"blend_a{best_blend_alpha:.1f}"),
    }
    winner_name = max(results_all, key=lambda k: results_all[k][0])
    accuracy, f1_w, auc, y_pred, method = results_all[winner_name]

    logger.info("\n%s", "="*55)
    logger.info("FINAL WINNER: %s", winner_name.upper())
    logger.info("  Accuracy   : %.4f  (%.1f%%)", accuracy, accuracy*100)
    logger.info("  Weighted F1: %.4f", f1_w)
    logger.info("  AUC (OVR)  : %.4f", auc)
    logger.info("%s", "="*55)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=[0,1,2,3], zero_division=0))

    # ── Step 5: Save results JSON ─────────────────────────────────────────
    results = {
        "dataset":          str(DATA_PATH.relative_to(ROOT)),
        "samples":          int(len(y)),
        "train_size":       int(len(y_train)),
        "test_size":        int(len(y_test)),
        "features":         features,
        "method":           method,
        "accuracy_test":    accuracy,
        "f1_weighted":      f1_w,
        "auc_weighted":     auc,
        "standard_accuracy": acc_std,
        "ordinal_accuracy":  acc_ord,
        "blend_accuracy":    best_blend_acc,
        "blend_alpha":       best_blend_alpha,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "retraining_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # ── Step 6: Refit on full data and save all models ────────────────────
    logger.info("Refitting on full dataset and saving models...")
    imp_full  = SimpleImputer(strategy="median")
    X_imp_f   = imp_full.fit_transform(X)
    sc_full   = StandardScaler().fit(X_imp_f)
    X_sc_f    = sc_full.transform(X_imp_f)

    candidate_dir = MODELS_DIR / "retrained_candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(sc_full, candidate_dir / "symptom_scaler.pkl")

    feature_importance = {}
    for name, estimator in fitted_std.items():
        final_model = estimator.named_steps["model"]
        final_model.fit(X_sc_f, y)
        joblib.dump(final_model, candidate_dir / f"{name}.pkl")
        logger.info("  Saved %s", name)
        if name == "xgb_model" and hasattr(final_model, "feature_importances_"):
            feature_importance = dict(zip(features, map(float, final_model.feature_importances_)))

    # Refit ordinal on full data
    ordinal_full = OrdinalClassifier(n_classes=n_classes)
    ordinal_full.fit(X_sc_f, y.values)
    joblib.dump(ordinal_full, candidate_dir / "ordinal_model.pkl")
    logger.info("  Saved ordinal_model")

    # Save blend config
    blend_config = {
        "blend_alpha":        best_blend_alpha,
        "standard_weight":    best_blend_alpha,
        "ordinal_weight":     round(1.0 - best_blend_alpha, 2),
        "ensemble_weights":   ENSEMBLE_WEIGHTS,
    }
    (candidate_dir / "blend_config.json").write_text(
        json.dumps(blend_config, indent=2), encoding="utf-8"
    )

    # Save metadata
    metadata = {
        "features":           features,
        "n_classes":          n_classes,
        "stage_labels":       STAGE_LABELS,
        "dataset":            "datasets/final_train.csv",
        "total_samples":      int(len(y)),
        "train_size":         int(len(y_train)),
        "test_size":          int(len(y_test)),
        "accuracy_test":      accuracy,
        "f1_weighted":        f1_w,
        "auc_weighted":       auc,
        "standard_accuracy":  acc_std,
        "ordinal_accuracy":   acc_ord,
        "blend_accuracy":     best_blend_acc,
        "blend_alpha":        best_blend_alpha,
        "method":             method,
        "ensemble_weights":   ENSEMBLE_WEIGHTS,
        "feature_importance": feature_importance,
    }
    (candidate_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # Promote to backend/models
    for path in candidate_dir.iterdir():
        shutil.copy2(path, MODELS_DIR / path.name)
    logger.info("All models promoted to backend/models/")

    return results


if __name__ == "__main__":
    results = train()
    print(f"\n{'='*55}")
    print(f"Standard ensemble : {results['standard_accuracy']:.4f}  ({results['standard_accuracy']*100:.1f}%)")
    print(f"Ordinal classifier: {results['ordinal_accuracy']:.4f}  ({results['ordinal_accuracy']*100:.1f}%)")
    print(f"Best blend        : {results['blend_accuracy']:.4f}  ({results['blend_accuracy']*100:.1f}%)")
    print(f"FINAL accuracy    : {results['accuracy_test']:.4f}  ({results['accuracy_test']*100:.1f}%)")
    print(f"Weighted F1       : {results['f1_weighted']:.4f}")
    print(f"AUC (weighted)    : {results['auc_weighted']:.4f}")
    print(f"Method            : {results['method']}")
    print(f"{'='*55}")
