"""Train and evaluate the API-compatible cirrhosis stage classifiers."""

import json
import logging
import shutil
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "cirrhosis.csv"
MODELS_DIR = ROOT / "backend" / "models"
FEATURES = [
    "Age", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema",
    "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos",
    "SGOT", "Tryglicerides", "Platelets", "Prothrombin",
]
CAT_ENCODINGS = {
    "Sex": {"F": 0, "M": 1},
    "Ascites": {"N": 0, "Y": 1},
    "Hepatomegaly": {"N": 0, "Y": 1},
    "Spiders": {"N": 0, "Y": 1},
    "Edema": {"N": 0, "S": 1, "Y": 2},
}
ENSEMBLE_WEIGHTS = {"xgb_model": 0.4, "gb_model": 0.4, "rf_model": 0.2}
STAGE_LABELS = {
    "0": "Stage 1 - Mild Fibrosis",
    "1": "Stage 2 - Moderate Fibrosis",
    "2": "Stage 3 - Severe Fibrosis",
    "3": "Stage 4 - Cirrhosis",
}


def load_dataset():
    df = pd.read_csv(DATA_PATH).dropna(subset=["Stage"]).copy()
    y = df["Stage"].astype(int) - 1
    X = df[FEATURES].copy()
    for column, mapping in CAT_ENCODINGS.items():
        X[column] = X[column].map(mapping)
    for column in FEATURES:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    return X, y


def model_searches():
    common = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    return {
        "rf_model": (
            Pipeline(common + [("model", RandomForestClassifier(random_state=42, n_jobs=-1))]),
            {
                "model__n_estimators": [250],
                "model__max_depth": [None, 8],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt", 0.8],
            },
        ),
        "gb_model": (
            Pipeline(common + [("model", GradientBoostingClassifier(random_state=42))]),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.08],
                "model__max_depth": [1, 2],
            },
        ),
        "xgb_model": (
            Pipeline(common + [("model", xgb.XGBClassifier(
                objective="multi:softprob", num_class=4, eval_metric="mlogloss",
                tree_method="hist", random_state=42, n_jobs=1,
            ))]),
            {
                "model__n_estimators": [120, 220],
                "model__learning_rate": [0.03, 0.08],
                "model__max_depth": [2, 3],
            },
        ),
    }


def ensemble_probabilities(models, X):
    probabilities = []
    weights = []
    for name, model in models.items():
        probabilities.append(model.predict_proba(X))
        weights.append(ENSEMBLE_WEIGHTS[name])
    result = np.average(probabilities, axis=0, weights=weights)
    return result / result.sum(axis=1, keepdims=True)


def existing_ensemble_accuracy(X_train, X_test, y_test):
    scaler_path = MODELS_DIR / "symptom_scaler.pkl"
    if not scaler_path.exists():
        return None
    scaler = joblib.load(scaler_path)
    X_test = X_test.fillna(X_train.median())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_scaled = scaler.transform(X_test)
        models = {}
        for name in ENSEMBLE_WEIGHTS:
            path = MODELS_DIR / f"{name}.pkl"
            if path.exists():
                models[name] = joblib.load(path)
        if not models:
            return None
        probas, weights = [], []
        for name, model in models.items():
            probas.append(model.predict_proba(X_scaled))
            weights.append(ENSEMBLE_WEIGHTS[name])
        y_pred = np.argmax(np.average(probas, axis=0, weights=weights), axis=1)
    return float(accuracy_score(y_test, y_pred))


def train():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fitted_models = {}
    cv_scores = {}
    best_params = {}

    for name, (pipeline, parameter_grid) in model_searches().items():
        search = GridSearchCV(
            pipeline, parameter_grid, scoring="accuracy", cv=cv, n_jobs=-1,
            refit=True, error_score="raise",
        )
        search.fit(X_train, y_train)
        fitted_models[name] = search.best_estimator_
        cv_scores[name] = float(search.best_score_)
        best_params[name] = search.best_params_
        print(f"{name}: CV accuracy={search.best_score_:.4f}; params={search.best_params_}")

    probabilities = ensemble_probabilities(fitted_models, X_test)
    y_pred = np.argmax(probabilities, axis=1)
    accuracy = float(accuracy_score(y_test, y_pred))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    baseline_accuracy = existing_ensemble_accuracy(X_train, X_test, y_test)

    print(f"\nHeld-out rows: {len(y_test)} / {len(y)}")
    print(f"Current saved ensemble accuracy: {baseline_accuracy if baseline_accuracy is not None else 'unavailable'}")
    print(f"Retrained ensemble accuracy: {accuracy:.4f}")
    print(f"Retrained weighted F1: {weighted_f1:.4f}")
    print(classification_report(y_test, y_pred, labels=[0, 1, 2, 3], zero_division=0))

    results = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "samples": int(len(y)),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "random_state": 42,
        "baseline_accuracy_test": baseline_accuracy,
        "accuracy_test": accuracy,
        "f1_weighted": weighted_f1,
        "cv_accuracy_by_model": cv_scores,
        "best_params": best_params,
    }
    result_path = MODELS_DIR / "retraining_results.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Refit preprocessing and models on all labeled rows after evaluation.
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler().fit(X_imputed)
    X_scaled = scaler.transform(X_imputed)
    candidate_dir = MODELS_DIR / "retrained_candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, candidate_dir / "symptom_scaler.pkl")
    feature_importance = {}
    for name, estimator in fitted_models.items():
        final_model = estimator.named_steps["model"]
        final_model.fit(X_scaled, y)
        joblib.dump(final_model, candidate_dir / f"{name}.pkl")
        if name == "xgb_model":
            feature_importance = dict(zip(FEATURES, map(float, final_model.feature_importances_)))

    metadata = {
        "features": FEATURES,
        "n_classes": 4,
        "stage_labels": STAGE_LABELS,
        "dataset": "datasets/cirrhosis.csv",
        "total_samples": int(len(y)),
        "real_samples": int(len(y)),
        "synthetic_samples": 0,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "accuracy_test": accuracy,
        "baseline_accuracy_test": baseline_accuracy,
        "f1_weighted": weighted_f1,
        "cv_mean": float(np.mean(list(cv_scores.values()))),
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "feature_importance": feature_importance,
    }
    (candidate_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    minimum_gain = 0.05
    if baseline_accuracy is None or accuracy >= baseline_accuracy + minimum_gain:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for path in candidate_dir.iterdir():
            shutil.copy2(path, MODELS_DIR / path.name)
        print("Promoted retrained models and scaler to backend/models.")
    else:
        print(
            f"Candidate saved to {candidate_dir.relative_to(ROOT)}; not promoted "
            f"because the held-out gain is below {minimum_gain:.0%}."
        )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()