"""
Complete ML Model Training and Inference
Includes Logistic Regression, Random Forest, SVM, and XGBoost
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import logging
from typing import Dict, Tuple, Any, List

logger = logging.getLogger(__name__)


class MLModelTrainer:
    """Train and manage machine learning models"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.models = {}
        self.scalers = {}

    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_test: np.ndarray, y_test: np.ndarray):
        """Train all ML models"""
        
        logger.info("Starting ML model training...")
        
        # Logistic Regression
        logger.info("Training Logistic Regression...")
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        
        # Random Forest
        logger.info("Training Random Forest...")
        self.train_random_forest(X_train, y_train, X_test, y_test)
        
        # SVM
        logger.info("Training SVM...")
        self.train_svm(X_train, y_train, X_test, y_test)
        
        # XGBoost
        logger.info("Training XGBoost...")
        self.train_xgboost(X_train, y_train, X_test, y_test)
        
        logger.info("ML model training completed")

    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray):
        """Train Logistic Regression model"""
        try:
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='multinomial',
                solver='lbfgs'
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Logistic Regression Accuracy: {accuracy:.4f}")
            
            # Save model
            joblib.dump(model, self.models_dir / "logistic_regression_model.pkl")
            self.models['logistic_regression'] = model
            
        except Exception as e:
            logger.error(f"Error training Logistic Regression: {e}")

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray):
        """Train Random Forest model"""
        try:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Random Forest Accuracy: {accuracy:.4f}")
            
            # Save model
            joblib.dump(model, self.models_dir / "random_forest_model.pkl")
            self.models['random_forest'] = model
            
        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")

    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray):
        """Train SVM model"""
        try:
            model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"SVM Accuracy: {accuracy:.4f}")
            
            # Save model
            joblib.dump(model, self.models_dir / "svm_model.pkl")
            self.models['svm'] = model
            
        except Exception as e:
            logger.error(f"Error training SVM: {e}")

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray):
        """Train XGBoost model"""
        try:
            # Determine number of classes
            n_classes = len(np.unique(y_train))
            objective = 'binary:logistic' if n_classes == 2 else 'multi:softprob'
            
            model = xgb.XGBClassifier(
                objective=objective,
                num_class=n_classes if n_classes > 2 else None,
                max_depth=7,
                learning_rate=0.1,
                n_estimators=200,
                random_state=42,
                eval_metric='mlogloss' if n_classes > 2 else 'logloss',
                tree_method='hist'
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"XGBoost Accuracy: {accuracy:.4f}")
            
            # Save model
            model.save_model(str(self.models_dir / "xgboost_model.json"))
            self.models['xgboost'] = model
            
        except Exception as e:
            logger.error(f"Error training XGBoost: {e}")

    def load_models(self):
        """Load all trained models"""
        try:
            if (self.models_dir / "logistic_regression_model.pkl").exists():
                self.models['logistic_regression'] = joblib.load(
                    self.models_dir / "logistic_regression_model.pkl"
                )
            
            if (self.models_dir / "random_forest_model.pkl").exists():
                self.models['random_forest'] = joblib.load(
                    self.models_dir / "random_forest_model.pkl"
                )
            
            if (self.models_dir / "svm_model.pkl").exists():
                self.models['svm'] = joblib.load(
                    self.models_dir / "svm_model.pkl"
                )
            
            if (self.models_dir / "xgboost_model.json").exists():
                self.models['xgboost'] = xgb.XGBClassifier()
                self.models['xgboost'].load_model(
                    str(self.models_dir / "xgboost_model.json")
                )
            
            logger.info(f"Loaded {len(self.models)} ML models")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions with all models and return averaged results
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with predictions and probabilities from all models
        """
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X)
                prob = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
                
                predictions[model_name] = pred[0] if len(pred.shape) > 1 else pred
                probabilities[model_name] = prob[0] if prob is not None else None
                
            except Exception as e:
                logger.error(f"Error in prediction with {model_name}: {e}")
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'ensemble_prediction': self._ensemble_predict(predictions)
        }

    def _ensemble_predict(self, predictions: Dict[str, Any]) -> int:
        """Ensemble prediction using majority voting"""
        if not predictions:
            return 0
        
        pred_values = list(predictions.values())
        unique, counts = np.unique(pred_values, return_counts=True)
        return int(unique[np.argmax(counts)])

    def get_feature_importance(self, model_name: str = 'random_forest') -> Dict[str, float]:
        """Get feature importance from tree-based models"""
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        if not hasattr(model, 'feature_importances_'):
            return {}
        
        importances = model.feature_importances_
        return {f"feature_{i}": importance for i, importance in enumerate(importances)}
