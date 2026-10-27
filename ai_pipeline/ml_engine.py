"""
Machine Learning Engine for Tabular Data
Trains XGBoost models for liver cirrhosis staging based on clinical features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.utils import class_weight
import xgboost as xgb
import joblib
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLEngine:
    """XGBoost-based ML engine for symptom-based liver cirrhosis prediction."""
    
    FEATURE_LIST = [
        'age', 'gender', 'alcohol_consumption', 'bilirubin', 'albumin',
        'ast', 'alp', 'alt', 'platelets', 'prothrombin', 'creatinine',
        'ascites', 'hepatomegaly', 'spiders', 'edema', 'encephalopathy',
        'fatigue', 'jaundice', 'abdominal_pain', 'weight_loss'
    ]
    
    STAGE_MAPPING = {
        0: "Normal",
        1: "Stage 1 - Mild Fibrosis",
        2: "Stage 2 - Moderate Fibrosis",
        3: "Stage 3 - Severe Fibrosis",
        4: "Stage 4 - Cirrhosis"
    }
    
    def __init__(self, model_path: str = 'models/ml_model.pkl', 
                 scaler_path: str = 'models/ml_scaler.pkl',
                 encoder_path: str = 'models/ml_encoder.pkl'):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.encoder_path = Path(encoder_path)
        
        self.model = None
        self.scaler = None
        self.encoder = None
        self.feature_importance = None
        
        # Ensure directories exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
    def load_and_preprocess_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess tabular data."""
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))
        df = df.fillna(df.mode().iloc[0])
        
        # Encode categorical variables if needed
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'Status':  # Don't encode target
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        
        logger.info(f"Data shape after preprocessing: {df.shape}")
        return df
    
    def prepare_features_and_target(self, df: pd.DataFrame, 
                                   target_col: str = 'Status') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target from dataframe."""
        # Select available features
        available_features = [f for f in self.FEATURE_LIST if f in df.columns]
        
        if len(available_features) == 0:
            # Use all numeric columns except target
            available_features = [col for col in df.columns 
                                 if col != target_col and df[col].dtype in ['int64', 'float64']]
        
        X = df[available_features].values.astype(np.float32)
        
        # Encode target if it's categorical or numeric labels are not zero-based
        if df[target_col].dtype == 'object':
            self.encoder = LabelEncoder()
            y = self.encoder.fit_transform(df[target_col].astype(str))
            joblib.dump(self.encoder, self.encoder_path)
        else:
            y_raw = df[target_col].values.astype(np.int32)
            unique_labels = np.unique(y_raw)
            if not np.array_equal(unique_labels, np.arange(unique_labels.size)):
                self.encoder = LabelEncoder()
                y = self.encoder.fit_transform(y_raw)
                joblib.dump(self.encoder, self.encoder_path)
            else:
                y = y_raw

        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X, y, available_features
    
    def train(self, data_path: str, target_col: str = 'Status', 
              test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """Train the XGBoost model."""
        logger.info("Starting ML model training...")
        
        # Load and preprocess data
        df = self.load_and_preprocess_data(data_path)
        X, y, feature_names = self.prepare_features_and_target(df, target_col)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Scaler saved to {self.scaler_path}")
        
        # Calculate class weights for imbalanced data
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        
        # Train XGBoost model
        logger.info("Training XGBoost classifier...")
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0,
            reg_alpha=0.1,
            reg_lambda=1,
            objective='multi:softprob',
            num_class=len(np.unique(y)),
            eval_metric='mlogloss',
            random_state=random_state,
            n_jobs=-1,
            tree_method='hist'
        )
        
        # Train with early stopping
        eval_set = [(X_test_scaled, y_test)]
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Save model
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        }
        
        # Calculate multi-class ROC-AUC if applicable
        if len(np.unique(y)) > 2:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted'))
            except:
                metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba[:, 1]))
        
        # Feature importance
        self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        logger.info("=" * 50)
        logger.info("MODEL TRAINING COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1']:.4f}")
        if metrics['roc_auc']:
            logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info("=" * 50)
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, zero_division=0))
        
        # Save metadata
        metadata = {
            'features': feature_names,
            'num_classes': len(np.unique(y)),
            'stage_mapping': self.STAGE_MAPPING,
            'metrics': metrics,
            'feature_importance': self.feature_importance
        }
        
        with open(self.model_path.parent / 'ml_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metrics
    
    def predict(self, features: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Make predictions with confidence scores."""
        if self.model is None:
            self.load()
        
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        prediction = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        confidence = np.max(probabilities)
        
        return prediction[0], float(confidence), probabilities[0]
    
    def save(self):
        """Save model artifacts."""
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
    
    def load(self):
        """Load model artifacts."""
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        
        if self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"Scaler loaded from {self.scaler_path}")
        
        if self.encoder_path.exists():
            self.encoder = joblib.load(self.encoder_path)
            logger.info(f"Encoder loaded from {self.encoder_path}")


if __name__ == "__main__":
    # Example usage
    ml_engine = MLEngine()
    
    # For actual training, you would use:
    # metrics = ml_engine.train('datasets/cirrhosis.csv')
    
    print("ML Engine initialized. Use ml_engine.train() to train the model.")
