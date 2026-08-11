"""
Backend Integration Module
Integrates ML, DL, XAI, and Fusion engines with the FastAPI backend
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import base64
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import AI engines
try:
    from ml_engine import MLEngine
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("MLEngine not available")

try:
    from dl_engine import DLEngine
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    logger.warning("DLEngine not available")

try:
    from xai_engine import create_xai_engine
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    logger.warning("XAIEngine not available")

try:
    from fusion_engine import create_fusion_engine
    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False
    logger.warning("FusionEngine not available")


class AISystemIntegration:
    """
    Integrated AI system that combines ML, DL, XAI, and Fusion engines
    for end-to-end liver cirrhosis diagnosis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the integrated AI system.
        
        Args:
            config: Configuration dictionary for model paths and settings
        """
        self.config = config or self._get_default_config()
        
        # Initialize engines
        self.ml_engine = None
        self.dl_engine = None
        self.xai_engine = None
        self.fusion_engine = None
        
        self._initialize_engines()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'ml_model_path': 'models/ml_model.pkl',
            'dl_model_path': 'models/dl_model.h5',
            'fusion_strategy': 'dynamic_weight',
            'ml_weight': 0.4,
            'dl_weight': 0.6
        }
    
    def _initialize_engines(self):
        """Initialize all AI engines."""
        logger.info("Initializing AI System...")
        
        # ML Engine
        if ML_AVAILABLE:
            try:
                self.ml_engine = MLEngine(
                    model_path=self.config.get('ml_model_path', 'models/ml_model.pkl')
                )
                self.ml_engine.load()
                logger.info("✓ ML Engine initialized")
            except Exception as e:
                logger.error(f"✗ ML Engine initialization failed: {e}")
        
        # DL Engine
        if DL_AVAILABLE:
            try:
                self.dl_engine = DLEngine(
                    model_path=self.config.get('dl_model_path', 'models/dl_model.h5')
                )
                self.dl_engine.load()
                logger.info("✓ DL Engine initialized")
            except Exception as e:
                logger.error(f"✗ DL Engine initialization failed: {e}")
        
        # XAI Engine
        if XAI_AVAILABLE:
            try:
                self.xai_engine = create_xai_engine()
                logger.info("✓ XAI Engine initialized")
            except Exception as e:
                logger.error(f"✗ XAI Engine initialization failed: {e}")
        
        # Fusion Engine
        if FUSION_AVAILABLE:
            try:
                fusion_strategy = self.config.get('fusion_strategy', 'dynamic_weight')
                self.fusion_engine = create_fusion_engine(fusion_strategy)
                logger.info(f"✓ Fusion Engine initialized ({fusion_strategy})")
            except Exception as e:
                logger.error(f"✗ Fusion Engine initialization failed: {e}")
        
        logger.info("AI System initialization complete")
    
    def predict_from_symptoms(self, symptoms: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get prediction from tabular symptom data.
        
        Args:
            symptoms: Dictionary containing symptom features
        
        Returns:
            Prediction dictionary with stage and probabilities
        """
        if self.ml_engine is None:
            logger.error("ML Engine not available")
            return {'error': 'ML Engine not available'}
        
        try:
            # Prepare feature vector
            features = self._prepare_symptom_features(symptoms)
            
            # Get prediction
            stage, confidence, probabilities = self.ml_engine.predict(features)
            
            prediction = {
                'source': 'ml',
                'stage': int(stage),
                'confidence': float(confidence),
                'probabilities': [float(p) for p in probabilities]
            }
            
            logger.info(f"ML Prediction: Stage {stage} (confidence: {confidence:.3f})")
            return prediction
            
        except Exception as e:
            logger.error(f"Error in symptom prediction: {e}")
            return {'error': str(e)}
    
    def predict_from_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Get prediction from medical image.
        
        Args:
            image_data: Image bytes (JPEG, PNG, etc.)
        
        Returns:
            Prediction dictionary with stage and probabilities
        """
        if self.dl_engine is None:
            logger.error("DL Engine not available")
            return {'error': 'DL Engine not available'}
        
        try:
            # Convert bytes to numpy array
            image = self._load_image(image_data)
            
            # Get prediction
            stage, confidence, probabilities = self.dl_engine.predict(image)
            
            prediction = {
                'source': 'dl',
                'stage': int(stage),
                'confidence': float(confidence),
                'probabilities': [float(p) for p in probabilities]
            }
            
            logger.info(f"DL Prediction: Stage {stage} (confidence: {confidence:.3f})")
            return prediction
            
        except Exception as e:
            logger.error(f"Error in image prediction: {e}")
            return {'error': str(e)}
    
    def predict_hybrid(self, symptoms: Optional[Dict] = None,
                      image_data: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Get hybrid prediction from both symptoms and image.
        
        Args:
            symptoms: Symptom data dictionary
            image_data: Image bytes
        
        Returns:
            Fused prediction with explanations
        """
        logger.info("Starting hybrid prediction...")
        
        ml_pred = None
        dl_pred = None
        
        # Get ML prediction
        if symptoms and self.ml_engine:
            ml_pred = self.predict_from_symptoms(symptoms)
        
        # Get DL prediction
        if image_data and self.dl_engine:
            dl_pred = self.predict_from_image(image_data)
        
        # Fuse predictions
        if self.fusion_engine and (ml_pred or dl_pred):
            if ml_pred and dl_pred:
                fused = self.fusion_engine.fuse_predictions(ml_pred, dl_pred)
            elif ml_pred:
                fused = self.fusion_engine.fuse_predictions(ml_pred, None)
            else:
                fused = self.fusion_engine.fuse_predictions(dl_pred, None)
        else:
            logger.warning("Fusion engine not available, using single prediction")
            fused = ml_pred or dl_pred or {'error': 'No models available'}
        
        # Generate XAI explanations
        xai_report = self._generate_xai_explanations(
            symptoms, image_data, ml_pred, dl_pred, fused
        )
        
        # Combine results
        result = {
            'prediction': fused,
            'xai_report': xai_report,
            'ml_prediction': ml_pred,
            'dl_prediction': dl_pred
        }
        
        return result
    
    def _generate_xai_explanations(self, symptoms: Optional[Dict],
                                  image_data: Optional[bytes],
                                  ml_pred: Optional[Dict],
                                  dl_pred: Optional[Dict],
                                  fused_pred: Dict) -> Dict[str, Any]:
        """Generate XAI explanations for predictions."""
        if self.xai_engine is None:
            logger.warning("XAI Engine not available")
            return {}
        
        try:
            xai_report = {
                'tabular_analysis': {},
                'imaging_analysis': {}
            }
            
            # SHAP explanation for tabular data
            if symptoms and ml_pred and self.ml_engine:
                try:
                    ml_explanation = self.xai_engine.generate_shap_explanation(
                        self.ml_engine.model,
                        np.random.rand(100, len(self._prepare_symptom_features(symptoms)[0])),
                        self._prepare_symptom_features(symptoms),
                        self._get_feature_names()
                    )
                    xai_report['tabular_analysis'] = ml_explanation
                except Exception as e:
                    logger.error(f"Error generating SHAP explanation: {e}")
            
            # Grad-CAM explanation for image
            if image_data and dl_pred and self.dl_engine:
                try:
                    image = self._load_image(image_data)
                    dl_explanation = self.xai_engine.generate_gradcam_heatmap(
                        self.dl_engine.model,
                        image
                    )
                    xai_report['imaging_analysis'] = dl_explanation
                except Exception as e:
                    logger.error(f"Error generating Grad-CAM explanation: {e}")
            
            # Clinical interpretation
            xai_report['clinical_interpretation'] = \
                self.xai_engine._generate_clinical_interpretation(fused_pred)
            xai_report['recommendations'] = \
                self.xai_engine._generate_recommendations(fused_pred)
            
            return xai_report
            
        except Exception as e:
            logger.error(f"Error in XAI report generation: {e}")
            return {}
    
    def _prepare_symptom_features(self, symptoms: Dict) -> np.ndarray:
        """Prepare symptom features into ML-ready format."""
        # This should match the feature order used during training
        feature_order = [
            'age', 'gender', 'alcohol_consumption', 'bilirubin', 'albumin',
            'ast', 'alp', 'alt', 'platelets', 'prothrombin', 'creatinine',
            'ascites', 'hepatomegaly', 'spiders', 'edema', 'encephalopathy',
            'fatigue', 'jaundice', 'abdominal_pain', 'weight_loss'
        ]
        
        features = []
        for feature in feature_order:
            value = symptoms.get(feature, 0)
            # Convert boolean to int
            if isinstance(value, bool):
                value = int(value)
            features.append(float(value))
        
        return np.array([features], dtype=np.float32)
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for explanations."""
        return [
            'age', 'gender', 'alcohol_consumption', 'bilirubin', 'albumin',
            'ast', 'alp', 'alt', 'platelets', 'prothrombin', 'creatinine',
            'ascites', 'hepatomegaly', 'spiders', 'edema', 'encephalopathy',
            'fatigue', 'jaundice', 'abdominal_pain', 'weight_loss'
        ]
    
    def _load_image(self, image_data: bytes) -> np.ndarray:
        """Load image from bytes."""
        image = Image.open(BytesIO(image_data))
        image_array = np.array(image)
        
        # Convert RGBA to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        # Convert grayscale to RGB
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        
        return image_array
    
    def health_check(self) -> Dict[str, Any]:
        """Check health and status of all engines."""
        return {
            'ml_engine': 'available' if self.ml_engine else 'unavailable',
            'dl_engine': 'available' if self.dl_engine else 'unavailable',
            'xai_engine': 'available' if self.xai_engine else 'unavailable',
            'fusion_engine': 'available' if self.fusion_engine else 'unavailable',
            'overall_status': 'operational' if (self.ml_engine or self.dl_engine) else 'degraded'
        }


def create_ai_system(config: Optional[Dict] = None) -> AISystemIntegration:
    """Factory function to create integrated AI system."""
    return AISystemIntegration(config)


if __name__ == "__main__":
    system = create_ai_system()
    print("AI System Integration initialized")
    print("Health Status:", system.health_check())
