"""
Explainable AI (XAI) Module
Generates SHAP explanations for tabular data and Grad-CAM for medical images
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json
import base64
import io
import cv2
from PIL import Image

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

try:
    from pytorch_grad_cam import GradCAM, EigenCAM, LayerCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import torch
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XAIEngine:
    """Explainable AI engine for generating model explanations."""
    
    def __init__(self):
        self.shap_explainer = None
        self.feature_names = None
        
    # ==================== TABULAR DATA (SHAP) ====================
    
    def generate_shap_explanation(self, model, X_background: np.ndarray,
                                 X_explain: np.ndarray,
                                 feature_names: List[str],
                                 max_samples: int = 100) -> Dict[str, Any]:
        """Generate SHAP explanations for tabular predictions."""
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
            return self._fallback_feature_importance(model, feature_names)
        
        logger.info("Generating SHAP explanations...")
        
        try:
            # Use background data for SHAP explainer
            background = shap.sample(X_background, min(max_samples, len(X_background)))
            
            # Create explainer (TreeExplainer for XGBoost)
            if hasattr(model, 'get_booster'):  # XGBoost
                self.shap_explainer = shap.TreeExplainer(model)
            else:
                self.shap_explainer = shap.KernelExplainer(
                    model.predict_proba, 
                    background
                )
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(X_explain)
            
            # Handle multi-class SHAP values
            if isinstance(shap_values, list):
                # Use the positive class (cirrhosis) SHAP values
                shap_array = shap_values[-1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_array = shap_values
            
            # Calculate mean absolute SHAP values for each feature
            mean_abs_shap = np.abs(shap_array).mean(axis=0)
            
            # Create explanation dictionary
            explanation = {
                'feature_importance': {
                    feature_names[i]: float(mean_abs_shap[i])
                    for i in range(len(feature_names))
                },
                'shap_values': shap_array[0].tolist() if len(shap_array.shape) > 1 else shap_array.tolist(),
                'base_value': float(self.shap_explainer.expected_value),
                'method': 'SHAP TreeExplainer'
            }
            
            logger.info("SHAP explanation generated successfully")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return self._fallback_feature_importance(model, feature_names)
    
    def _fallback_feature_importance(self, model, feature_names: List[str]) -> Dict[str, Any]:
        """Fallback feature importance if SHAP is unavailable."""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                explanation = {
                    'feature_importance': {
                        feature_names[i]: float(importance[i])
                        for i in range(len(feature_names))
                    },
                    'method': 'Model Feature Importance (Fallback)'
                }
                return explanation
        except Exception as e:
            logger.error(f"Error in fallback feature importance: {e}")
        
        return {
            'feature_importance': {f: 1.0/len(feature_names) for f in feature_names},
            'method': 'Uniform Weights (Fallback)'
        }
    
    # ==================== MEDICAL IMAGES (Grad-CAM) ====================
    
    def generate_gradcam_heatmap(self, model, image: np.ndarray,
                                target_layer: Optional[str] = None) -> Dict[str, Any]:
        """Generate Grad-CAM heatmap for CNN predictions."""
        
        try:
            import tensorflow as tf
            return self._gradcam_tensorflow(model, image, target_layer)
        except Exception as e:
            logger.warning(f"TensorFlow Grad-CAM failed: {e}. Trying fallback...")
            return self._gradcam_fallback(model, image)
    
    def _gradcam_tensorflow(self, model, image: np.ndarray,
                           target_layer: Optional[str] = None) -> Dict[str, Any]:
        """Grad-CAM using TensorFlow."""
        import tensorflow as tf
        
        logger.info("Generating Grad-CAM heatmap (TensorFlow)...")
        
        # Preprocess image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        original_image = image.copy()
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Get target layer (last convolutional layer)
        if target_layer is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    target_layer = layer
                    break
        
        if target_layer is None:
            logger.warning("No convolutional layer found")
            return self._gradcam_fallback(model, original_image)
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [target_layer.output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            class_channel = tf.argmax(predictions[0])
            class_loss = predictions[:, class_channel]
        
        # Get gradients
        grads = tape.gradient(class_loss, conv_outputs)
        
        # Global average pooling
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Compute Grad-CAM
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        heatmap_np = heatmap.numpy()
        
        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(heatmap_np, (original_image.shape[1], original_image.shape[0]))
        heatmap_resized = (heatmap_resized * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # Overlay on original image
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) if len(original_image.shape) == 3 else cv2.cvtColor(cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original_rgb, 0.7, cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), 0.3, 0)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', overlay)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        explanation = {
            'heatmap_base64': img_base64,
            'heatmap_overlay': True,
            'method': 'Grad-CAM (TensorFlow)',
            'intensity_map': heatmap_np.tolist()
        }
        
        logger.info("Grad-CAM heatmap generated successfully")
        return explanation
    
    def _gradcam_fallback(self, model, image: np.ndarray) -> Dict[str, Any]:
        """Fallback Grad-CAM visualization."""
        logger.info("Using Grad-CAM fallback method...")
        
        # Resize image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_resized = cv2.resize(image, (224, 224))
        
        # Compute Laplacian (edge detection as proxy for attention)
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Normalize to 0-1
        heatmap = np.abs(laplacian)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
        
        # Apply colormap
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Overlay
        overlay = cv2.addWeighted(image_resized, 0.7, heatmap_color, 0.3, 0)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', overlay)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'heatmap_base64': img_base64,
            'heatmap_overlay': True,
            'method': 'Edge Detection (Fallback)',
            'intensity_map': heatmap.tolist()
        }
    
    # ==================== REPORT GENERATION ====================
    
    def generate_xai_report(self, ml_explanation: Dict,
                           dl_explanation: Dict,
                           prediction: Dict) -> Dict[str, Any]:
        """Generate comprehensive XAI report."""
        
        report = {
            'prediction': prediction,
            'tabular_analysis': {
                'method': ml_explanation.get('method', 'Unknown'),
                'top_features': self._get_top_features(ml_explanation.get('feature_importance', {}), top_k=5),
                'full_feature_importance': ml_explanation.get('feature_importance', {}),
                'base_value': ml_explanation.get('base_value', None)
            },
            'imaging_analysis': {
                'method': dl_explanation.get('method', 'Unknown'),
                'heatmap_present': 'heatmap_base64' in dl_explanation,
                'attention_regions': self._extract_attention_regions(dl_explanation)
            },
            'clinical_interpretation': self._generate_clinical_interpretation(prediction),
            'recommendations': self._generate_recommendations(prediction)
        }
        
        if 'heatmap_base64' in dl_explanation:
            report['imaging_analysis']['heatmap_base64'] = dl_explanation['heatmap_base64']
        
        return report
    
    def _get_top_features(self, feature_importance: Dict, top_k: int = 5) -> List[Dict]:
        """Get top k important features."""
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [
            {'feature': name, 'importance': float(value)}
            for name, value in sorted_features[:top_k]
        ]
    
    def _extract_attention_regions(self, dl_explanation: Dict) -> Dict[str, Any]:
        """Extract high-attention regions from heatmap."""
        try:
            intensity_map = np.array(dl_explanation.get('intensity_map', []))
            if intensity_map.size == 0:
                return {'high_attention_percentage': 0}
            
            high_attention = np.sum(intensity_map > 0.7)
            total_pixels = intensity_map.size
            percentage = float((high_attention / total_pixels) * 100)
            
            return {
                'high_attention_percentage': percentage,
                'max_intensity': float(np.max(intensity_map)),
                'mean_intensity': float(np.mean(intensity_map))
            }
        except Exception as e:
            logger.error(f"Error extracting attention regions: {e}")
            return {}
    
    def _generate_clinical_interpretation(self, prediction: Dict) -> str:
        """Generate clinical interpretation of prediction."""
        
        stage = prediction.get('stage', 'Unknown')
        confidence = prediction.get('confidence', 0)
        risk_level = prediction.get('risk_level', 'unknown')
        
        interpretations = {
            'stage_0': "No cirrhosis detected. Liver appears normal.",
            'stage_1': "Mild fibrosis detected. Early intervention may help.",
            'stage_2': "Moderate fibrosis detected. Medical management is recommended.",
            'stage_3': "Severe fibrosis detected. Specialist consultation is urgent.",
            'stage_4': "Cirrhosis detected. Immediate medical management is critical."
        }
        
        base_interpretation = interpretations.get(stage, "Unable to classify.")
        
        confidence_text = "High" if confidence > 0.8 else "Moderate" if confidence > 0.6 else "Low"
        
        return f"{base_interpretation} (Confidence: {confidence_text})"
    
    def _generate_recommendations(self, prediction: Dict) -> List[str]:
        """Generate clinical recommendations based on prediction."""
        
        stage = prediction.get('stage', 'stage_0')
        recommendations = {
            'stage_0': [
                "Continue regular health check-ups",
                "Maintain healthy lifestyle",
                "Limit alcohol consumption",
                "Repeat screening annually"
            ],
            'stage_1': [
                "Schedule follow-up ultrasound in 6-12 months",
                "Consult hepatologist for monitoring plan",
                "Maintain healthy weight",
                "Avoid hepatotoxic substances"
            ],
            'stage_2': [
                "Urgent hepatologist consultation required",
                "Consider antiviral therapy if applicable",
                "Monitor for portal hypertension",
                "Follow-up imaging in 3-6 months"
            ],
            'stage_3': [
                "Urgent specialist referral required",
                "Evaluate for treatment options",
                "Screen for varices and HCC",
                "Consider liver transplant evaluation"
            ],
            'stage_4': [
                "Immediate hospitalization may be needed",
                "Comprehensive cirrhosis management plan",
                "Hepatic encephalopathy screening",
                "Urgent transplant evaluation"
            ]
        }
        
        return recommendations.get(stage, [])


def create_xai_engine() -> XAIEngine:
    """Factory function to create XAI engine."""
    return XAIEngine()


if __name__ == "__main__":
    engine = create_xai_engine()
    print("XAI Engine initialized successfully")
    print(f"SHAP available: {SHAP_AVAILABLE}")
    print(f"Grad-CAM available: {GRAD_CAM_AVAILABLE}")
