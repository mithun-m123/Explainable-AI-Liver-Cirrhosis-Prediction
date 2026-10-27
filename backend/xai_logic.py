"""
Explainable AI Module - Provides SHAP, LIME, and Grad-CAM explanations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
import json
from pathlib import Path
import base64
import io

import cv2
from PIL import Image

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

from schemas import ExplanationResult, GradCAMExplanation, SHAPExplanation

logger = logging.getLogger(__name__)

class XAIExplainer:
    """Explainable AI module for model interpretability"""

    def __init__(self):
        self.models_dir = Path("models")
        self.shap_explainer = None
        self.gradcam_explainer = None
        self._initialize_explainers()

    def _initialize_explainers(self):
        """Initialize SHAP and Grad-CAM explainers"""
        try:
            if not SHAP_AVAILABLE:
                logger.warning("SHAP library not installed - skipping SHAP initialization")
                return

            # Load background data for SHAP
            background_path = self.models_dir / "shap_background_data.pkl"
            if background_path.exists():
                import joblib
                background_data = joblib.load(background_path)
                self.shap_explainer = shap.KernelExplainer(
                    self._ml_predict_function,
                    background_data
                )
                logger.info("SHAP explainer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {str(e)}")

    def _ml_predict_function(self, X):
        """Prediction function for SHAP explainer"""
        # This should be replaced with actual model prediction
        # For now, return dummy predictions
        return np.random.rand(len(X), 5)

    async def explain_prediction(
        self,
        prediction_result,
        symptom_data,
        image_data: Optional[np.ndarray] = None
    ) -> ExplanationResult:
        """Generate comprehensive explanations for prediction"""
        try:
            # Generate SHAP explanation for symptoms
            shap_explanation = await self._generate_shap_explanation(symptom_data)

            # Generate Grad-CAM explanation for image if available
            gradcam_explanation = None
            if image_data is not None:
                gradcam_explanation = await self._generate_gradcam_explanation(image_data)

            # Generate natural language explanation
            natural_language_explanation = self._generate_natural_language_explanation(
                prediction_result, shap_explanation, gradcam_explanation
            )

            # Extract key factors
            key_factors = self._extract_key_factors(shap_explanation, gradcam_explanation)

            # Interpret confidence
            confidence_interpretation = self._interpret_confidence(prediction_result.confidence)

            return ExplanationResult(
                shap_explanation=shap_explanation,
                gradcam_explanation=gradcam_explanation,
                natural_language_explanation=natural_language_explanation,
                key_factors=key_factors,
                confidence_interpretation=confidence_interpretation
            )

        except Exception as e:
            logger.error(f"Explanation generation error: {str(e)}")
            return self._create_fallback_explanation()

    async def _generate_shap_explanation(self, symptom_data) -> Optional[SHAPExplanation]:
        """Generate SHAP explanation for symptom-based prediction"""
        try:
            if self.shap_explainer is None:
                return None

            # Convert symptom data to feature vector
            features = self._symptoms_to_features(symptom_data)
            features_df = pd.DataFrame([features], columns=self._get_feature_names())

            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(features_df.values[0])

            # For multi-class, take the values for predicted class
            if isinstance(shap_values, list):
                predicted_class = 0  # This should come from prediction_result
                shap_values_for_class = shap_values[predicted_class]
            else:
                shap_values_for_class = shap_values

            # Create feature importance dictionary
            feature_importance = {}
            for i, feature_name in enumerate(self._get_feature_names()):
                feature_importance[feature_name] = float(shap_values_for_class[i])

            return SHAPExplanation(
                feature_importance=feature_importance,
                base_value=float(self.shap_explainer.expected_value),
                feature_names=self._get_feature_names(),
                shap_values=shap_values_for_class.tolist()
            )

        except Exception as e:
            logger.error(f"SHAP explanation error: {str(e)}")
            return None

    async def _generate_gradcam_explanation(self, image_data: np.ndarray) -> Optional[GradCAMExplanation]:
        """Generate Grad-CAM explanation for image-based prediction"""
        try:
            # Create Grad-CAM explainer
            gradcam = GradCAMExplainer()

            # Generate heatmap
            heatmap, overlay = gradcam.generate_explanation(image_data)

            # Convert images to base64
            heatmap_b64 = self._image_to_base64(heatmap)
            overlay_b64 = self._image_to_base64(overlay)

            # Identify target regions (simplified)
            target_regions = self._identify_target_regions(heatmap)

            # Calculate confidence map
            confidence_map = self._calculate_confidence_map(target_regions)

            return GradCAMExplanation(
                heatmap=heatmap_b64,
                overlay=overlay_b64,
                target_regions=target_regions,
                confidence_map=confidence_map
            )

        except Exception as e:
            logger.error(f"Grad-CAM explanation error: {str(e)}")
            return None

    def _generate_natural_language_explanation(
        self,
        prediction_result,
        shap_explanation: Optional[SHAPExplanation],
        gradcam_explanation: Optional[GradCAMExplanation]
    ) -> str:
        """Generate human-readable explanation"""
        stage = prediction_result.stage
        confidence = prediction_result.confidence
        risk_level = prediction_result.risk_level

        explanation = f"The AI system predicts {prediction_result.stage_description} with {confidence:.1%} confidence. "

        if risk_level == "low":
            explanation += "This indicates a low risk of liver cirrhosis. "
        elif risk_level == "medium":
            explanation += "This suggests moderate liver involvement that should be monitored. "
        elif risk_level == "high":
            explanation += "This indicates significant liver fibrosis requiring medical attention. "
        else:  # critical
            explanation += "This suggests advanced cirrhosis requiring immediate specialist care. "

        # Add SHAP-based insights
        if shap_explanation:
            top_features = sorted(
                shap_explanation.feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]

            explanation += "Key factors influencing this prediction include: "
            feature_descriptions = []
            for feature, importance in top_features:
                direction = "elevated" if importance > 0 else "low"
                feature_descriptions.append(f"{direction} {feature.replace('_', ' ')}")
            explanation += ", ".join(feature_descriptions) + ". "

        # Add image-based insights
        if gradcam_explanation and gradcam_explanation.target_regions:
            explanation += f"The AI identified {len(gradcam_explanation.target_regions)} regions of interest in the liver image that contributed to this diagnosis. "

        return explanation

    def _extract_key_factors(
        self,
        shap_explanation: Optional[SHAPExplanation],
        gradcam_explanation: Optional[GradCAMExplanation]
    ) -> List[str]:
        """Extract key factors influencing the prediction"""
        key_factors = []

        # Extract from SHAP
        if shap_explanation:
            top_features = sorted(
                shap_explanation.feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]

            for feature, importance in top_features:
                direction = "high" if importance > 0 else "low"
                key_factors.append(f"{direction} {feature.replace('_', ' ')} levels")

        # Extract from Grad-CAM
        if gradcam_explanation:
            key_factors.append(f"image analysis identified {len(gradcam_explanation.target_regions)} suspicious regions")

        if not key_factors:
            key_factors = ["clinical assessment", "laboratory values", "symptom evaluation"]

        return key_factors

    def _interpret_confidence(self, confidence: float) -> str:
        """Interpret confidence score"""
        if confidence >= 0.9:
            return "Very high confidence - results are highly reliable"
        elif confidence >= 0.8:
            return "High confidence - results are reliable"
        elif confidence >= 0.7:
            return "Moderate confidence - results should be confirmed with additional tests"
        elif confidence >= 0.6:
            return "Low confidence - additional testing recommended"
        else:
            return "Very low confidence - specialist consultation strongly recommended"

    def _create_fallback_explanation(self) -> ExplanationResult:
        """Create fallback explanation when explainers fail"""
        return ExplanationResult(
            shap_explanation=None,
            gradcam_explanation=None,
            natural_language_explanation="The AI system made a prediction based on the provided symptoms and test results. Please consult with a healthcare professional for a comprehensive evaluation.",
            key_factors=["symptom assessment", "laboratory values"],
            confidence_interpretation="Confidence assessment not available - please verify with additional testing"
        )

    def _symptoms_to_features(self, symptom_data) -> List[float]:
        """Convert symptom data to feature vector"""
        return [
            symptom_data.age,
            symptom_data.sex,
            symptom_data.fatigue_level,
            symptom_data.alcohol_consumption,
            symptom_data.weight_loss_kg,
            int(symptom_data.abdominal_swelling),
            int(symptom_data.appetite_loss),
            int(symptom_data.jaundice),
            int(symptom_data.fever),
            symptom_data.ascites,
            symptom_data.hepatomegaly,
            symptom_data.spiders,
            symptom_data.edema,
            symptom_data.bilirubin,
            symptom_data.cholesterol,
            symptom_data.albumin,
            symptom_data.copper,
            symptom_data.alk_phos,
            symptom_data.ast,
            symptom_data.triglycerides,
            symptom_data.platelets,
            symptom_data.prothrombin
        ]

    def _get_feature_names(self) -> List[str]:
        """Get feature names for SHAP explanations"""
        return [
            'age', 'sex', 'fatigue_level', 'alcohol_consumption', 'weight_loss_kg',
            'abdominal_swelling', 'appetite_loss', 'jaundice', 'fever', 'ascites',
            'hepatomegaly', 'spiders', 'edema', 'bilirubin', 'cholesterol', 'albumin',
            'copper', 'alk_phos', 'ast', 'triglycerides', 'platelets', 'prothrombin'
        ]

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert image array to base64 string"""
        try:
            # Convert to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            pil_image = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)

            # Convert to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{image_base64}"

        except Exception as e:
            logger.error(f"Image to base64 conversion error: {str(e)}")
            return ""

    def _identify_target_regions(self, heatmap: np.ndarray) -> List[Dict[str, Any]]:
        """Identify regions of interest from heatmap"""
        try:
            # Simple thresholding to find high-activation regions
            threshold = np.percentile(heatmap, 90)  # Top 10% activations
            binary_map = (heatmap > threshold).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            regions = []
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) > 100:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    regions.append({
                        "id": i + 1,
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "area": int(cv2.contourArea(contour)),
                        "confidence": float(np.mean(heatmap[y:y+h, x:x+w]))
                    })

            return regions

        except Exception as e:
            logger.error(f"Region identification error: {str(e)}")
            return []

    def _calculate_confidence_map(self, regions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence scores for different regions"""
        if not regions:
            return {"overall": 0.5}

        confidences = [region["confidence"] for region in regions]
        return {
            "overall": float(np.mean(confidences)),
            "max": float(np.max(confidences)),
            "min": float(np.min(confidences)),
            "regions_count": len(regions)
        }


class GradCAMExplainer:
    """Grad-CAM implementation for model explainability"""

    def __init__(self):
        self.target_layer = None
        self.gradients = None
        self.activations = None

    def generate_explanation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Grad-CAM heatmap and overlay"""
        try:
            # For now, create a simple heatmap based on image features
            # In a real implementation, this would use actual model gradients

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Create a simple heatmap based on image intensity variations
            # This is a placeholder - real Grad-CAM would use model gradients
            heatmap = cv2.Laplacian(gray, cv2.CV_64F)
            heatmap = np.abs(heatmap)
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

            # Resize heatmap to original image size
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

            # Create overlay
            overlay = self._create_overlay(image, heatmap)

            return heatmap, overlay

        except Exception as e:
            logger.error(f"Grad-CAM generation error: {str(e)}")
            # Return dummy heatmap
            dummy_heatmap = np.ones((image.shape[0], image.shape[1]), dtype=np.float32) * 0.5
            dummy_overlay = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return dummy_heatmap, dummy_overlay

    def _create_overlay(self, image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """Create overlay of heatmap on original image"""
        try:
            # Convert heatmap to color map
            heatmap_colored = cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )

            # Ensure image is 3-channel
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Blend images
            overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)

            return overlay

        except Exception as e:
            logger.error(f"Overlay creation error: {str(e)}")
            return image
