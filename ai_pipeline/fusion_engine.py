"""
Fusion Engine for Hybrid AI Predictions
Combines ML (tabular) and DL (imaging) predictions with intelligent weighting
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Different fusion strategies for combining ML and DL predictions."""
    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    STACKING = "stacking"
    DYNAMIC_WEIGHT = "dynamic_weight"


class FusionEngine:
    """
    Hybrid fusion engine that combines ML and DL predictions
    using intelligent weighting and ensemble techniques.
    """
    
    STAGE_MAPPING = {
        0: "Stage 0 - Normal",
        1: "Stage 1 - Mild Fibrosis",
        2: "Stage 2 - Moderate Fibrosis",
        3: "Stage 3 - Severe Fibrosis",
        4: "Stage 4 - Cirrhosis"
    }
    
    RISK_MAPPING = {
        0: "low",
        1: "low",
        2: "medium",
        3: "high",
        4: "critical"
    }
    
    def __init__(self, strategy: FusionStrategy = FusionStrategy.DYNAMIC_WEIGHT,
                 ml_weight: float = 0.4, dl_weight: float = 0.6,
                 config_path: Optional[str] = None):
        """
        Initialize fusion engine.
        
        Args:
            strategy: Fusion strategy to use
            ml_weight: Weight for ML predictions (if using weighted average)
            dl_weight: Weight for DL predictions (if using weighted average)
            config_path: Path to load fusion configuration
        """
        self.strategy = strategy
        self.ml_weight = ml_weight
        self.dl_weight = dl_weight
        self.meta_learner = None
        
        if config_path and Path(config_path).exists():
            self._load_config(config_path)
    
    def fuse_predictions(self,
                        ml_prediction: Dict[str, Any],
                        dl_prediction: Optional[Dict[str, Any]] = None,
                        confidence_scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Fuse ML and DL predictions using the configured strategy.
        
        Args:
            ml_prediction: ML model prediction dict with 'stage' and 'probabilities'
            dl_prediction: DL model prediction dict with 'stage' and 'probabilities'
            confidence_scores: Custom confidence scores for each model
        
        Returns:
            Fused prediction dictionary with final stage and recommendation
        """
        logger.info(f"Fusing predictions using {self.strategy.value} strategy...")
        
        # If only ML prediction available, return it with lower confidence
        if dl_prediction is None:
            logger.warning("No DL prediction available. Using ML prediction only.")
            return self._single_model_prediction(ml_prediction, source='ML')
        
        # Route to appropriate fusion method
        if self.strategy == FusionStrategy.WEIGHTED_AVERAGE:
            fused = self._weighted_average_fusion(ml_prediction, dl_prediction)
        elif self.strategy == FusionStrategy.VOTING:
            fused = self._voting_fusion(ml_prediction, dl_prediction)
        elif self.strategy == FusionStrategy.DYNAMIC_WEIGHT:
            fused = self._dynamic_weight_fusion(ml_prediction, dl_prediction, confidence_scores)
        elif self.strategy == FusionStrategy.STACKING:
            fused = self._stacking_fusion(ml_prediction, dl_prediction)
        else:
            logger.warning(f"Unknown strategy: {self.strategy}. Using weighted average.")
            fused = self._weighted_average_fusion(ml_prediction, dl_prediction)
        
        # Calculate ensemble confidence
        fused['ensemble_confidence'] = self._calculate_ensemble_confidence(
            ml_prediction, dl_prediction, fused
        )
        
        # Generate recommendations
        fused['recommendations'] = self._generate_recommendations(fused)
        
        return fused
    
    def _weighted_average_fusion(self, ml_pred: Dict,
                                dl_pred: Dict) -> Dict[str, Any]:
        """Weighted average of ML and DL probability distributions."""
        logger.info(f"Using weighted average: ML={self.ml_weight}, DL={self.dl_weight}")
        
        # Get probability distributions
        ml_probs = np.array(ml_pred.get('probabilities', []))
        dl_probs = np.array(dl_pred.get('probabilities', []))
        
        # Ensure same length
        min_len = min(len(ml_probs), len(dl_probs))
        ml_probs = ml_probs[:min_len]
        dl_probs = dl_probs[:min_len]
        
        # Weighted average
        fused_probs = (self.ml_weight * ml_probs + self.dl_weight * dl_probs)
        
        # Normalize
        fused_probs = fused_probs / np.sum(fused_probs)
        
        # Get predicted stage
        predicted_stage = int(np.argmax(fused_probs))
        confidence = float(np.max(fused_probs))
        
        return {
            'stage': predicted_stage,
            'stage_name': self.STAGE_MAPPING[predicted_stage],
            'confidence': confidence,
            'probabilities': fused_probs.tolist(),
            'ml_contribution': self.ml_weight,
            'dl_contribution': self.dl_weight,
            'fusion_method': 'Weighted Average'
        }
    
    def _voting_fusion(self, ml_pred: Dict, dl_pred: Dict) -> Dict[str, Any]:
        """Majority voting between ML and DL predictions."""
        logger.info("Using majority voting strategy")
        
        ml_stage = int(ml_pred.get('stage', 0))
        dl_stage = int(dl_pred.get('stage', 0))
        
        ml_conf = ml_pred.get('confidence', 0.5)
        dl_conf = dl_pred.get('confidence', 0.5)
        
        # Confidence-weighted voting
        if ml_conf * ml_stage + dl_conf * dl_stage > 2:
            predicted_stage = max(ml_stage, dl_stage)
        else:
            predicted_stage = min(ml_stage, dl_stage)
        
        average_conf = (ml_conf + dl_conf) / 2
        
        # Get probability distributions if available
        ml_probs = np.array(ml_pred.get('probabilities', []))
        dl_probs = np.array(dl_pred.get('probabilities', []))
        
        if len(ml_probs) > 0 and len(dl_probs) > 0:
            fused_probs = (ml_probs + dl_probs) / 2
        else:
            fused_probs = np.zeros(5)
            fused_probs[predicted_stage] = 1.0
        
        return {
            'stage': predicted_stage,
            'stage_name': self.STAGE_MAPPING[predicted_stage],
            'confidence': average_conf,
            'probabilities': fused_probs.tolist(),
            'ml_stage': ml_stage,
            'dl_stage': dl_stage,
            'fusion_method': 'Majority Voting'
        }
    
    def _dynamic_weight_fusion(self, ml_pred: Dict, dl_pred: Dict,
                              confidence_scores: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Dynamic weighting based on model confidence.
        Adjusts weights based on individual model's confidence in their predictions.
        """
        logger.info("Using dynamic weight strategy")
        
        # Get confidence scores
        ml_conf = ml_pred.get('confidence', 0.5)
        dl_conf = dl_pred.get('confidence', 0.5)
        
        # Normalize confidences as weights
        total_conf = ml_conf + dl_conf
        if total_conf > 0:
            dynamic_ml_weight = ml_conf / total_conf
            dynamic_dl_weight = dl_conf / total_conf
        else:
            dynamic_ml_weight = self.ml_weight
            dynamic_dl_weight = self.dl_weight
        
        logger.info(f"Dynamic weights: ML={dynamic_ml_weight:.3f}, DL={dynamic_dl_weight:.3f}")
        
        # Get probability distributions
        ml_probs = np.array(ml_pred.get('probabilities', []))
        dl_probs = np.array(dl_pred.get('probabilities', []))
        
        # Ensure same length
        min_len = min(len(ml_probs), len(dl_probs))
        ml_probs = ml_probs[:min_len]
        dl_probs = dl_probs[:min_len]
        
        # Dynamic weighted average
        fused_probs = (dynamic_ml_weight * ml_probs + dynamic_dl_weight * dl_probs)
        
        # Normalize
        fused_probs = fused_probs / np.sum(fused_probs)
        
        # Get predicted stage
        predicted_stage = int(np.argmax(fused_probs))
        confidence = float(np.max(fused_probs))
        
        return {
            'stage': predicted_stage,
            'stage_name': self.STAGE_MAPPING[predicted_stage],
            'confidence': confidence,
            'probabilities': fused_probs.tolist(),
            'ml_weight': dynamic_ml_weight,
            'dl_weight': dynamic_dl_weight,
            'ml_confidence': ml_conf,
            'dl_confidence': dl_conf,
            'fusion_method': 'Dynamic Weighting'
        }
    
    def _stacking_fusion(self, ml_pred: Dict, dl_pred: Dict) -> Dict[str, Any]:
        """
        Stacking approach - use predictions as features for meta-learner.
        Currently uses a simple rule-based approach.
        """
        logger.info("Using stacking approach")
        
        ml_stage = int(ml_pred.get('stage', 0))
        dl_stage = int(dl_pred.get('stage', 0))
        ml_conf = ml_pred.get('confidence', 0.5)
        dl_conf = dl_pred.get('confidence', 0.5)
        
        # Meta-learner rule: if predictions agree, increase confidence
        if ml_stage == dl_stage:
            predicted_stage = ml_stage
            # Boost confidence if models agree
            confidence = (ml_conf + dl_conf) / 2 * 1.1
        else:
            # If predictions differ, prefer high-confidence prediction
            if ml_conf > dl_conf * 1.2:
                predicted_stage = ml_stage
                confidence = ml_conf * 0.95  # Reduce confidence due to disagreement
            elif dl_conf > ml_conf * 1.2:
                predicted_stage = dl_stage
                confidence = dl_conf * 0.95
            else:
                # Both roughly similar, take average
                predicted_stage = int(np.round((ml_stage + dl_stage) / 2))
                confidence = (ml_conf + dl_conf) / 2 * 0.9
        
        confidence = float(np.clip(confidence, 0, 1))
        
        fused_probs = np.zeros(5)
        fused_probs[predicted_stage] = confidence
        fused_probs = fused_probs / np.sum(fused_probs)
        
        return {
            'stage': predicted_stage,
            'stage_name': self.STAGE_MAPPING[predicted_stage],
            'confidence': confidence,
            'probabilities': fused_probs.tolist(),
            'agreement': ml_stage == dl_stage,
            'ml_stage': ml_stage,
            'dl_stage': dl_stage,
            'fusion_method': 'Stacking'
        }
    
    def _single_model_prediction(self, prediction: Dict,
                                source: str = 'Unknown') -> Dict[str, Any]:
        """Handle single model prediction when other is unavailable."""
        stage = int(prediction.get('stage', 0))
        confidence = prediction.get('confidence', 0.5) * 0.85  # Reduce confidence
        probs = prediction.get('probabilities', [])
        
        return {
            'stage': stage,
            'stage_name': self.STAGE_MAPPING[stage],
            'confidence': confidence,
            'probabilities': probs,
            'source': source,
            'warning': f"Only {source} prediction available. DL/ML prediction was missing.",
            'fusion_method': 'Single Model (Fallback)'
        }
    
    def _calculate_ensemble_confidence(self, ml_pred: Dict, dl_pred: Dict,
                                      fused: Dict) -> float:
        """Calculate overall ensemble confidence."""
        ml_conf = ml_pred.get('confidence', 0.5)
        dl_conf = dl_pred.get('confidence', 0.5)
        fused_conf = fused.get('confidence', 0.5)
        
        # Check agreement
        ml_stage = int(ml_pred.get('stage', 0))
        dl_stage = int(dl_pred.get('stage', 0))
        fused_stage = fused.get('stage', 0)
        
        # Boost confidence if all agree
        agreement_bonus = 0
        if ml_stage == dl_stage == fused_stage:
            agreement_bonus = 0.05
        elif (ml_stage == fused_stage or dl_stage == fused_stage):
            agreement_bonus = 0.02
        
        ensemble_conf = float(np.clip(fused_conf + agreement_bonus, 0, 1))
        return ensemble_conf
    
    def _generate_recommendations(self, fused_pred: Dict) -> Dict[str, Any]:
        """Generate clinical recommendations based on fused prediction."""
        stage = fused_pred.get('stage', 0)
        confidence = fused_pred.get('ensemble_confidence', fused_pred.get('confidence', 0.5))
        
        recommendations = {
            0: {
                'clinical': "No cirrhosis detected. Liver appears normal.",
                'actions': ["Routine health check-ups", "Maintain lifestyle"],
                'follow_up_months': 12
            },
            1: {
                'clinical': "Mild fibrosis detected. Early intervention may help prevent progression.",
                'actions': ["Follow-up ultrasound in 6-12 months", "Hepatologist consultation"],
                'follow_up_months': 12
            },
            2: {
                'clinical': "Moderate fibrosis detected. Medical management is recommended.",
                'actions': ["Urgent hepatologist consultation", "Consider treatment options"],
                'follow_up_months': 6
            },
            3: {
                'clinical': "Severe fibrosis/early cirrhosis detected. Immediate specialist care is urgent.",
                'actions': ["Emergency hepatology referral", "Screen for complications"],
                'follow_up_months': 3
            },
            4: {
                'clinical': "Cirrhosis detected. Comprehensive cirrhosis management is critical.",
                'actions': ["Immediate hospitalization if needed", "Transplant evaluation", "Varices screening"],
                'follow_up_months': 1
            }
        }
        
        rec = recommendations.get(stage, recommendations[0])
        
        confidence_text = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "low"
        
        return {
            'clinical_summary': rec['clinical'],
            'recommended_actions': rec['actions'],
            'follow_up_interval_months': rec['follow_up_months'],
            'confidence_level': confidence_text,
            'specialist_referral_required': stage >= 2
        }
    
    def _load_config(self, config_path: str):
        """Load fusion configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if 'ml_weight' in config:
                self.ml_weight = config['ml_weight']
            if 'dl_weight' in config:
                self.dl_weight = config['dl_weight']
            
            logger.info(f"Fusion config loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading fusion config: {e}")
    
    def save_config(self, output_path: str):
        """Save fusion configuration to JSON file."""
        config = {
            'strategy': self.strategy.value,
            'ml_weight': self.ml_weight,
            'dl_weight': self.dl_weight
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Fusion config saved to {output_path}")


def create_fusion_engine(strategy: str = 'dynamic_weight') -> FusionEngine:
    """Factory function to create fusion engine."""
    strategy_enum = FusionStrategy(strategy.lower())
    return FusionEngine(strategy=strategy_enum)


if __name__ == "__main__":
    engine = create_fusion_engine()
    print("Fusion Engine initialized successfully")
    print(f"Strategy: {engine.strategy.value}")
