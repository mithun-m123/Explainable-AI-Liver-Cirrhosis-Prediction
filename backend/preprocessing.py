"""
Data Preprocessing Module
Handles missing values, data cleaning, feature scaling, and encoding
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
import cv2
from PIL import Image
import io
import base64
from typing import Dict, Tuple, Any, List
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess patient data and medical records"""

    def __init__(self):
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        self.feature_ranges = {}

    def preprocess_patient_data(self, patient_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess patient symptoms and medical history
        
        Args:
            patient_data: Dictionary with patient information
            
        Returns:
            Tuple of (processed_features, metadata)
        """
        try:
            # Convert to DataFrame for easier processing
            df = pd.DataFrame([patient_data])
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Clean outliers
            df = self._remove_outliers(df)
            
            # Scale features
            features_scaled = self._scale_features(df)
            
            # Get metadata
            metadata = {
                'original_values': patient_data,
                'missing_values_filled': df.isnull().sum().to_dict(),
                'scaling_applied': True
            }
            
            return features_scaled[0], metadata
            
        except Exception as e:
            logger.error(f"Error in patient data preprocessing: {e}")
            raise

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using median imputation"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df

    def _scale_features(self, df: pd.DataFrame) -> np.ndarray:
        """Scale numerical features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df.values


class ImagePreprocessor:
    """Preprocess medical images (Ultrasound, CT, MRI)"""

    def __init__(self):
        self.target_size = (224, 224)  # Standard size for CNN
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def preprocess_image(self, image_data: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess medical image
        
        Args:
            image_data: PIL Image, numpy array, or base64 string
            
        Returns:
            Tuple of (processed_image, metadata)
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(image_data, str):
                image = self._decode_base64(image_data)
            elif isinstance(image_data, np.ndarray):
                image = Image.fromarray(image_data)
            else:
                image = image_data

            # Store original dimensions
            original_size = image.size
            
            # Resize
            image_resized = self._resize_image(image)
            
            # Normalize
            image_normalized = self._normalize_image(image_resized)
            
            # Apply enhancement techniques
            image_enhanced = self._enhance_image(image_normalized)
            
            # Remove noise
            image_denoised = self._denoise_image(image_enhanced)
            
            metadata = {
                'original_size': original_size,
                'processed_size': self.target_size,
                'preprocessing_steps': ['resize', 'normalize', 'enhance', 'denoise'],
                'contrast_enhanced': True,
                'noise_removed': True
            }
            
            return image_denoised, metadata
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            raise

    def _decode_base64(self, image_data: str) -> Image.Image:
        """Decode base64 image string"""
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_bytes))

    def _resize_image(self, image: Image.Image) -> np.ndarray:
        """Resize image to standard size"""
        image = image.resize(self.target_size, Image.LANCZOS)
        return np.array(image)

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image using ImageNet statistics"""
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply channel-wise normalization
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = (image - self.mean) / self.std
        
        return image

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast and brightness"""
        # Convert to uint8 for OpenCV operations
        image_uint8 = ((image + 1) * 127.5).astype(np.uint8)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(image_uint8.shape) == 3:
            # Convert to grayscale, apply CLAHE, then back to color
            gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
            enhanced = clahe.apply(gray)
            image_uint8 = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        else:
            image_uint8 = clahe.apply(image_uint8)
        
        # Normalize back to [-1, 1]
        image = image_uint8.astype(np.float32) / 127.5 - 1
        
        return image

    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from image"""
        image_uint8 = ((image + 1) * 127.5).astype(np.uint8)
        
        if len(image_uint8.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                image_uint8, 
                None, 
                h=10, 
                hForColorComponents=10, 
                templateWindowSize=7, 
                searchWindowSize=21
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                image_uint8, 
                None, 
                h=10, 
                templateWindowSize=7, 
                searchWindowSize=21
            )
        
        # Normalize back to [-1, 1]
        image = denoised.astype(np.float32) / 127.5 - 1
        
        return image


class FeatureExtractor:
    """Extract features from patient data and images"""

    def __init__(self):
        self.medical_features = [
            'age', 'sex', 'fatigue_level', 'alcohol_consumption', 'weight_loss_kg',
            'abdominal_swelling', 'appetite_loss', 'jaundice', 'fever', 'ascites',
            'hepatomegaly', 'spiders', 'edema', 'bilirubin', 'cholesterol', 'albumin',
            'copper', 'alk_phos', 'ast', 'triglycerides', 'platelets', 'prothrombin'
        ]

    def extract_medical_features(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """Extract medical features from patient data"""
        features = []
        for feature in self.medical_features:
            value = patient_data.get(feature, 0)
            features.append(value if value is not None else 0)
        
        return np.array(features)

    def extract_image_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract hand-crafted features from medical image
        (Used alongside CNN features)
        """
        features = {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            image_gray = (image * 255).astype(np.uint8)
        
        # Edge detection
        edges = cv2.Canny(image_gray, 100, 200)
        features['edge_density'] = np.sum(edges) / edges.size
        
        # Texture features (Haralick)
        features['texture_variance'] = np.var(image_gray)
        features['texture_mean'] = np.mean(image_gray)
        features['texture_std'] = np.std(image_gray)
        
        # Shape features
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            features['contour_area'] = cv2.contourArea(largest_contour)
            features['contour_perimeter'] = cv2.arcLength(largest_contour, True)
            
            # Circularity
            if features['contour_perimeter'] > 0:
                features['circularity'] = 4 * np.pi * features['contour_area'] / (features['contour_perimeter'] ** 2)
        
        # Histogram features
        hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
        features['histogram_energy'] = np.sum(hist ** 2) / hist.size
        features['histogram_entropy'] = -np.sum(hist * np.log(hist + 1e-10)) / hist.size
        
        return features

    def combine_features(self, medical_features: np.ndarray, image_features: Dict[str, Any]) -> np.ndarray:
        """Combine medical and image features"""
        image_feature_values = np.array(list(image_features.values()))
        combined = np.concatenate([medical_features, image_feature_values])
        return combined
