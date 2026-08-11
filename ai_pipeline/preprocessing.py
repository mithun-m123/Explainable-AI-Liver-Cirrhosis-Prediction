"""
Data Preprocessing Module
Handles preparation of both tabular (symptoms) and imaging data
"""

import numpy as np
import pandas as pd
import cv2
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import logging
from PIL import Image
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TabularPreprocessor:
    """Preprocessing for tabular/symptom data."""
    
    # Define symptom features
    SYMPTOM_FEATURES = [
        'age', 'gender', 'alcohol_consumption', 'bilirubin', 'albumin',
        'ast', 'alp', 'alt', 'platelets', 'prothrombin', 'creatinine',
        'ascites', 'hepatomegaly', 'spiders', 'edema', 'encephalopathy',
        'fatigue', 'jaundice', 'abdominal_pain', 'weight_loss'
    ]
    
    # Feature value ranges for normalization
    FEATURE_RANGES = {
        'age': (18, 100),
        'alcohol_consumption': (0, 50),
        'bilirubin': (0, 20),
        'albumin': (2, 5.5),
        'ast': (10, 400),
        'alp': (20, 150),
        'alt': (10, 400),
        'platelets': (20, 450),
        'prothrombin': (11, 20),
        'creatinine': (0.5, 5),
    }
    
    def __init__(self):
        self.feature_stats = {}
        self.scaler_params = {}
    
    def handle_missing_values(self, df: pd.DataFrame,
                             strategy: str = 'median') -> pd.DataFrame:
        """Handle missing values in dataframe."""
        df_copy = df.copy()
        
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        
        if strategy == 'median':
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(
                df_copy[numeric_cols].median()
            )
        elif strategy == 'mean':
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(
                df_copy[numeric_cols].mean()
            )
        elif strategy == 'forward_fill':
            df_copy = df_copy.fillna(method='ffill')
        
        # Handle categorical
        categorical_cols = df_copy.select_dtypes(include=['object']).columns
        df_copy[categorical_cols] = df_copy[categorical_cols].fillna('unknown')
        
        logger.info(f"Missing values handled using {strategy} strategy")
        return df_copy
    
    def handle_outliers(self, df: pd.DataFrame,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """Handle outliers using IQR or Z-score."""
        df_copy = df.copy()
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                
                # Clip values
                df_copy[col] = df_copy[col].clip(lower=lower, upper=upper)
        
        elif method == 'zscore':
            from scipy import stats
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df_copy[col]))
                df_copy[col] = df_copy[col][z_scores < threshold]
        
        logger.info(f"Outliers handled using {method} method")
        return df_copy
    
    def normalize_features(self, df: pd.DataFrame,
                          method: str = 'minmax') -> pd.DataFrame:
        """Normalize/scale numerical features."""
        df_copy = df.copy()
        
        for feature in self.SYMPTOM_FEATURES:
            if feature not in df_copy.columns:
                continue
            
            if feature in ['gender', 'ascites', 'hepatomegaly', 'spiders', 
                          'edema', 'encephalopathy', 'fatigue', 'jaundice',
                          'abdominal_pain', 'weight_loss']:
                # Binary features
                df_copy[feature] = df_copy[feature].astype(int)
            else:
                # Normalize numerical features
                if method == 'minmax':
                    min_val = self.FEATURE_RANGES.get(feature, (0, 1))[0]
                    max_val = self.FEATURE_RANGES.get(feature, (0, 1))[1]
                    df_copy[feature] = (df_copy[feature] - min_val) / (max_val - min_val)
                    df_copy[feature] = np.clip(df_copy[feature], 0, 1)
                
                elif method == 'zscore':
                    mean = df_copy[feature].mean()
                    std = df_copy[feature].std()
                    if std > 0:
                        df_copy[feature] = (df_copy[feature] - mean) / std
        
        logger.info(f"Features normalized using {method} method")
        return df_copy
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        df_copy = df.copy()
        
        categorical_cols = ['gender']  # Add other categorical columns
        
        for col in categorical_cols:
            if col in df_copy.columns:
                # Simple label encoding
                unique_values = df_copy[col].unique()
                mapping = {val: idx for idx, val in enumerate(unique_values)}
                df_copy[col] = df_copy[col].map(mapping)
        
        return df_copy
    
    def preprocess(self, df: pd.DataFrame,
                  handle_missing: bool = True,
                  handle_outliers: bool = True,
                  normalize: bool = True,
                  encode: bool = True) -> pd.DataFrame:
        """Complete preprocessing pipeline."""
        df_processed = df.copy()
        
        logger.info("Starting tabular data preprocessing...")
        
        if handle_missing:
            df_processed = self.handle_missing_values(df_processed)
        
        if handle_outliers:
            df_processed = self.handle_outliers(df_processed)
        
        if encode:
            df_processed = self.encode_categorical(df_processed)
        
        if normalize:
            df_processed = self.normalize_features(df_processed)
        
        logger.info("Tabular preprocessing complete")
        return df_processed


class ImagePreprocessor:
    """Preprocessing for medical imaging data."""
    
    IMG_SIZE = (224, 224)
    
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def load_image(self, image_path_or_bytes) -> np.ndarray:
        """Load image from file path or bytes."""
        if isinstance(image_path_or_bytes, str):
            image = cv2.imread(image_path_or_bytes)
            if image is None:
                raise ValueError(f"Cannot load image from {image_path_or_bytes}")
        elif isinstance(image_path_or_bytes, bytes):
            nparr = np.frombuffer(image_path_or_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            image = image_path_or_bytes
        
        return image
    
    def resize(self, image: np.ndarray,
              size: Tuple[int, int] = None) -> np.ndarray:
        """Resize image to target size."""
        if size is None:
            size = self.IMG_SIZE
        
        return cv2.resize(image, size)
    
    def normalize(self, image: np.ndarray,
                 method: str = 'imagenet') -> np.ndarray:
        """Normalize image pixels."""
        image = image.astype(np.float32) / 255.0
        
        if method == 'imagenet':
            # ImageNet normalization
            image = (image - self.mean) / self.std
        elif method == 'standard':
            # Simple standardization
            image = (image - 0.5) / 0.5
        
        return image
    
    def enhance_contrast(self, image: np.ndarray,
                        method: str = 'clahe',
                        clip_limit: float = 2.0) -> np.ndarray:
        """Enhance image contrast."""
        if len(image.shape) == 3:
            # Convert to LAB color space
            image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = image_lab[:, :, 0]
        else:
            l_channel = image
        
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_channel)
        elif method == 'histogram':
            l_enhanced = cv2.equalizeHist(l_channel)
        else:
            l_enhanced = l_channel
        
        if len(image.shape) == 3:
            image_lab[:, :, 0] = l_enhanced
            image = cv2.cvtColor(image_lab, cv2.COLOR_LAB2BGR)
        else:
            image = l_enhanced
        
        return image
    
    def denoise(self, image: np.ndarray,
               method: str = 'bilateral') -> np.ndarray:
        """Denoise image."""
        if method == 'bilateral':
            image = cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            image = cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'morphological':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        return image
    
    def standardize_color(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is RGB."""
        if len(image.shape) == 2:
            # Grayscale -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            # BGR -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def preprocess(self, image_path_or_bytes,
                  resize: bool = True,
                  enhance: bool = True,
                  denoise: bool = True,
                  normalize: bool = True) -> np.ndarray:
        """Complete preprocessing pipeline for medical images."""
        logger.info("Starting image preprocessing...")
        
        # Load
        image = self.load_image(image_path_or_bytes)
        
        # Standardize color
        image = self.standardize_color(image)
        
        # Denoise
        if denoise:
            image = self.denoise(image, method='bilateral')
        
        # Enhance contrast
        if enhance:
            image = self.enhance_contrast(image, method='clahe')
        
        # Resize
        if resize:
            image = self.resize(image)
        
        # Normalize
        if normalize:
            image = self.normalize(image)
        
        logger.info("Image preprocessing complete")
        return image
    
    def preprocess_batch(self, image_paths_or_bytes: List,
                        **kwargs) -> np.ndarray:
        """Preprocess batch of images."""
        processed_images = []
        
        for img_data in image_paths_or_bytes:
            try:
                processed = self.preprocess(img_data, **kwargs)
                processed_images.append(processed)
            except Exception as e:
                logger.error(f"Error preprocessing image: {e}")
                continue
        
        return np.array(processed_images)


class DataAugmentation:
    """Data augmentation for improving model robustness."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
    
    def rotate(self, image: np.ndarray, angle_range: Tuple = (-15, 15)) -> np.ndarray:
        """Random rotation."""
        angle = np.random.uniform(*angle_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated
    
    def flip(self, image: np.ndarray,
            flip_h: bool = True,
            flip_v: bool = False) -> np.ndarray:
        """Random flip."""
        if flip_h and np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
        if flip_v and np.random.rand() > 0.5:
            image = cv2.flip(image, 0)
        return image
    
    def brightness(self, image: np.ndarray,
                  factor_range: Tuple = (0.8, 1.2)) -> np.ndarray:
        """Adjust brightness."""
        factor = np.random.uniform(*factor_range)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
        return image
    
    def contrast(self, image: np.ndarray,
                factor_range: Tuple = (0.8, 1.2)) -> np.ndarray:
        """Adjust contrast."""
        factor = np.random.uniform(*factor_range)
        mean = np.mean(image)
        image = mean + factor * (image - mean)
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image
    
    def augment_tabular(self, X: np.ndarray,
                       y: np.ndarray,
                       oversample_minority: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Augment tabular data (SMOTE-like)."""
        from sklearn.preprocessing import StandardScaler
        
        if not oversample_minority:
            return X, y
        
        unique, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)
        
        X_augmented = [X]
        y_augmented = [y]
        
        for class_idx, count in zip(unique, counts):
            if count < max_count:
                # Find samples of this class
                class_mask = y == class_idx
                class_samples = X[class_mask]
                
                # Generate synthetic samples
                samples_needed = max_count - count
                indices = np.random.choice(len(class_samples), samples_needed)
                synthetic_samples = class_samples[indices]
                
                # Add small random noise
                noise = np.random.normal(0, 0.01, synthetic_samples.shape)
                synthetic_samples = synthetic_samples + noise
                
                X_augmented.append(synthetic_samples)
                y_augmented.append(np.full(samples_needed, class_idx))
        
        X_balanced = np.vstack(X_augmented)
        y_balanced = np.concatenate(y_augmented)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(y_balanced))
        return X_balanced[shuffle_idx], y_balanced[shuffle_idx]


def create_preprocessors() -> Dict[str, Any]:
    """Factory function to create all preprocessors."""
    return {
        'tabular': TabularPreprocessor(),
        'image': ImagePreprocessor(),
        'augmentation': DataAugmentation()
    }


if __name__ == "__main__":
    print("Data preprocessing module loaded successfully")
    preprocessors = create_preprocessors()
    print(f"Available preprocessors: {list(preprocessors.keys())}")
