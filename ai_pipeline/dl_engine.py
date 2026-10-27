"""
Deep Learning Engine for Medical Imaging
Uses EfficientNet-B0 for liver cirrhosis staging from ultrasound/MRI images
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import numpy as np
import logging
from pathlib import Path
import json
from typing import Tuple, Dict, Any, List, Optional
import matplotlib.pyplot as plt
import cv2
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DLEngine:
    """EfficientNet-based Deep Learning engine for image-based liver cirrhosis prediction."""
    
    IMG_SIZE = (224, 224)
    NUM_CLASSES = 5  # Stages F0, F1, F2, F3, F4
    
    STAGE_MAPPING = {
        0: "F0 - Normal",
        1: "F1 - Mild Fibrosis",
        2: "F2 - Moderate Fibrosis",
        3: "F3 - Severe Fibrosis",
        4: "F4 - Cirrhosis"
    }
    
    def __init__(self, model_path: str = 'models/dl_model.h5',
                 architecture: str = 'efficientnet_b0'):
        self.model_path = Path(model_path)
        self.architecture = architecture
        self.model = None
        self.history = None
        
        # Ensure directories exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
    def build_model(self, input_shape: Tuple = (224, 224, 3),
                   num_classes: int = 5,
                   fine_tune_layers: int = 50) -> models.Model:
        """Build EfficientNet model with custom head."""
        logger.info(f"Building {self.architecture} model...")
        
        # Load base model
        if self.architecture == 'efficientnet_b0':
            base_model = EfficientNetB0(
                input_shape=input_shape,
                weights='imagenet',
                include_top=False
            )
        elif self.architecture == 'efficientnet_b1':
            base_model = EfficientNetB1(
                input_shape=input_shape,
                weights='imagenet',
                include_top=False
            )
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        # Freeze base model layers for initial training
        base_model.trainable = False
        
        # Build custom head
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # Data augmentation layers
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Normalization
        x = layers.Normalization()(x)
        
        # Base model
        x = base_model(x, training=False)
        
        # Custom classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile with initial settings
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.model = model
        logger.info(f"Model built successfully. Total parameters: {model.count_params()}")
        
        return model
    
    def create_data_generators(self, batch_size: int = 32) -> Tuple:
        """Create data augmentation generators."""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        return train_datagen, val_datagen, test_datagen
    
    def train(self, train_dir: str, val_dir: Optional[str] = None, epochs: int = 50,
             batch_size: int = 32, fine_tune_from: int = 50, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the DL model."""
        logger.info("Starting DL model training...")
        
        if self.model is None:
            self.build_model(num_classes=self.NUM_CLASSES)
        
        # Load training and validation data from one directory if val_dir is not provided
        if val_dir is None:
            logger.info(f"Loading data from {train_dir} with validation split {validation_split}")
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=validation_split
            )
            val_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split
            )
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.IMG_SIZE,
                batch_size=batch_size,
                class_mode='categorical',
                subset='training',
                shuffle=True,
                seed=42
            )
            val_generator = val_datagen.flow_from_directory(
                train_dir,
                target_size=self.IMG_SIZE,
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False,
                seed=42
            )
        else:
            # Create data generators
            train_datagen, val_datagen, _ = self.create_data_generators(batch_size)
            
            # Load training data
            logger.info(f"Loading training data from {train_dir}")
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.IMG_SIZE,
                batch_size=batch_size,
                class_mode='categorical'
            )
            
            # Load validation data
            logger.info(f"Loading validation data from {val_dir}")
            val_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=self.IMG_SIZE,
                batch_size=batch_size,
                class_mode='categorical'
            )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                str(self.model_path),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
        ]
        
        # Initial training phase
        logger.info("Phase 1: Training with frozen base model...")
        history1 = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=min(20, epochs),
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning phase
        if epochs > 20:
            logger.info(f"Phase 2: Fine-tuning with unfrozen layers (from {fine_tune_from})...")
            
            # Unfreeze base model layers
            base_model = self.model.layers[3]  # EfficientNet layer
            base_model.trainable = True
            
            # Freeze early layers
            for layer in base_model.layers[:-fine_tune_from]:
                layer.trainable = False
            
            # Recompile with lower learning rate
            self.model.compile(
                optimizer=Adam(learning_rate=1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
            )
            
            history2 = self.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs - 20,
                callbacks=callbacks,
                verbose=1
            )
            
            self.history = {
                'phase1': history1.history,
                'phase2': history2.history
            }
        else:
            self.history = history1.history
        
        logger.info("=" * 50)
        logger.info("DL MODEL TRAINING COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Model saved to {self.model_path}")
        
        # Save training metadata
        metadata = {
            'architecture': self.architecture,
            'img_size': self.IMG_SIZE,
            'num_classes': self.NUM_CLASSES,
            'stage_mapping': self.STAGE_MAPPING,
            'training_date': datetime.now().isoformat(),
            'epochs': epochs,
            'batch_size': batch_size,
            'fine_tune_from': fine_tune_from
        }
        
        with open(self.model_path.parent / 'dl_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'status': 'success',
            'model_path': str(self.model_path),
            'history': self.history
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference."""
        if isinstance(image, str):
            image = cv2.imread(image)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, self.IMG_SIZE)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        return np.expand_dims(image, axis=0)
    
    def predict(self, image: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Make prediction on image."""
        if self.model is None:
            self.load()
        
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Predict
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        return predicted_class, confidence, predictions[0]
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple]:
        """Batch prediction."""
        results = []
        for image in images:
            pred_class, confidence, probs = self.predict(image)
            results.append({
                'stage': pred_class,
                'stage_name': self.STAGE_MAPPING[pred_class],
                'confidence': confidence,
                'probabilities': probs.tolist()
            })
        return results
    
    def save(self):
        """Save model."""
        if self.model is not None:
            self.model.save(str(self.model_path))
            logger.info(f"Model saved to {self.model_path}")
    
    def load(self):
        """Load model."""
        if self.model_path.exists():
            self.model = keras.models.load_model(str(self.model_path))
            logger.info(f"Model loaded from {self.model_path}")
        else:
            logger.warning(f"Model not found at {self.model_path}")
    
    def evaluate(self, test_dir: str, batch_size: int = 32) -> Dict[str, Any]:
        """Evaluate model on test set."""
        if self.model is None:
            self.load()
        
        _, _, test_datagen = self.create_data_generators(batch_size)
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.IMG_SIZE,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        loss, accuracy, precision, recall = self.model.evaluate(test_generator)
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall)
        }


if __name__ == "__main__":
    # Example usage
    dl_engine = DLEngine()
    
    # Build model
    dl_engine.build_model()
    
    # Print model summary
    dl_engine.model.summary()
    
    print("\nDL Engine initialized. Use dl_engine.train() to train on image dataset.")
