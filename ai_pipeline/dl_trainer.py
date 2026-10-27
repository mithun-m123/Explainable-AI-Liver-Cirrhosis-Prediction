"""
Complete Deep Learning Model Training and Inference
Includes CNN, ResNet50, EfficientNet, and Liver Region Detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, Any, List
import json

logger = logging.getLogger(__name__)


class LiverCNN(nn.Module):
    """Custom CNN for liver cirrhosis detection"""
    
    def __init__(self, num_classes=5):
        super(LiverCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LiverRegionDetector(nn.Module):
    """U-Net style network for liver region segmentation"""
    
    def __init__(self):
        super(LiverRegionDetector, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = self._conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = self._conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Output
        self.final_conv = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool3)
        
        # Decoder
        upconv3 = self.upconv3(bottleneck)
        dec3 = self.dec3(torch.cat([upconv3, enc3], dim=1))
        
        upconv2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat([upconv2, enc2], dim=1))
        
        upconv1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([upconv1, enc1], dim=1))
        
        # Output
        out = self.final_conv(dec1)
        out = self.sigmoid(out)
        
        return out


class DLModelTrainer:
    """Train and manage deep learning models"""
    
    def __init__(self, models_dir: str = "models", device: str = None):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.models = {}
        logger.info(f"Using device: {self.device}")
    
    def create_cnn_model(self, num_classes: int = 5) -> nn.Module:
        """Create custom CNN model"""
        model = LiverCNN(num_classes=num_classes).to(self.device)
        self.models['cnn'] = model
        return model
    
    def create_resnet_model(self, num_classes: int = 5, pretrained: bool = True) -> nn.Module:
        """Create ResNet50 model"""
        model = models.resnet50(pretrained=pretrained)
        
        # Modify final layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        model = model.to(self.device)
        self.models['resnet50'] = model
        return model
    
    def create_efficientnet_model(self, num_classes: int = 5, pretrained: bool = True) -> nn.Module:
        """Create EfficientNet-B0 model"""
        try:
            from torchvision.models import efficientnet_b0
            model = efficientnet_b0(pretrained=pretrained)
            
            # Modify final layer
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            
            model = model.to(self.device)
            self.models['efficientnet'] = model
            return model
        except Exception as e:
            logger.warning(f"EfficientNet not available: {e}")
            return None
    
    def create_liver_detector(self) -> nn.Module:
        """Create liver region detection model"""
        model = LiverRegionDetector().to(self.device)
        self.models['liver_detector'] = model
        return model
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, epochs: int = 50, 
                   learning_rate: float = 0.001, model_name: str = 'model'):
        """Train a model"""
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.save_model(model, model_name)
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return history
    
    def save_model(self, model: nn.Module, model_name: str):
        """Save model"""
        model_path = self.models_dir / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved: {model_path}")
    
    def load_model(self, model: nn.Module, model_name: str):
        """Load model"""
        model_path = self.models_dir / f"{model_name}.pth"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Model loaded: {model_path}")
        else:
            logger.warning(f"Model not found: {model_path}")
    
    def predict(self, model: nn.Module, image: torch.Tensor) -> Tuple[int, np.ndarray]:
        """Make prediction"""
        model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.item(), probabilities.cpu().numpy()


# Import torch here for type hints
try:
    import torch
except ImportError:
    torch = None
