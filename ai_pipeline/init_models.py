import joblib
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os

# Create the target directory if it doesn't exist
target_dir = '../backend/models/'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

print("--- Generating Dummy Models for Development ---")

# 1. Create a dummy ML Model (Symptom Model)
# We simulate 15 features (Age, Bilirubin, etc.)
X_dummy = np.random.rand(100, 15) 
y_dummy = np.random.randint(0, 4, 100) # 4 stages
ml_model = RandomForestClassifier()
ml_model.fit(X_dummy, y_dummy)

# Save to backend/models/
joblib.dump(ml_model, os.path.join(target_dir, 'symptom_model.pkl'))
print("Created: backend/models/symptom_model.pkl")

# 2. Create a dummy DL Model (Imaging Model)
# We use a very small MobileNet to save space and time
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(4, activation='softmax')(x)
dl_model = tf.keras.Model(inputs=base_model.input, outputs=output)

# Save to backend/models/
dl_model.save(os.path.join(target_dir, 'liver_cnn_model.h5'))
print("Created: backend/models/liver_cnn_model.h5")

print("\nSuccess! Your backend is now ready to run predictions.")