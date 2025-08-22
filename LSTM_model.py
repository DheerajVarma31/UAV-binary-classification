import os
import numpy as np
import librosa
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from glob import glob

# =======================
# Dataset Paths
# =======================
UAV_PATH = "C:/Users/dheer/UAV-binary-classification/drone-dataset-uav/Drone sound"
NON_UAV_PATH = "C:/Users/dheer/UAV-binary-classification/drone-dataset-uav/Other sounds"

# =======================
# Preprocessing - Extract MFCCs
# =======================
def preprocess_audio_lstm(file_path, sample_rate=16000, n_mfcc=40, max_len=100):
    y, sr = librosa.load(file_path, sr=sample_rate)
    y = librosa.util.normalize(y)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)

    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T  # shape: (time, n_mfcc)
    
    # Pad or truncate sequences for uniform length
    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len, :]
    
    return mfcc

def load_dataset_lstm():
    X, y = [], []
    uav_files = glob(os.path.join(UAV_PATH, "*.wav"))[:1300]
    non_uav_files = glob(os.path.join(NON_UAV_PATH, "*.wav"))[:1300]

    print(f"Loading UAV samples: {len(uav_files)}")
    print(f"Loading NON-UAV samples: {len(non_uav_files)}")

    for f in uav_files:
        try:
            X.append(preprocess_audio_lstm(f))
            y.append(1)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    for f in non_uav_files:
        try:
            X.append(preprocess_audio_lstm(f))
            y.append(0)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    print(f"✅ Total loaded samples: {len(X)}")
    return np.array(X), np.array(y)

# =======================
# Load Data
# =======================
X, y = load_dataset_lstm()
X = X.astype(np.float32)
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

print("Train class distribution:", Counter(y_train))
print("Test class distribution:", Counter(y_test))

# Class Weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weights_dict)

# =======================
# Build LSTM Model
# =======================
def build_lstm(input_shape=(100, 40)):
    model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_lstm()
model.summary()

# =======================
# Train Model
# =======================
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=16,
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)

# Save Model
model.save("uav_binary_lstm_model.h5")
print("✅ Model saved as uav_binary_lstm_model.h5")

# Evaluate
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))
print("Predictions:", np.unique(y_pred, return_counts=True))
