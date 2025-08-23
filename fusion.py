import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter

# =======================
# Load Preprocessed Data
# =======================

# NOTE:
# - X_img, y are spectrogram images (for CNN & VGG)
# - X_seq, y are MFCC sequences (for LSTM)
# Ensure X_img and X_seq are aligned with y

# Example: If you already have datasets loaded separately
# Load spectrogram data for CNN & VGG
X_img = np.load("X_spectrogram.npy")
y = np.load("y_labels.npy")

# Load MFCC sequences for LSTM
X_seq = np.load("X_mfcc.npy")

# Split train-test consistently
X_img_train, X_img_test, y_train, y_test = train_test_split(
    X_img, y, stratify=y, test_size=0.2, random_state=42
)

X_seq_train, X_seq_test, _, _ = train_test_split(
    X_seq, y, stratify=y, test_size=0.2, random_state=42
)

print("Train class distribution:", Counter(y_train))
print("Test class distribution:", Counter(y_test))

# =======================
# Load Trained Models
# =======================
cnn_model = load_model("uav_binary_cnn_model.h5")
lstm_model = load_model("uav_binary_lstm_model.h5")
vgg_model = load_model("uav_binary_vgg_model.h5")

# =======================
# Predict with Each Model
# =======================
cnn_preds = cnn_model.predict(X_img_test)
vgg_preds = vgg_model.predict(X_img_test)
lstm_preds = lstm_model.predict(X_seq_test)

# =======================
# Simple Averaging Fusion
# =======================
avg_preds = (cnn_preds + vgg_preds + lstm_preds) / 3
final_preds = (avg_preds > 0.5).astype("int32")

print("\n--- Late Fusion (Average) ---")
print(confusion_matrix(y_test, final_preds))
print(classification_report(y_test, final_preds, zero_division=1))

# =======================
# Stacked Meta-Learner Fusion
# =======================
from keras import layers, models

# Stack model outputs as features
stacked_features_train = np.hstack([
    cnn_model.predict(X_img_train),
    vgg_model.predict(X_img_train),
    lstm_model.predict(X_seq_train)
])

stacked_features_test = np.hstack([
    cnn_preds,
    vgg_preds,
    lstm_preds
])

# Build a small dense classifier
meta_model = models.Sequential([
    layers.Input(shape=(stacked_features_train.shape[1],)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

meta_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

meta_model.fit(stacked_features_train, y_train, epochs=20, batch_size=16, validation_split=0.2)

# Meta-model predictions
meta_preds = (meta_model.predict(stacked_features_test) > 0.5).astype("int32")

print("\n--- Stacked Fusion (Meta-Learner) ---")
print(confusion_matrix(y_test, meta_preds))
print(classification_report(y_test, meta_preds, zero_division=1))
