import os
import numpy as np
import librosa
import tensorflow as tf
from keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from glob import glob

# 📁 Dataset Path Configuration
UAV_PATH = "C:/Users/dheer/UAV-binary-classification/drone-dataset-uav/Drone sound"
NON_UAV_PATH = "C:/Users/dheer/UAV-binary-classification/drone-dataset-uav/Other sounds"

# 🔊 1️⃣ Preprocess Audio (less clean)
def preprocess_audio(file_path, sample_rate=16000):
    y, sr = librosa.load(file_path, sr=sample_rate)
    return y, sr  # No normalization, no trimming

# 🖼️ 2️⃣ Extract Spectrogram Image FAST (no matplotlib)
def extract_spectrogram_image_fast(y, sr, img_size=(227, 227)):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Normalize to [0,1]
    S_dB_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())

    # Resize to match model input shape
    S_resized = tf.image.resize(S_dB_norm[..., np.newaxis], img_size).numpy()

    # Convert to RGB
    image = np.repeat(S_resized, 3, axis=-1)

    # Add Gaussian noise
    noise = np.random.normal(0, 0.05, image.shape)
    image = np.clip(image + noise, 0, 1)
    return image

# 📂 3️⃣ Load Dataset (fast split)
def load_dataset():
    X_train, y_train, X_test, y_test = [], [], [], []

    uav_files = glob(os.path.join(UAV_PATH, "*.wav"))
    non_uav_files = glob(os.path.join(NON_UAV_PATH, "*.wav"))

    # Deterministic 80/20 split
    split_uav = int(0.8 * len(uav_files))
    split_non_uav = int(0.8 * len(non_uav_files))

    train_uav = uav_files[:split_uav]
    test_uav = uav_files[split_uav:]

    train_non_uav = non_uav_files[:split_non_uav]
    test_non_uav = non_uav_files[split_non_uav:]

    print(f"📂 UAV train: {len(train_uav)}, test: {len(test_uav)}")
    print(f"📂 Non-UAV train: {len(train_non_uav)}, test: {len(test_non_uav)}")

    # Training UAV
    for f in train_uav:
        try:
            y_audio, sr = preprocess_audio(f)
            img = extract_spectrogram_image_fast(y_audio, sr)
            X_train.append(img)
            y_train.append(1)
        except Exception as e:
            print(f"⚠️ Error loading {f}: {e}")

    # Training Non-UAV
    for f in train_non_uav:
        try:
            y_audio, sr = preprocess_audio(f)
            img = extract_spectrogram_image_fast(y_audio, sr)
            X_train.append(img)
            y_train.append(0)
        except Exception as e:
            print(f"⚠️ Error loading {f}: {e}")

    # Testing UAV
    for f in test_uav:
        try:
            y_audio, sr = preprocess_audio(f)
            img = extract_spectrogram_image_fast(y_audio, sr)
            X_test.append(img)
            y_test.append(1)
        except Exception as e:
            print(f"⚠️ Error loading {f}: {e}")

    # Testing Non-UAV
    for f in test_non_uav:
        try:
            y_audio, sr = preprocess_audio(f)
            img = extract_spectrogram_image_fast(y_audio, sr)
            X_test.append(img)
            y_test.append(0)
        except Exception as e:
            print(f"⚠️ Error loading {f}: {e}")

    print(f"✅ Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# 🚀 Load Data
X_train, y_train, X_test, y_test = load_dataset()
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print("Train class distribution:", Counter(y_train))
print("Test class distribution:", Counter(y_test))

# ⚖️ Class Weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print("📊 Class weights:", class_weights_dict)

# 🧠 4️⃣ Smaller CNN (for ~80% accuracy)
def build_cnn(input_shape=(227, 227, 3)):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_cnn()

# 🏁 Train
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=16,
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)

# 💾 Save Model
model.save("uav_binary_cnn_model_reduced.h5")
print("✅ Model saved as uav_binary_cnn_model_reduced.h5")

# 📈 Evaluate
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))
print("Predictions:", np.unique(y_pred, return_counts=True))
