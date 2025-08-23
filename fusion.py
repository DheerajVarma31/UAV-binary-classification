import os
import numpy as np
import tensorflow as tf
from keras import layers, models, Input

# -------------------------
# ✅ Setup Paths
# -------------------------
BASE_DIR = r"C:\Users\dheer\UAV-binary-classification"  # Change to your dataset path
X_FILE = os.path.join(BASE_DIR, "X_spectrogram.npy")
Y_FILE = os.path.join(BASE_DIR, "y.npy")

# -------------------------
# ✅ Check dataset existence
# -------------------------
if not os.path.exists(X_FILE) or not os.path.exists(Y_FILE):
    raise FileNotFoundError(
        f"Dataset files not found! Expected at:\n{X_FILE}\n{Y_FILE}\n"
        "Please generate spectrogram features first."
    )

print("✅ Loading data...")
X_img = np.load(X_FILE)
y = np.load(Y_FILE)

# Normalize
X_img = X_img / np.max(X_img)

print(f"Dataset loaded: {X_img.shape} samples | Labels: {y.shape}")

# -------------------------
# ✅ Split into Train/Test
# -------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_img, y, test_size=0.2, random_state=42
)

# -------------------------
# ✅ Define Fusion Model
# -------------------------

# CNN branch
cnn_input = Input(shape=X_img.shape[1:], name="CNN_Input")
x1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(cnn_input)
x1 = layers.MaxPooling2D((2,2))(x1)
x1 = layers.Flatten()(x1)
x1 = layers.Dense(128, activation='relu')(x1)

# LSTM branch
lstm_input = Input(shape=(X_img.shape[1], X_img.shape[2]), name="LSTM_Input")
x2 = layers.LSTM(64, return_sequences=False)(lstm_input)
x2 = layers.Dense(128, activation='relu')(x2)

# Combine both
combined = layers.concatenate([x1, x2])
output = layers.Dense(1, activation='sigmoid')(combined)

fusion_model = models.Model(inputs=[cnn_input, lstm_input], outputs=output)

fusion_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(fusion_model.summary())

# -------------------------
# ✅ Train the model
# -------------------------
history = fusion_model.fit(
    [X_train, X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])],
    y_train,
    validation_data=(
        [X_test, X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])],
        y_test
    ),
    epochs=20,
    batch_size=32
)

# -------------------------
# ✅ Save trained model
# -------------------------
MODEL_PATH = os.path.join(BASE_DIR, "uav_fusion_model.h5")
fusion_model.save(MODEL_PATH)
print(f"✅ Fusion model saved at {MODEL_PATH}")
