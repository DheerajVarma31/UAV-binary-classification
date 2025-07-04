import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
import streamlit as st
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras import layers, models
from glob import glob
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter


# 📁 Dataset Path Configuration
UAV_PATH = "C:/Users/dheer/UAV-binary-classification/drone-dataset-uav/Drone sound"
NON_UAV_PATH = "C:/Users/dheer/UAV-binary-classification/drone-dataset-uav/Other sounds"

# 1️⃣ Preprocess Audio
def preprocess_audio(file_path, sample_rate=16000):
    y, sr = librosa.load(file_path, sr=sample_rate)
    y = librosa.util.normalize(y)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    return y_trimmed, sr

# 2️⃣ Extract Spectrogram Image
def extract_spectrogram_image(y, sr):
    import matplotlib
    matplotlib.use("Agg")  # ✅ Use non-interactive backend

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(2.27, 2.27), dpi=100)
    canvas = FigureCanvas(fig)  # ✅ Create a canvas object
    ax = fig.add_subplot(111)
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, ax=ax)
    ax.axis('off')

    canvas.draw()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    # remove alpha channel
    return image[:, :, :3] / 255.0


# 3️⃣ Load Dataset
def load_dataset():
    X, y = [], []
    for label, path in zip([1, 0], [UAV_PATH, NON_UAV_PATH]):
        print(f"📂 Scanning: {path}")
        files = glob(os.path.join(path, "*.wav"))
        print(f"🔎 Found {len(files)} .wav files for label {label}")
        for f in files[:300]:
            try:
                y_audio, sr = preprocess_audio(f)
                img = extract_spectrogram_image(y_audio, sr)
                X.append(img)
                y.append(label)
            except Exception as e:
                print(f"⚠️ Error loading {f}: {e}")
    print(f"✅ Total loaded samples: {len(X)}")
    return np.array(X), np.array(y)

X, y = load_dataset()
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Compute weights for classes: 0 (non-UAV), 1 (UAV)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print("📊 Class weights:", class_weights_dict)

print("Train class distribution:", Counter(y_train))
print("Test class distribution:", Counter(y_test))


# 4️⃣ Build CNN Model
def build_cnn(input_shape=(227, 227, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_cnn()
model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=15,
    batch_size=16,
    class_weight=class_weights_dict  # ✅ This is the key
)

model.save("uav_binary_cnn_model.h5")

# 5️⃣ Evaluate Model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred,zero_division=1))

# 6️⃣ Streamlit App

def run_streamlit_app():
    st.set_page_config(page_title="UAV Binary Classification", layout="centered")
    st.title("🛸 UAV Sound Classifier (Binary)")

    uploaded_file = st.file_uploader("Upload a WAV audio file", type=[".wav"])

    if uploaded_file is not None:
        try:
            y, sr = librosa.load(uploaded_file, sr=16000)
            y = librosa.util.normalize(y)
            y, _ = librosa.effects.trim(y, top_db=20)
            img = extract_spectrogram_image(y, sr)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            model = tf.keras.models.load_model("uav_binary_cnn_model.h5")
            pred = model.predict(img)
            label = "🛸 UAV Detected" if pred[0][0] > 0.5 else "✅ Non-UAV"
            st.success(label)
            st.metric(label="Confidence", value=f"{pred[0][0] * 100:.2f}%")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload a WAV file.")



