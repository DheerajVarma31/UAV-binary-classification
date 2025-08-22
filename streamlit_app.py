# streamlit_app.py
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("uav_binary_lstm_model.h5")

# 🔊 Preprocess audio
def preprocess_audio(file_path, sample_rate=16000):
    y, sr = librosa.load(file_path, sr=sample_rate)
    y = librosa.util.normalize(y)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    return y_trimmed, sr

# 📊 Convert audio to spectrogram image
def extract_spectrogram_image(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig = plt.figure(figsize=(2.27, 2.27), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    librosa.display.specshow(S_dB, sr=sr, ax=ax)
    ax.axis('off')
    canvas.draw()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    image = image.reshape((height, width, 4))
    plt.close(fig)
    return image[:, :, :3] / 255.0

# 🖼 Streamlit UI
st.set_page_config(page_title="🛸 UAV Detector", layout="centered")
st.title("🛸 UAV Sound Classifier (Binary)")
st.write("Upload a `.wav` file to detect whether it's a **UAV** or **Non-UAV** sound.")

uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file is not None:
    try:
        y_audio, sr = preprocess_audio(uploaded_file)
        img = extract_spectrogram_image(y_audio, sr)
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img)[0][0]
        label = "🛸 UAV Detected" if prediction > 0.5 else "✅ Non-UAV"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # Output
        st.success(label)
        st.metric("Confidence", f"{confidence * 100:.2f}%")

    except Exception as e:
        st.error(f"❌ Error processing audio: {str(e)}")
else:
    st.info("Please upload a valid `.wav` file.")
