import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# ================================
# Load trained LSTM model
# ================================
st.set_page_config(page_title="🛸 UAV Detector (LSTM)", layout="centered")
st.title("🛸 UAV Sound Classifier (LSTM)")
st.write("Upload a `.wav` file to detect whether it's a **UAV** or **Non-UAV** sound.")

@st.cache_resource
def load_lstm_model():
    return tf.keras.models.load_model("uav_binary_lstm_model.h5")

model = load_lstm_model()

# ================================
# Preprocess Audio for LSTM
# ================================
def preprocess_audio_lstm(file_path, sample_rate=16000, n_mfcc=40, max_len=100):
    """
    Converts raw audio into a padded MFCC feature sequence
    compatible with LSTM input.
    """
    y, sr = librosa.load(file_path, sr=sample_rate)
    y = librosa.util.normalize(y)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T  # Shape: (time_steps, n_mfcc)

    # Pad or truncate to fixed length
    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len, :]

    return np.expand_dims(mfcc, axis=0)  # Shape: (1, time_steps, n_mfcc)

# ================================
# Streamlit App Logic
# ================================
uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file is not None:
    try:
        # Preprocess audio
        lstm_input = preprocess_audio_lstm(uploaded_file)

        # Predict
        prediction = model.predict(lstm_input)[0][0]
        label = "🛸 UAV Detected" if prediction > 0.5 else "✅ Non-UAV"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # Show Results
        st.success(label)
        st.metric("Confidence", f"{confidence * 100:.2f}%")

    except Exception as e:
        st.error(f"❌ Error processing audio: {str(e)}")

else:
    st.info("Please upload a valid `.wav` file.")