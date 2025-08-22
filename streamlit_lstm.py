import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# ================================
# Streamlit Page Setup
# ================================
st.set_page_config(page_title="🛸 UAV Detector (LSTM)", layout="centered")
st.title("🛸 UAV Sound Classifier (LSTM)")
st.write("Upload a `.wav` file to detect whether it's a **UAV** or **Non-UAV** sound.")

# ================================
# Load LSTM Model
# ================================
@st.cache_resource
def load_lstm_model():
    model = tf.keras.models.load_model("uav_binary_lstm_model.h5")
    return model

model = load_lstm_model()

# Display model summary in debug
with st.expander("Model Summary (Debug)"):
    model.summary(print_fn=lambda x: st.text(x))

# ================================
# Preprocessing Function
# ================================
def preprocess_audio_lstm(file_path, sample_rate=16000, n_mfcc=40, max_len=100):
    """
    Converts raw audio into a normalized, padded MFCC sequence for LSTM input.
    """
    # Load and normalize audio
    y, sr = librosa.load(file_path, sr=sample_rate)
    y = librosa.util.normalize(y)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T  # shape: (time_steps, n_mfcc)

    # Normalize MFCCs to match training
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)

    # Pad or truncate
    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len, :]

    return np.expand_dims(mfcc, axis=0)  # Shape: (1, time_steps, n_mfcc)

# ================================
# File Upload and Prediction
# ================================
uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file is not None:
    try:
        st.info("🔄 Processing audio file...")
        
        # Preprocess input
        lstm_input = preprocess_audio_lstm(uploaded_file)
        st.write("**Debug:** Input shape to model →", lstm_input.shape)

        # Prediction
        prediction = float(model.predict(lstm_input)[0][0])
        st.write("**Debug:** Raw prediction score →", prediction)

        # Classification
        label = "🛸 UAV Detected" if prediction > 0.5 else "✅ Non-UAV"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # Display results
        st.subheader("Prediction Result")
        st.success(label)
        st.metric("Confidence", f"{confidence * 100:.2f}%")

    except Exception as e:
        st.error(f"❌ Error processing audio: {str(e)}")
else:
    st.info("📂 Please upload a valid `.wav` file.")
