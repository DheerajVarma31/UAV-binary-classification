import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.models import load_model

# =======================
# Load Pre-trained Models
# =======================
@st.cache_resource
def load_models():
    cnn_model = load_model("uav_binary_cnn_model.h5")
    lstm_model = load_model("uav_binary_lstm_model.h5")
    vgg_model = load_model("uav_binary_vgg_model.h5")
    return cnn_model, lstm_model, vgg_model

cnn_model, lstm_model, vgg_model = load_models()

# =======================
# Audio Preprocessing
# =======================
def preprocess_for_cnn_vgg(y, sr):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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

def preprocess_for_lstm(y, sr, n_mfcc=40, max_len=100):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len, :]
    return mfcc

# =======================
# Prediction Function
# =======================
def predict_fusion(audio_file):
    y, sr = librosa.load(audio_file, sr=16000)
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y, top_db=20)

    img_input = np.expand_dims(preprocess_for_cnn_vgg(y, sr), axis=0)
    lstm_input = np.expand_dims(preprocess_for_lstm(y, sr), axis=0)

    # Get model predictions
    cnn_pred = cnn_model.predict(img_input)[0][0]
    vgg_pred = vgg_model.predict(img_input)[0][0]
    lstm_pred = lstm_model.predict(lstm_input)[0][0]

    # Average Fusion
    avg_score = (cnn_pred + vgg_pred + lstm_pred) / 3
    label = "UAV Detected" if avg_score > 0.5 else "No UAV Detected"

    return cnn_pred, vgg_pred, lstm_pred, avg_score, label

# =======================
# Streamlit UI
# =======================
st.title("🚁 UAV Audio Detection - Fusion Model")
st.markdown("Upload an audio file to detect UAV sounds using CNN + LSTM + VGG fusion.")

audio_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if audio_file:
    st.audio(audio_file, format="audio/wav")
    with st.spinner("Analyzing..."):
        cnn_pred, vgg_pred, lstm_pred, avg_score, label = predict_fusion(audio_file)

    st.subheader("🔍 Prediction Results")
    st.write(f"**CNN Prediction:** {cnn_pred:.2f}")
    st.write(f"**VGG Prediction:** {vgg_pred:.2f}")
    st.write(f"**LSTM Prediction:** {lstm_pred:.2f}")
    st.write(f"**Fusion Score (Average):** {avg_score:.2f}")
    st.markdown(f"### 🏷️ **Result: {label}**")

    if avg_score > 0.5:
        st.success("Drone/UAV sound detected with high confidence!")
    else:
        st.info("No UAV sound detected.")
