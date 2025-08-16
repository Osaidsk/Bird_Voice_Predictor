import streamlit as st
import os
import json
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import joblib
import pandas as pd

# Optional: Lottie animation
try:
    from streamlit_lottie import st_lottie  # pip install streamlit-lottie
except Exception:
    st_lottie = None

# ---------------------------
# Page config + styles
# ---------------------------
st.set_page_config(page_title="Bird Species Predictor", page_icon="🦜", layout="wide")

st.markdown(
    """
    <style>
    .title {
        text-align: center; 
        color: #2c3e50; 
        font-size: 2.5rem; 
        font-weight: 800; 
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center; 
        color: #34495e; 
        font-size: 1.1rem; 
        margin-bottom: 2rem;
    }
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 25px;
        margin-bottom: 2rem;
    }
    .nav-card {
        padding: 16px 32px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        text-decoration: none;
        color: white !important;
        box-shadow: 0px 6px 12px rgba(0,0,0,0.25);
        cursor: pointer;
    }
    .nav-card.prediction { background: #e74c3c; }
    .nav-card.performance { background: #3498db; }
    .nav-card.about { background: #27ae60; }
    .nav-card:hover {
        opacity: 0.9;
        transform: translateY(-3px);
        transition: 0.2s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Session + Navigation
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "Prediction"

# Header
st.markdown('<div class="title">🦜 Bird Species Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an audio file and let AI predict the most likely bird species with confidence scores</div>', unsafe_allow_html=True)

# Navigation with styled buttons
nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    if st.button("🔮 Prediction"):
        st.session_state.page = "Prediction"
with nav_col2:
    if st.button("📊 Model Performance"):
        st.session_state.page = "Performance"
with nav_col3:
    if st.button("ℹ️ About"):
        st.session_state.page = "About"

page = st.session_state.page

# ---------------------------
# Load model artifacts
# ---------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("rf_model_filtered.pkl")
    label_encoder = joblib.load("label_encoder_filtered.pkl")
    scaler = joblib.load("scaler_filtered.pkl")
    return model, label_encoder, scaler

try:
    model, label_encoder, scaler = load_assets()
except Exception as e:
    st.error(f"❌ Failed to load model assets: {e}")
    st.stop()

# ---------------------------
# Helpers
# ---------------------------
def load_lottie(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_features_from_bytes(audio_bytes: bytes):
    if not audio_bytes:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=None)
        peak = np.max(np.abs(y)) if y.size else 0.0
        if peak < 0.01:
            return None
        y = y / peak

        n = len(y)
        Y = np.fft.fft(y)
        magnitude = np.abs(Y)
        freq = np.fft.fftfreq(n, d=1 / sr)
        valid = (freq > 100) & (freq < sr / 2)
        positive_freqs = freq[valid]
        positive_mag = magnitude[valid]
        dominant_freq = positive_freqs[np.argmax(positive_mag)] if positive_mag.size else 0.0

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spec_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        rms = np.mean(librosa.feature.rms(y=y))
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        features = np.concatenate(
            [[dominant_freq], [spec_bw], [spec_flatness], [rms], mfcc_mean, chroma_mean]
        ).reshape(1, -1)

        return features
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ---------------------------
# Pages
# ---------------------------
if page == "Prediction":
    uploaded = st.file_uploader("🎵 Upload Bird Audio (MP3/WAV)", type=["mp3", "wav"])

    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        audio_format = uploaded.type if uploaded.type else "audio/mp3"
        st.audio(audio_bytes, format=audio_format)

        if st.button("🚀 Start Prediction"):
            lottie_obj = load_lottie("bird_animation.json") if st_lottie else None
            with st.spinner("Analyzing the audio..."):
                if st_lottie and lottie_obj:
                    st_lottie(lottie_obj, height=160, key="loading")

                features = extract_features_from_bytes(audio_bytes)
                if features is None:
                    st.error("❌ Could not extract usable audio features (file may be too silent). Try another recording.")
                    st.stop()

                try:
                    scaled = scaler.transform(features)
                    proba = model.predict_proba(scaled)[0]
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    st.stop()

                top_idx = np.argsort(proba)[-5:][::-1]
                top_species = label_encoder.inverse_transform(top_idx)
                top_probs = proba[top_idx]

                st.subheader("📢 Most Likely Bird Species")
                st.success(f"**{top_species[0]}** ({top_probs[0]:.2%})")

                st.subheader("📊 Top 5 Predictions")
                fig, ax = plt.subplots(figsize=(6, 3.8))
                ax.barh(top_species[::-1], top_probs[::-1], color="#6c5ce7")
                ax.set_xlabel("Probability")
                ax.set_xlim(0, 1)
                ax.set_title("Top Predictions")
                for i, v in enumerate(top_probs[::-1]):
                    ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9)
                st.pyplot(fig)

elif page == "Performance":
    st.subheader("📊 Model Performance Report")

    # --- Summary Table (manual) ---
    st.write("### 📌 Summary Metrics")
    summary_data = {
        "Metric": ["Accuracy", "Macro Avg", "Weighted Avg"],
        "Precision": ["-", 0.78, 0.78],
        "Recall": ["-", 0.79, 0.79],
        "F1-Score": [0.79, 0.77, 0.77],
        "Support": [84, 84, 84],
    }
    st.table(summary_data)

    # --- Confusion Matrix ---
    st.write("### 🔀 Confusion Matrix")
    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png", caption="Normalized Confusion Matrix", use_container_width=True)
    else:
        st.info("Confusion matrix image not found. Make sure it's saved as 'confusion_matrix.png'.")

    # --- Dynamic Classification Report ---
    st.write("### 📑 Final Classification Report (Filtered Strong Species)")
    if os.path.exists("classification_report.csv"):
        df_report = pd.read_csv("classification_report.csv")
        st.dataframe(
            df_report.style.format(
                {"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}
            )
        )
    else:
        st.warning("⚠️ classification_report.csv not found. Please save it during training.")

elif page == "About":
    st.subheader("ℹ️ About This Project")
    st.markdown(
        """
        **Bird Species Predictor**  
        Built with a **RandomForest classifier** trained on acoustic features extracted using **Librosa**:

        - Dominant frequency  
        - MFCCs  
        - Spectral bandwidth, flatness, RMS  
        - Chroma STFT  

        **Features**
        - Upload `.mp3` or `.wav`  
        - Top-1 species + Top-5 probabilities  

        **Credits / GitHub**  
        Add your repository link here.
        """
    )
