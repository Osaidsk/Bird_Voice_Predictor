import librosa
import numpy as np
import joblib

# ======= LOAD MODEL, ENCODER, SCALER =======
model = joblib.load(r"C:\Users\user\Desktop\AI_ML\new repo for model 1\rf_model_filtered.pkl")
label_encoder = joblib.load(r"C:\Users\user\Desktop\AI_ML\new repo for model 1\label_encoder_filtered.pkl")
scaler = joblib.load(r"C:\Users\user\Desktop\AI_ML\new repo for model 1\scaler_filtered.pkl")

# ======= AUDIO PATH =======
test_audio_path = r"C:\Users\user\Downloads\XC979248 - Andean Guan - Penelope montagnii.mp3"

# ======= FEATURE EXTRACTION =======
def extract_features_for_prediction(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    if np.max(np.abs(y)) < 0.01:
        print("Audio too silent — skipping...")
        return None

    y = y / np.max(np.abs(y))

    # --- Dominant Frequency ---
    n = len(y)
    Y = np.fft.fft(y)
    magnitude = np.abs(Y)
    freq = np.fft.fftfreq(n, d=1/sr)
    valid = (freq > 100) & (freq < sr/2)
    positive_freqs = freq[valid]
    positive_mag = magnitude[valid]
    dominant_freq = positive_freqs[np.argmax(positive_mag)] if len(positive_mag) > 0 else 0.0

    # --- MFCCs ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # --- Spectral Bandwidth ---
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # --- Spectral Flatness ---
    spec_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    # --- RMS Energy ---
    rms = np.mean(librosa.feature.rms(y=y))

    # --- Chroma STFT ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # --- Combine all features ---
    features = np.concatenate([
        [dominant_freq],
        [spec_bw],
        [spec_flatness],
        [rms],
        mfcc_mean,
        chroma_mean
    ])

    return features.reshape(1, -1)

# ======= EXTRACT & PREDICT =======
features = extract_features_for_prediction(test_audio_path)

if features is not None:
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    species_name = label_encoder.inverse_transform([prediction[0]])[0]
    print(f"\n🔊 Predicted Bird Species: {species_name}")
else:
    print("❌ No prediction made.")
