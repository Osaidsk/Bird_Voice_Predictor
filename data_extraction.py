from pathlib import Path
import librosa
import numpy as np
import pandas as pd

data_dir = Path(r"C:\Users\user\Desktop\AI_ML\Voice of Birds")
combined_data = []
for file in data_dir.rglob("*"):
    if file.suffix.lower() not in (".mp3", ".wav"):
        continue
    # Assume species is the parent directory name
    species = file.parent.name
    filename = file.name
    try:
        y, sr = librosa.load(file, sr=None)

        if np.max(np.abs(y)) < 0.01:
            print(f"Skipping silent file: {filename}")
            continue

        # Normalize
        y = y / np.max(np.abs(y))

        # --- Feature: Dominant Frequency ---
        n = len(y)
        Y = np.fft.fft(y)
        magnitude = np.abs(Y)
        freq = np.fft.fftfreq(n, d=1/sr)
        valid_indices = (freq > 100) & (freq < sr / 2)
        if np.any(valid_indices):
            dom_freq = freq[valid_indices][np.argmax(magnitude[valid_indices])]
        else:
            dom_freq = 0.0

        # --- MFCCs ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1)

        # --- Spectral Bandwidth ---
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

        # --- Spectral Flatness ---
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

        # --- RMS Energy ---
        rms = np.mean(librosa.feature.rms(y=y))

        # --- Chroma STFT ---
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_means = np.mean(chroma, axis=1)

        # Combine features
        feature_row = {
            "Filename": filename,
            "Species": species,
            "Dominant_Frequency_Hz": dom_freq,
            "Spectral_Bandwidth": spectral_bandwidth,
            "Spectral_Flatness": spectral_flatness,
            "RMS_Energy": rms,
        }

        # Add MFCCs
        for i in range(13):
            feature_row[f"MFCC_{i+1}"] = mfcc_means[i]

        # Add Chroma
        for i in range(12):
            feature_row[f"Chroma_{i+1}"] = chroma_means[i]

        combined_data.append(feature_row)

    except Exception as e:
        print(f"Error with {filename}: {e}")

# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

# Save to CSV
output_path = Path("bird_audio_selected_features.csv")
df_combined.to_csv(output_path, index=False)

# Show sample
print("\n✅ Dataset with selected features saved as:")
print(output_path)
print(df_combined.head())
