
# ==== STEP 1: LOAD AND FILTER ====
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Load the dataset from CSV
df = pd.read_csv("bird_audio_selected_features.csv")

# Keep only species with at least 30 samples
df = df.groupby('Species').filter(lambda x: len(x) >= 30)


# ==== STEP 2: INITIAL ENCODING ====
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Species'])


# ==== STEP 3: FEATURE SELECTION ====
X = df.drop(columns=['Filename', 'Species', 'Label'])    # Features (input)
y = df['Label']                                          # Labels (output)


# ==== STEP 4: TRAIN-TEST SPLIT ====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)                                                        # train= 80% , test= 20% 


# ==== STEP 5: SCALING ====
scaler = StandardScaler()     #
X_train = scaler.fit_transform(X_train)                  # Fit and transform on training data
X_test = scaler.transform(X_test)                        # Transform on test data


# ==== STEP 6: RANDOM FOREST MODEL ====
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

# Train the RF model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)


# ==== STEP 6: INITIAL REPORT ====
print("\n=== Initial Classification Report ===")
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)

# ==== STEP 7: IDENTIFY STRONG SPECIES (precision >= 0.8), excluding known weaker species ====
report_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

weaker_species = {
    'Speckled Chachalaca_sound',
    'Little Chachalaca_sound',
    'Crested Guan_sound',
    'Colombian Chachalaca_sound',
    'Rusty-margined Guan_sound',
    'West Mexican Chachalaca_sound',
    'Little Tinamou_sound',
    'Orange-footed Scrubfowl_sound',
    'Tawny-breasted Tinamou_sound'
}
strong_species = [
    species for species in label_encoder.classes_
    if report_dict.get(species, {}).get('precision', 0) >= 0.8 and species not in weaker_species
]

print("\n✅ Species kept for second run:", strong_species)

# ==== STEP 8: FILTER DATA AGAIN BASED ON STRONG SPECIES ====
df_filtered = df[df['Species'].isin(strong_species)].reset_index(drop=True)

# Re-encode labels after filtering
label_encoder_filtered = LabelEncoder()
df_filtered['Label'] = label_encoder_filtered.fit_transform(df_filtered['Species'])

# Separate features and labels (no scaling yet)
X_filt = df_filtered.drop(columns=['Filename', 'Species', 'Label'])
y_filt = df_filtered['Label']

# ==== STEP 9: SPLIT FIRST, THEN SCALE ====
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_filt, y_filt, test_size=0.2, stratify=y_filt, random_state=42
)

# Scale features (fit on train only, transform test)
scaler_filtered = StandardScaler()
X_train_f_scaled = scaler_filtered.fit_transform(X_train_f)
X_test_f_scaled = scaler_filtered.transform(X_test_f)

# ==== STEP 10: FINAL MODEL TRAINING AND PREDICTION ====
rf_model.fit(X_train_f_scaled, y_train_f)
y_pred_f = rf_model.predict(X_test_f_scaled)


# ==== FINAL REPORT ====
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv("classification_report.csv", index=True)
print("\n=== Final Classification Report (Filtered Strong Species) ===")
print(classification_report(y_test_f, y_pred_f, target_names=label_encoder_filtered.classes_))

# ==== CONFUSION MATRIX ====
cm = confusion_matrix(y_test_f, y_pred_f, normalize='true')
np.save("confusion_matrix.npy", cm)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt='.2f',
            xticklabels=label_encoder_filtered.classes_,
            yticklabels=label_encoder_filtered.classes_,
            cmap='YlGnBu')
plt.title("Normalized Confusion Matrix - Final Ensemble Model")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()
# ✅ Save confusion matrix image (instead of just showing)
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()


# ==== SAVE FINAL ARTIFACTS ====
joblib.dump(rf_model, 'rf_model_filtered.pkl')
joblib.dump(scaler, 'scaler_filtered.pkl')
joblib.dump(label_encoder_filtered, 'label_encoder_filtered.pkl')
print("✅ Final model, scaler, and encoder saved.")


