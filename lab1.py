import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


print("=" * 70)
print("LAB EXPERIMENT 1")
print("AI IN HEALTHCARE")
print("Reading Tabular, Textual, Image, Signal and Medical Data")
print("=" * 70)


# ============================================================
# 1. TABULAR / MEDICAL DATA
# UCI CLEVELAND HEART DISEASE DATASET
# ============================================================

print("\n1. TABULAR / MEDICAL DATA")
print("-" * 50)

data_path = "heart+disease/processed.cleveland.data"

columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

df = pd.read_csv(
    data_path,
    header=None,
    names=columns,
    na_values="?"
)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())


# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

print("\n2. DATA PREPROCESSING")
print("-" * 50)

print("\nMissing values before preprocessing:")
print(df.isnull().sum())

for column in df.columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].median())

print("\nMissing values after preprocessing:")
print(df.isnull().sum())


# Convert target:
# 0 = No Heart Disease
# 1 = Heart Disease

df["target"] = df["target"].apply(
    lambda x: 0 if x == 0 else 1
)

print("\nTarget distribution:")
print(df["target"].value_counts())


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n3. EXPLORATORY DATA ANALYSIS")
print("-" * 50)

plt.figure(figsize=(8, 5))

sns.countplot(
    x="target",
    data=df
)

plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease")
plt.ylabel("Number of Patients")

plt.xticks(
    [0, 1],
    ["No Disease", "Disease"]
)

plt.tight_layout()
plt.show()


# Correlation matrix

plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()


# ============================================================
# 4. HEART DISEASE PREDICTION
# LOGISTIC REGRESSION
# ============================================================

print("\n4. HEART DISEASE PREDICTION")
print("-" * 50)

X = df.drop("target", axis=1)
y = df["target"]


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Feature scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Create Logistic Regression model

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# Predictions

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# Accuracy

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ROC AUC

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\nModel Performance")
print("Accuracy:", round(accuracy, 4))
print("ROC AUC:", round(roc_auc, 4))


# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Disease"]
)

disp.plot()

plt.title("Heart Disease Prediction - Confusion Matrix")

plt.tight_layout()
plt.show()


# ============================================================
# 6. SIGNAL DATA
# MIT-BIH ARRHYTHMIA DATABASE
# ECG RECORD 100
# ============================================================

print("\n6. SIGNAL DATA - ECG")
print("-" * 50)

print("\nDownloading MIT-BIH ECG record 100...")

record = wfdb.rdrecord(
    "100",
    pn_dir="mitdb"
)

annotation = wfdb.rdann(
    "100",
    "atr",
    pn_dir="mitdb"
)


print("\nECG Record Information")

print("Number of samples:", record.sig_len)

print("Sampling frequency:", record.fs)

print("Number of channels:", record.n_sig)

print("Channel names:", record.sig_name)

print("Number of annotations:", len(annotation.sample))


# ============================================================
# 7. PLOT ECG SIGNAL
# ============================================================

sampling_frequency = record.fs

start_sample = 0

end_sample = int(
    10 * sampling_frequency
)

ecg_signal = record.p_signal[
    start_sample:end_sample,
    0
]

time = np.arange(
    len(ecg_signal)
) / sampling_frequency


plt.figure(figsize=(14, 5))

plt.plot(
    time,
    ecg_signal
)

plt.title("MIT-BIH ECG Signal - Record 100")

plt.xlabel("Time (seconds)")

plt.ylabel("ECG Amplitude")

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 8. ECG WITH BEAT ANNOTATIONS
# ============================================================

annotation_samples = annotation.sample

annotation_samples = annotation_samples[
    annotation_samples < end_sample
]

annotation_times = (
    annotation_samples / sampling_frequency
)

annotation_values = record.p_signal[
    annotation_samples,
    0
]


plt.figure(figsize=(14, 5))

plt.plot(
    time,
    ecg_signal,
    label="ECG Signal"
)

plt.scatter(
    annotation_times,
    annotation_values,
    marker="o",
    label="Beat Annotations"
)

plt.title("ECG Signal with Beat Annotations")

plt.xlabel("Time (seconds)")

plt.ylabel("ECG Amplitude")

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 9. BASIC HEART RATE STATISTICS
# ============================================================

beat_intervals = np.diff(
    annotation_samples
) / sampling_frequency


average_beat_interval = np.mean(
    beat_intervals
)


heart_rate = (
    60 / average_beat_interval
)


print("\nECG Statistics")

print(
    "Average beat interval:",
    round(average_beat_interval, 4),
    "seconds"
)

print(
    "Approximate heart rate:",
    round(heart_rate, 2),
    "BPM"
)


# ============================================================
# 10. TEXTUAL DATA
# CLINICAL NOTE
# ============================================================

print("\n10. TEXTUAL DATA")
print("-" * 50)


clinical_note = """
Patient is a 58 year old male.
The patient reports chest pain and shortness of breath.
Blood pressure is elevated.
The patient also reports occasional fatigue.
The doctor recommends further cardiac evaluation.
"""


print("\nClinical Note:")

print(clinical_note)


# ============================================================
# 11. TEXT PREPROCESSING
# ============================================================

clean_text = clinical_note.lower()

clean_text = re.sub(
    r"[^a-zA-Z\s]",
    "",
    clean_text
)

words = clean_text.split()


print("\nTokens:")

print(words)


# ============================================================
# 12. WORD FREQUENCY
# ============================================================

word_frequency = Counter(words)


print("\nWord Frequency:")

for word, frequency in word_frequency.most_common():

    print(
        word,
        ":",
        frequency
    )


# ============================================================
# 13. MEDICAL TERMS
# ============================================================

medical_terms = [
    "chest",
    "pain",
    "shortness",
    "breath",
    "blood",
    "pressure",
    "fatigue",
    "cardiac"
]


found_terms = []

for word in medical_terms:

    if word in words:

        found_terms.append(word)


print("\nImportant Medical Terms:")

print(found_terms)


# ============================================================
# 14. IMAGE DATA
# SYNTHETIC MEDICAL IMAGE
# ============================================================

print("\n14. IMAGE DATA")
print("-" * 50)

print("\nCreating synthetic medical image...")


image_array = np.random.randint(
    0,
    256,
    (256, 256),
    dtype=np.uint8
)


image = Image.fromarray(
    image_array,
    mode="L"
)


print(
    "Image format:",
    image.format
)

print(
    "Image size:",
    image.size
)

print(
    "Image mode:",
    image.mode
)


# ============================================================
# 15. DISPLAY IMAGE
# ============================================================

plt.figure(figsize=(6, 6))

plt.imshow(
    image,
    cmap="gray"
)

plt.title("Synthetic Medical Image")

plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# 16. RESIZE IMAGE
# ============================================================

resized_image = image.resize(
    (128, 128)
)


print(
    "\nOriginal image size:",
    image.size
)

print(
    "Resized image size:",
    resized_image.size
)


plt.figure(figsize=(6, 6))

plt.imshow(
    resized_image,
    cmap="gray"
)

plt.title("Resized Medical Image")

plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# 17. MULTIMODAL HEALTHCARE DATA
# ============================================================

print("\n17. MULTIMODAL DATA INTEGRATION")
print("-" * 50)

print("""
The experiment demonstrates four different healthcare data
modalities:

1. Tabular / Medical Data
   Cleveland Heart Disease Dataset

2. Signal Data
   MIT-BIH ECG Signal

3. Textual Data
   Clinical Patient Note

4. Image Data
   Synthetic Medical Image

These modalities can be combined in a future multimodal
healthcare system.

For example:

Tabular Features
        +
ECG Features
        +
Clinical Text Features
        +
Medical Image Features
        |
        v
Multimodal Fusion Model
        |
        v
Healthcare Prediction
""")


print("\n" + "=" * 70)

print("LAB EXPERIMENT 1 COMPLETED")

print("=" * 70)