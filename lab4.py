# Core libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing & modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

pd.set_option("display.max_columns", 40)



# Path to the raw data file (update if needed)
DATA_PATH = "wdbc.data"  # place wdbc.data in the same folder as this notebook

# Column names: id, diagnosis, then 10 base features x 3 (mean, se, worst)
base_features = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
]
suffixes = ["mean", "se", "worst"]
feature_cols = [f"{feat}_{suf}" for suf in suffixes for feat in base_features]
columns = ["id", "diagnosis"] + feature_cols

df = pd.read_csv(DATA_PATH, header=None, names=columns)
print("Shape:", df.shape)
df.head()



df.info()



summary_stats = df.drop(columns=["id"]).describe().T
summary_stats["median"] = df.drop(columns=["id", "diagnosis"]).median()
summary_stats = summary_stats[["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]]
summary_stats



# Missing values
print("Missing values per column:")
print(df.isnull().sum().sum(), "total missing values")

# Duplicates (based on all feature columns, ignoring id)
n_dupes = df.drop(columns=["id"]).duplicated().sum()
print(f"\nDuplicate rows (excluding id): {n_dupes}")

if n_dupes > 0:
    df = df.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)
    print("Duplicates removed. New shape:", df.shape)
else:
    print("No duplicate rows found — no action needed.")



# Outlier detection using IQR method
def iqr_outlier_count(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum()

outlier_counts = df[feature_cols].apply(iqr_outlier_count).sort_values(ascending=False)
outlier_counts_pct = (outlier_counts / len(df) * 100).round(1)

outlier_summary = pd.DataFrame({
    "outlier_count": outlier_counts,
    "outlier_pct": outlier_counts_pct
})
outlier_summary.head(15)



# Sanity check: no impossible values (e.g., negative measurements)
print("Any negative values in feature columns?", (df[feature_cols] < 0).any().any())
print("Any zero values in feature columns (could indicate missing/erroneous encoding)?")
print((df[feature_cols] == 0).sum()[lambda x: x > 0])



diagnosis_counts = df["diagnosis"].value_counts()
print(diagnosis_counts)
print((diagnosis_counts / len(df) * 100).round(1))

plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x="diagnosis", hue="diagnosis", palette={"B": "#4C72B0", "M": "#C44E52"}, legend=False)
ax.set_title("Class Distribution: Benign (B) vs Malignant (M)")
ax.set_xlabel("Diagnosis")
ax.set_ylabel("Count")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom")
plt.tight_layout()
plt.show()



key_mean_features = ["radius_mean", "texture_mean", "perimeter_mean",
                      "area_mean", "smoothness_mean", "concavity_mean"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(key_mean_features):
    sns.histplot(data=df, x=col, hue="diagnosis", kde=True, ax=axes[i],
                 palette={"B": "#4C72B0", "M": "#C44E52"}, alpha=0.6)
    axes[i].set_title(col)
plt.tight_layout()
plt.show()



fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(key_mean_features):
    sns.boxplot(data=df, x="diagnosis", y=col, hue="diagnosis", ax=axes[i],
                palette={"B": "#4C72B0", "M": "#C44E52"}, legend=False)
    axes[i].set_title(col)
plt.tight_layout()
plt.show()



plt.figure(figsize=(18, 14))
corr = df[feature_cols].corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, linewidths=0.3)
plt.title("Correlation Heatmap of All 30 Features")
plt.tight_layout()
plt.show()

# Top correlated feature pairs (excluding self-correlation)
corr_pairs = corr.abs().unstack().sort_values(ascending=False)
corr_pairs = corr_pairs[corr_pairs < 1.0]
print("Top 10 most correlated feature pairs:")
print(corr_pairs.drop_duplicates().head(10))



# 2.1 Encode target: M -> 1 (malignant / positive class), B -> 0 (benign)
le = LabelEncoder()
df["target"] = le.fit_transform(df["diagnosis"])  # B=0, M=1 (alphabetical -> verify below)
print(dict(zip(le.classes_, le.transform(le.classes_))))

X = df[feature_cols].copy()
y = df["target"].copy()



# 2.2 Train/test split (80/20) with stratified sampling to preserve class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
print("\nTrain class distribution:\n", y_train.value_counts(normalize=True).round(3))
print("\nTest class distribution:\n", y_test.value_counts(normalize=True).round(3))



# 2.3 Standardize features (fit scaler on TRAIN only to avoid data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for readability
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

X_train_scaled.describe().loc[["mean", "std"]].round(3)



# Train logistic regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

y_pred = log_reg.predict(X_test_scaled)
y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]  # probability of malignant (class 1)



acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")



report = classification_report(y_test, y_pred, target_names=["Benign (0)", "Malignant (1)"])
print(report)

# As a tidy dataframe too
report_dict = classification_report(y_test, y_pred, target_names=["Benign (0)", "Malignant (1)"], output_dict=True)
pd.DataFrame(report_dict).T.round(3)



cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=True)
plt.title("Confusion Matrix — Logistic Regression")
plt.tight_layout()
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}  <-- clinically most critical (missed cancers)")
print(f"True Positives:  {tp}")



fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6.5, 5.5))
plt.plot(fpr, tpr, color="#C44E52", lw=2, label=f"ROC curve (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Logistic Regression")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

print(f"AUC Score: {auc_score:.4f}")

