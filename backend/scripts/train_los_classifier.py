# /home/adomin/kaizen/backend/scripts/train_los_classifier.py
# ==============================================================================
# KAIZEN HEALTHCARE - OPERATIONAL LOS CATEGORICAL CLASSIFIER
#
# Architecture:
#   Multiclass LightGBM with Class-Weighted Cross-Entropy
#   Target: 4 Operational LOS Tiers (<24h, 24-72h, 72-168h, >168h)
#   Split: Chronological 70k Train / 15k Calibration / 15k Test
# ==============================================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PATHS AND CONFIGURATION
# ==============================================================================

RANDOM_STATE = 42
TRAIN_SIZE = 70000
CALIBRATION_SIZE = 15000
TEST_SIZE = 15000

SCRIPT_DIR = Path(__file__).resolve().parent          # kaizen/backend/scripts
BACKEND_DIR = SCRIPT_DIR.parent                      # kaizen/backend
PROJECT_ROOT = SCRIPT_DIR.parent.parent               # kaizen

# Resolved Path objects
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_patient_stays_100k_fixed.csv"
OUTPUT_DIR = BACKEND_DIR / "outputs"
MODEL_DIR = BACKEND_DIR / "models"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIER_LABELS = ["<24h (Obs/ED)", "24-72h (1-3d)", "72-168h (3-7d)", ">168h (>7d)"]

# ==============================================================================
# 2. DATA PREPARATION AND DERIVED FEATURE ENGINEERING
# ==============================================================================

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

df["arrival_time"] = pd.to_datetime(df["arrival_time"])
df = df.sort_values("arrival_time").reset_index(drop=True)
df["arrival_hour"] = df["arrival_time"].dt.hour

# Clinical derived features
df["shock_index"] = df["heart_rate"] / df["sbp"].replace(0, np.nan)
df["pulse_pressure"] = df["sbp"] - df["dbp"]
df["mean_arterial_pressure"] = df["dbp"] + (df["sbp"] - df["dbp"]) / 3.0
df["oxygen_deficit"] = 100.0 - df["o2_sat"]
df["temperature_deviation"] = np.abs(df["temp_c"] - 37.0)
df["arrival_hour_sin"] = np.sin(2 * np.pi * df["arrival_hour"] / 24.0)
df["arrival_hour_cos"] = np.cos(2 * np.pi * df["arrival_hour"] / 24.0)

# Severity score
hr_s = np.clip((df["heart_rate"] - 80.0) / 60.0, 0.0, 1.0)
sbp_s = np.clip((120.0 - df["sbp"]) / 60.0, 0.0, 1.0)
rr_s = np.clip((df["resp_rate"] - 18.0) / 20.0, 0.0, 1.0)
o2_s = np.clip((95.0 - df["o2_sat"]) / 25.0, 0.0, 1.0)
temp_s = np.clip(np.abs(df["temp_c"] - 37.0) / 4.0, 0.0, 1.0)
comorb_s = np.clip(df["charlson_index"] / 10.0, 0.0, 1.0)
esi_s = (6.0 - df["triage_acuity"]) / 5.0

df["severity_score"] = np.clip(
    0.25 * esi_s
    + 0.15 * hr_s
    + 0.20 * sbp_s
    + 0.15 * rr_s
    + 0.15 * o2_s
    + 0.05 * temp_s
    + 0.05 * comorb_s,
    0.0,
    1.0,
)

if "requires_ventilation" in df.columns:
    df["requires_ventilation"] = df["requires_ventilation"].astype(int)
else:
    df["requires_ventilation"] = 0

# Construct 4 operational target tiers
df["los_tier"] = pd.cut(
    df["los_hours"],
    bins=[-np.inf, 24.0, 72.0, 168.0, np.inf],
    labels=[0, 1, 2, 3],
).astype(int)

NUMERIC_FEATURES = [
    "age", "charlson_index", "heart_rate", "sbp", "dbp", "o2_sat", "resp_rate",
    "temp_c", "triage_acuity", "shock_index", "pulse_pressure",
    "mean_arterial_pressure", "oxygen_deficit", "temperature_deviation",
    "severity_score", "requires_ventilation", "arrival_hour_sin", "arrival_hour_cos",
]
CATEGORICAL_FEATURES = ["gender", "chief_complaint"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ]
)

# Chronological split
train_df = df.iloc[:TRAIN_SIZE].copy()
test_df = df.iloc[TRAIN_SIZE + CALIBRATION_SIZE : TRAIN_SIZE + CALIBRATION_SIZE + TEST_SIZE].copy()

X_train = preprocessor.fit_transform(train_df[FEATURES])
y_train = train_df["los_tier"].values

X_test = preprocessor.transform(test_df[FEATURES])
y_test = test_df["los_tier"].values
test_actual_hours = test_df["los_hours"].values

# ==============================================================================
# 3. MODEL TRAINING (MULTICLASS LIGHTGBM)
# ==============================================================================

print("Training Multiclass LightGBM Classifier...")
clf = LGBMClassifier(
    objective="multiclass",
    num_class=4,
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=7,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
)

clf.fit(X_train, y_train)

# Inference
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

# ==============================================================================
# 4. METRIC COMPUTATION
# ==============================================================================

acc = float(accuracy_score(y_test, y_pred))
bal_acc = float(balanced_accuracy_score(y_test, y_pred))
adjacent_acc = float(np.mean(np.abs(y_test - y_pred) <= 1))
loss = float(log_loss(y_test, y_prob))
macro_auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))

# Outlier screening metric (>168h binary AUC)
binary_long_actual = (test_actual_hours > 168.0).astype(int)
binary_long_prob = y_prob[:, 3]
long_stay_auc = float(roc_auc_score(binary_long_actual, binary_long_prob))

print("\n" + "=" * 70)
print("OPERATIONAL LOS CLASSIFICATION RESULTS")
print("=" * 70)
print(f"Overall Accuracy          : {acc * 100:.2f}%")
print(f"Balanced Accuracy         : {bal_acc * 100:.2f}%")
print(f"Adjacent-Tier Accuracy    : {adjacent_acc * 100:.2f}% (Exact or off-by-1 tier)")
print(f"Multiclass Log Loss       : {loss:.4f}")
print(f"Macro ROC-AUC (4 Tiers)   : {macro_auc:.4f}")
print(f"Long-Stay (>7d) Alert AUC : {long_stay_auc:.4f}")

print("\nDetailed Tier Breakdown:")
print(classification_report(y_test, y_pred, target_names=TIER_LABELS))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=TIER_LABELS, columns=TIER_LABELS)
print("Confusion Matrix (Rows: Actual, Columns: Predicted):")
print(cm_df)

# ==============================================================================
# 5. SAVE PREDICTIONS AND METRICS
# ==============================================================================

out_df = test_df[["stay_id", "patient_id", "arrival_time", "triage_acuity"]].copy()
out_df["actual_los_hours"] = test_actual_hours
out_df["actual_tier"] = y_test
out_df["predicted_tier"] = y_pred
out_df["prob_tier_0_lt24h"] = y_prob[:, 0]
out_df["prob_tier_1_24_72h"] = y_prob[:, 1]
out_df["prob_tier_2_72_168h"] = y_prob[:, 2]
out_df["prob_tier_3_gt168h"] = y_prob[:, 3]

pred_csv_path = OUTPUT_DIR / "los_classification_test_predictions.csv"
metrics_json_path = OUTPUT_DIR / "los_classification_metrics.json"

out_df.to_csv(pred_csv_path, index=False)

metrics_report = {
    "model": "4-Tier Operational LightGBM Classifier",
    "accuracy": acc,
    "balanced_accuracy": bal_acc,
    "adjacent_tier_accuracy": adjacent_acc,
    "log_loss": loss,
    "macro_roc_auc": macro_auc,
    "long_stay_screening_auc": long_stay_auc,
    "tier_labels": TIER_LABELS,
}

with open(metrics_json_path, "w", encoding="utf-8") as f:
    json.dump(metrics_report, f, indent=4)

print(f"\nSaved predictions : {pred_csv_path}")
print(f"Saved metrics     : {metrics_json_path}")