# ============================================================
# Predictive Resource Optimization System for Healthcare
# MODEL #9 - FINAL TAIL-AWARE TWO-STAGE LOS MODEL
#
# Architecture:
#
# Patient clinical features
#          ↓
# Preprocessing
#          ↓
# Long-stay classifier: P(LOS > 168h)
#          ↓
# ┌───────────────────────┐
# │                       │
# Normal-stay models    Long-stay models
# │                       │
# P10/P50/P90           P10/P50/P90
# │                       │
# └───────────┬───────────┘
#             ↓
# Probability-weighted quantiles
#             ↓
# MAPIE Conformalized Quantile Regression
#             ↓
# Calibrated 80% LOS prediction interval
#
# Target:
#     los_hours
#
# Final prediction:
#     P50
#
# Prediction interval:
#     80%
#     Lower = P10
#     Upper = P90
#
# Chronological split:
#     70,000 train
#     15,000 calibration
#     15,000 test
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor, LGBMClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    log_loss,
    roc_auc_score,
)

from mapie.regression import ConformalizedQuantileRegressor


warnings.filterwarnings("ignore")


# ============================================================
# 2. GLOBAL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TRAIN_SIZE = 70_000
CALIBRATION_SIZE = 15_000
TEST_SIZE = 15_000

CONFIDENCE_LEVEL = 0.80

LOWER_QUANTILE = 0.10
MEDIAN_QUANTILE = 0.50
UPPER_QUANTILE = 0.90

LONG_STAY_THRESHOLD = 168.0


# ============================================================
# 3. PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "synthetic_patient_stays_100k.csv"


# ============================================================
# 4. FEATURE DEFINITIONS
# ============================================================

NUMERIC_FEATURES = [
    "age",
    "charlson_index",
    "heart_rate",
    "sbp",
    "dbp",
    "o2_sat",
    "resp_rate",
    "temp_c",
    "triage_acuity",
    "requires_ventilation",
    "severity_score",
]

CATEGORICAL_FEATURES = [
    "gender",
    "chief_complaint",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "los_hours"


# ============================================================
# 5. LIGHTGBM CONFIGURATION
# ============================================================

BASE_LGB_PARAMS = {
    "num_leaves": 31,
    "max_depth": 7,
    "learning_rate": 0.03,
    "n_estimators": 800,
    "min_child_samples": 100,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.10,
    "reg_lambda": 1.00,
    "random_state": RANDOM_STATE,
    "verbosity": -1,
    "n_jobs": -1,
}


# ============================================================
# 6. TWO-STAGE QUANTILE MODEL
# ============================================================
#
# This class combines:
#
#     Normal-stay quantile model
#     Long-stay quantile model
#
# using:
#
#     P(long stay) = classifier probability
#
# final_quantile =
#
#     (1 - P_long) * normal_quantile
#     + P_long * long_quantile
#
# This gives the model a smooth tail-aware transition.
# ============================================================


class TwoStageQuantileModel:

    def __init__(
        self,
        classifier,
        normal_model,
        long_model,
    ):
        self.classifier = classifier
        self.normal_model = normal_model
        self.long_model = long_model

        self.is_fitted_ = True

    def fit(self, X, y):
        self.is_fitted_ = True
        return self

    def predict(self, X):

        long_probability = self.classifier.predict_proba(X)[:, 1]

        normal_prediction = self.normal_model.predict(X)

        long_prediction = self.long_model.predict(X)

        prediction = (
            (1.0 - long_probability) * normal_prediction
            + long_probability * long_prediction
        )

        return prediction


# ============================================================
# 7. CREATE SEVERITY SCORE
# ============================================================
#
# This follows the severity construction used by the
# synthetic patient generator.
#
# It is calculated BEFORE LOS generation and is therefore
# available as an arrival-time clinical feature in this
# synthetic dataset.
# ============================================================


def create_severity_score(df):

    heart_rate_severity = np.clip(
        (df["heart_rate"] - 80.0) / 60.0,
        0.0,
        1.0,
    )

    blood_pressure_severity = np.clip(
        (120.0 - df["sbp"]) / 60.0,
        0.0,
        1.0,
    )

    respiratory_severity = np.clip(
        (df["resp_rate"] - 18.0) / 20.0,
        0.0,
        1.0,
    )

    oxygen_severity = np.clip(
        (95.0 - df["o2_sat"]) / 25.0,
        0.0,
        1.0,
    )

    temperature_severity = np.clip(
        np.abs(df["temp_c"] - 37.0) / 4.0,
        0.0,
        1.0,
    )

    comorbidity_severity = np.clip(
        df["charlson_index"] / 10.0,
        0.0,
        1.0,
    )

    esi_severity = (
        (6.0 - df["triage_acuity"]) / 5.0
    )

    severity = (
        0.25 * esi_severity
        + 0.15 * heart_rate_severity
        + 0.20 * blood_pressure_severity
        + 0.15 * respiratory_severity
        + 0.15 * oxygen_severity
        + 0.05 * temperature_severity
        + 0.05 * comorbidity_severity
    )

    return np.clip(severity, 0.0, 1.0)


# ============================================================
# 8. QUANTILE MODEL TRAINING FUNCTION
# ============================================================


def train_quantile_model(
    X,
    y,
    alpha,
    sample_weight=None,
):

    params = BASE_LGB_PARAMS.copy()

    params["objective"] = "quantile"
    params["alpha"] = alpha

    model = LGBMRegressor(**params)

    model.fit(
        X,
        y,
        sample_weight=sample_weight,
    )

    return model


# ============================================================
# 9. PRINT SECTION
# ============================================================


def section(title):

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# 10. LOAD DATA
# ============================================================

section("MODEL #9 - FINAL TAIL-AWARE TWO-STAGE LOS MODEL")

print("\nWorking directory:")
print(BASE_DIR)

print("\nDataset:")
print(DATA_PATH)

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}"
    )


df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 11. BASIC DATA VALIDATION
# ============================================================

section("STEP 1 - DATA VALIDATION")

required_columns = [
    "stay_id",
    "patient_id",
    "age",
    "gender",
    "charlson_index",
    "heart_rate",
    "sbp",
    "dbp",
    "o2_sat",
    "resp_rate",
    "temp_c",
    "triage_acuity",
    "chief_complaint",
    "arrival_time",
    "los_hours",
    "requires_ventilation",
]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

print("Required columns: PASS")

print(
    "Missing values:",
    int(df[required_columns].isna().sum().sum())
)

print(
    "Duplicate stay IDs:",
    int(df["stay_id"].duplicated().sum())
)

print(
    "Negative LOS:",
    int((df["los_hours"] < 0).sum())
)


# ============================================================
# 12. CHRONOLOGICAL ORDER
# ============================================================

section("STEP 2 - CHRONOLOGICAL ORDER")

df["arrival_time"] = pd.to_datetime(
    df["arrival_time"]
)

df = df.sort_values(
    "arrival_time"
).reset_index(drop=True)

print(
    "First arrival:",
    df["arrival_time"].min()
)

print(
    "Last arrival:",
    df["arrival_time"].max()
)


# ============================================================
# 13. CREATE CAUSAL FEATURES
# ============================================================

section("STEP 3 - CREATE CAUSAL FEATURES")

print("Creating severity_score...")

df["severity_score"] = create_severity_score(df)

df["requires_ventilation"] = (
    df["requires_ventilation"]
    .astype(int)
)

print(
    "Severity range:",
    round(df["severity_score"].min(), 4),
    "to",
    round(df["severity_score"].max(), 4)
)

print(
    "Requires ventilation:",
    df["requires_ventilation"].value_counts()
    .sort_index()
    .to_dict()
)


# ============================================================
# 14. CHRONOLOGICAL 70/15/15 SPLIT
# ============================================================

section("STEP 4 - CHRONOLOGICAL 70/15/15 SPLIT")

train_end = TRAIN_SIZE

calibration_end = (
    TRAIN_SIZE + CALIBRATION_SIZE
)

train_df = df.iloc[:train_end].copy()

calibration_df = df.iloc[
    train_end:calibration_end
].copy()

test_df = df.iloc[
    calibration_end:
].copy()

print(
    f"Training:     {len(train_df):,}"
)

print(
    f"Calibration:  {len(calibration_df):,}"
)

print(
    f"Test:         {len(test_df):,}"
)

print("\nTraining period:")
print(
    train_df["arrival_time"].min(),
    "->",
    train_df["arrival_time"].max()
)

print("\nCalibration period:")
print(
    calibration_df["arrival_time"].min(),
    "->",
    calibration_df["arrival_time"].max()
)

print("\nTest period:")
print(
    test_df["arrival_time"].min(),
    "->",
    test_df["arrival_time"].max()
)


# ============================================================
# 15. PREPARE X / Y
# ============================================================

X_train_raw = train_df[FEATURES].copy()
y_train = train_df[TARGET].values

X_calibration_raw = calibration_df[FEATURES].copy()
y_calibration = calibration_df[TARGET].values

X_test_raw = test_df[FEATURES].copy()
y_test = test_df[TARGET].values


# ============================================================
# 16. PREPROCESSING
# ============================================================

section("STEP 5 - FEATURE PREPROCESSING")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="drop",
)

X_train = preprocessor.fit_transform(
    X_train_raw
)

X_calibration = preprocessor.transform(
    X_calibration_raw
)

X_test = preprocessor.transform(
    X_test_raw
)

print(
    "Training matrix:",
    X_train.shape
)

print(
    "Calibration matrix:",
    X_calibration.shape
)

print(
    "Test matrix:",
    X_test.shape
)


# ============================================================
# 17. CREATE LONG-STAY TARGET
# ============================================================

section("STEP 6 - LONG-STAY CLASSIFICATION TARGET")

y_train_long = (
    y_train > LONG_STAY_THRESHOLD
).astype(int)

y_calibration_long = (
    y_calibration > LONG_STAY_THRESHOLD
).astype(int)

y_test_long = (
    y_test > LONG_STAY_THRESHOLD
).astype(int)

print(
    f"Long-stay threshold: > {LONG_STAY_THRESHOLD:.0f} hours"
)

print(
    "\nTraining long stays:",
    int(y_train_long.sum()),
    f"({100 * y_train_long.mean():.2f}%)"
)

print(
    "Calibration long stays:",
    int(y_calibration_long.sum()),
    f"({100 * y_calibration_long.mean():.2f}%)"
)

print(
    "Test long stays:",
    int(y_test_long.sum()),
    f"({100 * y_test_long.mean():.2f}%)"
)


# ============================================================
# 18. TRAIN LONG-STAY CLASSIFIER
# ============================================================

section("STEP 7 - TRAIN LONG-STAY CLASSIFIER")

positive_count = int(y_train_long.sum())
negative_count = int(
    len(y_train_long) - positive_count
)

scale_pos_weight = (
    negative_count / positive_count
)

print(
    "Scale positive weight:",
    round(scale_pos_weight, 3)
)

classifier = LGBMClassifier(
    objective="binary",
    n_estimators=800,
    num_leaves=31,
    max_depth=7,
    learning_rate=0.03,
    min_child_samples=100,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.10,
    reg_lambda=1.00,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    verbosity=-1,
    n_jobs=-1,
)

classifier.fit(
    X_train,
    y_train_long,
)

calibration_long_probability = (
    classifier.predict_proba(X_calibration)[:, 1]
)

test_long_probability = (
    classifier.predict_proba(X_test)[:, 1]
)

print("Long-stay classifier trained.")

try:

    calibration_auc = roc_auc_score(
        y_calibration_long,
        calibration_long_probability,
    )

    test_auc = roc_auc_score(
        y_test_long,
        test_long_probability,
    )

    print(
        f"Calibration ROC-AUC: {calibration_auc:.4f}"
    )

    print(
        f"Test ROC-AUC:        {test_auc:.4f}"
    )

except Exception:

    calibration_auc = None
    test_auc = None


# ============================================================
# 19. SPLIT TRAINING DATA INTO NORMAL / LONG STAYS
# ============================================================

section("STEP 8 - CREATE NORMAL AND LONG-STAY REGRESSION DATA")

normal_mask = (
    y_train <= LONG_STAY_THRESHOLD
)

long_mask = (
    y_train > LONG_STAY_THRESHOLD
)

X_train_normal = X_train[normal_mask]
y_train_normal = y_train[normal_mask]

X_train_long = X_train[long_mask]
y_train_long_los = y_train[long_mask]

print(
    "Normal-stay training records:",
    len(y_train_normal)
)

print(
    "Long-stay training records:",
    len(y_train_long_los)
)

print(
    "\nNormal LOS mean:",
    round(y_train_normal.mean(), 3)
)

print(
    "Long-stay LOS mean:",
    round(y_train_long_los.mean(), 3)
)

print(
    "Long-stay LOS median:",
    round(np.median(y_train_long_los), 3)
)


# ============================================================
# 20. TRAIN NORMAL-STAY QUANTILE MODELS
# ============================================================

section("STEP 9 - TRAIN NORMAL-STAY QUANTILE MODELS")

print("Training normal P10...")

normal_p10 = train_quantile_model(
    X_train_normal,
    y_train_normal,
    LOWER_QUANTILE,
)

print("Normal P10 complete.")

print("Training normal P50...")

normal_p50 = train_quantile_model(
    X_train_normal,
    y_train_normal,
    MEDIAN_QUANTILE,
)

print("Normal P50 complete.")

print("Training normal P90...")

normal_p90 = train_quantile_model(
    X_train_normal,
    y_train_normal,
    UPPER_QUANTILE,
)

print("Normal P90 complete.")


# ============================================================
# 21. TRAIN LONG-STAY QUANTILE MODELS
# ============================================================

section("STEP 10 - TRAIN LONG-STAY QUANTILE MODELS")

print("Training long-stay P10...")

long_p10 = train_quantile_model(
    X_train_long,
    y_train_long_los,
    LOWER_QUANTILE,
)

print("Long-stay P10 complete.")

print("Training long-stay P50...")

long_p50 = train_quantile_model(
    X_train_long,
    y_train_long_los,
    MEDIAN_QUANTILE,
)

print("Long-stay P50 complete.")

print("Training long-stay P90...")

long_p90 = train_quantile_model(
    X_train_long,
    y_train_long_los,
    UPPER_QUANTILE,
)

print("Long-stay P90 complete.")


# ============================================================
# 22. CREATE TWO-STAGE QUANTILE ESTIMATORS
# ============================================================

section("STEP 11 - CREATE TWO-STAGE QUANTILE ESTIMATORS")

model_p10 = TwoStageQuantileModel(
    classifier=classifier,
    normal_model=normal_p10,
    long_model=long_p10,
)

model_p50 = TwoStageQuantileModel(
    classifier=classifier,
    normal_model=normal_p50,
    long_model=long_p50,
)

model_p90 = TwoStageQuantileModel(
    classifier=classifier,
    normal_model=normal_p90,
    long_model=long_p90,
)

print(
    "Created P10 / P50 / P90 two-stage estimators."
)


# ============================================================
# 23. RAW CALIBRATION PREDICTIONS
# ============================================================

section("STEP 12 - RAW CALIBRATION QUANTILE PREDICTIONS")

cal_p10 = model_p10.predict(
    X_calibration
)

cal_p50 = model_p50.predict(
    X_calibration
)

cal_p90 = model_p90.predict(
    X_calibration
)

# Force non-negative LOS.
cal_p10 = np.maximum(cal_p10, 0.0)
cal_p50 = np.maximum(cal_p50, 0.0)
cal_p90 = np.maximum(cal_p90, 0.0)

# Quantile ordering correction.
cal_lower = np.minimum(
    cal_p10,
    np.minimum(cal_p50, cal_p90)
)

cal_upper = np.maximum(
    cal_p90,
    np.maximum(cal_p50, cal_p10)
)

cal_median = np.clip(
    cal_p50,
    cal_lower,
    cal_upper,
)

cal_ordering = (
    (cal_p10 <= cal_p50)
    & (cal_p50 <= cal_p90)
)

print(
    "Raw calibration ordering:",
    f"{100 * cal_ordering.mean():.3f}%"
)

print("Corrected calibration ordering: 100%")


# ============================================================
# 24. RAW TEST PREDICTIONS
# ============================================================

section("STEP 13 - RAW TEST QUANTILE PREDICTIONS")

test_p10 = model_p10.predict(
    X_test
)

test_p50 = model_p50.predict(
    X_test
)

test_p90 = model_p90.predict(
    X_test
)

test_p10 = np.maximum(
    test_p10,
    0.0,
)

test_p50 = np.maximum(
    test_p50,
    0.0,
)

test_p90 = np.maximum(
    test_p90,
    0.0,
)

test_ordering = (
    (test_p10 <= test_p50)
    & (test_p50 <= test_p90)
)

print(
    "Raw test ordering:",
    f"{100 * test_ordering.mean():.3f}%"
)


# ============================================================
# 25. QUANTILE CROSSING CORRECTION
# ============================================================

section("STEP 14 - QUANTILE ORDERING CORRECTION")

test_lower = np.minimum(
    test_p10,
    np.minimum(test_p50, test_p90)
)

test_upper = np.maximum(
    test_p90,
    np.maximum(test_p50, test_p10)
)

test_median = np.clip(
    test_p50,
    test_lower,
    test_upper,
)

print(
    "Corrected test ordering: 100%"
)


# ============================================================
# 26. RAW POINT-PREDICTION METRICS
# ============================================================

section("STEP 15 - RAW MODEL #9 TEST METRICS")

raw_mae = mean_absolute_error(
    y_test,
    test_median,
)

raw_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_median,
    )
)

raw_r2 = r2_score(
    y_test,
    test_median,
)

print(
    f"P50 MAE:  {raw_mae:.3f} h"
)

print(
    f"P50 RMSE: {raw_rmse:.3f} h"
)

print(
    f"R²:       {raw_r2:.4f}"
)


# ============================================================
# 27. LONG-STAY PERFORMANCE BEFORE CONFORMALIZATION
# ============================================================

section("STEP 16 - LONG-STAY PERFORMANCE")

tail_mask = (
    y_test > LONG_STAY_THRESHOLD
)

tail_actual = y_test[tail_mask]

tail_predicted = test_median[tail_mask]

tail_mae = mean_absolute_error(
    tail_actual,
    tail_predicted,
)

tail_rmse = np.sqrt(
    mean_squared_error(
        tail_actual,
        tail_predicted,
    )
)

tail_bias = np.mean(
    tail_predicted - tail_actual
)

print(
    f">168h cases: {int(tail_mask.sum())}"
)

print(
    f"Actual mean: {tail_actual.mean():.3f} h"
)

print(
    f"Predicted mean: {tail_predicted.mean():.3f} h"
)

print(
    f">168h MAE: {tail_mae:.3f} h"
)

print(
    f">168h RMSE: {tail_rmse:.3f} h"
)

print(
    f">168h Bias: {tail_bias:.3f} h"
)


# ============================================================
# 28. MAPIE CONFORMALIZED QUANTILE REGRESSION
# ============================================================

section("STEP 17 - MAPIE CONFORMALIZED QUANTILE REGRESSION")

print(
    "Confidence level:",
    CONFIDENCE_LEVEL
)

print(
    "\nMAPIE estimator order:"
)

print("  1. P10 -> lower")
print("  2. P90 -> upper")
print("  3. P50 -> median")

mapie_cqr = ConformalizedQuantileRegressor(
    estimator=[
        model_p10,
        model_p90,
        model_p50,
    ],
    confidence_level=CONFIDENCE_LEVEL,
    prefit=True,
)

print(
    "\nCalibrating MAPIE on 15,000 calibration records..."
)

mapie_cqr.conformalize(
    X_calibration,
    y_calibration,
)

print("MAPIE calibration completed.")


# ============================================================
# 29. FINAL CONFORMAL TEST PREDICTIONS
# ============================================================

section("STEP 18 - FINAL CONFORMAL TEST PREDICTIONS")

mapie_point, mapie_interval = (
    mapie_cqr.predict_interval(
        X_test
    )
)

final_point = np.asarray(
    mapie_point
).reshape(-1)

final_lower = np.asarray(
    mapie_interval[:, 0, 0]
).reshape(-1)

final_upper = np.asarray(
    mapie_interval[:, 1, 0]
).reshape(-1)


# Safety correction for numerical / interval ordering issues.
original_lower = final_lower.copy()
original_upper = final_upper.copy()

final_lower = np.minimum(
    original_lower,
    original_upper,
)

final_upper = np.maximum(
    original_lower,
    original_upper,
)

final_lower = np.maximum(
    final_lower,
    0.0,
)

final_upper = np.maximum(
    final_upper,
    final_lower,
)


# ============================================================
# 30. FINAL METRICS
# ============================================================

section("STEP 19 - FINAL MODEL #9 RESULTS")

final_mae = mean_absolute_error(
    y_test,
    final_point,
)

final_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        final_point,
    )
)

final_r2 = r2_score(
    y_test,
    final_point,
)

coverage = np.mean(
    (y_test >= final_lower)
    & (y_test <= final_upper)
)

interval_width = (
    final_upper - final_lower
)

mean_width = np.mean(
    interval_width
)

median_width = np.median(
    interval_width
)

p90_width = np.percentile(
    interval_width,
    90,
)

ordering_validity = np.mean(
    final_lower <= final_upper
)

mean_error = np.mean(
    final_point - y_test
)

underprediction_rate = np.mean(
    final_point < y_test
)

print(
    f"MAE:                    {final_mae:.3f} h"
)

print(
    f"RMSE:                   {final_rmse:.3f} h"
)

print(
    f"R²:                      {final_r2:.4f}"
)

print(
    f"80% Coverage:            {100 * coverage:.3f}%"
)

print(
    f"Mean Interval Width:     {mean_width:.3f} h"
)

print(
    f"Median Interval Width:   {median_width:.3f} h"
)

print(
    f"90th Percentile Width:    {p90_width:.3f} h"
)

print(
    f"Mean Prediction Error:    {mean_error:.3f} h"
)

print(
    f"Underprediction:           {100 * underprediction_rate:.3f}%"
)

print(
    f"Interval Ordering:         {100 * ordering_validity:.3f}%"
)


# ============================================================
# 31. FINAL >168H PERFORMANCE
# ============================================================

section("STEP 20 - FINAL >168H PERFORMANCE")

final_tail_pred = final_point[
    tail_mask
]

final_tail_lower = final_lower[
    tail_mask
]

final_tail_upper = final_upper[
    tail_mask
]

final_tail_mae = mean_absolute_error(
    tail_actual,
    final_tail_pred,
)

final_tail_rmse = np.sqrt(
    mean_squared_error(
        tail_actual,
        final_tail_pred,
    )
)

final_tail_bias = np.mean(
    final_tail_pred - tail_actual
)

tail_coverage = np.mean(
    (tail_actual >= final_tail_lower)
    & (tail_actual <= final_tail_upper)
)

print(
    f">168h cases:        {int(tail_mask.sum())}"
)

print(
    f"Actual mean:        {tail_actual.mean():.3f} h"
)

print(
    f"Predicted mean:     {final_tail_pred.mean():.3f} h"
)

print(
    f">168h MAE:          {final_tail_mae:.3f} h"
)

print(
    f">168h RMSE:         {final_tail_rmse:.3f} h"
)

print(
    f">168h Bias:         {final_tail_bias:.3f} h"
)

print(
    f">168h Coverage:     {100 * tail_coverage:.3f}%"
)


# ============================================================
# 32. PERFORMANCE BY LOS RANGE
# ============================================================

section("STEP 21 - PERFORMANCE BY ACTUAL LOS RANGE")

def evaluate_range(
    name,
    mask,
):

    if mask.sum() == 0:
        return {
            "LOS_Range": name,
            "Count": 0,
        }

    actual = y_test[mask]
    predicted = final_point[mask]

    lower = final_lower[mask]
    upper = final_upper[mask]

    return {
        "LOS_Range": name,
        "Count": int(mask.sum()),
        "Actual_Mean_hours": float(
            actual.mean()
        ),
        "Predicted_Mean_hours": float(
            predicted.mean()
        ),
        "MAE_hours": float(
            mean_absolute_error(
                actual,
                predicted,
            )
        ),
        "RMSE_hours": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        ),
        "Bias_hours": float(
            np.mean(
                predicted - actual
            )
        ),
        "Coverage_percent": float(
            100
            * np.mean(
                (actual >= lower)
                & (actual <= upper)
            )
        ),
        "Mean_Interval_Width_hours": float(
            np.mean(
                upper - lower
            )
        ),
    }


range_results = [

    evaluate_range(
        "0-24h",
        y_test <= 24,
    ),

    evaluate_range(
        "24-72h",
        (y_test > 24)
        & (y_test <= 72),
    ),

    evaluate_range(
        "72-168h",
        (y_test > 72)
        & (y_test <= 168),
    ),

    evaluate_range(
        ">168h",
        y_test > 168,
    ),
]

range_df = pd.DataFrame(
    range_results
)

print(
    range_df.to_string(
        index=False
    )
)


# ============================================================
# 33. PERFORMANCE BY TRIAGE ACUITY
# ============================================================

section("STEP 22 - PERFORMANCE BY TRIAGE ACUITY")

triage_results = []

for acuity in sorted(
    test_df["triage_acuity"].unique()
):

    mask = (
        test_df["triage_acuity"].values
        == acuity
    )

    actual = y_test[mask]
    predicted = final_point[mask]

    lower = final_lower[mask]
    upper = final_upper[mask]

    triage_results.append(
        {
            "Triage_Acuity": int(acuity),
            "N": int(mask.sum()),
            "Actual_Mean_hours": float(
                actual.mean()
            ),
            "Predicted_Mean_hours": float(
                predicted.mean()
            ),
            "MAE_hours": float(
                mean_absolute_error(
                    actual,
                    predicted,
                )
            ),
            "Bias_hours": float(
                np.mean(
                    predicted - actual
                )
            ),
            "Coverage_percent": float(
                100
                * np.mean(
                    (actual >= lower)
                    & (actual <= upper)
                )
            ),
        }
    )


triage_df = pd.DataFrame(
    triage_results
)

print(
    triage_df.to_string(
        index=False
    )
)


# ============================================================
# 34. EXTREME UNDERPREDICTION CHECK
# ============================================================

section("STEP 23 - EXTREME UNDERPREDICTION CHECK")

extreme_mask = (
    y_test > 168
)

extreme_actual = y_test[
    extreme_mask
]

extreme_pred = final_point[
    extreme_mask
]

at_least_2x_low = (
    extreme_pred
    < extreme_actual / 2
)

print(
    ">168h cases:",
    int(extreme_mask.sum())
)

print(
    "Predicted at least 2x too low:",
    int(at_least_2x_low.sum()),
    f"({100 * at_least_2x_low.mean():.3f}%)"
)

print(
    "Actual mean:",
    round(
        extreme_actual.mean(),
        3,
    ),
)

print(
    "Predicted mean:",
    round(
        extreme_pred.mean(),
        3,
    ),
)


# ============================================================
# 35. SAVE TEST PREDICTIONS
# ============================================================

section("STEP 24 - SAVE TEST PREDICTIONS")

predictions = pd.DataFrame(
    {
        "stay_id": test_df["stay_id"].values,
        "patient_id": test_df["patient_id"].values,
        "triage_acuity": test_df[
            "triage_acuity"
        ].values,
        "actual_los_hours": y_test,
        "long_stay_probability": (
            test_long_probability
        ),
        "predicted_los_hours": final_point,
        "los_lower_80": final_lower,
        "los_upper_80": final_upper,
        "prediction_error_hours": (
            final_point - y_test
        ),
    }
)

predictions_path = (
    OUTPUT_DIR
    / "los_predictions_model9_test.csv"
)

predictions.to_csv(
    predictions_path,
    index=False,
)

print(
    "Saved:",
    predictions_path
)


# ============================================================
# 36. SAVE LOS RANGE RESULTS
# ============================================================

range_path = (
    OUTPUT_DIR
    / "los_model9_by_los_range.csv"
)

range_df.to_csv(
    range_path,
    index=False,
)

print(
    "Saved:",
    range_path
)


# ============================================================
# 37. SAVE TRIAGE RESULTS
# ============================================================

triage_path = (
    OUTPUT_DIR
    / "los_model9_by_triage_acuity.csv"
)

triage_df.to_csv(
    triage_path,
    index=False,
)

print(
    "Saved:",
    triage_path
)


# ============================================================
# 38. SAVE MODELS
# ============================================================

section("STEP 25 - SAVE MODEL ARTIFACTS")

joblib.dump(
    classifier,
    MODEL_DIR
    / "los_model9_long_stay_classifier.joblib",
)

joblib.dump(
    normal_p10,
    MODEL_DIR
    / "los_model9_normal_p10.joblib",
)

joblib.dump(
    normal_p50,
    MODEL_DIR
    / "los_model9_normal_p50.joblib",
)

joblib.dump(
    normal_p90,
    MODEL_DIR
    / "los_model9_normal_p90.joblib",
)

joblib.dump(
    long_p10,
    MODEL_DIR
    / "los_model9_long_p10.joblib",
)

joblib.dump(
    long_p50,
    MODEL_DIR
    / "los_model9_long_p50.joblib",
)

joblib.dump(
    long_p90,
    MODEL_DIR
    / "los_model9_long_p90.joblib",
)

joblib.dump(
    preprocessor,
    MODEL_DIR
    / "los_model9_preprocessor.joblib",
)

joblib.dump(
    model_p10,
    MODEL_DIR
    / "los_model9_p10.joblib",
)

joblib.dump(
    model_p50,
    MODEL_DIR
    / "los_model9_p50.joblib",
)

joblib.dump(
    model_p90,
    MODEL_DIR
    / "los_model9_p90.joblib",
)

joblib.dump(
    mapie_cqr,
    MODEL_DIR
    / "los_model9_mapie_cqr.joblib",
)

print(
    "All Model #9 artifacts saved."
)


# ============================================================
# 39. SAVE FINAL REPORT
# ============================================================

section("STEP 26 - SAVE FINAL REPORT")

report = {

    "model": "LOS Prediction - Model #9",

    "architecture":
        "Tail-Aware Two-Stage LightGBM Quantile Regression + MAPIE CQR",

    "target":
        TARGET,

    "long_stay_threshold_hours":
        LONG_STAY_THRESHOLD,

    "confidence_level":
        CONFIDENCE_LEVEL,

    "quantiles": {
        "lower": LOWER_QUANTILE,
        "median": MEDIAN_QUANTILE,
        "upper": UPPER_QUANTILE,
    },

    "dataset_records":
        int(len(df)),

    "training_records":
        int(len(train_df)),

    "calibration_records":
        int(len(calibration_df)),

    "test_records":
        int(len(test_df)),

    "features":
        FEATURES,

    "chronological_split":
        "70/15/15",

    "long_stay_classifier": {

        "training_long_stays":
            int(y_train_long.sum()),

        "test_long_stays":
            int(y_test_long.sum()),

        "test_roc_auc":
            None
            if test_auc is None
            else float(test_auc),
    },

    "final_test_metrics": {

        "MAE_hours":
            float(final_mae),

        "RMSE_hours":
            float(final_rmse),

        "R2":
            float(final_r2),

        "Coverage_percent":
            float(100 * coverage),

        "Mean_Interval_Width_hours":
            float(mean_width),

        "Median_Interval_Width_hours":
            float(median_width),

        "P90_Interval_Width_hours":
            float(p90_width),

        "Mean_Prediction_Error_hours":
            float(mean_error),

        "Underprediction_percent":
            float(
                100
                * underprediction_rate
            ),

        "Interval_Ordering_Validity_percent":
            float(
                100
                * ordering_validity
            ),
    },

    "over_168h_metrics": {

        "count":
            int(tail_mask.sum()),

        "actual_mean_hours":
            float(tail_actual.mean()),

        "predicted_mean_hours":
            float(final_tail_pred.mean()),

        "MAE_hours":
            float(final_tail_mae),

        "RMSE_hours":
            float(final_tail_rmse),

        "bias_hours":
            float(final_tail_bias),

        "coverage_percent":
            float(100 * tail_coverage),
    },
}


report_path = (
    OUTPUT_DIR
    / "los_model9_final_report.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        report,
        file,
        indent=4,
    )


# ============================================================
# 40. SAVE CONFIGURATION
# ============================================================

configuration = {

    "model":
        "Tail-Aware Two-Stage LOS Prediction",

    "algorithm":
        "LightGBM Quantile Regression",

    "conformal_method":
        "MAPIE Conformalized Quantile Regression",

    "confidence_level":
        CONFIDENCE_LEVEL,

    "long_stay_threshold_hours":
        LONG_STAY_THRESHOLD,

    "features":
        FEATURES,

    "numeric_features":
        NUMERIC_FEATURES,

    "categorical_features":
        CATEGORICAL_FEATURES,

    "target":
        TARGET,

    "split":
        "Chronological 70/15/15",

    "training_records":
        TRAIN_SIZE,

    "calibration_records":
        CALIBRATION_SIZE,

    "test_records":
        TEST_SIZE,

    "lightgbm_parameters":
        BASE_LGB_PARAMS,
}


configuration_path = (
    OUTPUT_DIR
    / "los_model9_configuration.json"
)

with open(
    configuration_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        configuration,
        file,
        indent=4,
    )


# ============================================================
# 41. FINAL SUMMARY
# ============================================================

section("MODEL #9 COMPLETE")

print("\nFINAL MODEL #9 RESULTS")

print(
    f"MAE:                  {final_mae:.3f} h"
)

print(
    f"RMSE:                 {final_rmse:.3f} h"
)

print(
    f"R²:                   {final_r2:.4f}"
)

print(
    f"80% Coverage:          {100 * coverage:.3f}%"
)

print(
    f"Mean Interval Width:   {mean_width:.3f} h"
)

print(
    f">168h MAE:             {final_tail_mae:.3f} h"
)

print(
    f">168h Coverage:        {100 * tail_coverage:.3f}%"
)

print(
    f"Interval Ordering:     {100 * ordering_validity:.3f}%"
)

print("\nSaved report:")
print(report_path)

print("\nSaved configuration:")
print(configuration_path)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\n" + "=" * 75)
print("MODEL #9 TRAINING FINISHED SUCCESSFULLY")
print("=" * 75)