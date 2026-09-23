# ============================================================
# Predictive Resource Optimization System for Healthcare
# Model 1: Length of Stay (LOS) Prediction
#
# Architecture:
#
# Patient-level features
#          ↓
# LightGBM Quantile Regression
#          ↓
# P10 / P50 / P90
#          ↓
# MAPIE Conformalized Quantile Regression
#          ↓
# Calibrated 80% LOS Prediction Interval
#
# Dataset:
#   synthetic_patient_stays_100k.csv
#
# Target:
#   los_hours
#
# Confidence level:
#   80%
#
# Quantiles:
#   P10 = 0.10
#   P50 = 0.50
#   P90 = 0.90
#
# Split:
#   70% Training
#   15% Conformal Calibration
#   15% Final Test
#
# IMPORTANT:
#   This script predicts LOS using information available at
#   the initial patient assessment. Variables representing
#   downstream outcomes or later events are excluded to
#   prevent target leakage.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from mapie.regression import ConformalizedQuantileRegressor
from mapie.metrics.regression import (
    regression_coverage_score,
    regression_mean_width_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. VERSION CHECK
# ============================================================

print("=" * 80)
print("PREDICTIVE RESOURCE OPTIMIZATION SYSTEM FOR HEALTHCARE")
print("MODEL 1 — LENGTH OF STAY (LOS) PREDICTION")
print("=" * 80)

print("\nChecking installed packages...")

import pandas as pd
import sklearn
import lightgbm
import mapie

print(f"pandas        : {pd.__version__}")
print(f"scikit-learn  : {sklearn.__version__}")
print(f"LightGBM      : {lightgbm.__version__}")
print(f"MAPIE         : {mapie.__version__}")

print("✓ Required packages imported successfully")


# ============================================================
# 3. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# ------------------------------------------------------------
# Prediction interval
# ------------------------------------------------------------

CONFIDENCE_LEVEL = 0.80

LOWER_QUANTILE = (1.0 - CONFIDENCE_LEVEL) / 2.0
MEDIAN_QUANTILE = 0.50
UPPER_QUANTILE = (1.0 + CONFIDENCE_LEVEL) / 2.0

# Expected:
# LOWER_QUANTILE = 0.10
# MEDIAN_QUANTILE = 0.50
# UPPER_QUANTILE = 0.90


# ------------------------------------------------------------
# Required project dataset size
# ------------------------------------------------------------

MIN_RECORDS = 50_000
MAX_RECORDS = 150_000


# ------------------------------------------------------------
# Maximum acceptable rounding difference when checking LOS
# against arrival/discharge timestamps.
# ------------------------------------------------------------

MAX_ALLOWED_ROUNDING_ERROR = 0.02


# ============================================================
# 4. PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "synthetic_patient_stays_100k.csv"


# ============================================================
# 5. REQUIRED DATASET COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
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
    "triage_start_time",
    "bed_assigned_time",
    "discharge_time",
    "los_hours",
    "disposition",
    "icu_transfer_flag",
    "initial_care_unit",
    "requires_ventilation",
]


# ============================================================
# 6. LOS MODEL FEATURES
# ============================================================
#
# The project document defines patient-level clinical features
# for LOS/routing prediction.
#
# For this LOS model, we define the prediction point as the
# initial patient assessment.
#
# Therefore, we use:
#
#   Demographics
#   Initial clinical condition
#   Triage severity
#   Chief complaint
#
# We intentionally exclude downstream variables such as:
#
#   bed_assigned_time
#   discharge_time
#   disposition
#   icu_transfer_flag
#   initial_care_unit
#   requires_ventilation
#   los_hours
#
# because they can contain information that becomes available
# after the prediction point or is directly related to the
# eventual outcome.
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
]


CATEGORICAL_FEATURES = [
    "gender",
    "chief_complaint",
]


FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


TARGET_COLUMN = "los_hours"


# ============================================================
# 7. LIGHTGBM PARAMETERS
# ============================================================
#
# Quantile regression is implemented using:
#
#     objective = "quantile"
#
# The alpha value changes for P10, P50 and P90.
# ============================================================

BASE_LIGHTGBM_PARAMETERS = {
    "objective": "quantile",

    "n_estimators": 500,

    "learning_rate": 0.05,

    "num_leaves": 31,

    "max_depth": -1,

    "min_child_samples": 30,

    "subsample": 0.9,

    "subsample_freq": 1,

    "colsample_bytree": 0.9,

    "reg_alpha": 0.0,

    "reg_lambda": 0.1,

    "random_state": RANDOM_STATE,

    "n_jobs": -1,

    "verbosity": -1,
}


# ============================================================
# 8. HELPER FUNCTIONS
# ============================================================


def create_preprocessor():
    """
    Create a fresh preprocessing object.

    A separate preprocessor is created for each model so that
    P10, P50 and P90 models do not share mutable fitted state.
    """

    return ColumnTransformer(
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
                    sparse_output=True,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def create_quantile_model(alpha):
    """
    Create a preprocessing + LightGBM quantile regression
    pipeline for a specific quantile.
    """

    model_parameters = BASE_LIGHTGBM_PARAMETERS.copy()

    model_parameters["alpha"] = alpha

    estimator = LGBMRegressor(
        **model_parameters
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                estimator,
            ),
        ]
    )

    return pipeline


def pinball_loss(y_true, y_pred, alpha):
    """
    Calculate pinball loss for a quantile model.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    errors = y_true - y_pred

    return float(
        np.mean(
            np.where(
                errors >= 0,
                alpha * errors,
                (alpha - 1.0) * errors,
            )
        )
    )


def validate_prediction_order(
    lower,
    median,
    upper,
    name="prediction",
):
    """
    Check that:
        lower <= median <= upper

    Returns the percentage of records satisfying the condition.
    """

    valid = (
        (lower <= median)
        &
        (median <= upper)
    )

    percentage = float(
        np.mean(valid)
    )

    print(
        f"{name} ordering validity: "
        f"{percentage:.3%}"
    )

    return percentage


def safe_non_negative(array):
    """
    LOS cannot be negative.
    """

    return np.maximum(
        np.asarray(array, dtype=float),
        0.0,
    )


# ============================================================
# 9. CHECK DATASET PATH
# ============================================================

print("\n" + "-" * 80)
print("[1/12] Checking dataset")
print("-" * 80)

print(f"Expected CSV:\n{DATA_PATH}")

if not DATA_PATH.exists():

    raise FileNotFoundError(
        "\nCSV file not found.\n\n"
        f"Expected location:\n{DATA_PATH}\n\n"
        "Make sure the file is named:\n"
        "synthetic_patient_stays_100k.csv\n\n"
        "and is inside:\n"
        f"{DATA_DIR}"
    )


# ============================================================
# 10. LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

print("\nDataset loaded successfully.")

print(
    f"Rows    : {len(df):,}"
)

print(
    f"Columns : {len(df.columns)}"
)


# ============================================================
# 11. DATASET VALIDATION
# ============================================================

print("\n" + "-" * 80)
print("[2/12] Validating dataset structure")
print("-" * 80)


# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_columns
        )
    )

print("✓ All required columns are present")


# ------------------------------------------------------------
# Dataset size
# ------------------------------------------------------------

if not (
    MIN_RECORDS
    <= len(df)
    <= MAX_RECORDS
):

    raise ValueError(
        f"Dataset contains {len(df):,} records.\n"
        f"Project requirement: "
        f"{MIN_RECORDS:,}–{MAX_RECORDS:,} records."
    )

print(
    f"✓ Dataset size is valid "
    f"({len(df):,} records)"
)


# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

missing_values = (
    df[REQUIRED_COLUMNS]
    .isna()
    .sum()
)

missing_values = (
    missing_values[
        missing_values > 0
    ]
)

if len(missing_values) > 0:

    print(
        "\nMissing values found:"
    )

    print(
        missing_values
    )

    raise ValueError(
        "Missing values detected. "
        "Do not train the model until the "
        "source dataset is corrected."
    )

print("✓ No missing values")


# ------------------------------------------------------------
# Duplicate IDs
# ------------------------------------------------------------

duplicate_stays = (
    df["stay_id"]
    .duplicated()
    .sum()
)

duplicate_patients = (
    df["patient_id"]
    .duplicated()
    .sum()
)

if duplicate_stays > 0:

    raise ValueError(
        f"{duplicate_stays:,} duplicate stay_id values detected."
    )

if duplicate_patients > 0:

    raise ValueError(
        f"{duplicate_patients:,} duplicate patient_id values detected."
    )

print("✓ No duplicate stay_id values")
print("✓ No duplicate patient_id values")


# ============================================================
# 12. TARGET VALIDATION
# ============================================================

print("\n" + "-" * 80)
print("[3/12] Validating LOS target")
print("-" * 80)


if not pd.api.types.is_numeric_dtype(
    df[TARGET_COLUMN]
):

    raise ValueError(
        "los_hours must be numeric."
    )


if not np.isfinite(
    df[TARGET_COLUMN]
).all():

    raise ValueError(
        "los_hours contains NaN or infinite values."
    )


if (
    df[TARGET_COLUMN] <= 0
).any():

    invalid_count = int(
        (
            df[TARGET_COLUMN] <= 0
        ).sum()
    )

    raise ValueError(
        f"{invalid_count:,} LOS values are "
        "zero or negative."
    )


print("✓ LOS target is numeric")
print("✓ LOS target contains only finite values")
print("✓ LOS target contains only positive values")


# ------------------------------------------------------------
# LOS distribution
# ------------------------------------------------------------

print("\nLOS distribution:")

print(
    f"  Mean   : {df[TARGET_COLUMN].mean():.3f} hours"
)

print(
    f"  Median : {df[TARGET_COLUMN].median():.3f} hours"
)

print(
    f"  90th   : "
    f"{df[TARGET_COLUMN].quantile(0.90):.3f} hours"
)

print(
    f"  95th   : "
    f"{df[TARGET_COLUMN].quantile(0.95):.3f} hours"
)

print(
    f"  99th   : "
    f"{df[TARGET_COLUMN].quantile(0.99):.3f} hours"
)

print(
    f"  Max    : {df[TARGET_COLUMN].max():.3f} hours"
)


# ============================================================
# 13. TRIAGE AND PHYSIOLOGICAL VALIDATION
# ============================================================

print("\n" + "-" * 80)
print("[4/12] Validating clinical constraints")
print("-" * 80)


# ------------------------------------------------------------
# Triage acuity
# ------------------------------------------------------------

valid_triage_values = {
    1,
    2,
    3,
    4,
    5,
}

actual_triage_values = set(
    df["triage_acuity"]
    .astype(int)
    .unique()
)

if not actual_triage_values.issubset(
    valid_triage_values
):

    raise ValueError(
        "Unexpected triage_acuity values: "
        f"{actual_triage_values}"
    )

print(
    "✓ Triage acuity values are valid "
    "(ESI 1–5)"
)


# ------------------------------------------------------------
# Blood pressure
# ------------------------------------------------------------

invalid_bp = int(
    (
        df["sbp"]
        <=
        df["dbp"]
    ).sum()
)

if invalid_bp > 0:

    raise ValueError(
        f"{invalid_bp:,} records have "
        "SBP <= DBP."
    )

print("✓ SBP > DBP for all records")


# ============================================================
# 14. TIMESTAMP VALIDATION
# ============================================================

print("\n" + "-" * 80)
print("[5/12] Validating timestamps")
print("-" * 80)


TIMESTAMP_COLUMNS = [
    "arrival_time",
    "triage_start_time",
    "bed_assigned_time",
    "discharge_time",
]


for column in TIMESTAMP_COLUMNS:

    df[column] = pd.to_datetime(
        df[column],
        errors="raise",
    )


# ------------------------------------------------------------
# Chronological sequence
# ------------------------------------------------------------

invalid_triage = int(
    (
        df["triage_start_time"]
        <
        df["arrival_time"]
    ).sum()
)


invalid_bed = int(
    (
        df["bed_assigned_time"]
        <
        df["triage_start_time"]
    ).sum()
)


invalid_discharge = int(
    (
        df["discharge_time"]
        <
        df["bed_assigned_time"]
    ).sum()
)


if invalid_triage > 0:

    raise ValueError(
        "Some triage_start_time values "
        "occur before arrival_time."
    )


if invalid_bed > 0:

    raise ValueError(
        "Some bed_assigned_time values "
        "occur before triage_start_time."
    )


if invalid_discharge > 0:

    raise ValueError(
        "Some discharge_time values "
        "occur before bed_assigned_time."
    )


print("✓ Timestamp chronology is valid")


# ============================================================
# 15. VERIFY LOS AGAINST TIMESTAMPS
# ============================================================

calculated_los = (
    (
        df["discharge_time"]
        -
        df["arrival_time"]
    )
    .dt.total_seconds()
    / 3600.0
)


los_difference = (
    calculated_los
    -
    df["los_hours"]
).abs()


bad_los_count = int(
    (
        los_difference
        >
        MAX_ALLOWED_ROUNDING_ERROR
    ).sum()
)


if bad_los_count > 0:

    raise ValueError(
        f"{bad_los_count:,} LOS records do not "
        "match arrival_time → discharge_time."
    )


print(
    "✓ LOS matches arrival_time → discharge_time "
    "within rounding tolerance"
)


# ============================================================
# 16. SORT CHRONOLOGICALLY
# ============================================================

print("\n" + "-" * 80)
print("[6/12] Sorting data chronologically")
print("-" * 80)


df = (
    df
    .sort_values(
        by="arrival_time"
    )
    .reset_index(
        drop=True
    )
)


print(
    "Earliest arrival:"
)

print(
    df["arrival_time"].min()
)

print(
    "Latest arrival:"
)

print(
    df["arrival_time"].max()
)


# ============================================================
# 17. CHRONOLOGICAL TRAIN / CALIBRATION / TEST SPLIT
# ============================================================
#
# 70% Training
# 15% Conformal calibration
# 15% Final test
#
# No random shuffling.
#
# This is intentional because the synthetic dataset contains
# a chronological hospital simulation.
# ============================================================

print("\n" + "-" * 80)
print("[7/12] Creating chronological data splits")
print("-" * 80)


n_records = len(df)

train_end = int(
    n_records * 0.70
)

calibration_end = int(
    n_records * 0.85
)


train_df = (
    df
    .iloc[
        :train_end
    ]
    .copy()
)


calibration_df = (
    df
    .iloc[
        train_end:
        calibration_end
    ]
    .copy()
)


test_df = (
    df
    .iloc[
        calibration_end:
    ]
    .copy()
)


print(
    f"Training records    : "
    f"{len(train_df):,}"
)

print(
    f"Calibration records : "
    f"{len(calibration_df):,}"
)

print(
    f"Test records        : "
    f"{len(test_df):,}"
)


print("\nTraining period:")
print(
    train_df["arrival_time"].min(),
    "→",
    train_df["arrival_time"].max(),
)


print("\nCalibration period:")
print(
    calibration_df["arrival_time"].min(),
    "→",
    calibration_df["arrival_time"].max(),
)


print("\nTest period:")
print(
    test_df["arrival_time"].min(),
    "→",
    test_df["arrival_time"].max(),
)


# ============================================================
# 18. PREPARE X AND y
# ============================================================

print("\n" + "-" * 80)
print("[8/12] Preparing model features")
print("-" * 80)


X_train = (
    train_df[
        FEATURE_COLUMNS
    ]
    .copy()
)

y_train = (
    train_df[
        TARGET_COLUMN
    ]
    .copy()
)


X_calibration = (
    calibration_df[
        FEATURE_COLUMNS
    ]
    .copy()
)

y_calibration = (
    calibration_df[
        TARGET_COLUMN
    ]
    .copy()
)


X_test = (
    test_df[
        FEATURE_COLUMNS
    ]
    .copy()
)

y_test = (
    test_df[
        TARGET_COLUMN
    ]
    .copy()
)


print("\nFeatures used by LOS model:")

for feature in FEATURE_COLUMNS:

    print(
        f"  ✓ {feature}"
    )


print(
    f"\nTarget: {TARGET_COLUMN}"
)


# ============================================================
# 19. TRAIN P10 LIGHTGBM MODEL
# ============================================================

print("\n" + "-" * 80)
print("[9/12] Training LightGBM quantile models")
print("-" * 80)


print(
    "\nTraining P10 model "
    "(alpha = 0.10)..."
)


model_p10 = create_quantile_model(
    LOWER_QUANTILE
)


model_p10.fit(
    X_train,
    y_train,
)


print(
    "✓ P10 model trained"
)


# ============================================================
# 20. TRAIN P50 LIGHTGBM MODEL
# ============================================================

print(
    "\nTraining P50 model "
    "(alpha = 0.50)..."
)


model_p50 = create_quantile_model(
    MEDIAN_QUANTILE
)


model_p50.fit(
    X_train,
    y_train,
)


print(
    "✓ P50 model trained"
)


# ============================================================
# 21. TRAIN P90 LIGHTGBM MODEL
# ============================================================

print(
    "\nTraining P90 model "
    "(alpha = 0.90)..."
)


model_p90 = create_quantile_model(
    UPPER_QUANTILE
)


model_p90.fit(
    X_train,
    y_train,
)


print(
    "✓ P90 model trained"
)


# ============================================================
# 22. BASE QUANTILE PREDICTIONS
# ============================================================

print("\n" + "-" * 80)
print("[10/12] Evaluating base LightGBM quantile predictions")
print("-" * 80)


p10_test = safe_non_negative(
    model_p10.predict(
        X_test
    )
)


p50_test = safe_non_negative(
    model_p50.predict(
        X_test
    )
)


p90_test = safe_non_negative(
    model_p90.predict(
        X_test
    )
)


# ------------------------------------------------------------
# Quantile ordering diagnostic
# ------------------------------------------------------------

raw_ordering_validity = validate_prediction_order(
    p10_test,
    p50_test,
    p90_test,
    name="Raw P10/P50/P90",
)


# ------------------------------------------------------------
# Base metrics
# ------------------------------------------------------------

base_mae = mean_absolute_error(
    y_test,
    p50_test,
)


base_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        p50_test,
    )
)


base_pinball_p10 = pinball_loss(
    y_test,
    p10_test,
    LOWER_QUANTILE,
)


base_pinball_p90 = pinball_loss(
    y_test,
    p90_test,
    UPPER_QUANTILE,
)


# For the raw interval, use the lower/upper predictions
# as produced by LightGBM.
#
# If occasional quantile crossing exists, we report it rather
# than silently modifying the raw model outputs.


base_interval_coverage = float(
    np.mean(
        (
            y_test.to_numpy()
            >=
            p10_test
        )
        &
        (
            y_test.to_numpy()
            <=
            p90_test
        )
    )
)


base_interval_width = float(
    np.mean(
        p90_test
        -
        p10_test
    )
)


print("\nBase LightGBM results:")

print(
    f"  P50 MAE          : "
    f"{base_mae:.3f} hours"
)

print(
    f"  P50 RMSE         : "
    f"{base_rmse:.3f} hours"
)

print(
    f"  P10 Pinball Loss : "
    f"{base_pinball_p10:.3f}"
)

print(
    f"  P90 Pinball Loss : "
    f"{base_pinball_p90:.3f}"
)

print(
    f"  P10-P90 Coverage : "
    f"{base_interval_coverage:.3%}"
)

print(
    f"  Mean Width       : "
    f"{base_interval_width:.3f} hours"
)


# ============================================================
# 23. MAPIE CONFORMALIZED QUANTILE REGRESSION
# ============================================================
#
# MAPIE CQR with prefit=True requires three already-fitted
# quantile regressors in this order:
#
#   1. Lower quantile
#   2. Upper quantile
#   3. Median quantile
#
# This is exactly:
#
#   P10
#   P90
#   P50
#
# Calibration is performed ONLY on the calibration dataset.
# The test set remains untouched until final evaluation.
# ============================================================

print("\n" + "-" * 80)
print("[11/12] Applying MAPIE Conformalized Quantile Regression")
print("-" * 80)


print(
    "\nCreating MAPIE CQR object..."
)


mapie_cqr = (
    ConformalizedQuantileRegressor(
        estimator=[
            model_p10,
            model_p90,
            model_p50,
        ],
        confidence_level=CONFIDENCE_LEVEL,
        prefit=True,
    )
)


print(
    "✓ MAPIE CQR object created"
)


# ------------------------------------------------------------
# Conformal calibration
# ------------------------------------------------------------

print(
    "\nCalibrating using dedicated "
    "15% calibration dataset..."
)


mapie_cqr.conformalize(
    X_calibration,
    y_calibration,
)


print(
    "✓ Conformal calibration completed"
)


# ============================================================
# 24. FINAL TEST PREDICTIONS
# ============================================================

print(
    "\nGenerating final test prediction intervals..."
)


y_pred_mapie, y_interval_mapie = (
    mapie_cqr.predict_interval(
        X_test
    )
)


# MAPIE returns:
#
# y_pred:
#     shape = (n_samples,)
#
# y_interval:
#     shape = (n_samples, 2, 1)
#
# where:
#
#     [:, 0, 0] = lower bound
#     [:, 1, 0] = upper bound


los_predicted = safe_non_negative(
    y_pred_mapie
)


los_lower = safe_non_negative(
    y_interval_mapie[
        :,
        0,
        0,
    ]
)


los_upper = safe_non_negative(
    y_interval_mapie[
        :,
        1,
        0,
    ]
)


# ------------------------------------------------------------
# Ensure final interval contains point prediction
# ------------------------------------------------------------
#
# This is a presentation/safety ordering step.
#
# It does not change the conformal calibration process.
# ------------------------------------------------------------

los_lower = np.minimum(
    los_lower,
    los_predicted,
)


los_upper = np.maximum(
    los_upper,
    los_predicted,
)


# ============================================================
# 25. FINAL CONFORMAL METRICS
# ============================================================

print(
    "\nCalculating final conformal metrics..."
)


conformal_mae = (
    mean_absolute_error(
        y_test,
        los_predicted,
    )
)


conformal_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        los_predicted,
    )
)


# MAPIE metric functions accept intervals in the
# shape returned by predict_interval:
#
#     (n_samples, 2, 1)


conformal_coverage = (
    regression_coverage_score(
        y_test.to_numpy(),
        y_interval_mapie,
    )[0]
)


conformal_width = (
    regression_mean_width_score(
        y_interval_mapie,
    )[0]
)


# Manual coverage verification
manual_coverage = float(
    np.mean(
        (
            y_test.to_numpy()
            >=
            los_lower
        )
        &
        (
            y_test.to_numpy()
            <=
            los_upper
        )
    )
)


# ============================================================
# 26. FINAL MODEL RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FINAL LOS MODEL RESULTS")
print("=" * 80)


print(
    f"\nTarget confidence level : "
    f"{CONFIDENCE_LEVEL:.0%}"
)


print(
    f"Lower quantile          : "
    f"P{int(LOWER_QUANTILE * 100)}"
)


print(
    f"Median quantile         : "
    f"P{int(MEDIAN_QUANTILE * 100)}"
)


print(
    f"Upper quantile          : "
    f"P{int(UPPER_QUANTILE * 100)}"
)


print(
    f"\nTest P50 MAE            : "
    f"{conformal_mae:.3f} hours"
)


print(
    f"Test P50 RMSE           : "
    f"{conformal_rmse:.3f} hours"
)


print(
    f"MAPIE coverage          : "
    f"{conformal_coverage:.3%}"
)


print(
    f"Manual coverage         : "
    f"{manual_coverage:.3%}"
)


print(
    f"Mean interval width     : "
    f"{conformal_width:.3f} hours"
)


print(
    "\nExpected coverage target: "
    f"{CONFIDENCE_LEVEL:.0%}"
)


# ============================================================
# 27. CREATE FINAL TEST RESULTS
# ============================================================

print("\n" + "-" * 80)
print("Creating final LOS prediction dataset")
print("-" * 80)


results = (
    test_df[
        [
            "stay_id",
            "patient_id",
            "arrival_time",
            "triage_acuity",
            "disposition",
            "los_hours",
        ]
    ]
    .copy()
)


results = results.rename(
    columns={
        "los_hours":
            "actual_los_hours",
    }
)


results[
    "predicted_los_hours"
] = los_predicted


results[
    "los_lower_80"
] = los_lower


results[
    "los_upper_80"
] = los_upper


results[
    "interval_width_hours"
] = (
    results[
        "los_upper_80"
    ]
    -
    results[
        "los_lower_80"
    ]
)


results[
    "covered_by_80_interval"
] = (
    (
        results[
            "actual_los_hours"
        ]
        >=
        results[
            "los_lower_80"
        ]
    )
    &
    (
        results[
            "actual_los_hours"
        ]
        <=
        results[
            "los_upper_80"
        ]
    )
)


# ============================================================
# 28. SAVE TEST PREDICTIONS
# ============================================================

prediction_path = (
    OUTPUT_DIR
    /
    "los_predictions_test.csv"
)


results.to_csv(
    prediction_path,
    index=False,
)


print(
    f"✓ Saved:\n"
    f"{prediction_path}"
)


# ============================================================
# 29. EVALUATION BY TRIAGE ACUITY
# ============================================================

print("\n" + "-" * 80)
print("Evaluating LOS model by triage acuity")
print("-" * 80)


acuity_results = []


for acuity in sorted(
    results[
        "triage_acuity"
    ].unique()
):

    group = results[
        results[
            "triage_acuity"
        ]
        ==
        acuity
    ]


    actual = group[
        "actual_los_hours"
    ].to_numpy()


    predicted = group[
        "predicted_los_hours"
    ].to_numpy()


    lower = group[
        "los_lower_80"
    ].to_numpy()


    upper = group[
        "los_upper_80"
    ].to_numpy()


    group_coverage = float(
        np.mean(
            (
                actual
                >=
                lower
            )
            &
            (
                actual
                <=
                upper
            )
        )
    )


    group_width = float(
        np.mean(
            upper
            -
            lower
        )
    )


    group_mae = float(
        mean_absolute_error(
            actual,
            predicted,
        )
    )


    acuity_results.append(
        {
            "triage_acuity":
                int(acuity),

            "n_test_records":
                int(len(group)),

            "actual_median_los_hours":
                float(np.median(actual)),

            "predicted_median_los_hours":
                float(np.median(predicted)),

            "mae_hours":
                group_mae,

            "coverage_80":
                group_coverage,

            "mean_interval_width_hours":
                group_width,
        }
    )


acuity_results_df = pd.DataFrame(
    acuity_results
)


acuity_path = (
    OUTPUT_DIR
    /
    "los_model_by_triage_acuity.csv"
)


acuity_results_df.to_csv(
    acuity_path,
    index=False,
)


print(
    acuity_results_df.to_string(
        index=False
    )
)


print(
    f"\n✓ Saved:\n"
    f"{acuity_path}"
)


# ============================================================
# 30. SAVE MODEL ARTIFACTS
# ============================================================

print("\n" + "-" * 80)
print("Saving trained model artifacts")
print("-" * 80)


p10_model_path = (
    MODEL_DIR
    /
    "lightgbm_los_p10.joblib"
)


p50_model_path = (
    MODEL_DIR
    /
    "lightgbm_los_p50.joblib"
)


p90_model_path = (
    MODEL_DIR
    /
    "lightgbm_los_p90.joblib"
)


mapie_model_path = (
    MODEL_DIR
    /
    "mapie_los_cqr.joblib"
)


joblib.dump(
    model_p10,
    p10_model_path,
)


joblib.dump(
    model_p50,
    p50_model_path,
)


joblib.dump(
    model_p90,
    p90_model_path,
)


joblib.dump(
    mapie_cqr,
    mapie_model_path,
)


print(
    f"✓ {p10_model_path}"
)

print(
    f"✓ {p50_model_path}"
)

print(
    f"✓ {p90_model_path}"
)

print(
    f"✓ {mapie_model_path}"
)


# ============================================================
# 31. SAVE MODEL CONFIGURATION
# ============================================================

configuration = {

    "project":
        "Predictive Resource Optimization System for Healthcare",

    "model":
        "Length of Stay Prediction",

    "algorithm":
        "LightGBM Quantile Regression",

    "uncertainty_method":
        "MAPIE Conformalized Quantile Regression",

    "confidence_level":
        CONFIDENCE_LEVEL,

    "lower_quantile":
        LOWER_QUANTILE,

    "median_quantile":
        MEDIAN_QUANTILE,

    "upper_quantile":
        UPPER_QUANTILE,

    "total_records":
        int(len(df)),

    "training_records":
        int(len(train_df)),

    "calibration_records":
        int(len(calibration_df)),

    "test_records":
        int(len(test_df)),

    "features":
        FEATURE_COLUMNS,

    "numeric_features":
        NUMERIC_FEATURES,

    "categorical_features":
        CATEGORICAL_FEATURES,

    "target":
        TARGET_COLUMN,

    "split_method":
        "chronological 70/15/15",

    "random_state":
        RANDOM_STATE,

    "lightgbm_parameters":
        BASE_LIGHTGBM_PARAMETERS,

    "prediction_point":
        "initial patient assessment",

    "excluded_downstream_features": [
        "bed_assigned_time",
        "discharge_time",
        "disposition",
        "icu_transfer_flag",
        "initial_care_unit",
        "requires_ventilation",
        "los_hours",
    ],
}


configuration_path = (
    OUTPUT_DIR
    /
    "los_model_configuration.json"
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


print(
    f"✓ Saved:\n"
    f"{configuration_path}"
)


# ============================================================
# 32. SAVE METRICS
# ============================================================

metrics = {

    "confidence_level":
        CONFIDENCE_LEVEL,

    "target_coverage":
        CONFIDENCE_LEVEL,

    "actual_conformal_coverage":
        float(conformal_coverage),

    "manual_test_coverage":
        float(manual_coverage),

    "mean_conformal_interval_width_hours":
        float(conformal_width),

    "test_p50_mae_hours":
        float(conformal_mae),

    "test_p50_rmse_hours":
        float(conformal_rmse),

    "base_p10_p90_coverage":
        float(base_interval_coverage),

    "base_p10_p90_mean_width_hours":
        float(base_interval_width),

    "base_p10_pinball_loss":
        float(base_pinball_p10),

    "base_p90_pinball_loss":
        float(base_pinball_p90),

    "raw_quantile_ordering_validity":
        float(raw_ordering_validity),
}


metrics_path = (
    OUTPUT_DIR
    /
    "los_model_metrics.json"
)


with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=4,
    )


print(
    f"✓ Saved:\n"
    f"{metrics_path}"
)


# ============================================================
# 33. SAVE FEATURE LIST
# ============================================================

feature_list_path = (
    OUTPUT_DIR
    /
    "los_model_features.json"
)


feature_information = {

    "target":
        TARGET_COLUMN,

    "numeric_features":
        NUMERIC_FEATURES,

    "categorical_features":
        CATEGORICAL_FEATURES,

    "all_features":
        FEATURE_COLUMNS,
}


with open(
    feature_list_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        feature_information,
        file,
        indent=4,
    )


print(
    f"✓ Saved:\n"
    f"{feature_list_path}"
)


# ============================================================
# 34. FINAL EXAMPLE
# ============================================================

print("\n" + "=" * 80)
print("LOS MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 80)


print("\nFinal architecture:")

print(
    "Patient clinical features"
)

print(
    "        ↓"
)

print(
    "LightGBM Quantile Regression"
)

print(
    "        ↓"
)

print(
    "P10 / P50 / P90"
)

print(
    "        ↓"
)

print(
    "MAPIE Conformalized Quantile Regression"
)

print(
    "        ↓"
)

print(
    "Calibrated 80% LOS Prediction Interval"
)

print(
    "        ↓"
)

print(
    "[LOS Lower, LOS Predicted, LOS Upper]"
)


if len(results) > 0:

    example = results.iloc[0]

    print("\nExample test prediction:")

    print(
        f"Stay ID              : "
        f"{example['stay_id']}"
    )

    print(
        f"Actual LOS           : "
        f"{example['actual_los_hours']:.2f} hours"
    )

    print(
        f"Predicted LOS        : "
        f"{example['predicted_los_hours']:.2f} hours"
    )

    print(
        f"80% Lower Bound      : "
        f"{example['los_lower_80']:.2f} hours"
    )

    print(
        f"80% Upper Bound      : "
        f"{example['los_upper_80']:.2f} hours"
    )


# ============================================================
# 35. OUTPUT FILE SUMMARY
# ============================================================

print("\nOutput files:")

print(
    f"  1. {prediction_path.name}"
)

print(
    f"  2. {acuity_path.name}"
)

print(
    f"  3. {configuration_path.name}"
)

print(
    f"  4. {metrics_path.name}"
)

print(
    f"  5. {feature_list_path.name}"
)

print(
    f"  6. {p10_model_path.name}"
)

print(
    f"  7. {p50_model_path.name}"
)

print(
    f"  8. {p90_model_path.name}"
)

print(
    f"  9. {mapie_model_path.name}"
)


print("\nDone.")