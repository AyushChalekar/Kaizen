# ============================================================
# Predictive Resource Optimization System for Healthcare
# Model 1B: Length of Stay (LOS) Prediction
#
# IMPROVED LOS MODEL — LOG-TRANSFORMED TARGET
#
# Architecture:
#
# Patient-level features
#          ↓
# log1p(LOS)
#          ↓
# LightGBM Quantile Regression
#          ↓
# P10 / P50 / P90
#          ↓
# MAPIE Conformalized Quantile Regression
#          ↓
# Inverse transformation: expm1()
#          ↓
# Calibrated 80% LOS Prediction Interval
#
# Dataset:
#   synthetic_patient_stays_100k.csv
#
# Target:
#   los_hours
#
# Model target:
#   log1p(los_hours)
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
#   This is a separate improvement experiment.
#   It does NOT overwrite the baseline model artifacts.
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

from lightgbm import LGBMRegressor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from mapie.regression import ConformalizedQuantileRegressor
from mapie.metrics.regression import (
    regression_coverage_score,
    regression_mean_width_score,
)


warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42


# ------------------------------------------------------------
# Prediction interval
# ------------------------------------------------------------

CONFIDENCE_LEVEL = 0.80

LOWER_QUANTILE = (
    1.0 - CONFIDENCE_LEVEL
) / 2.0

MEDIAN_QUANTILE = 0.50

UPPER_QUANTILE = (
    1.0 + CONFIDENCE_LEVEL
) / 2.0


# Expected:
#
# LOWER_QUANTILE  = 0.10
# MEDIAN_QUANTILE = 0.50
# UPPER_QUANTILE  = 0.90


# ------------------------------------------------------------
# Dataset size requirement
# ------------------------------------------------------------

MIN_RECORDS = 50_000
MAX_RECORDS = 150_000


# ------------------------------------------------------------
# LOS timestamp validation tolerance
# ------------------------------------------------------------

MAX_ALLOWED_ROUNDING_ERROR = 0.02


# ============================================================
# 3. PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

MODEL_DIR = (
    BASE_DIR
    /
    "models"
)

OUTPUT_DIR = (
    BASE_DIR
    /
    "outputs"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Input dataset
# ------------------------------------------------------------

DATA_PATH = (
    DATA_DIR
    /
    "synthetic_patient_stays_100k.csv"
)


# ============================================================
# 4. OUTPUT PATHS
# ============================================================
#
# IMPORTANT:
# These names contain "_log" so the baseline model is not
# overwritten.
# ============================================================


PREDICTION_PATH = (
    OUTPUT_DIR
    /
    "los_predictions_test_log.csv"
)


ACUITY_RESULTS_PATH = (
    OUTPUT_DIR
    /
    "los_model_by_triage_acuity_log.csv"
)


LOS_RANGE_RESULTS_PATH = (
    OUTPUT_DIR
    /
    "los_model_by_los_range_log.csv"
)


METRICS_PATH = (
    OUTPUT_DIR
    /
    "los_model_metrics_log.json"
)


CONFIGURATION_PATH = (
    OUTPUT_DIR
    /
    "los_model_configuration_log.json"
)


FEATURES_PATH = (
    OUTPUT_DIR
    /
    "los_model_features_log.json"
)


# ============================================================
# 5. MODEL ARTIFACT PATHS
# ============================================================


P10_MODEL_PATH = (
    MODEL_DIR
    /
    "lightgbm_los_log_p10.joblib"
)


P50_MODEL_PATH = (
    MODEL_DIR
    /
    "lightgbm_los_log_p50.joblib"
)


P90_MODEL_PATH = (
    MODEL_DIR
    /
    "lightgbm_los_log_p90.joblib"
)


MAPIE_MODEL_PATH = (
    MODEL_DIR
    /
    "mapie_los_log_cqr.joblib"
)


# ============================================================
# 6. REQUIRED DATASET COLUMNS
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
# 7. LOS MODEL FEATURES
# ============================================================
#
# Prediction point:
#   Initial patient assessment
#
# Only features available at/around initial assessment are used.
#
# Downstream variables are deliberately excluded:
#
#   bed_assigned_time
#   discharge_time
#   disposition
#   icu_transfer_flag
#   initial_care_unit
#   requires_ventilation
#   los_hours
#
# This prevents target leakage.
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
    +
    CATEGORICAL_FEATURES
)


TARGET_COLUMN = "los_hours"


# ============================================================
# 8. LIGHTGBM PARAMETERS
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
# 9. HELPER FUNCTIONS
# ============================================================


def create_preprocessor():
    """
    Create a fresh preprocessing pipeline.

    Numeric variables:
        passed directly to LightGBM.

    Categorical variables:
        one-hot encoded.

    Unknown categories:
        safely ignored.
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
    Create a fresh LightGBM quantile regression pipeline.

    The model learns log1p(LOS), not raw LOS.
    """

    parameters = (
        BASE_LIGHTGBM_PARAMETERS.copy()
    )

    parameters["alpha"] = alpha

    model = LGBMRegressor(
        **parameters
    )

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",

                create_preprocessor(),
            ),

            (
                "model",

                model,
            ),

        ]

    )

    return pipeline


def safe_non_negative(values):
    """
    LOS cannot be negative after inverse transformation.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    return np.maximum(
        values,
        0.0,
    )


def pinball_loss(
    y_true,
    y_pred,
    alpha,
):
    """
    Calculate quantile pinball loss.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    error = (
        y_true
        -
        y_pred
    )

    return float(
        np.mean(
            np.where(
                error >= 0,

                alpha * error,

                (alpha - 1.0)
                * error,
            )
        )
    )


# ============================================================
# 10. HEADER
# ============================================================


print("=" * 80)

print(
    "PREDICTIVE RESOURCE OPTIMIZATION SYSTEM FOR HEALTHCARE"
)

print(
    "MODEL 1B — LOG-TRANSFORMED LOS PREDICTION"
)

print("=" * 80)


# ============================================================
# 11. PACKAGE VERSION CHECK
# ============================================================


print(
    "\nChecking installed packages..."
)


import pandas
import sklearn
import lightgbm
import mapie


print(
    f"pandas        : "
    f"{pandas.__version__}"
)


print(
    f"scikit-learn  : "
    f"{sklearn.__version__}"
)


print(
    f"LightGBM      : "
    f"{lightgbm.__version__}"
)


print(
    f"MAPIE         : "
    f"{mapie.__version__}"
)


print(
    "✓ Required packages imported successfully"
)


# ============================================================
# 12. CHECK DATASET
# ============================================================


print("\n" + "-" * 80)

print(
    "[1/14] Checking dataset"
)

print("-" * 80)


print(
    f"Expected CSV:\n"
    f"{DATA_PATH}"
)


if not DATA_PATH.exists():

    raise FileNotFoundError(

        "\nDataset not found.\n\n"

        f"Expected location:\n"
        f"{DATA_PATH}\n\n"

        "Make sure the dataset is named:\n"
        "synthetic_patient_stays_100k.csv\n\n"

        "and placed inside:\n"
        f"{DATA_DIR}"

    )


# ============================================================
# 13. LOAD DATA
# ============================================================


df = pd.read_csv(
    DATA_PATH
)


print(
    "\nDataset loaded successfully."
)


print(
    f"Rows    : "
    f"{len(df):,}"
)


print(
    f"Columns : "
    f"{len(df.columns)}"
)


# ============================================================
# 14. DATASET STRUCTURE VALIDATION
# ============================================================


print("\n" + "-" * 80)

print(
    "[2/14] Validating dataset structure"
)

print("-" * 80)


missing_columns = [

    column

    for column in REQUIRED_COLUMNS

    if column not in df.columns

]


if missing_columns:

    raise ValueError(

        "Missing required columns:\n"

        +
        "\n".join(
            f"  - {column}"
            for column in missing_columns
        )

    )


print(
    "✓ All required columns are present"
)


# ------------------------------------------------------------
# Dataset size
# ------------------------------------------------------------


if not (
    MIN_RECORDS
    <=
    len(df)
    <=
    MAX_RECORDS
):

    raise ValueError(

        f"Dataset contains "
        f"{len(df):,} records.\n"

        f"Required range: "
        f"{MIN_RECORDS:,}–"
        f"{MAX_RECORDS:,}."

    )


print(
    f"✓ Dataset size is valid "
    f"({len(df):,} records)"
)


# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------


missing_values = (
    df[
        REQUIRED_COLUMNS
    ]
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
        "\nMissing values:"
    )

    print(
        missing_values
    )

    raise ValueError(
        "Missing values detected."
    )


print(
    "✓ No missing values"
)


# ------------------------------------------------------------
# Duplicate IDs
# ------------------------------------------------------------


if (
    df["stay_id"]
    .duplicated()
    .any()
):

    raise ValueError(
        "Duplicate stay_id values detected."
    )


if (
    df["patient_id"]
    .duplicated()
    .any()
):

    raise ValueError(
        "Duplicate patient_id values detected."
    )


print(
    "✓ No duplicate stay_id values"
)

print(
    "✓ No duplicate patient_id values"
)


# ============================================================
# 15. LOS VALIDATION
# ============================================================


print("\n" + "-" * 80)

print(
    "[3/14] Validating LOS target"
)

print("-" * 80)


if not pd.api.types.is_numeric_dtype(
    df[TARGET_COLUMN]
):

    raise ValueError(
        "los_hours must be numeric."
    )


if not np.isfinite(
    df[TARGET_COLUMN]
    .to_numpy()
).all():

    raise ValueError(
        "los_hours contains NaN or infinite values."
    )


if (
    df[TARGET_COLUMN]
    <= 0
).any():

    raise ValueError(
        "LOS contains zero or negative values."
    )


print(
    "✓ LOS is numeric"
)

print(
    "✓ LOS contains only finite values"
)

print(
    "✓ LOS contains only positive values"
)


# ------------------------------------------------------------
# LOS distribution
# ------------------------------------------------------------


print(
    "\nLOS distribution:"
)


print(
    f"  Mean   : "
    f"{df[TARGET_COLUMN].mean():.3f} hours"
)


print(
    f"  Median : "
    f"{df[TARGET_COLUMN].median():.3f} hours"
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
    f"  Max    : "
    f"{df[TARGET_COLUMN].max():.3f} hours"
)


# ============================================================
# 16. CLINICAL VALIDATION
# ============================================================


print("\n" + "-" * 80)

print(
    "[4/14] Validating clinical constraints"
)

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
    df[
        "triage_acuity"
    ]
    .astype(int)
    .unique()
)


if not actual_triage_values.issubset(
    valid_triage_values
):

    raise ValueError(
        "Unexpected triage acuity values: "
        f"{actual_triage_values}"
    )


print(
    "✓ Triage acuity values are valid (ESI 1–5)"
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


print(
    "✓ SBP > DBP for all records"
)


# ============================================================
# 17. TIMESTAMP VALIDATION
# ============================================================


print("\n" + "-" * 80)

print(
    "[5/14] Validating timestamps"
)

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
# Chronological order
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
        "Invalid triage timestamp ordering."
    )


if invalid_bed > 0:

    raise ValueError(
        "Invalid bed assignment timestamp ordering."
    )


if invalid_discharge > 0:

    raise ValueError(
        "Invalid discharge timestamp ordering."
    )


print(
    "✓ Timestamp chronology is valid"
)


# ============================================================
# 18. VERIFY LOS AGAINST TIMESTAMPS
# ============================================================


calculated_los = (

    (
        df["discharge_time"]
        -
        df["arrival_time"]
    )
    .dt.total_seconds()
    /
    3600.0

)


los_difference = (

    calculated_los
    -
    df[TARGET_COLUMN]

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

        f"{bad_los_count:,} records have "
        "LOS values that do not agree with "
        "arrival_time → discharge_time."

    )


print(
    "✓ LOS agrees with timestamps"
)


# ============================================================
# 19. SORT CHRONOLOGICALLY
# ============================================================


print("\n" + "-" * 80)

print(
    "[6/14] Sorting data chronologically"
)

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
    f"Earliest arrival: "
    f"{df['arrival_time'].min()}"
)


print(
    f"Latest arrival: "
    f"{df['arrival_time'].max()}"
)


# ============================================================
# 20. CHRONOLOGICAL TRAIN / CALIBRATION / TEST SPLIT
# ============================================================


print("\n" + "-" * 80)

print(
    "[7/14] Creating chronological 70/15/15 split"
)

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


print(
    "\nTraining period:"
)


print(
    train_df["arrival_time"].min(),
    "→",
    train_df["arrival_time"].max(),
)


print(
    "\nCalibration period:"
)


print(
    calibration_df["arrival_time"].min(),
    "→",
    calibration_df["arrival_time"].max(),
)


print(
    "\nTest period:"
)


print(
    test_df["arrival_time"].min(),
    "→",
    test_df["arrival_time"].max(),
)


# ============================================================
# 21. PREPARE FEATURES
# ============================================================


print("\n" + "-" * 80)

print(
    "[8/14] Preparing features and log-transformed target"
)

print("-" * 80)


X_train = (
    train_df[
        FEATURE_COLUMNS
    ]
    .copy()
)


X_calibration = (
    calibration_df[
        FEATURE_COLUMNS
    ]
    .copy()
)


X_test = (
    test_df[
        FEATURE_COLUMNS
    ]
    .copy()
)


# ------------------------------------------------------------
# Raw LOS targets
# ------------------------------------------------------------


y_train_raw = (
    train_df[
        TARGET_COLUMN
    ]
    .to_numpy(
        dtype=float
    )
)


y_calibration_raw = (
    calibration_df[
        TARGET_COLUMN
    ]
    .to_numpy(
        dtype=float
    )
)


y_test_raw = (
    test_df[
        TARGET_COLUMN
    ]
    .to_numpy(
        dtype=float
    )
)


# ------------------------------------------------------------
# Log transformation
# ------------------------------------------------------------
#
# log1p(y) = log(1 + y)
#
# This reduces the effect of the extreme right tail while
# preserving the ordering of LOS values.
# ------------------------------------------------------------


y_train = np.log1p(
    y_train_raw
)


y_calibration = np.log1p(
    y_calibration_raw
)


y_test = np.log1p(
    y_test_raw
)


print(
    "\nFeatures used:"
)


for feature in FEATURE_COLUMNS:

    print(
        f"  ✓ {feature}"
    )


print(
    "\nOriginal target:"
)

print(
    "  los_hours"
)


print(
    "\nModel target:"
)

print(
    "  log1p(los_hours)"
)


print(
    "✓ Log transformation applied only to the target"
)


# ============================================================
# 22. TRAIN P10
# ============================================================


print("\n" + "-" * 80)

print(
    "[9/14] Training LightGBM P10 model"
)

print("-" * 80)


print(
    "Quantile alpha = 0.10"
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
# 23. TRAIN P50
# ============================================================


print(
    "\nTraining LightGBM P50 model"
)


print(
    "Quantile alpha = 0.50"
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
# 24. TRAIN P90
# ============================================================


print(
    "\nTraining LightGBM P90 model"
)


print(
    "Quantile alpha = 0.90"
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
# 25. BASE LOG-SCALE PREDICTIONS
# ============================================================


print("\n" + "-" * 80)

print(
    "[10/14] Evaluating base quantile predictions"
)

print("-" * 80)


p10_log = (
    model_p10
    .predict(
        X_test
    )
)


p50_log = (
    model_p50
    .predict(
        X_test
    )
)


p90_log = (
    model_p90
    .predict(
        X_test
    )
)


# ------------------------------------------------------------
# Convert predictions back to hours
# ------------------------------------------------------------
#
# inverse of log1p:
#
# expm1(x) = exp(x) - 1
# ------------------------------------------------------------


p10_test = safe_non_negative(
    np.expm1(
        p10_log
    )
)


p50_test = safe_non_negative(
    np.expm1(
        p50_log
    )
)


p90_test = safe_non_negative(
    np.expm1(
        p90_log
    )
)


# ------------------------------------------------------------
# Raw quantile crossing diagnostic
# ------------------------------------------------------------


valid_quantile_order = (

    (p10_test <= p50_test)
    &
    (p50_test <= p90_test)

)


quantile_ordering_rate = float(
    np.mean(
        valid_quantile_order
    )
)


quantile_crossing_count = int(
    np.sum(
        ~valid_quantile_order
    )
)


print(
    f"\nRaw P10/P50/P90 ordering validity: "
    f"{quantile_ordering_rate:.3%}"
)


print(
    f"Quantile crossing records: "
    f"{quantile_crossing_count:,}"
)


# ------------------------------------------------------------
# Base metrics
# ------------------------------------------------------------


base_mae = float(
    mean_absolute_error(
        y_test_raw,
        p50_test,
    )
)


base_rmse = float(
    np.sqrt(
        mean_squared_error(
            y_test_raw,
            p50_test,
        )
    )
)


base_r2 = float(
    r2_score(
        y_test_raw,
        p50_test,
    )
)


base_pinball_p10 = pinball_loss(
    y_test_raw,
    p10_test,
    LOWER_QUANTILE,
)


base_pinball_p90 = pinball_loss(
    y_test_raw,
    p90_test,
    UPPER_QUANTILE,
)


base_coverage = float(
    np.mean(
        (
            y_test_raw
            >=
            p10_test
        )
        &
        (
            y_test_raw
            <=
            p90_test
        )
    )
)


base_width = float(
    np.mean(
        p90_test
        -
        p10_test
    )
)


print(
    "\nBase log-target LightGBM results:"
)


print(
    f"  P50 MAE          : "
    f"{base_mae:.3f} hours"
)


print(
    f"  P50 RMSE         : "
    f"{base_rmse:.3f} hours"
)


print(
    f"  P50 R²           : "
    f"{base_r2:.4f}"
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
    f"{base_coverage:.3%}"
)


print(
    f"  Mean Width       : "
    f"{base_width:.3f} hours"
)


# ============================================================
# 26. MAPIE CONFORMALIZED QUANTILE REGRESSION
# ============================================================
#
# IMPORTANT:
#
# MAPIE receives the target on the SAME SCALE on which the
# quantile models were trained.
#
# Therefore:
#
#     calibration target = log1p(LOS)
#
# and MAPIE constructs the conformal interval in log space.
#
# Because expm1() is strictly increasing, the interval can
# safely be transformed back to LOS hours afterward.
# ============================================================


print("\n" + "-" * 80)

print(
    "[11/14] Applying MAPIE Conformalized Quantile Regression"
)

print("-" * 80)


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
    "\nCalibrating on dedicated 15% calibration dataset..."
)


mapie_cqr.conformalize(

    X_calibration,

    y_calibration,

)


print(
    "✓ MAPIE conformal calibration completed"
)


# ============================================================
# 27. FINAL TEST PREDICTIONS
# ============================================================


print(
    "\nGenerating final calibrated test predictions..."
)


y_pred_log, y_interval_log = (

    mapie_cqr.predict_interval(
        X_test
    )

)


# ------------------------------------------------------------
# Extract log-scale interval
# ------------------------------------------------------------


lower_log = (
    y_interval_log[
        :,
        0,
        0,
    ]
)


upper_log = (
    y_interval_log[
        :,
        1,
        0,
    ]
)


predicted_log = (
    y_pred_log
)


# ------------------------------------------------------------
# Convert everything back to LOS hours
# ------------------------------------------------------------


los_predicted = safe_non_negative(
    np.expm1(
        predicted_log
    )
)


los_lower = safe_non_negative(
    np.expm1(
        lower_log
    )
)


los_upper = safe_non_negative(
    np.expm1(
        upper_log
    )
)


# ------------------------------------------------------------
# Final interval ordering
# ------------------------------------------------------------
#
# The final interval must contain the point prediction.
#
# This is only a final ordering/safety operation.
# The conformal calibration itself was performed before this
# presentation step.
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
# 28. FINAL CONFORMAL METRICS
# ============================================================


print(
    "\nCalculating final conformal metrics..."
)


conformal_mae = float(
    mean_absolute_error(
        y_test_raw,
        los_predicted,
    )
)


conformal_rmse = float(
    np.sqrt(
        mean_squared_error(
            y_test_raw,
            los_predicted,
        )
    )
)


conformal_r2 = float(
    r2_score(
        y_test_raw,
        los_predicted,
    )
)


# ------------------------------------------------------------
# MAPIE coverage and width
#
# These are calculated on MAPIE's original log-scale interval
# because that is the interval MAPIE actually calibrated.
# ------------------------------------------------------------


mapie_log_coverage = float(
    regression_coverage_score(
        y_test,
        y_interval_log,
    )[0]
)


mapie_log_width = float(
    regression_mean_width_score(
        y_interval_log,
    )[0]
)


# ------------------------------------------------------------
# Manual coverage on original LOS scale
# ------------------------------------------------------------


manual_coverage = float(
    np.mean(
        (
            y_test_raw
            >=
            los_lower
        )
        &
        (
            y_test_raw
            <=
            los_upper
        )
    )
)


# ------------------------------------------------------------
# Original-scale interval width
# ------------------------------------------------------------


mean_interval_width = float(
    np.mean(
        los_upper
        -
        los_lower
    )
)


median_interval_width = float(
    np.median(
        los_upper
        -
        los_lower
    )
)


p90_interval_width = float(
    np.percentile(
        los_upper
        -
        los_lower,
        90,
    )
)


# ------------------------------------------------------------
# Prediction bias
# ------------------------------------------------------------


prediction_error = (
    los_predicted
    -
    y_test_raw
)


mean_prediction_error = float(
    np.mean(
        prediction_error
    )
)


underprediction_rate = float(
    np.mean(
        los_predicted
        <
        y_test_raw
    )
)


overprediction_rate = float(
    np.mean(
        los_predicted
        >
        y_test_raw
    )
)


# ============================================================
# 29. FINAL RESULTS
# ============================================================


print("\n" + "=" * 80)

print(
    "FINAL LOG-TRANSFORMED LOS MODEL RESULTS"
)

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
    "\nTarget transformation:"
)


print(
    "  log1p(los_hours)"
)


print(
    "\nTest P50 MAE:"
)


print(
    f"  {conformal_mae:.3f} hours"
)


print(
    "\nTest P50 RMSE:"
)


print(
    f"  {conformal_rmse:.3f} hours"
)


print(
    "\nTest P50 R²:"
)


print(
    f"  {conformal_r2:.4f}"
)


print(
    "\nMAPIE coverage on log scale:"
)


print(
    f"  {mapie_log_coverage:.3%}"
)


print(
    "\nManual coverage on original LOS scale:"
)


print(
    f"  {manual_coverage:.3%}"
)


print(
    "\nTarget coverage:"
)


print(
    f"  {CONFIDENCE_LEVEL:.3%}"
)


print(
    "\nMean interval width on original LOS scale:"
)


print(
    f"  {mean_interval_width:.3f} hours"
)


print(
    "\nMedian interval width:"
)


print(
    f"  {median_interval_width:.3f} hours"
)


print(
    "\n90th percentile interval width:"
)


print(
    f"  {p90_interval_width:.3f} hours"
)


print(
    "\nMean prediction error:"
)


print(
    f"  {mean_prediction_error:+.3f} hours"
)


print(
    "\nUnderprediction rate:"
)


print(
    f"  {underprediction_rate:.3%}"
)


print(
    "\nOverprediction rate:"
)


print(
    f"  {overprediction_rate:.3%}"
)


# ============================================================
# 30. CREATE FINAL PREDICTION DATAFRAME
# ============================================================


print("\n" + "-" * 80)

print(
    "[12/14] Creating final prediction dataset"
)

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
    "prediction_error_hours"
] = (

    results[
        "predicted_los_hours"
    ]

    -

    results[
        "actual_los_hours"
    ]

)


results[
    "absolute_error_hours"
] = np.abs(
    results[
        "prediction_error_hours"
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
# 31. SAVE FINAL PREDICTIONS
# ============================================================


results.to_csv(
    PREDICTION_PATH,
    index=False,
)


print(
    f"✓ Saved:\n"
    f"{PREDICTION_PATH}"
)


# ============================================================
# 32. EVALUATE BY TRIAGE ACUITY
# ============================================================


print("\n" + "-" * 80)

print(
    "[13/14] Evaluating by triage acuity"
)

print("-" * 80)


acuity_results = []


for acuity in sorted(
    results[
        "triage_acuity"
    ]
    .unique()
):

    group = results[
        results[
            "triage_acuity"
        ]
        ==
        acuity
    ]


    actual_group = (
        group[
            "actual_los_hours"
        ]
        .to_numpy(
            dtype=float
        )
    )


    predicted_group = (
        group[
            "predicted_los_hours"
        ]
        .to_numpy(
            dtype=float
        )
    )


    lower_group = (
        group[
            "los_lower_80"
        ]
        .to_numpy(
            dtype=float
        )
    )


    upper_group = (
        group[
            "los_upper_80"
        ]
        .to_numpy(
            dtype=float
        )
    )


    group_error = (
        predicted_group
        -
        actual_group
    )


    group_mae = float(
        mean_absolute_error(
            actual_group,
            predicted_group,
        )
    )


    group_rmse = float(
        np.sqrt(
            mean_squared_error(
                actual_group,
                predicted_group,
            )
        )
    )


    group_bias = float(
        np.mean(
            group_error
        )
    )


    group_coverage = float(
        np.mean(
            (
                actual_group
                >=
                lower_group
            )
            &
            (
                actual_group
                <=
                upper_group
            )
        )
    )


    group_width = float(
        np.mean(
            upper_group
            -
            lower_group
        )
    )


    group_underprediction = float(
        np.mean(
            predicted_group
            <
            actual_group
        )
    )


    acuity_results.append(
        {

            "triage_acuity":
                int(acuity),

            "n_test_records":
                int(len(group)),

            "actual_mean_los_hours":
                float(
                    np.mean(
                        actual_group
                    )
                ),

            "actual_median_los_hours":
                float(
                    np.median(
                        actual_group
                    )
                ),

            "predicted_mean_los_hours":
                float(
                    np.mean(
                        predicted_group
                    )
                ),

            "predicted_median_los_hours":
                float(
                    np.median(
                        predicted_group
                    )
                ),

            "mae_hours":
                group_mae,

            "rmse_hours":
                group_rmse,

            "bias_hours":
                group_bias,

            "underprediction_rate":
                group_underprediction,

            "coverage_80":
                group_coverage,

            "mean_interval_width_hours":
                group_width,

        }
    )


acuity_results_df = pd.DataFrame(
    acuity_results
)


print(
    "\nPerformance by triage acuity:"
)


print(
    acuity_results_df.to_string(
        index=False
    )
)


acuity_results_df.to_csv(
    ACUITY_RESULTS_PATH,
    index=False,
)


print(
    f"\n✓ Saved:\n"
    f"{ACUITY_RESULTS_PATH}"
)


# ============================================================
# 33. EVALUATE BY LOS RANGE
# ============================================================


print("\n" + "-" * 80)

print(
    "Evaluating by actual LOS range"
)

print("-" * 80)


def classify_los(hours):

    if hours <= 24:

        return "0-24 hours"

    elif hours <= 72:

        return "24-72 hours"

    elif hours <= 168:

        return "72-168 hours"

    else:

        return ">168 hours"


results[
    "los_range"
] = (
    results[
        "actual_los_hours"
    ]
    .apply(
        classify_los
    )
)


LOS_RANGE_ORDER = [

    "0-24 hours",

    "24-72 hours",

    "72-168 hours",

    ">168 hours",

]


los_range_results = []


for los_range in LOS_RANGE_ORDER:

    group = results[
        results[
            "los_range"
        ]
        ==
        los_range
    ]


    if len(group) == 0:

        continue


    actual_group = (
        group[
            "actual_los_hours"
        ]
        .to_numpy(
            dtype=float
        )
    )


    predicted_group = (
        group[
            "predicted_los_hours"
        ]
        .to_numpy(
            dtype=float
        )
    )


    lower_group = (
        group[
            "los_lower_80"
        ]
        .to_numpy(
            dtype=float
        )
    )


    upper_group = (
        group[
            "los_upper_80"
        ]
        .to_numpy(
            dtype=float
        )
    )


    group_error = (
        predicted_group
        -
        actual_group
    )


    group_mae = float(
        mean_absolute_error(
            actual_group,
            predicted_group,
        )
    )


    group_rmse = float(
        np.sqrt(
            mean_squared_error(
                actual_group,
                predicted_group,
            )
        )
    )


    group_bias = float(
        np.mean(
            group_error
        )
    )


    group_coverage = float(
        np.mean(
            (
                actual_group
                >=
                lower_group
            )
            &
            (
                actual_group
                <=
                upper_group
            )
        )
    )


    group_width = float(
        np.mean(
            upper_group
            -
            lower_group
        )
    )


    group_underprediction = float(
        np.mean(
            predicted_group
            <
            actual_group
        )
    )


    los_range_results.append(
        {

            "los_range":
                los_range,

            "n_test_records":
                int(len(group)),

            "actual_mean_los_hours":
                float(
                    np.mean(
                        actual_group
                    )
                ),

            "actual_median_los_hours":
                float(
                    np.median(
                        actual_group
                    )
                ),

            "predicted_mean_los_hours":
                float(
                    np.mean(
                        predicted_group
                    )
                ),

            "predicted_median_los_hours":
                float(
                    np.median(
                        predicted_group
                    )
                ),

            "mae_hours":
                group_mae,

            "rmse_hours":
                group_rmse,

            "bias_hours":
                group_bias,

            "underprediction_rate":
                group_underprediction,

            "coverage_80":
                group_coverage,

            "mean_interval_width_hours":
                group_width,

        }
    )


los_range_results_df = pd.DataFrame(
    los_range_results
)


print(
    "\nPerformance by actual LOS range:"
)


print(
    los_range_results_df.to_string(
        index=False
    )
)


los_range_results_df.to_csv(
    LOS_RANGE_RESULTS_PATH,
    index=False,
)


print(
    f"\n✓ Saved:\n"
    f"{LOS_RANGE_RESULTS_PATH}"
)


# ============================================================
# 34. SAVE MODEL ARTIFACTS AND CONFIGURATION
# ============================================================


print("\n" + "-" * 80)

print(
    "[14/14] Saving model artifacts and configuration"
)

print("-" * 80)


# ------------------------------------------------------------
# Save models
# ------------------------------------------------------------


joblib.dump(
    model_p10,
    P10_MODEL_PATH,
)


joblib.dump(
    model_p50,
    P50_MODEL_PATH,
)


joblib.dump(
    model_p90,
    P90_MODEL_PATH,
)


joblib.dump(
    mapie_cqr,
    MAPIE_MODEL_PATH,
)


print(
    f"✓ {P10_MODEL_PATH}"
)


print(
    f"✓ {P50_MODEL_PATH}"
)


print(
    f"✓ {P90_MODEL_PATH}"
)


print(
    f"✓ {MAPIE_MODEL_PATH}"
)


# ============================================================
# 35. SAVE CONFIGURATION
# ============================================================


configuration = {

    "project":
        "Predictive Resource Optimization System for Healthcare",

    "model":
        "Length of Stay Prediction — Log Target",

    "algorithm":
        "LightGBM Quantile Regression",

    "conformal_method":
        "MAPIE Conformalized Quantile Regression",

    "target":
        TARGET_COLUMN,

    "target_transformation":
        "log1p(los_hours)",

    "inverse_transformation":
        "expm1(prediction)",

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

    "split_method":
        "chronological 70/15/15",

    "prediction_point":
        "initial patient assessment",

    "random_state":
        RANDOM_STATE,

    "lightgbm_parameters":
        BASE_LIGHTGBM_PARAMETERS,

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


with open(
    CONFIGURATION_PATH,
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
    f"{CONFIGURATION_PATH}"
)


# ============================================================
# 36. SAVE FEATURE INFORMATION
# ============================================================


feature_information = {

    "target":
        TARGET_COLUMN,

    "model_target":
        "log1p(los_hours)",

    "inverse_transform":
        "expm1",

    "numeric_features":
        NUMERIC_FEATURES,

    "categorical_features":
        CATEGORICAL_FEATURES,

    "all_features":
        FEATURE_COLUMNS,

}


with open(
    FEATURES_PATH,
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
    f"{FEATURES_PATH}"
)


# ============================================================
# 37. SAVE METRICS
# ============================================================


coverage_difference = (
    manual_coverage
    -
    CONFIDENCE_LEVEL
)


metrics = {

    "model":
        "LOS Log-Transformed Quantile Model",

    "target_transformation":
        "log1p(los_hours)",

    "confidence_level":
        CONFIDENCE_LEVEL,

    "target_coverage":
        CONFIDENCE_LEVEL,

    "test_records":
        int(len(test_df)),

    "mae_hours":
        conformal_mae,

    "rmse_hours":
        conformal_rmse,

    "r2":
        conformal_r2,

    "mapie_log_scale_coverage":
        mapie_log_coverage,

    "manual_original_scale_coverage":
        manual_coverage,

    "coverage_difference_from_target":
        coverage_difference,

    "mean_interval_width_hours":
        mean_interval_width,

    "median_interval_width_hours":
        median_interval_width,

    "p90_interval_width_hours":
        p90_interval_width,

    "mean_prediction_error_hours":
        mean_prediction_error,

    "underprediction_rate":
        underprediction_rate,

    "overprediction_rate":
        overprediction_rate,

    "raw_quantile_ordering_validity":
        quantile_ordering_rate,

    "raw_quantile_crossing_records":
        quantile_crossing_count,

    "base_lightgbm_mae_hours":
        base_mae,

    "base_lightgbm_rmse_hours":
        base_rmse,

    "base_lightgbm_r2":
        base_r2,

    "base_p10_p90_coverage":
        base_coverage,

    "base_p10_p90_mean_width_hours":
        base_width,

    "base_p10_pinball_loss":
        base_pinball_p10,

    "base_p90_pinball_loss":
        base_pinball_p90,

}


with open(
    METRICS_PATH,
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
    f"{METRICS_PATH}"
)


# ============================================================
# 38. FINAL SUMMARY
# ============================================================


print("\n" + "=" * 80)

print(
    "LOG-TRANSFORMED LOS MODEL TRAINING COMPLETED"
)

print("=" * 80)


print(
    "\nArchitecture:"
)


print(
    "Patient clinical features"
)


print(
    "        ↓"
)


print(
    "log1p(LOS)"
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
    "expm1()"
)


print(
    "        ↓"
)


print(
    "Calibrated 80% LOS prediction interval"
)


print(
    "\nFinal results:"
)


print(
    f"  MAE                : "
    f"{conformal_mae:.3f} hours"
)


print(
    f"  RMSE               : "
    f"{conformal_rmse:.3f} hours"
)


print(
    f"  R²                 : "
    f"{conformal_r2:.4f}"
)


print(
    f"  80% Coverage       : "
    f"{manual_coverage:.3%}"
)


print(
    f"  Mean Interval Width: "
    f"{mean_interval_width:.3f} hours"
)


print(
    f"  Underprediction    : "
    f"{underprediction_rate:.3%}"
)


print(
    "\nExample test prediction:"
)


if len(results) > 0:

    example = results.iloc[0]


    print(
        f"  Stay ID        : "
        f"{example['stay_id']}"
    )


    print(
        f"  Actual LOS     : "
        f"{example['actual_los_hours']:.2f} hours"
    )


    print(
        f"  Predicted LOS  : "
        f"{example['predicted_los_hours']:.2f} hours"
    )


    print(
        f"  Lower 80%      : "
        f"{example['los_lower_80']:.2f} hours"
    )


    print(
        f"  Upper 80%      : "
        f"{example['los_upper_80']:.2f} hours"
    )


print(
    "\nNew files created:"
)


print(
    f"  {PREDICTION_PATH.name}"
)


print(
    f"  {ACUITY_RESULTS_PATH.name}"
)


print(
    f"  {LOS_RANGE_RESULTS_PATH.name}"
)


print(
    f"  {METRICS_PATH.name}"
)


print(
    f"  {CONFIGURATION_PATH.name}"
)


print(
    f"  {FEATURES_PATH.name}"
)


print(
    f"  {P10_MODEL_PATH.name}"
)


print(
    f"  {P50_MODEL_PATH.name}"
)


print(
    f"  {P90_MODEL_PATH.name}"
)


print(
    f"  {MAPIE_MODEL_PATH.name}"
)


print("\n" + "=" * 80)

print(
    "DONE — LOG LOS MODEL READY FOR COMPARISON"
)

print("=" * 80)