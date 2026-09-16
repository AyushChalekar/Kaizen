# ============================================================
# LOS Prediction - Tail-Weighted LightGBM Quantile Regression
#                     + MAPIE Conformal Prediction
#
# Kaizen Healthcare Predictive Resource Optimization System
#
# EXPERIMENT:
#   Tail-weighted training to improve prediction of long LOS.
#
# IMPORTANT:
#   - Existing baseline is NOT overwritten.
#   - Existing tuned model is NOT overwritten.
#   - Test set is NOT used for model/weight selection.
#
# DATA SPLIT:
#   70,000 -> Training
#   15,000 -> Calibration
#   15,000 -> Final Test
#
# MODEL:
#   LightGBM Quantile Regression
#       P10 = 0.10
#       P50 = 0.50
#       P90 = 0.90
#
# UNCERTAINTY:
#   MAPIE Conformalized Quantile Regression
#   Confidence level = 80%
#
# MAIN OBJECTIVE:
#   Improve long-LOS performance while maintaining
#   acceptable overall accuracy and interval coverage.
# ============================================================


import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_pinball_loss,
)

from mapie.regression import ConformalizedQuantileRegressor

from mapie.metrics.regression import (
    regression_coverage_score,
    regression_mean_width_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "synthetic_patient_stays_100k.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. GLOBAL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TRAIN_SIZE = 70_000
CALIBRATION_SIZE = 15_000
TEST_SIZE = 15_000

# Internal validation portion of training data.
#
# Used ONLY for selecting the best tail-weight scheme.
#
# 60,000 -> internal training
# 10,000 -> internal validation

TUNING_VALIDATION_SIZE = 10_000

MAPIE_CONFIDENCE_LEVEL = 0.80


# ============================================================
# 3. TARGET AND FEATURES
# ============================================================

TARGET = "los_hours"

# Strict arrival / triage-time features.
#
# We deliberately exclude downstream variables:
#
#   discharge_time
#   bed_assigned_time
#   disposition
#   icu_transfer_flag
#   initial_care_unit
#   requires_ventilation
#
# These are not available reliably at the initial
# LOS prediction point.

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

FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ============================================================
# 4. QUANTILES
# ============================================================

QUANTILES = {
    "p10": 0.10,
    "p50": 0.50,
    "p90": 0.90,
}


# ============================================================
# 5. LIGHTGBM BASE CONFIGURATION
# ============================================================
#
# Hyperparameter tuning already showed that the conservative
# configuration was the strongest among the tested settings.
#
# Therefore this experiment does NOT repeat the same broad
# hyperparameter search.
#
# Instead, it concentrates on the training-weight strategy.

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
}


# ============================================================
# 6. TAIL-WEIGHT SCHEMES
# ============================================================
#
# Weight is determined ONLY from the training target.
#
# The model is therefore encouraged to pay more attention
# to long stays during training.
#
# Scheme 0 is the unweighted reference.
#
# Scheme 1 = mild
# Scheme 2 = moderate
# Scheme 3 = strong
# Scheme 4 = aggressive
#
# We allow the validation process to decide whether weighting
# actually helps.
#
# If weighting does not help, the reference scheme can win.
#
# This prevents us from forcing a worse model simply because
# it is "tail weighted".

WEIGHT_SCHEMES = {
    "unweighted_reference": {
        "0_24h": 1.00,
        "24_72h": 1.00,
        "72_168h": 1.00,
        "over_168h": 1.00,
    },

    "mild_tail": {
        "0_24h": 1.00,
        "24_72h": 1.10,
        "72_168h": 1.35,
        "over_168h": 1.75,
    },

    "moderate_tail": {
        "0_24h": 1.00,
        "24_72h": 1.20,
        "72_168h": 1.60,
        "over_168h": 2.25,
    },

    "strong_tail": {
        "0_24h": 1.00,
        "24_72h": 1.25,
        "72_168h": 1.85,
        "over_168h": 2.75,
    },

    "aggressive_tail": {
        "0_24h": 1.00,
        "24_72h": 1.35,
        "72_168h": 2.10,
        "over_168h": 3.50,
    },
}


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ------------------------------------------------------------
# Prepare features
# ------------------------------------------------------------

def prepare_features(df):
    """
    Prepare LightGBM input features.

    Categorical variables are converted to pandas category
    dtype so LightGBM can process them natively.
    """

    X = df[FEATURES].copy()

    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].astype("category")

    return X


# ------------------------------------------------------------
# Create LightGBM quantile model
# ------------------------------------------------------------

def create_quantile_model(alpha):
    """
    Create LightGBM quantile regression model.
    """

    params = BASE_LGB_PARAMS.copy()

    params.update({
        "objective": "quantile",
        "alpha": alpha,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    })

    return lgb.LGBMRegressor(
        **params
    )


# ------------------------------------------------------------
# Validate dataset
# ------------------------------------------------------------

def validate_dataset(df):

    print_section(
        "DATASET VALIDATION"
    )

    required_columns = (
        FEATURES
        + [
            TARGET,
            "arrival_time",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    if df[TARGET].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    if (df[TARGET] < 0).any():
        raise ValueError(
            "Target contains negative LOS values."
        )

    if df[FEATURES].isna().any().any():
        raise ValueError(
            "Feature columns contain missing values."
        )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        "Required columns: PASS"
    )

    print(
        "Missing values: PASS"
    )

    print(
        "Negative LOS: PASS"
    )


# ------------------------------------------------------------
# Create tail weights
# ------------------------------------------------------------

def create_tail_weights(
    y,
    scheme,
):
    """
    Create observation weights based on LOS.

    Important:
        These weights are used only during model training.

    They are NOT used during:
        - calibration
        - final test evaluation
    """

    y = np.asarray(y)

    weights = np.ones(
        len(y),
        dtype=float,
    )

    mask_0_24 = (
        y < 24
    )

    mask_24_72 = (
        (y >= 24)
        & (y < 72)
    )

    mask_72_168 = (
        (y >= 72)
        & (y < 168)
    )

    mask_over_168 = (
        y >= 168
    )

    weights[mask_0_24] = (
        scheme["0_24h"]
    )

    weights[mask_24_72] = (
        scheme["24_72h"]
    )

    weights[mask_72_168] = (
        scheme["72_168h"]
    )

    weights[mask_over_168] = (
        scheme["over_168h"]
    )

    return weights


# ------------------------------------------------------------
# Print weight distribution
# ------------------------------------------------------------

def print_weight_distribution(
    y,
    weights,
):

    y = np.asarray(y)

    masks = {
        "0-24h": y < 24,

        "24-72h": (
            (y >= 24)
            & (y < 72)
        ),

        "72-168h": (
            (y >= 72)
            & (y < 168)
        ),

        ">168h": y >= 168,
    }

    print(
        "\nTraining weight distribution:"
    )

    for name, mask in masks.items():

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        print(
            f"{name:10s} | "
            f"n={count:6,d} | "
            f"weight={weights[mask][0]:.2f}"
        )


# ------------------------------------------------------------
# Point metrics
# ------------------------------------------------------------

def point_metrics(
    y_true,
    y_pred,
):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    median_abs_error = np.median(
        np.abs(
            y_true - y_pred
        )
    )

    mean_error = np.mean(
        y_pred - y_true
    )

    underprediction = (
        np.mean(
            y_pred < y_true
        )
        * 100
    )

    overprediction = (
        np.mean(
            y_pred > y_true
        )
        * 100
    )

    return {
        "MAE_hours": float(mae),

        "RMSE_hours": float(rmse),

        "R2": float(r2),

        "Median_Absolute_Error_hours":
            float(
                median_abs_error
            ),

        "Mean_Prediction_Error_hours":
            float(
                mean_error
            ),

        "Underprediction_percent":
            float(
                underprediction
            ),

        "Overprediction_percent":
            float(
                overprediction
            ),
    }


# ------------------------------------------------------------
# Tail metrics
# ------------------------------------------------------------

def tail_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    masks = {
        "0_24h": y_true < 24,

        "24_72h": (
            (y_true >= 24)
            & (y_true < 72)
        ),

        "72_168h": (
            (y_true >= 72)
            & (y_true < 168)
        ),

        "over_168h": y_true >= 168,
    }

    results = {}

    for group, mask in masks.items():

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        actual = y_true[mask]

        predicted = y_pred[mask]

        results[group] = {
            "count": count,

            "actual_mean_hours":
                float(
                    np.mean(actual)
                ),

            "predicted_mean_hours":
                float(
                    np.mean(predicted)
                ),

            "MAE_hours":
                float(
                    mean_absolute_error(
                        actual,
                        predicted,
                    )
                ),

            "RMSE_hours":
                float(
                    np.sqrt(
                        mean_squared_error(
                            actual,
                            predicted,
                        )
                    )
                ),

            "bias_hours":
                float(
                    np.mean(
                        predicted - actual
                    )
                ),

            "underprediction_percent":
                float(
                    np.mean(
                        predicted < actual
                    )
                    * 100
                ),
        }

    return results


# ------------------------------------------------------------
# Pinball metrics
# ------------------------------------------------------------

def calculate_pinball_metrics(
    y_true,
    p10,
    p90,
):

    p10_loss = mean_pinball_loss(
        y_true,
        p10,
        alpha=0.10,
    )

    p90_loss = mean_pinball_loss(
        y_true,
        p90,
        alpha=0.90,
    )

    return (
        float(p10_loss),
        float(p90_loss),
    )


# ------------------------------------------------------------
# Interval metrics
# ------------------------------------------------------------

def interval_metrics(
    y_true,
    lower,
    upper,
):

    # MAPIE 1.5.0 expects:
    #
    # (n_samples, 2, 1)

    intervals = np.column_stack(
        [
            lower,
            upper,
        ]
    )[:, :, np.newaxis]

    coverage = (
        regression_coverage_score(
            y_true,
            intervals,
        )
    )

    mean_width = (
        regression_mean_width_score(
            intervals,
        )
    )

    width = (
        upper - lower
    )

    ordering_validity = (
        np.mean(
            lower <= upper
        )
        * 100
    )

    return {
        "Coverage_percent":
            float(
                coverage * 100
            ),

        "Mean_Interval_Width_hours":
            float(
                mean_width
            ),

        "Median_Interval_Width_hours":
            float(
                np.median(width)
            ),

        "P90_Interval_Width_hours":
            float(
                np.percentile(
                    width,
                    90,
                )
            ),

        "Interval_Ordering_Validity_percent":
            float(
                ordering_validity
            ),
    }


# ------------------------------------------------------------
# Tail interval coverage
# ------------------------------------------------------------

def interval_tail_metrics(
    y_true,
    lower,
    upper,
):

    y_true = np.asarray(
        y_true
    )

    lower = np.asarray(
        lower
    )

    upper = np.asarray(
        upper
    )

    masks = {
        "0_24h": y_true < 24,

        "24_72h": (
            (y_true >= 24)
            & (y_true < 72)
        ),

        "72_168h": (
            (y_true >= 72)
            & (y_true < 168)
        ),

        "over_168h": y_true >= 168,
    }

    results = {}

    for group, mask in masks.items():

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        actual = y_true[mask]

        low = lower[mask]

        high = upper[mask]

        coverage = (
            np.mean(
                (actual >= low)
                & (actual <= high)
            )
            * 100
        )

        width = (
            high - low
        )

        results[group] = {
            "count": count,

            "coverage_percent":
                float(
                    coverage
                ),

            "mean_width_hours":
                float(
                    np.mean(width)
                ),
        }

    return results


# ============================================================
# 8. LOAD DATA
# ============================================================

print_section(
    "LOS MODEL - TAIL WEIGHTED LIGHTGBM + MAPIE"
)

print(
    f"Data path: {DATA_PATH}"
)

print(
    f"Output directory: {OUTPUT_DIR}"
)

print(
    f"Model directory: {MODEL_DIR}"
)

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n"
        f"{DATA_PATH}"
    )

df = pd.read_csv(
    DATA_PATH
)

print(
    "\nDataset loaded successfully."
)

print(
    f"Shape: {df.shape}"
)

validate_dataset(
    df
)


# ============================================================
# 9. CHRONOLOGICAL ORDER
# ============================================================

print_section(
    "CHRONOLOGICAL ORDERING"
)

df["arrival_time"] = pd.to_datetime(
    df["arrival_time"],
    errors="raise",
)

df = (
    df
    .sort_values(
        "arrival_time"
    )
    .reset_index(
        drop=True
    )
)

print(
    "Sorted by arrival_time: PASS"
)

print(
    f"First arrival: "
    f"{df['arrival_time'].min()}"
)

print(
    f"Last arrival:  "
    f"{df['arrival_time'].max()}"
)


# ============================================================
# 10. DATA SPLIT
# ============================================================

print_section(
    "CHRONOLOGICAL DATA SPLIT"
)

required_rows = (
    TRAIN_SIZE
    + CALIBRATION_SIZE
    + TEST_SIZE
)

if len(df) < required_rows:

    raise ValueError(
        f"Dataset contains only "
        f"{len(df):,} rows but "
        f"{required_rows:,} are required."
    )


train_df = (
    df
    .iloc[
        :TRAIN_SIZE
    ]
    .copy()
)

calibration_df = (
    df
    .iloc[
        TRAIN_SIZE:
        TRAIN_SIZE
        + CALIBRATION_SIZE
    ]
    .copy()
)

test_df = (
    df
    .iloc[
        TRAIN_SIZE
        + CALIBRATION_SIZE:
        TRAIN_SIZE
        + CALIBRATION_SIZE
        + TEST_SIZE
    ]
    .copy()
)


print(
    f"Training:     "
    f"{len(train_df):,} rows"
)

print(
    f"Calibration:  "
    f"{len(calibration_df):,} rows"
)

print(
    f"Test:         "
    f"{len(test_df):,} rows"
)


print(
    f"\nTraining period:"
    f" {train_df['arrival_time'].min()}"
    f" -> "
    f"{train_df['arrival_time'].max()}"
)

print(
    f"Calibration period:"
    f" {calibration_df['arrival_time'].min()}"
    f" -> "
    f"{calibration_df['arrival_time'].max()}"
)

print(
    f"Test period:"
    f" {test_df['arrival_time'].min()}"
    f" -> "
    f"{test_df['arrival_time'].max()}"
)


# ============================================================
# 11. INTERNAL VALIDATION SPLIT
# ============================================================

print_section(
    "INTERNAL VALIDATION SPLIT"
)

tail_train_df = (
    train_df
    .iloc[
        :-TUNING_VALIDATION_SIZE
    ]
    .copy()
)

tail_validation_df = (
    train_df
    .iloc[
        -TUNING_VALIDATION_SIZE:
    ]
    .copy()
)


print(
    f"Tail-weight training rows: "
    f"{len(tail_train_df):,}"
)

print(
    f"Tail-weight validation rows: "
    f"{len(tail_validation_df):,}"
)


print(
    f"\nTail training period:"
    f" {tail_train_df['arrival_time'].min()}"
    f" -> "
    f"{tail_train_df['arrival_time'].max()}"
)

print(
    f"Tail validation period:"
    f" {tail_validation_df['arrival_time'].min()}"
    f" -> "
    f"{tail_validation_df['arrival_time'].max()}"
)


# ============================================================
# 12. PREPARE TRAINING DATA
# ============================================================

X_tail_train = prepare_features(
    tail_train_df
)

y_tail_train = (
    tail_train_df[TARGET]
    .to_numpy()
)

X_tail_validation = prepare_features(
    tail_validation_df
)

y_tail_validation = (
    tail_validation_df[TARGET]
    .to_numpy()
)


# ============================================================
# 13. SHOW VALIDATION LOS DISTRIBUTION
# ============================================================

print_section(
    "TAIL VALIDATION DISTRIBUTION"
)

validation_masks = {
    "0-24h":
        y_tail_validation < 24,

    "24-72h":
        (
            (y_tail_validation >= 24)
            &
            (y_tail_validation < 72)
        ),

    "72-168h":
        (
            (y_tail_validation >= 72)
            &
            (y_tail_validation < 168)
        ),

    ">168h":
        y_tail_validation >= 168,
}

for group, mask in validation_masks.items():

    print(
        f"{group:10s}: "
        f"{int(mask.sum()):,} rows"
    )


# ============================================================
# 14. TAIL-WEIGHT EXPERIMENT
# ============================================================

print_section(
    "TAIL-WEIGHT EXPERIMENT"
)

print(
    f"Testing {len(WEIGHT_SCHEMES)} "
    f"weight schemes."
)

print(
    "\nThe validation process will evaluate:"
)

print(
    "1. Overall P50 MAE"
)

print(
    "2. >168h P50 MAE"
)

print(
    "3. 72-168h P50 MAE"
)

print(
    "4. P10 pinball loss"
)

print(
    "5. P90 pinball loss"
)

print(
    "\nThe final test set is NOT used "
    "during this selection."
)


tail_tuning_rows = []

best_scheme_name = None
best_scheme_score = np.inf
best_scheme_details = None


# ------------------------------------------------------------
# Selection philosophy
# ------------------------------------------------------------
#
# For each scheme we calculate:
#
# overall MAE
# >168h MAE
# 72-168h MAE
#
# We normalize tail errors relative to the unweighted
# reference model.
#
# Score:
#
#   40% overall MAE
#   40% >168h MAE
#   20% 72-168h MAE
#
# Lower is better.
#
# This prevents the model from becoming excellent on only
# the extreme tail while becoming much worse everywhere else.
#
# P10/P90 are evaluated separately using pinball loss.
#
# The selected scheme is therefore tail-aware but still
# constrained by overall performance.


# ============================================================
# 15. TRAIN REFERENCE MODEL FIRST
# ============================================================

print_section(
    "TRAINING UNWEIGHTED REFERENCE"
)

reference_predictions = {}

reference_models = {}

reference_weights = np.ones(
    len(y_tail_train),
    dtype=float,
)

for quantile_name, alpha in QUANTILES.items():

    print(
        f"\nTraining reference "
        f"{quantile_name.upper()}..."
    )

    model = create_quantile_model(
        alpha
    )

    model.fit(
        X_tail_train,
        y_tail_train,
        sample_weight=reference_weights,
        categorical_feature=CATEGORICAL_FEATURES,
    )

    prediction = model.predict(
        X_tail_validation
    )

    reference_models[
        quantile_name
    ] = model

    reference_predictions[
        quantile_name
    ] = prediction


reference_p50 = (
    reference_predictions["p50"]
)

reference_p10 = (
    reference_predictions["p10"]
)

reference_p90 = (
    reference_predictions["p90"]
)


reference_overall_mae = (
    mean_absolute_error(
        y_tail_validation,
        reference_p50,
    )
)

reference_tail_results = tail_metrics(
    y_tail_validation,
    reference_p50,
)

reference_tail_mae_72_168 = (
    reference_tail_results[
        "72_168h"
    ]["MAE_hours"]
)

reference_tail_mae_over_168 = (
    reference_tail_results[
        "over_168h"
    ]["MAE_hours"]
)

reference_p10_loss = (
    mean_pinball_loss(
        y_tail_validation,
        reference_p10,
        alpha=0.10,
    )
)

reference_p90_loss = (
    mean_pinball_loss(
        y_tail_validation,
        reference_p90,
        alpha=0.90,
    )
)


print(
    f"\nReference overall MAE: "
    f"{reference_overall_mae:.4f} h"
)

print(
    f"Reference 72-168h MAE: "
    f"{reference_tail_mae_72_168:.4f} h"
)

print(
    f"Reference >168h MAE: "
    f"{reference_tail_mae_over_168:.4f} h"
)

print(
    f"Reference P10 pinball: "
    f"{reference_p10_loss:.4f}"
)

print(
    f"Reference P90 pinball: "
    f"{reference_p90_loss:.4f}"
)


# ============================================================
# 16. TEST EACH WEIGHT SCHEME
# ============================================================

for scheme_name, scheme in WEIGHT_SCHEMES.items():

    # We already trained the reference model.
    #
    # Do not train it twice.

    if scheme_name == "unweighted_reference":

        p10 = reference_p10
        p50 = reference_p50
        p90 = reference_p90

    else:

        print()
        print(
            "-" * 70
        )

        print(
            f"Testing weight scheme: "
            f"{scheme_name}"
        )

        print(
            f"0-24h       = "
            f"{scheme['0_24h']:.2f}"
        )

        print(
            f"24-72h      = "
            f"{scheme['24_72h']:.2f}"
        )

        print(
            f"72-168h     = "
            f"{scheme['72_168h']:.2f}"
        )

        print(
            f">168h       = "
            f"{scheme['over_168h']:.2f}"
        )

        sample_weights = (
            create_tail_weights(
                y_tail_train,
                scheme,
            )
        )

        print_weight_distribution(
            y_tail_train,
            sample_weights,
        )

        scheme_predictions = {}

        for quantile_name, alpha in QUANTILES.items():

            print(
                f"\nTraining "
                f"{quantile_name.upper()}..."
            )

            model = create_quantile_model(
                alpha
            )

            model.fit(
                X_tail_train,
                y_tail_train,
                sample_weight=sample_weights,
                categorical_feature=CATEGORICAL_FEATURES,
            )

            scheme_predictions[
                quantile_name
            ] = model.predict(
                X_tail_validation
            )

        p10 = scheme_predictions[
            "p10"
        ]

        p50 = scheme_predictions[
            "p50"
        ]

        p90 = scheme_predictions[
            "p90"
        ]


    # --------------------------------------------------------
    # Calculate validation metrics
    # --------------------------------------------------------

    overall_mae = (
        mean_absolute_error(
            y_tail_validation,
            p50,
        )
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            y_tail_validation,
            p50,
        )
    )

    overall_r2 = (
        r2_score(
            y_tail_validation,
            p50,
        )
    )

    tail_result = tail_metrics(
        y_tail_validation,
        p50,
    )

    mae_72_168 = tail_result[
        "72_168h"
    ]["MAE_hours"]

    mae_over_168 = tail_result[
        "over_168h"
    ]["MAE_hours"]

    p10_pinball = (
        mean_pinball_loss(
            y_tail_validation,
            p10,
            alpha=0.10,
        )
    )

    p90_pinball = (
        mean_pinball_loss(
            y_tail_validation,
            p90,
            alpha=0.90,
        )
    )

    # --------------------------------------------------------
    # Tail-aware selection score
    # --------------------------------------------------------
    #
    # Normalize against reference.
    #
    # A score below 1.0 means improvement over reference.
    #
    # 40% overall MAE
    # 40% >168h MAE
    # 20% 72-168h MAE

    normalized_overall = (
        overall_mae
        / reference_overall_mae
    )

    normalized_over_168 = (
        mae_over_168
        / reference_tail_mae_over_168
    )

    normalized_72_168 = (
        mae_72_168
        / reference_tail_mae_72_168
    )

    selection_score = (
        0.40 * normalized_overall
        + 0.40 * normalized_over_168
        + 0.20 * normalized_72_168
    )

    tail_improvement_percent = (
        (
            reference_tail_mae_over_168
            - mae_over_168
        )
        / reference_tail_mae_over_168
    ) * 100

    overall_change_percent = (
        (
            overall_mae
            - reference_overall_mae
        )
        / reference_overall_mae
    ) * 100

    print(
        f"\nRESULT - {scheme_name}"
    )

    print(
        f"Overall MAE: "
        f"{overall_mae:.4f} h"
    )

    print(
        f"72-168h MAE: "
        f"{mae_72_168:.4f} h"
    )

    print(
        f">168h MAE: "
        f"{mae_over_168:.4f} h"
    )

    print(
        f"R²: "
        f"{overall_r2:.4f}"
    )

    print(
        f"P10 Pinball: "
        f"{p10_pinball:.4f}"
    )

    print(
        f"P90 Pinball: "
        f"{p90_pinball:.4f}"
    )

    print(
        f">168h improvement vs reference: "
        f"{tail_improvement_percent:+.2f}%"
    )

    print(
        f"Overall MAE change vs reference: "
        f"{overall_change_percent:+.2f}%"
    )

    print(
        f"Tail-aware selection score: "
        f"{selection_score:.6f}"
    )

    tail_tuning_rows.append({
        "weight_scheme": scheme_name,

        "weight_0_24h":
            scheme["0_24h"],

        "weight_24_72h":
            scheme["24_72h"],

        "weight_72_168h":
            scheme["72_168h"],

        "weight_over_168h":
            scheme["over_168h"],

        "overall_MAE_hours":
            float(overall_mae),

        "overall_RMSE_hours":
            float(overall_rmse),

        "overall_R2":
            float(overall_r2),

        "MAE_72_168h":
            float(mae_72_168),

        "MAE_over_168h":
            float(mae_over_168),

        "P10_pinball":
            float(p10_pinball),

        "P90_pinball":
            float(p90_pinball),

        "tail_improvement_over_168_percent":
            float(
                tail_improvement_percent
            ),

        "overall_MAE_change_percent":
            float(
                overall_change_percent
            ),

        "tail_aware_selection_score":
            float(selection_score),
    })

    if (
        selection_score
        < best_scheme_score
    ):

        best_scheme_score = (
            selection_score
        )

        best_scheme_name = (
            scheme_name
        )

        best_scheme_details = {
            "scheme":
                scheme.copy(),

            "overall_MAE_hours":
                float(overall_mae),

            "overall_RMSE_hours":
                float(overall_rmse),

            "overall_R2":
                float(overall_r2),

            "MAE_72_168h":
                float(mae_72_168),

            "MAE_over_168h":
                float(mae_over_168),

            "P10_pinball":
                float(p10_pinball),

            "P90_pinball":
                float(p90_pinball),

            "selection_score":
                float(selection_score),
        }


# ============================================================
# 17. SAVE TAIL-WEIGHT SEARCH RESULTS
# ============================================================

tail_tuning_df = pd.DataFrame(
    tail_tuning_rows
)

tail_tuning_path = (
    OUTPUT_DIR
    / "los_tail_weight_experiment_results.csv"
)

tail_tuning_df.to_csv(
    tail_tuning_path,
    index=False,
)


# ============================================================
# 18. DISPLAY BEST SCHEME
# ============================================================

print_section(
    "BEST TAIL-WEIGHT SCHEME"
)

print(
    f"Selected scheme: "
    f"{best_scheme_name}"
)

print(
    f"Selection score: "
    f"{best_scheme_score:.6f}"
)

print(
    "\nSelected weights:"
)

for key, value in (
    best_scheme_details[
        "scheme"
    ].items()
):

    print(
        f"{key:15s}: "
        f"{value:.2f}"
    )

print(
    "\nValidation performance:"
)

print(
    f"Overall MAE: "
    f"{best_scheme_details['overall_MAE_hours']:.4f} h"
)

print(
    f"72-168h MAE: "
    f"{best_scheme_details['MAE_72_168h']:.4f} h"
)

print(
    f">168h MAE: "
    f"{best_scheme_details['MAE_over_168h']:.4f} h"
)

print(
    f"R²: "
    f"{best_scheme_details['overall_R2']:.4f}"
)


# ============================================================
# 19. TRAIN FINAL MODELS ON FULL 70K TRAINING DATA
# ============================================================

print_section(
    "TRAINING FINAL TAIL-WEIGHTED MODELS"
)

X_train = prepare_features(
    train_df
)

y_train = (
    train_df[TARGET]
    .to_numpy()
)

X_calibration = prepare_features(
    calibration_df
)

y_calibration = (
    calibration_df[TARGET]
    .to_numpy()
)

X_test = prepare_features(
    test_df
)

y_test = (
    test_df[TARGET]
    .to_numpy()
)


# Create final training weights.

final_weights = (
    create_tail_weights(
        y_train,
        best_scheme_details[
            "scheme"
        ],
    )
)

print_weight_distribution(
    y_train,
    final_weights,
)


# ------------------------------------------------------------
# Train P10/P50/P90
# ------------------------------------------------------------

final_models = {}

for quantile_name, alpha in QUANTILES.items():

    print(
        f"\nTraining final "
        f"{quantile_name.upper()} "
        f"model..."
    )

    model = create_quantile_model(
        alpha
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=final_weights,
        categorical_feature=CATEGORICAL_FEATURES,
    )

    final_models[
        quantile_name
    ] = model

    model_path = (
        MODEL_DIR
        / (
            "los_lightgbm_tail_weighted_"
            f"{quantile_name}.joblib"
        )
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"Saved: "
        f"{model_path.name}"
    )


# ============================================================
# 20. CALIBRATION PREDICTIONS
# ============================================================

print_section(
    "RAW QUANTILE PREDICTIONS"
)

p10_cal = (
    final_models["p10"]
    .predict(
        X_calibration
    )
)

p50_cal = (
    final_models["p50"]
    .predict(
        X_calibration
    )
)

p90_cal = (
    final_models["p90"]
    .predict(
        X_calibration
    )
)


# ============================================================
# 21. TEST PREDICTIONS
# ============================================================

p10_test = (
    final_models["p10"]
    .predict(
        X_test
    )
)

p50_test = (
    final_models["p50"]
    .predict(
        X_test
    )
)

p90_test = (
    final_models["p90"]
    .predict(
        X_test
    )
)


# ============================================================
# 22. RAW QUANTILE ORDERING
# ============================================================

cal_ordering = (
    (p10_cal <= p50_cal)
    &
    (p50_cal <= p90_cal)
)

test_ordering = (
    (p10_test <= p50_test)
    &
    (p50_test <= p90_test)
)

print(
    f"Calibration raw ordering validity: "
    f"{np.mean(cal_ordering) * 100:.3f}%"
)

print(
    f"Test raw ordering validity: "
    f"{np.mean(test_ordering) * 100:.3f}%"
)


# ============================================================
# 23. RAW TEST METRICS
# ============================================================

raw_point = point_metrics(
    y_test,
    p50_test,
)

raw_p10_pinball = (
    mean_pinball_loss(
        y_test,
        p10_test,
        alpha=0.10,
    )
)

raw_p90_pinball = (
    mean_pinball_loss(
        y_test,
        p90_test,
        alpha=0.90,
    )
)

print_section(
    "RAW TAIL-WEIGHTED TEST METRICS"
)

print(
    f"P50 MAE: "
    f"{raw_point['MAE_hours']:.3f} hours"
)

print(
    f"P50 RMSE: "
    f"{raw_point['RMSE_hours']:.3f} hours"
)

print(
    f"P50 R²: "
    f"{raw_point['R2']:.4f}"
)

print(
    f"P10 Pinball: "
    f"{raw_p10_pinball:.3f}"
)

print(
    f"P90 Pinball: "
    f"{raw_p90_pinball:.3f}"
)


# ============================================================
# 24. MAPIE CQR
# ============================================================

print_section(
    "MAPIE CONFORMAL CALIBRATION"
)

print(
    "Creating MAPIE CQR model:"
)

print(
    "P10 -> lower quantile"
)

print(
    "P50 -> median"
)

print(
    "P90 -> upper quantile"
)

print(
    "Confidence level = 0.80"
)


mapie_model = (
    ConformalizedQuantileRegressor(
        estimator=[
            final_models["p10"],
            final_models["p90"],
            final_models["p50"],
        ],
        confidence_level=MAPIE_CONFIDENCE_LEVEL,
        prefit=True,
    )
)


print(
    "\nCalibrating MAPIE on the "
    "15,000-row calibration dataset..."
)

mapie_model.conformalize(
    X_calibration,
    y_calibration,
)

print(
    "MAPIE calibration completed."
)


# ============================================================
# 25. MAPIE TEST PREDICTIONS
# ============================================================

print_section(
    "FINAL MAPIE TEST PREDICTIONS"
)

mapie_result = (
    mapie_model
    .predict_interval(
        X_test
    )
)


# MAPIE 1.5.0 returns:
#
#   y_pred, y_interval
#
# y_interval can be represented as:
#
#   (n_samples, 2, 1)
#
# We handle both 2D and 3D safely.

mapie_prediction, mapie_interval = (
    mapie_result
)

mapie_interval = np.asarray(
    mapie_interval
)


if mapie_interval.ndim == 3:

    lower = (
        mapie_interval[
            :, 0, 0
        ]
    )

    upper = (
        mapie_interval[
            :, 1, 0
        ]
    )

elif mapie_interval.ndim == 2:

    lower = (
        mapie_interval[
            :, 0
        ]
    )

    upper = (
        mapie_interval[
            :, 1
        ]
    )

else:

    raise ValueError(
        "Unexpected MAPIE interval shape: "
        f"{mapie_interval.shape}"
    )


# P50 is the final point prediction.

median_prediction = (
    p50_test
)


# ============================================================
# 26. FINAL POINT METRICS
# ============================================================

final_point = point_metrics(
    y_test,
    median_prediction,
)


# ============================================================
# 27. FINAL INTERVAL METRICS
# ============================================================

final_interval = interval_metrics(
    y_test,
    lower,
    upper,
)


print_section(
    "FINAL TAIL-WEIGHTED MODEL - TEST RESULTS"
)

print(
    f"P50 MAE: "
    f"{final_point['MAE_hours']:.3f} hours"
)

print(
    f"P50 RMSE: "
    f"{final_point['RMSE_hours']:.3f} hours"
)

print(
    f"R²: "
    f"{final_point['R2']:.4f}"
)

print(
    f"Median Absolute Error: "
    f"{final_point['Median_Absolute_Error_hours']:.3f} hours"
)

print(
    f"Mean Prediction Error: "
    f"{final_point['Mean_Prediction_Error_hours']:.3f} hours"
)

print(
    f"Underprediction: "
    f"{final_point['Underprediction_percent']:.3f}%"
)

print(
    f"Overprediction: "
    f"{final_point['Overprediction_percent']:.3f}%"
)

print(
    f"80% Coverage: "
    f"{final_interval['Coverage_percent']:.3f}%"
)

print(
    f"Mean Interval Width: "
    f"{final_interval['Mean_Interval_Width_hours']:.3f} hours"
)

print(
    f"Median Interval Width: "
    f"{final_interval['Median_Interval_Width_hours']:.3f} hours"
)

print(
    f"90th Percentile Interval Width: "
    f"{final_interval['P90_Interval_Width_hours']:.3f} hours"
)

print(
    f"Interval Ordering Validity: "
    f"{final_interval['Interval_Ordering_Validity_percent']:.3f}%"
)


# ============================================================
# 28. FINAL LOS-RANGE PERFORMANCE
# ============================================================

print_section(
    "PERFORMANCE BY ACTUAL LOS RANGE"
)

final_tail_results = tail_metrics(
    y_test,
    median_prediction,
)

final_tail_intervals = (
    interval_tail_metrics(
        y_test,
        lower,
        upper,
    )
)


tail_output_rows = []


for group in [
    "0_24h",
    "24_72h",
    "72_168h",
    "over_168h",
]:

    if group not in final_tail_results:
        continue

    point_result = (
        final_tail_results[
            group
        ]
    )

    interval_result = (
        final_tail_intervals.get(
            group,
            {},
        )
    )

    tail_output_rows.append({

        "LOS_Range":
            group,

        "Count":
            point_result[
                "count"
            ],

        "Actual_Mean_hours":
            point_result[
                "actual_mean_hours"
            ],

        "Predicted_Mean_hours":
            point_result[
                "predicted_mean_hours"
            ],

        "MAE_hours":
            point_result[
                "MAE_hours"
            ],

        "RMSE_hours":
            point_result[
                "RMSE_hours"
            ],

        "Bias_hours":
            point_result[
                "bias_hours"
            ],

        "Underprediction_percent":
            point_result[
                "underprediction_percent"
            ],

        "Coverage_percent":
            interval_result.get(
                "coverage_percent",
                np.nan,
            ),

        "Mean_Interval_Width_hours":
            interval_result.get(
                "mean_width_hours",
                np.nan,
            ),
    })


tail_output_df = pd.DataFrame(
    tail_output_rows
)


print(
    tail_output_df.to_string(
        index=False
    )
)


tail_output_path = (
    OUTPUT_DIR
    / "los_tail_weighted_by_los_range.csv"
)

tail_output_df.to_csv(
    tail_output_path,
    index=False,
)


# ============================================================
# 29. COMPARE AGAINST BASELINE VALUES
# ============================================================
#
# These are the previously established baseline test metrics.
#
# They are included here ONLY for comparison.
#
# They are NOT used for training or model selection.

BASELINE_RESULTS = {

    "MAE_hours":
        30.970,

    "RMSE_hours":
        52.166,

    "R2":
        0.2142,

    "Coverage_percent":
        79.653,

    "Mean_Interval_Width_hours":
        97.396,

    "MAE_over_168h":
        163.866,

    "Coverage_over_168h":
        22.90,
}


# Extract final >168h metrics.

final_over_168 = (
    final_tail_results[
        "over_168h"
    ]
)

final_over_168_interval = (
    final_tail_intervals[
        "over_168h"
    ]
)


comparison_rows = [

    {
        "Metric":
            "Overall MAE (hours)",

        "Baseline":
            BASELINE_RESULTS[
                "MAE_hours"
            ],

        "Tail_Weighted":
            final_point[
                "MAE_hours"
            ],
    },

    {
        "Metric":
            "Overall RMSE (hours)",

        "Baseline":
            BASELINE_RESULTS[
                "RMSE_hours"
            ],

        "Tail_Weighted":
            final_point[
                "RMSE_hours"
            ],
    },

    {
        "Metric":
            "R2",

        "Baseline":
            BASELINE_RESULTS[
                "R2"
            ],

        "Tail_Weighted":
            final_point[
                "R2"
            ],
    },

    {
        "Metric":
            "80% Coverage (%)",

        "Baseline":
            BASELINE_RESULTS[
                "Coverage_percent"
            ],

        "Tail_Weighted":
            final_interval[
                "Coverage_percent"
            ],
    },

    {
        "Metric":
            "Mean Interval Width (hours)",

        "Baseline":
            BASELINE_RESULTS[
                "Mean_Interval_Width_hours"
            ],

        "Tail_Weighted":
            final_interval[
                "Mean_Interval_Width_hours"
            ],
    },

    {
        "Metric":
            ">168h MAE (hours)",

        "Baseline":
            BASELINE_RESULTS[
                "MAE_over_168h"
            ],

        "Tail_Weighted":
            final_over_168[
                "MAE_hours"
            ],
    },

    {
        "Metric":
            ">168h Coverage (%)",

        "Baseline":
            BASELINE_RESULTS[
                "Coverage_over_168h"
            ],

        "Tail_Weighted":
            final_over_168_interval[
                "coverage_percent"
            ],
    },
]


comparison_df = pd.DataFrame(
    comparison_rows
)


print_section(
    "BASELINE VS TAIL-WEIGHTED"
)

print(
    comparison_df.to_string(
        index=False
    )
)


comparison_path = (
    OUTPUT_DIR
    / "los_baseline_vs_tail_weighted.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False,
)


# ============================================================
# 30. SAVE TEST PREDICTIONS
# ============================================================

print_section(
    "SAVING TEST PREDICTIONS"
)

prediction_output = pd.DataFrame({

    "stay_id":
        test_df[
            "stay_id"
        ].values,

    "patient_id":
        test_df[
            "patient_id"
        ].values,

    "arrival_time":
        test_df[
            "arrival_time"
        ].values,

    "triage_acuity":
        test_df[
            "triage_acuity"
        ].values,

    "actual_los_hours":
        y_test,

    "predicted_los_p50_hours":
        median_prediction,

    "predicted_los_p10_raw_hours":
        p10_test,

    "predicted_los_p90_raw_hours":
        p90_test,

    "conformal_lower_hours":
        lower,

    "conformal_upper_hours":
        upper,

    "prediction_error_hours":
        (
            median_prediction
            - y_test
        ),
})


prediction_path = (
    OUTPUT_DIR
    / "los_predictions_test_tail_weighted.csv"
)

prediction_output.to_csv(
    prediction_path,
    index=False,
)

print(
    f"Saved: "
    f"{prediction_path.name}"
)

print(
    f"Rows saved: "
    f"{len(prediction_output):,}"
)


# ============================================================
# 31. SAVE MAPIE MODEL
# ============================================================

mapie_model_path = (
    MODEL_DIR
    / "los_mapie_cqr_tail_weighted.joblib"
)

joblib.dump(
    mapie_model,
    mapie_model_path,
)

print(
    f"Saved: "
    f"{mapie_model_path.name}"
)


# ============================================================
# 32. SAVE MODEL CONFIGURATION
# ============================================================

configuration = {

    "experiment":
        "Tail-weighted LightGBM Quantile Regression + MAPIE CQR",

    "target":
        TARGET,

    "features":
        FEATURES,

    "numeric_features":
        NUMERIC_FEATURES,

    "categorical_features":
        CATEGORICAL_FEATURES,

    "train_size":
        TRAIN_SIZE,

    "calibration_size":
        CALIBRATION_SIZE,

    "test_size":
        TEST_SIZE,

    "internal_validation_size":
        TUNING_VALIDATION_SIZE,

    "quantiles":
        QUANTILES,

    "mapie_confidence_level":
        MAPIE_CONFIDENCE_LEVEL,

    "base_lightgbm_parameters":
        BASE_LGB_PARAMS,

    "all_weight_schemes":
        WEIGHT_SCHEMES,

    "selected_weight_scheme":
        best_scheme_name,

    "selected_weights":
        best_scheme_details[
            "scheme"
        ],

    "selection_score":
        best_scheme_score,

    "selection_method":
        {
            "overall_MAE_weight":
                0.40,

            "over_168h_MAE_weight":
                0.40,

            "72_168h_MAE_weight":
                0.20,

            "description":
                (
                    "Tail-aware validation score "
                    "based on normalized MAE relative "
                    "to the unweighted reference."
                ),
        },

    "final_test_metrics":
        {
            **final_point,
            **final_interval,
        },

    "final_over_168h_metrics":
        final_over_168,

    "final_over_168h_interval_metrics":
        final_over_168_interval,

    "baseline_comparison":
        BASELINE_RESULTS,
}


configuration_path = (
    OUTPUT_DIR
    / "los_tail_weighted_configuration.json"
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
    f"Saved: "
    f"{configuration_path.name}"
)


# ============================================================
# 33. SAVE FINAL SUMMARY
# ============================================================

summary = {

    "model":
        "Tail-Weighted LightGBM "
        "Quantile Regression + MAPIE CQR",

    "selected_weight_scheme":
        best_scheme_name,

    "selected_weights":
        best_scheme_details[
            "scheme"
        ],

    "test_rows":
        int(len(test_df)),

    "MAE_hours":
        final_point[
            "MAE_hours"
        ],

    "RMSE_hours":
        final_point[
            "RMSE_hours"
        ],

    "R2":
        final_point[
            "R2"
        ],

    "Coverage_percent":
        final_interval[
            "Coverage_percent"
        ],

    "Mean_Interval_Width_hours":
        final_interval[
            "Mean_Interval_Width_hours"
        ],

    "Median_Interval_Width_hours":
        final_interval[
            "Median_Interval_Width_hours"
        ],

    "P90_Interval_Width_hours":
        final_interval[
            "P90_Interval_Width_hours"
        ],

    "Interval_Ordering_Validity_percent":
        final_interval[
            "Interval_Ordering_Validity_percent"
        ],

    "MAE_over_168h":
        final_over_168[
            "MAE_hours"
        ],

    "Coverage_over_168h":
        final_over_168_interval[
            "coverage_percent"
        ],

    "Predicted_Mean_over_168h":
        final_over_168[
            "predicted_mean_hours"
        ],

    "Actual_Mean_over_168h":
        final_over_168[
            "actual_mean_hours"
        ],
}


summary_path = (
    OUTPUT_DIR
    / "los_tail_weighted_metrics.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        summary,
        file,
        indent=4,
    )


print(
    f"Saved: "
    f"{summary_path.name}"
)


# ============================================================
# 34. FINAL INTERPRETATION
# ============================================================

print_section(
    "TAIL-WEIGHTED LOS MODEL COMPLETE"
)

print(
    "Training completed successfully."
)

print(
    "\nArchitecture:"
)

print(
    "Tail-weighted LightGBM P10/P50/P90 "
    "-> MAPIE CQR"
)

print(
    "\nData split:"
)

print(
    "70,000 train | "
    "15,000 calibration | "
    "15,000 test"
)

print(
    "\nSelected weight scheme:"
)

print(
    best_scheme_name
)

print(
    "\nFINAL TEST:"
)

print(
    f"MAE       = "
    f"{final_point['MAE_hours']:.3f} h"
)

print(
    f"RMSE      = "
    f"{final_point['RMSE_hours']:.3f} h"
)

print(
    f"R²        = "
    f"{final_point['R2']:.4f}"
)

print(
    f"Coverage  = "
    f"{final_interval['Coverage_percent']:.3f}%"
)

print(
    f"Mean Width = "
    f"{final_interval['Mean_Interval_Width_hours']:.3f} h"
)

print(
    "\n>168h PERFORMANCE:"
)

print(
    f"Actual mean LOS = "
    f"{final_over_168['actual_mean_hours']:.3f} h"
)

print(
    f"Predicted mean LOS = "
    f"{final_over_168['predicted_mean_hours']:.3f} h"
)

print(
    f">168h MAE = "
    f"{final_over_168['MAE_hours']:.3f} h"
)

print(
    f">168h Coverage = "
    f"{final_over_168_interval['coverage_percent']:.3f}%"
)

print(
    "\nMain output:"
)

print(
    prediction_path
)

print(
    "\nComparison output:"
)

print(
    comparison_path
)

print(
    "\nDONE."
)
