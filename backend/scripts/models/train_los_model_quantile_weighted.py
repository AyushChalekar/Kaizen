# ============================================================
# LOS Prediction - Quantile-Specific Tail-Weighted LightGBM
#                     + MAPIE Conformal Prediction
#
# Kaizen Healthcare Predictive Resource Optimization System
#
# EXPERIMENT:
#   Different LOS weights for P10, P50 and P90.
#
# MAIN IDEA:
#   P50 -> moderate emphasis on long stays
#   P90 -> stronger emphasis on long stays
#   P10 -> lighter weighting
#
# QUANTILES:
#   P10 = 0.10
#   P50 = 0.50
#   P90 = 0.90
#
# UNCERTAINTY:
#   MAPIE Conformalized Quantile Regression
#   Confidence = 80%
#
# DATA:
#   70,000 training
#   15,000 calibration
#   15,000 final test
#
# IMPORTANT:
#   Existing baseline/tuned/tail-weighted files are NOT
#   overwritten.
# ============================================================


import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_pinball_loss,
)

from mapie.regression import (
    ConformalizedQuantileRegressor,
)

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
# 2. GENERAL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TRAIN_SIZE = 70_000
CALIBRATION_SIZE = 15_000
TEST_SIZE = 15_000

INTERNAL_VALIDATION_SIZE = 10_000

MAPIE_CONFIDENCE_LEVEL = 0.80


# ============================================================
# 3. TARGET
# ============================================================

TARGET = "los_hours"


# ============================================================
# 4. STRICT ARRIVAL/Triage FEATURES
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

FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ============================================================
# 5. QUANTILES
# ============================================================

QUANTILES = {
    "p10": 0.10,
    "p50": 0.50,
    "p90": 0.90,
}


# ============================================================
# 6. LIGHTGBM PARAMETERS
# ============================================================
#
# The previous hyperparameter experiment showed that the
# conservative configuration was the strongest configuration
# tested.
#
# Therefore this experiment focuses on the more meaningful
# change: quantile-specific weighting.

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
# 7. QUANTILE-SPECIFIC WEIGHT SCHEMES
# ============================================================
#
# Each scheme contains separate weights for:
#
#   P10
#   P50
#   P90
#
# and four LOS ranges:
#
#   0-24h
#   24-72h
#   72-168h
#   >168h
#
# P90 receives stronger tail emphasis because the upper
# quantile should learn the upper LOS distribution more
# strongly.
#
# A few schemes are tested rather than blindly selecting
# one arbitrary weighting level.
#
# Scheme 1 = balanced
# Scheme 2 = moderate
# Scheme 3 = strong P90
# Scheme 4 = strong P50 + P90

WEIGHT_SCHEMES = {

    "balanced_quantile_tail": {

        "p10": {
            "0_24h": 1.00,
            "24_72h": 1.05,
            "72_168h": 1.15,
            "over_168h": 1.25,
        },

        "p50": {
            "0_24h": 1.00,
            "24_72h": 1.15,
            "72_168h": 1.50,
            "over_168h": 2.00,
        },

        "p90": {
            "0_24h": 1.00,
            "24_72h": 1.20,
            "72_168h": 1.70,
            "over_168h": 2.75,
        },
    },


    "moderate_quantile_tail": {

        "p10": {
            "0_24h": 1.00,
            "24_72h": 1.05,
            "72_168h": 1.20,
            "over_168h": 1.40,
        },

        "p50": {
            "0_24h": 1.00,
            "24_72h": 1.20,
            "72_168h": 1.60,
            "over_168h": 2.25,
        },

        "p90": {
            "0_24h": 1.00,
            "24_72h": 1.25,
            "72_168h": 1.90,
            "over_168h": 3.25,
        },
    },


    "strong_p90_tail": {

        "p10": {
            "0_24h": 1.00,
            "24_72h": 1.05,
            "72_168h": 1.15,
            "over_168h": 1.30,
        },

        "p50": {
            "0_24h": 1.00,
            "24_72h": 1.15,
            "72_168h": 1.55,
            "over_168h": 2.10,
        },

        "p90": {
            "0_24h": 1.00,
            "24_72h": 1.30,
            "72_168h": 2.10,
            "over_168h": 3.75,
        },
    },


    "strong_p50_p90_tail": {

        "p10": {
            "0_24h": 1.00,
            "24_72h": 1.05,
            "72_168h": 1.20,
            "over_168h": 1.35,
        },

        "p50": {
            "0_24h": 1.00,
            "24_72h": 1.25,
            "72_168h": 1.80,
            "over_168h": 2.50,
        },

        "p90": {
            "0_24h": 1.00,
            "24_72h": 1.30,
            "72_168h": 2.20,
            "over_168h": 4.00,
        },
    },
}


# ============================================================
# 8. BASELINE RESULTS FOR FINAL COMPARISON
# ============================================================

BASELINE_RESULTS = {

    "MAE_hours": 30.970,

    "RMSE_hours": 52.166,

    "R2": 0.2142,

    "Coverage_percent": 79.653,

    "Mean_Interval_Width_hours": 97.396,

    "MAE_over_168h": 163.866,

    "Coverage_over_168h": 22.90,
}


# ============================================================
# 9. HELPER: SECTION
# ============================================================

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 10. PREPARE FEATURES
# ============================================================

def prepare_features(df):

    X = df[
        FEATURES
    ].copy()

    for column in CATEGORICAL_FEATURES:

        X[column] = (
            X[column]
            .astype("category")
        )

    return X


# ============================================================
# 11. CREATE LIGHTGBM MODEL
# ============================================================

def create_quantile_model(alpha):

    params = (
        BASE_LGB_PARAMS.copy()
    )

    params.update({

        "objective": "quantile",

        "alpha": alpha,

        "random_state":
            RANDOM_STATE,

        "n_jobs": -1,

        "verbosity": -1,
    })

    return lgb.LGBMRegressor(
        **params
    )


# ============================================================
# 12. CREATE QUANTILE-SPECIFIC SAMPLE WEIGHTS
# ============================================================

def create_quantile_weights(
    y,
    scheme,
    quantile_name,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    weights = np.ones(
        len(y),
        dtype=float,
    )

    q_weights = (
        scheme[
            quantile_name
        ]
    )

    mask_0_24 = (
        y < 24
    )

    mask_24_72 = (
        (y >= 24)
        &
        (y < 72)
    )

    mask_72_168 = (
        (y >= 72)
        &
        (y < 168)
    )

    mask_over_168 = (
        y >= 168
    )

    weights[
        mask_0_24
    ] = q_weights[
        "0_24h"
    ]

    weights[
        mask_24_72
    ] = q_weights[
        "24_72h"
    ]

    weights[
        mask_72_168
    ] = q_weights[
        "72_168h"
    ]

    weights[
        mask_over_168
    ] = q_weights[
        "over_168h"
    ]

    return weights


# ============================================================
# 13. QUANTILE ORDERING WRAPPER
# ============================================================
#
# MAPIE expects three quantile estimators.
#
# Some independent quantile models can cross:
#
#   P10 > P50
#   P50 > P90
#
# The previous experiment showed this issue.
#
# This wrapper ensures that the three predictions are ordered:
#
#   P10 <= P50 <= P90
#
# BEFORE MAPIE sees them.
#
# For each prediction call, all three underlying LightGBM
# models are evaluated and then sorted.
#
# role:
#   "lower"  -> minimum
#   "median" -> median
#   "upper"  -> maximum
#
# This wrapper does NOT change the trained LightGBM models.
# It only guarantees valid quantile ordering at prediction
# time.

class QuantileOrderingWrapper(
    BaseEstimator,
    RegressorMixin,
):

    def __init__(
        self,
        p10_model,
        p50_model,
        p90_model,
        role,
    ):

        self.p10_model = (
            p10_model
        )

        self.p50_model = (
            p50_model
        )

        self.p90_model = (
            p90_model
        )

        self.role = role


    def fit(
        self,
        X,
        y=None,
    ):

        # Models are already fitted.
        #
        # This method exists for sklearn compatibility.

        self.is_fitted_ = True

        return self


    def predict(self, X):

        p10 = np.asarray(
            self.p10_model.predict(X)
        )

        p50 = np.asarray(
            self.p50_model.predict(X)
        )

        p90 = np.asarray(
            self.p90_model.predict(X)
        )

        stacked = np.column_stack(
            [
                p10,
                p50,
                p90,
            ]
        )

        sorted_predictions = (
            np.sort(
                stacked,
                axis=1,
            )
        )

        if self.role == "lower":

            return (
                sorted_predictions[:, 0]
            )

        if self.role == "median":

            return (
                sorted_predictions[:, 1]
            )

        if self.role == "upper":

            return (
                sorted_predictions[:, 2]
            )

        raise ValueError(
            "role must be "
            "'lower', 'median' or 'upper'"
        )


# ============================================================
# 14. VALIDATE DATASET
# ============================================================

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

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df[TARGET].isna().any():

        raise ValueError(
            "LOS target contains missing values."
        )

    if (
        df[TARGET] < 0
    ).any():

        raise ValueError(
            "LOS contains negative values."
        )

    if (
        df[FEATURES]
        .isna()
        .any()
        .any()
    ):

        raise ValueError(
            "Features contain missing values."
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


# ============================================================
# 15. POINT METRICS
# ============================================================

def calculate_point_metrics(
    y_true,
    y_pred,
):

    mae = (
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = (
        r2_score(
            y_true,
            y_pred,
        )
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

        "MAE_hours":
            float(mae),

        "RMSE_hours":
            float(rmse),

        "R2":
            float(r2),

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


# ============================================================
# 16. LOS RANGE METRICS
# ============================================================

def calculate_los_range_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    ranges = {

        "0_24h": (
            y_true < 24
        ),

        "24_72h": (
            (y_true >= 24)
            &
            (y_true < 72)
        ),

        "72_168h": (
            (y_true >= 72)
            &
            (y_true < 168)
        ),

        "over_168h": (
            y_true >= 168
        ),
    }

    results = {}

    for name, mask in ranges.items():

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        actual = (
            y_true[mask]
        )

        predicted = (
            y_pred[mask]
        )

        results[name] = {

            "count":
                count,

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


# ============================================================
# 17. PINBALL METRICS
# ============================================================

def calculate_pinball_metrics(
    y_true,
    p10,
    p90,
):

    p10_loss = (
        mean_pinball_loss(
            y_true,
            p10,
            alpha=0.10,
        )
    )

    p90_loss = (
        mean_pinball_loss(
            y_true,
            p90,
            alpha=0.90,
        )
    )

    return (
        float(p10_loss),
        float(p90_loss),
    )


# ============================================================
# 18. INTERVAL METRICS
# ============================================================

def calculate_interval_metrics(
    y_true,
    lower,
    upper,
):

    # MAPIE metrics expect:
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

    ordering = (
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
                ordering
            ),
    }


# ============================================================
# 19. TAIL INTERVAL METRICS
# ============================================================

def calculate_tail_interval_metrics(
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

    ranges = {

        "0_24h": (
            y_true < 24
        ),

        "24_72h": (
            (y_true >= 24)
            &
            (y_true < 72)
        ),

        "72_168h": (
            (y_true >= 72)
            &
            (y_true < 168)
        ),

        "over_168h": (
            y_true >= 168
        ),
    }

    results = {}

    for name, mask in ranges.items():

        if mask.sum() == 0:
            continue

        actual = (
            y_true[mask]
        )

        low = (
            lower[mask]
        )

        high = (
            upper[mask]
        )

        coverage = (
            np.mean(
                (actual >= low)
                &
                (actual <= high)
            )
            * 100
        )

        results[name] = {

            "coverage_percent":
                float(
                    coverage
                ),

            "mean_width_hours":
                float(
                    np.mean(
                        high - low
                    )
                ),
        }

    return results


# ============================================================
# 20. LOAD DATA
# ============================================================

print_section(
    "QUANTILE-SPECIFIC TAIL-WEIGHTED LOS MODEL"
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
# 21. CHRONOLOGICAL ORDER
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
    f"Last arrival: "
    f"{df['arrival_time'].max()}"
)


# ============================================================
# 22. CHRONOLOGICAL SPLIT
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
        f"At least {required_rows:,} "
        f"rows are required."
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
    f"Training: "
    f"{len(train_df):,} rows"
)

print(
    f"Calibration: "
    f"{len(calibration_df):,} rows"
)

print(
    f"Test: "
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
# 23. INTERNAL VALIDATION SPLIT
# ============================================================

print_section(
    "INTERNAL VALIDATION SPLIT"
)

quantile_train_df = (
    train_df
    .iloc[
        :-INTERNAL_VALIDATION_SIZE
    ]
    .copy()
)

quantile_validation_df = (
    train_df
    .iloc[
        -INTERNAL_VALIDATION_SIZE:
    ]
    .copy()
)


print(
    f"Internal training: "
    f"{len(quantile_train_df):,}"
)

print(
    f"Internal validation: "
    f"{len(quantile_validation_df):,}"
)


# ============================================================
# 24. PREPARE INTERNAL DATA
# ============================================================

X_internal_train = (
    prepare_features(
        quantile_train_df
    )
)

y_internal_train = (
    quantile_train_df[
        TARGET
    ].to_numpy()
)

X_internal_validation = (
    prepare_features(
        quantile_validation_df
    )
)

y_internal_validation = (
    quantile_validation_df[
        TARGET
    ].to_numpy()
)


# ============================================================
# 25. INTERNAL VALIDATION DISTRIBUTION
# ============================================================

print_section(
    "INTERNAL VALIDATION LOS DISTRIBUTION"
)

validation_groups = {

    "0-24h":
        y_internal_validation < 24,

    "24-72h":
        (
            (y_internal_validation >= 24)
            &
            (y_internal_validation < 72)
        ),

    "72-168h":
        (
            (y_internal_validation >= 72)
            &
            (y_internal_validation < 168)
        ),

    ">168h":
        y_internal_validation >= 168,
}

for name, mask in (
    validation_groups.items()
):

    print(
        f"{name:10s}: "
        f"{int(mask.sum()):,} rows"
    )


# ============================================================
# 26. QUANTILE-SPECIFIC WEIGHT EXPERIMENT
# ============================================================

print_section(
    "QUANTILE-SPECIFIC WEIGHT EXPERIMENT"
)

print(
    f"Testing "
    f"{len(WEIGHT_SCHEMES)} "
    f"weighting strategies."
)

print(
    "\nEach strategy trains separate:"
)

print(
    "P10 model"
)

print(
    "P50 model"
)

print(
    "P90 model"
)

print(
    "\nThe final test set is NOT used "
    "for selecting the strategy."
)


experiment_results = []

best_scheme_name = None
best_selection_score = np.inf
best_scheme_details = None


# ============================================================
# 27. TEST EACH WEIGHT SCHEME
# ============================================================

for scheme_index, (
    scheme_name,
    scheme,
) in enumerate(
    WEIGHT_SCHEMES.items(),
    start=1,
):

    print()
    print(
        "-" * 70
    )

    print(
        f"SCHEME "
        f"{scheme_index}/"
        f"{len(WEIGHT_SCHEMES)}: "
        f"{scheme_name}"
    )


    # --------------------------------------------------------
    # Train three quantile models
    # --------------------------------------------------------

    scheme_models = {}

    scheme_predictions = {}


    for quantile_name, alpha in (
        QUANTILES.items()
    ):

        print(
            f"\nTraining "
            f"{quantile_name.upper()}..."
        )

        weights = (
            create_quantile_weights(
                y_internal_train,
                scheme,
                quantile_name,
            )
        )

        model = (
            create_quantile_model(
                alpha
            )
        )

        model.fit(
            X_internal_train,
            y_internal_train,
            sample_weight=weights,
            categorical_feature=CATEGORICAL_FEATURES,
        )

        prediction = (
            model.predict(
                X_internal_validation
            )
        )

        scheme_models[
            quantile_name
        ] = model

        scheme_predictions[
            quantile_name
        ] = prediction


    # --------------------------------------------------------
    # Raw predictions
    # --------------------------------------------------------

    raw_p10 = (
        scheme_predictions["p10"]
    )

    raw_p50 = (
        scheme_predictions["p50"]
    )

    raw_p90 = (
        scheme_predictions["p90"]
    )


    # --------------------------------------------------------
    # Quantile crossing correction
    # --------------------------------------------------------
    #
    # Sort all three quantile predictions for every patient.
    #
    # This guarantees:
    #
    # P10 <= P50 <= P90

    sorted_predictions = (
        np.sort(
            np.column_stack(
                [
                    raw_p10,
                    raw_p50,
                    raw_p90,
                ]
            ),
            axis=1,
        )
    )

    p10 = (
        sorted_predictions[:, 0]
    )

    p50 = (
        sorted_predictions[:, 1]
    )

    p90 = (
        sorted_predictions[:, 2]
    )


    # --------------------------------------------------------
    # Ordering validity
    # --------------------------------------------------------

    ordering_validity = (
        np.mean(
            (p10 <= p50)
            &
            (p50 <= p90)
        )
        * 100
    )


    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    overall_mae = (
        mean_absolute_error(
            y_internal_validation,
            p50,
        )
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            y_internal_validation,
            p50,
        )
    )

    overall_r2 = (
        r2_score(
            y_internal_validation,
            p50,
        )
    )


    # --------------------------------------------------------
    # LOS-range metrics
    # --------------------------------------------------------

    range_results = (
        calculate_los_range_metrics(
            y_internal_validation,
            p50,
        )
    )

    mae_72_168 = (
        range_results[
            "72_168h"
        ]["MAE_hours"]
    )

    mae_over_168 = (
        range_results[
            "over_168h"
        ]["MAE_hours"]
    )


    # --------------------------------------------------------
    # Pinball losses
    # --------------------------------------------------------

    p10_pinball = (
        mean_pinball_loss(
            y_internal_validation,
            p10,
            alpha=0.10,
        )
    )

    p90_pinball = (
        mean_pinball_loss(
            y_internal_validation,
            p90,
            alpha=0.90,
        )
    )


    # --------------------------------------------------------
    # Tail-aware selection score
    # --------------------------------------------------------
    #
    # Lower is better.
    #
    # 40% overall MAE
    # 40% >168h MAE
    # 20% 72-168h MAE
    #
    # The score is normalized against the first
    # unweighted/reference-style experiment below.
    #
    # For the first scheme, it becomes the initial reference.

    if len(experiment_results) == 0:

        reference_overall_mae = (
            overall_mae
        )

        reference_72_168_mae = (
            mae_72_168
        )

        reference_over_168_mae = (
            mae_over_168
        )


    normalized_overall = (
        overall_mae
        / reference_overall_mae
    )

    normalized_72_168 = (
        mae_72_168
        / reference_72_168_mae
    )

    normalized_over_168 = (
        mae_over_168
        / reference_over_168_mae
    )

    selection_score = (
        0.40 * normalized_overall
        +
        0.20 * normalized_72_168
        +
        0.40 * normalized_over_168
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"\nRESULT - "
        f"{scheme_name}"
    )

    print(
        f"Overall MAE: "
        f"{overall_mae:.4f} h"
    )

    print(
        f"Overall RMSE: "
        f"{overall_rmse:.4f} h"
    )

    print(
        f"R²: "
        f"{overall_r2:.4f}"
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
        f"P10 Pinball: "
        f"{p10_pinball:.4f}"
    )

    print(
        f"P90 Pinball: "
        f"{p90_pinball:.4f}"
    )

    print(
        f"Ordering validity: "
        f"{ordering_validity:.3f}%"
    )

    print(
        f"Selection score: "
        f"{selection_score:.6f}"
    )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    experiment_results.append({

        "weight_scheme":
            scheme_name,

        "overall_MAE_hours":
            float(
                overall_mae
            ),

        "overall_RMSE_hours":
            float(
                overall_rmse
            ),

        "overall_R2":
            float(
                overall_r2
            ),

        "MAE_72_168h":
            float(
                mae_72_168
            ),

        "MAE_over_168h":
            float(
                mae_over_168
            ),

        "P10_pinball":
            float(
                p10_pinball
            ),

        "P90_pinball":
            float(
                p90_pinball
            ),

        "ordering_validity_percent":
            float(
                ordering_validity
            ),

        "selection_score":
            float(
                selection_score
            ),
    })


    # --------------------------------------------------------
    # Select best scheme
    # --------------------------------------------------------

    if (
        selection_score
        < best_selection_score
    ):

        best_selection_score = (
            selection_score
        )

        best_scheme_name = (
            scheme_name
        )

        best_scheme_details = {
            "scheme":
                scheme,

            "overall_MAE_hours":
                float(
                    overall_mae
                ),

            "overall_RMSE_hours":
                float(
                    overall_rmse
                ),

            "overall_R2":
                float(
                    overall_r2
                ),

            "MAE_72_168h":
                float(
                    mae_72_168
                ),

            "MAE_over_168h":
                float(
                    mae_over_168
                ),

            "P10_pinball":
                float(
                    p10_pinball
                ),

            "P90_pinball":
                float(
                    p90_pinball
                ),

            "ordering_validity":
                float(
                    ordering_validity
                ),

            "selection_score":
                float(
                    selection_score
                ),
        }


# ============================================================
# 28. SAVE EXPERIMENT RESULTS
# ============================================================

experiment_results_df = (
    pd.DataFrame(
        experiment_results
    )
)

experiment_results_path = (
    OUTPUT_DIR
    / "los_quantile_specific_weight_results.csv"
)

experiment_results_df.to_csv(
    experiment_results_path,
    index=False,
)


# ============================================================
# 29. DISPLAY BEST SCHEME
# ============================================================

print_section(
    "BEST QUANTILE-SPECIFIC WEIGHT SCHEME"
)

print(
    f"Selected scheme: "
    f"{best_scheme_name}"
)

print(
    f"Selection score: "
    f"{best_selection_score:.6f}"
)

print(
    "\nSelected P10 weights:"
)

for key, value in (
    best_scheme_details[
        "scheme"
    ]["p10"].items()
):

    print(
        f"{key:15s}: "
        f"{value:.2f}"
    )

print(
    "\nSelected P50 weights:"
)

for key, value in (
    best_scheme_details[
        "scheme"
    ]["p50"].items()
):

    print(
        f"{key:15s}: "
        f"{value:.2f}"
    )

print(
    "\nSelected P90 weights:"
)

for key, value in (
    best_scheme_details[
        "scheme"
    ]["p90"].items()
):

    print(
        f"{key:15s}: "
        f"{value:.2f}"
    )


print(
    "\nInternal validation performance:"
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
# 30. PREPARE FINAL DATA
# ============================================================

print_section(
    "PREPARING FINAL TRAINING DATA"
)

X_train = (
    prepare_features(
        train_df
    )
)

y_train = (
    train_df[
        TARGET
    ].to_numpy()
)

X_calibration = (
    prepare_features(
        calibration_df
    )
)

y_calibration = (
    calibration_df[
        TARGET
    ].to_numpy()
)

X_test = (
    prepare_features(
        test_df
    )
)

y_test = (
    test_df[
        TARGET
    ].to_numpy()
)


# ============================================================
# 31. TRAIN FINAL QUANTILE-SPECIFIC MODELS
# ============================================================

print_section(
    "TRAINING FINAL QUANTILE-SPECIFIC MODELS"
)

final_models = {}


for quantile_name, alpha in (
    QUANTILES.items()
):

    print(
        f"\nTraining final "
        f"{quantile_name.upper()}..."
    )

    weights = (
        create_quantile_weights(
            y_train,
            best_scheme_details[
                "scheme"
            ],
            quantile_name,
        )
    )

    model = (
        create_quantile_model(
            alpha
        )
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=weights,
        categorical_feature=CATEGORICAL_FEATURES,
    )

    final_models[
        quantile_name
    ] = model

    model_path = (
        MODEL_DIR
        /
        (
            "los_lightgbm_quantile_weighted_"
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
# 32. RAW CALIBRATION PREDICTIONS
# ============================================================

print_section(
    "RAW CALIBRATION QUANTILE PREDICTIONS"
)

raw_p10_cal = (
    final_models["p10"]
    .predict(
        X_calibration
    )
)

raw_p50_cal = (
    final_models["p50"]
    .predict(
        X_calibration
    )
)

raw_p90_cal = (
    final_models["p90"]
    .predict(
        X_calibration
    )
)


# ============================================================
# 33. RAW TEST PREDICTIONS
# ============================================================

print_section(
    "RAW TEST QUANTILE PREDICTIONS"
)

raw_p10_test = (
    final_models["p10"]
    .predict(
        X_test
    )
)

raw_p50_test = (
    final_models["p50"]
    .predict(
        X_test
    )
)

raw_p90_test = (
    final_models["p90"]
    .predict(
        X_test
    )
)


# ============================================================
# 34. QUANTILE ORDERING BEFORE MAPIE
# ============================================================

print_section(
    "QUANTILE CROSSING CORRECTION"
)

raw_calibration_ordering = (
    (
        raw_p10_cal
        <= raw_p50_cal
    )
    &
    (
        raw_p50_cal
        <= raw_p90_cal
    )
)

raw_test_ordering = (
    (
        raw_p10_test
        <= raw_p50_test
    )
    &
    (
        raw_p50_test
        <= raw_p90_test
    )
)


print(
    f"Raw calibration ordering: "
    f"{np.mean(raw_calibration_ordering) * 100:.3f}%"
)

print(
    f"Raw test ordering: "
    f"{np.mean(raw_test_ordering) * 100:.3f}%"
)


# Corrected quantiles.

calibration_sorted = (
    np.sort(
        np.column_stack(
            [
                raw_p10_cal,
                raw_p50_cal,
                raw_p90_cal,
            ]
        ),
        axis=1,
    )
)

test_sorted = (
    np.sort(
        np.column_stack(
            [
                raw_p10_test,
                raw_p50_test,
                raw_p90_test,
            ]
        ),
        axis=1,
    )
)


p10_cal_corrected = (
    calibration_sorted[:, 0]
)

p50_cal_corrected = (
    calibration_sorted[:, 1]
)

p90_cal_corrected = (
    calibration_sorted[:, 2]
)


p10_test_corrected = (
    test_sorted[:, 0]
)

p50_test_corrected = (
    test_sorted[:, 1]
)

p90_test_corrected = (
    test_sorted[:, 2]
)


corrected_calibration_ordering = (
    (
        p10_cal_corrected
        <= p50_cal_corrected
    )
    &
    (
        p50_cal_corrected
        <= p90_cal_corrected
    )
)

corrected_test_ordering = (
    (
        p10_test_corrected
        <= p50_test_corrected
    )
    &
    (
        p50_test_corrected
        <= p90_test_corrected
    )
)


print(
    f"Corrected calibration ordering: "
    f"{np.mean(corrected_calibration_ordering) * 100:.3f}%"
)

print(
    f"Corrected test ordering: "
    f"{np.mean(corrected_test_ordering) * 100:.3f}%"
)


# ============================================================
# 35. RAW TEST METRICS
# ============================================================

print_section(
    "RAW QUANTILE-SPECIFIC TEST METRICS"
)

raw_point_metrics = (
    calculate_point_metrics(
        y_test,
        p50_test_corrected,
    )
)

raw_p10_pinball = (
    mean_pinball_loss(
        y_test,
        p10_test_corrected,
        alpha=0.10,
    )
)

raw_p90_pinball = (
    mean_pinball_loss(
        y_test,
        p90_test_corrected,
        alpha=0.90,
    )
)


print(
    f"P50 MAE: "
    f"{raw_point_metrics['MAE_hours']:.3f} h"
)

print(
    f"P50 RMSE: "
    f"{raw_point_metrics['RMSE_hours']:.3f} h"
)

print(
    f"P50 R²: "
    f"{raw_point_metrics['R2']:.4f}"
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
# 36. CREATE MAPIE WRAPPERS
# ============================================================
#
# Important:
#
# MAPIE needs estimators with predict().
#
# The wrappers ensure that MAPIE always receives:
#
#   lower <= median <= upper
#
# even when the independent LightGBM quantile models cross.

print_section(
    "CREATING ORDERED QUANTILE ESTIMATORS"
)

ordered_lower_estimator = (
    QuantileOrderingWrapper(
        final_models["p10"],
        final_models["p50"],
        final_models["p90"],
        role="lower",
    )
)

ordered_median_estimator = (
    QuantileOrderingWrapper(
        final_models["p10"],
        final_models["p50"],
        final_models["p90"],
        role="median",
    )
)

ordered_upper_estimator = (
    QuantileOrderingWrapper(
        final_models["p10"],
        final_models["p50"],
        final_models["p90"],
        role="upper",
    )
)


# ============================================================
# 37. MAPIE CQR
# ============================================================

print_section(
    "MAPIE CONFORMAL CALIBRATION"
)

print(
    "P10 -> ordered lower quantile"
)

print(
    "P50 -> ordered median quantile"
)

print(
    "P90 -> ordered upper quantile"
)

print(
    "Confidence level = 0.80"
)


mapie_model = (
    ConformalizedQuantileRegressor(
        estimator=[
            ordered_lower_estimator,
            ordered_upper_estimator,
            ordered_median_estimator,
        ],
        confidence_level=(
            MAPIE_CONFIDENCE_LEVEL
        ),
        prefit=True,
    )
)


print(
    "\nCalibrating MAPIE..."
)

mapie_model.conformalize(
    X_calibration,
    y_calibration,
)

print(
    "MAPIE calibration completed."
)


# ============================================================
# 38. MAPIE TEST PREDICTION
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


# MAPIE 1.5.0:
#
#     y_pred, y_interval
#
# y_interval:
#
#     (n_samples, 2, 1)
#
# Handle both 3D and 2D safely.

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


# Final point prediction uses the corrected P50.

median_prediction = (
    p50_test_corrected
)


# ============================================================
# 39. FINAL POINT METRICS
# ============================================================

final_point = (
    calculate_point_metrics(
        y_test,
        median_prediction,
    )
)


# ============================================================
# 40. FINAL INTERVAL METRICS
# ============================================================

# Defensive final ordering.
#
# MAPIE should already return ordered intervals, but this
# guarantees a valid lower/upper pair.

final_lower = np.minimum(
    lower,
    upper,
)

final_upper = np.maximum(
    lower,
    upper,
)


final_interval = (
    calculate_interval_metrics(
        y_test,
        final_lower,
        final_upper,
    )
)


# ============================================================
# 41. FINAL RESULTS
# ============================================================

print_section(
    "FINAL QUANTILE-SPECIFIC MODEL - TEST RESULTS"
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
# 42. LOS-RANGE PERFORMANCE
# ============================================================

print_section(
    "PERFORMANCE BY ACTUAL LOS RANGE"
)

range_results = (
    calculate_los_range_metrics(
        y_test,
        median_prediction,
    )
)

tail_interval_results = (
    calculate_tail_interval_metrics(
        y_test,
        final_lower,
        final_upper,
    )
)


range_rows = []


for group in [
    "0_24h",
    "24_72h",
    "72_168h",
    "over_168h",
]:

    if group not in range_results:
        continue

    point_result = (
        range_results[group]
    )

    interval_result = (
        tail_interval_results.get(
            group,
            {},
        )
    )

    range_rows.append({

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


range_df = pd.DataFrame(
    range_rows
)


print(
    range_df.to_string(
        index=False
    )
)


range_path = (
    OUTPUT_DIR
    / "los_quantile_weighted_by_los_range.csv"
)

range_df.to_csv(
    range_path,
    index=False,
)


# ============================================================
# 43. BASELINE COMPARISON
# ============================================================

print_section(
    "BASELINE VS QUANTILE-SPECIFIC MODEL"
)

over_168_point = (
    range_results[
        "over_168h"
    ]
)

over_168_interval = (
    tail_interval_results[
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

        "Quantile_Specific":
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

        "Quantile_Specific":
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

        "Quantile_Specific":
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

        "Quantile_Specific":
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

        "Quantile_Specific":
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

        "Quantile_Specific":
            over_168_point[
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

        "Quantile_Specific":
            over_168_interval[
                "coverage_percent"
            ],
    },
]


comparison_df = (
    pd.DataFrame(
        comparison_rows
    )
)


print(
    comparison_df.to_string(
        index=False
    )
)


comparison_path = (
    OUTPUT_DIR
    / "los_baseline_vs_quantile_specific.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False,
)


# ============================================================
# 44. TEST PREDICTION FILE
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

    "raw_p10_hours":
        raw_p10_test,

    "raw_p50_hours":
        raw_p50_test,

    "raw_p90_hours":
        raw_p90_test,

    "corrected_p10_hours":
        p10_test_corrected,

    "corrected_p50_hours":
        p50_test_corrected,

    "corrected_p90_hours":
        p90_test_corrected,

    "conformal_lower_hours":
        final_lower,

    "conformal_upper_hours":
        final_upper,

    "prediction_error_hours":
        (
            median_prediction
            - y_test
        ),
})


prediction_path = (
    OUTPUT_DIR
    / "los_predictions_test_quantile_specific.csv"
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
# 45. SAVE MAPIE MODEL
# ============================================================

mapie_model_path = (
    MODEL_DIR
    / "los_mapie_cqr_quantile_specific.joblib"
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
# 46. SAVE CONFIGURATION
# ============================================================

configuration = {

    "experiment":
        (
            "Quantile-specific "
            "tail-weighted LightGBM "
            "+ MAPIE CQR"
        ),

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
        INTERNAL_VALIDATION_SIZE,

    "quantiles":
        QUANTILES,

    "mapie_confidence_level":
        MAPIE_CONFIDENCE_LEVEL,

    "lightgbm_parameters":
        BASE_LGB_PARAMS,

    "weight_schemes":
        WEIGHT_SCHEMES,

    "selected_scheme":
        best_scheme_name,

    "selected_scheme_details":
        best_scheme_details,

    "quantile_crossing_correction":
        (
            "Per-row sorting of P10/P50/P90 "
            "to enforce P10 <= P50 <= P90."
        ),

    "baseline_results":
        BASELINE_RESULTS,

    "final_test_metrics":
        {
            **final_point,
            **final_interval,
        },

    "over_168h_metrics":
        over_168_point,

    "over_168h_interval_metrics":
        over_168_interval,
}


configuration_path = (
    OUTPUT_DIR
    / "los_quantile_specific_configuration.json"
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
# 47. SAVE FINAL METRICS
# ============================================================

final_summary = {

    "model":
        (
            "Quantile-specific "
            "tail-weighted LightGBM "
            "+ MAPIE CQR"
        ),

    "selected_scheme":
        best_scheme_name,

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

    "Median_Absolute_Error_hours":
        final_point[
            "Median_Absolute_Error_hours"
        ],

    "Mean_Prediction_Error_hours":
        final_point[
            "Mean_Prediction_Error_hours"
        ],

    "Underprediction_percent":
        final_point[
            "Underprediction_percent"
        ],

    "Overprediction_percent":
        final_point[
            "Overprediction_percent"
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

    "over_168h_actual_mean_hours":
        over_168_point[
            "actual_mean_hours"
        ],

    "over_168h_predicted_mean_hours":
        over_168_point[
            "predicted_mean_hours"
        ],

    "over_168h_MAE_hours":
        over_168_point[
            "MAE_hours"
        ],

    "over_168h_coverage_percent":
        over_168_interval[
            "coverage_percent"
        ],
}


summary_path = (
    OUTPUT_DIR
    / "los_quantile_specific_metrics.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        final_summary,
        file,
        indent=4,
    )


print(
    f"Saved: "
    f"{summary_path.name}"
)


# ============================================================
# 48. FINAL COMPLETION
# ============================================================

print_section(
    "QUANTILE-SPECIFIC LOS MODEL COMPLETE"
)

print(
    "Training completed successfully."
)

print(
    "\nArchitecture:"
)

print(
    "Quantile-specific weighted "
    "LightGBM P10/P50/P90"
)

print(
    "-> Quantile ordering correction"
)

print(
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
    "\nSelected scheme:"
)

print(
    best_scheme_name
)

print(
    "\nFINAL TEST:"
)

print(
    f"MAE        = "
    f"{final_point['MAE_hours']:.3f} h"
)

print(
    f"RMSE       = "
    f"{final_point['RMSE_hours']:.3f} h"
)

print(
    f"R²         = "
    f"{final_point['R2']:.4f}"
)

print(
    f"Coverage   = "
    f"{final_interval['Coverage_percent']:.3f}%"
)

print(
    f"Mean Width = "
    f"{final_interval['Mean_Interval_Width_hours']:.3f} h"
)

print(
    f"Ordering   = "
    f"{final_interval['Interval_Ordering_Validity_percent']:.3f}%"
)

print(
    "\n>168h PERFORMANCE:"
)

print(
    f"Actual mean LOS = "
    f"{over_168_point['actual_mean_hours']:.3f} h"
)

print(
    f"Predicted mean LOS = "
    f"{over_168_point['predicted_mean_hours']:.3f} h"
)

print(
    f">168h MAE = "
    f"{over_168_point['MAE_hours']:.3f} h"
)

print(
    f">168h Coverage = "
    f"{over_168_interval['coverage_percent']:.3f}%"
)

print(
    "\nOutput:"
)

print(
    prediction_path
)

print(
    "\nDONE."
)