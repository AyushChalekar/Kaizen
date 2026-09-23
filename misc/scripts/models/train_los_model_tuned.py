# ============================================================
# LOS Prediction - Tuned LightGBM Quantile Regression + MAPIE
# Kaizen Healthcare Predictive Resource Optimization System
#
# Purpose:
#   Tune LightGBM quantile models for LOS prediction while
#   preserving the project's chronological train/calibration/test
#   architecture.
#
# Split:
#   70,000 -> Training
#   15,000 -> Calibration
#   15,000 -> Final Test
#
# Models:
#   P10  -> 10th percentile
#   P50  -> Median
#   P90  -> 90th percentile
#
# Final uncertainty:
#   MAPIE Conformalized Quantile Regression
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

DATA_PATH = BASE_DIR / "data" / "synthetic_patient_stays_100k.csv"

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TRAIN_SIZE = 70_000
CALIBRATION_SIZE = 15_000
TEST_SIZE = 15_000

# Internal validation split inside the 70k training set.
# This is ONLY used for hyperparameter tuning.
TUNING_VALIDATION_SIZE = 10_000

QUANTILES = {
    "p10": 0.10,
    "p50": 0.50,
    "p90": 0.90,
}


# ============================================================
# 3. FEATURES
# ============================================================

# Strict arrival/triage-time features.
#
# We intentionally do NOT use:
#   discharge_time
#   bed_assigned_time
#   disposition
#   icu_transfer_flag
#   initial_care_unit
#   requires_ventilation
#   los_hours
#
# These can contain information that becomes available only
# after or during the patient's stay/routing process.

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

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "los_hours"


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a clean section heading."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_dataset(df):
    """Validate required columns and basic data quality."""

    print_section("DATASET VALIDATION")

    required_columns = FEATURES + [TARGET, "arrival_time"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    if (df[TARGET] < 0).any():
        raise ValueError("Target contains negative LOS values.")

    if df[FEATURES].isna().any().any():
        raise ValueError("Feature columns contain missing values.")

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("Required columns: PASS")
    print("Missing values: PASS")
    print("Negative LOS: PASS")


def prepare_features(df):
    """
    Prepare model features.

    LightGBM can directly handle pandas categorical columns,
    so categorical columns are converted to pandas category dtype.
    """

    X = df[FEATURES].copy()

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    return X


def create_model(params, alpha):
    """Create a LightGBM quantile regression model."""

    model_params = params.copy()

    model_params.update({
        "objective": "quantile",
        "alpha": alpha,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    })

    return lgb.LGBMRegressor(**model_params)


def evaluate_point_prediction(y_true, y_pred):
    """Calculate point prediction metrics."""

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    median_absolute_error = np.median(
        np.abs(y_true - y_pred)
    )

    mean_error = np.mean(y_pred - y_true)

    underprediction_rate = np.mean(
        y_pred < y_true
    ) * 100

    overprediction_rate = np.mean(
        y_pred > y_true
    ) * 100

    return {
        "MAE_hours": float(mae),
        "RMSE_hours": float(rmse),
        "R2": float(r2),
        "Median_Absolute_Error_hours": float(
            median_absolute_error
        ),
        "Mean_Prediction_Error_hours": float(mean_error),
        "Underprediction_percent": float(
            underprediction_rate
        ),
        "Overprediction_percent": float(
            overprediction_rate
        ),
    }


def evaluate_interval(y_true, lower, upper):
    """Calculate conformal prediction interval metrics."""

    # MAPIE 1.5.0 expects intervals in the shape:
    # (n_samples, 2, 1)
    intervals = np.column_stack([
        lower,
        upper,
    ])[:, :, np.newaxis]

    coverage = regression_coverage_score(
        y_true,
        intervals,
    )

    mean_width = regression_mean_width_score(
        intervals,
    )

    ordering_validity = np.mean(
        lower <= upper
    ) * 100

    return {
        "Coverage_percent": float(coverage * 100),
        "Mean_Interval_Width_hours": float(mean_width),
        "Median_Interval_Width_hours": float(
            np.median(upper - lower)
        ),
        "P90_Interval_Width_hours": float(
            np.percentile(upper - lower, 90)
        ),
        "Interval_Ordering_Validity_percent": float(
            ordering_validity
        ),
    }

def evaluate_los_ranges(
    y_true,
    y_pred,
    lower,
    upper,
):
    """Evaluate model performance across LOS ranges."""

    result_rows = []

    ranges = [
        ("0-24h", 0, 24),
        ("24-72h", 24, 72),
        ("72-168h", 72, 168),
        (">168h", 168, np.inf),
    ]

    for range_name, low, high in ranges:

        mask = (
            (y_true >= low)
            & (y_true < high)
        )

        count = int(mask.sum())

        if count == 0:
            continue

        actual = y_true[mask]
        predicted = y_pred[mask]
        lower_range = lower[mask]
        upper_range = upper[mask]

        mae = mean_absolute_error(
            actual,
            predicted,
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        )

        bias = np.mean(
            predicted - actual
        )

        coverage = np.mean(
            (actual >= lower_range)
            & (actual <= upper_range)
        ) * 100

        underprediction = np.mean(
            predicted < actual
        ) * 100

        width = np.mean(
            upper_range - lower_range
        )

        result_rows.append({
            "LOS_Range": range_name,
            "Count": count,
            "Actual_Mean_hours": float(
                np.mean(actual)
            ),
            "Predicted_Mean_hours": float(
                np.mean(predicted)
            ),
            "MAE_hours": float(mae),
            "RMSE_hours": float(rmse),
            "Bias_hours": float(bias),
            "Underprediction_percent": float(
                underprediction
            ),
            "Coverage_percent": float(
                coverage
            ),
            "Mean_Interval_Width_hours": float(
                width
            ),
        })

    return pd.DataFrame(result_rows)


def evaluate_triage_groups(
    y_true,
    y_pred,
    lower,
    upper,
    acuity,
):
    """Evaluate model performance by triage acuity."""

    result_rows = []

    for level in sorted(
        pd.Series(acuity).dropna().unique()
    ):

        mask = acuity == level

        count = int(mask.sum())

        if count == 0:
            continue

        actual = y_true[mask]
        predicted = y_pred[mask]
        lower_group = lower[mask]
        upper_group = upper[mask]

        coverage = np.mean(
            (actual >= lower_group)
            & (actual <= upper_group)
        ) * 100

        result_rows.append({
            "Triage_Acuity": int(level),
            "Count": count,
            "Actual_Median_hours": float(
                np.median(actual)
            ),
            "Predicted_Median_hours": float(
                np.median(predicted)
            ),
            "MAE_hours": float(
                mean_absolute_error(
                    actual,
                    predicted,
                )
            ),
            "Coverage_percent": float(
                coverage
            ),
            "Mean_Interval_Width_hours": float(
                np.mean(
                    upper_group - lower_group
                )
            ),
        })

    return pd.DataFrame(result_rows)


# ============================================================
# 5. HYPERPARAMETER SEARCH SPACE
# ============================================================

# The search is deliberately controlled rather than enormous.
#
# The goal is to test:
#   - shallow/moderate/deeper trees
#   - different leaf counts
#   - learning rates
#   - minimum observations per leaf
#   - L1/L2 regularization
#   - row/feature subsampling
#
# P50 is optimized for MAE.
# P10/P90 are optimized for pinball loss.

PARAMETER_CONFIGS = [

    # --------------------------------------------------------
    # Configuration 1 - Conservative
    # --------------------------------------------------------
    {
        "name": "config_01_conservative",
        "num_leaves": 31,
        "max_depth": 7,
        "learning_rate": 0.03,
        "n_estimators": 800,
        "min_child_samples": 100,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.10,
        "reg_lambda": 1.00,
    },

    # --------------------------------------------------------
    # Configuration 2 - More leaves
    # --------------------------------------------------------
    {
        "name": "config_02_more_leaves",
        "num_leaves": 63,
        "max_depth": 9,
        "learning_rate": 0.03,
        "n_estimators": 900,
        "min_child_samples": 80,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.10,
        "reg_lambda": 1.00,
    },

    # --------------------------------------------------------
    # Configuration 3 - Deeper
    # --------------------------------------------------------
    {
        "name": "config_03_deeper",
        "num_leaves": 63,
        "max_depth": 12,
        "learning_rate": 0.025,
        "n_estimators": 1000,
        "min_child_samples": 80,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.20,
        "reg_lambda": 1.50,
    },

    # --------------------------------------------------------
    # Configuration 4 - High capacity
    # --------------------------------------------------------
    {
        "name": "config_04_high_capacity",
        "num_leaves": 127,
        "max_depth": 14,
        "learning_rate": 0.02,
        "n_estimators": 1200,
        "min_child_samples": 60,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.20,
        "reg_lambda": 2.00,
    },

    # --------------------------------------------------------
    # Configuration 5 - Stronger regularization
    # --------------------------------------------------------
    {
        "name": "config_05_regularized",
        "num_leaves": 63,
        "max_depth": 10,
        "learning_rate": 0.025,
        "n_estimators": 1000,
        "min_child_samples": 120,
        "subsample": 0.80,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.50,
        "reg_lambda": 3.00,
    },

    # --------------------------------------------------------
    # Configuration 6 - Balanced
    # --------------------------------------------------------
    {
        "name": "config_06_balanced",
        "num_leaves": 95,
        "max_depth": 11,
        "learning_rate": 0.025,
        "n_estimators": 1000,
        "min_child_samples": 75,
        "subsample": 0.90,
        "colsample_bytree": 0.95,
        "reg_alpha": 0.15,
        "reg_lambda": 1.50,
    },

    # --------------------------------------------------------
    # Configuration 7 - Tail-focused capacity
    # --------------------------------------------------------
    {
        "name": "config_07_tail_capacity",
        "num_leaves": 127,
        "max_depth": 16,
        "learning_rate": 0.015,
        "n_estimators": 1400,
        "min_child_samples": 50,
        "subsample": 0.90,
        "colsample_bytree": 1.00,
        "reg_alpha": 0.10,
        "reg_lambda": 1.50,
    },

    # --------------------------------------------------------
    # Configuration 8 - Lower learning rate
    # --------------------------------------------------------
    {
        "name": "config_08_low_learning_rate",
        "num_leaves": 63,
        "max_depth": 10,
        "learning_rate": 0.015,
        "n_estimators": 1400,
        "min_child_samples": 70,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.10,
        "reg_lambda": 1.00,
    },
]


# ============================================================
# 6. LOAD DATA
# ============================================================

print_section("LOS MODEL - TUNED LIGHTGBM + MAPIE")

print(f"Data path: {DATA_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Model directory: {MODEL_DIR}")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"\nDataset loaded successfully.")
print(f"Shape: {df.shape}")

validate_dataset(df)


# ============================================================
# 7. CHRONOLOGICAL SORT
# ============================================================

print_section("CHRONOLOGICAL ORDERING")

df["arrival_time"] = pd.to_datetime(
    df["arrival_time"],
    errors="raise",
)

df = df.sort_values(
    "arrival_time"
).reset_index(drop=True)

print("Sorted by arrival_time: PASS")

print(
    f"First arrival: {df['arrival_time'].min()}"
)

print(
    f"Last arrival:  {df['arrival_time'].max()}"
)


# ============================================================
# 8. CHECK SPLIT SIZE
# ============================================================

required_rows = (
    TRAIN_SIZE
    + CALIBRATION_SIZE
    + TEST_SIZE
)

if len(df) < required_rows:
    raise ValueError(
        f"Dataset needs at least {required_rows:,} rows."
    )


# ============================================================
# 9. CHRONOLOGICAL TRAIN / CALIBRATION / TEST SPLIT
# ============================================================

print_section("CHRONOLOGICAL DATA SPLIT")

train_df = df.iloc[
    :TRAIN_SIZE
].copy()

calibration_df = df.iloc[
    TRAIN_SIZE:
    TRAIN_SIZE + CALIBRATION_SIZE
].copy()

test_df = df.iloc[
    TRAIN_SIZE + CALIBRATION_SIZE:
    TRAIN_SIZE + CALIBRATION_SIZE + TEST_SIZE
].copy()

print(
    f"Training:     {len(train_df):,} rows"
)

print(
    f"Calibration:  {len(calibration_df):,} rows"
)

print(
    f"Test:         {len(test_df):,} rows"
)

print(
    f"\nTraining period:"
    f" {train_df['arrival_time'].min()}"
    f" -> {train_df['arrival_time'].max()}"
)

print(
    f"Calibration period:"
    f" {calibration_df['arrival_time'].min()}"
    f" -> {calibration_df['arrival_time'].max()}"
)

print(
    f"Test period:"
    f" {test_df['arrival_time'].min()}"
    f" -> {test_df['arrival_time'].max()}"
)


# ============================================================
# 10. INTERNAL VALIDATION SPLIT FOR TUNING
# ============================================================

print_section("INTERNAL VALIDATION SPLIT FOR TUNING")

tune_train_df = train_df.iloc[
    :-TUNING_VALIDATION_SIZE
].copy()

tune_valid_df = train_df.iloc[
    -TUNING_VALIDATION_SIZE:
].copy()

print(
    f"Tuning training rows:   {len(tune_train_df):,}"
)

print(
    f"Tuning validation rows: {len(tune_valid_df):,}"
)

print(
    f"\nTuning training period:"
    f" {tune_train_df['arrival_time'].min()}"
    f" -> {tune_train_df['arrival_time'].max()}"
)

print(
    f"Tuning validation period:"
    f" {tune_valid_df['arrival_time'].min()}"
    f" -> {tune_valid_df['arrival_time'].max()}"
)


# ============================================================
# 11. PREPARE FEATURES
# ============================================================

X_tune_train = prepare_features(
    tune_train_df
)

y_tune_train = tune_train_df[TARGET].values

X_tune_valid = prepare_features(
    tune_valid_df
)

y_tune_valid = tune_valid_df[TARGET].values


# ============================================================
# 12. HYPERPARAMETER TUNING
# ============================================================

print_section("HYPERPARAMETER TUNING")

print(
    f"Number of configurations: "
    f"{len(PARAMETER_CONFIGS)}"
)

print(
    "Each configuration will be tested for "
    "P10, P50 and P90."
)

print(
    "\nThis can take some time because multiple "
    "LightGBM models are being trained."
)


tuning_results = []

best_models = {
    "p10": None,
    "p50": None,
    "p90": None,
}

best_scores = {
    "p10": np.inf,
    "p50": np.inf,
    "p90": np.inf,
}


for config_index, config in enumerate(
    PARAMETER_CONFIGS,
    start=1,
):

    config_name = config["name"]

    print()
    print(
        "-" * 70
    )

    print(
        f"Configuration "
        f"{config_index}/{len(PARAMETER_CONFIGS)}: "
        f"{config_name}"
    )

    print(
        f"leaves={config['num_leaves']}, "
        f"depth={config['max_depth']}, "
        f"lr={config['learning_rate']}, "
        f"estimators={config['n_estimators']}, "
        f"min_child={config['min_child_samples']}"
    )

    # Remove internal name before passing to LightGBM.
    lgb_params = {
        key: value
        for key, value in config.items()
        if key != "name"
    }

    for quantile_name, alpha in QUANTILES.items():

        print(
            f"\nTraining {quantile_name.upper()} "
            f"(alpha={alpha})..."
        )

        model = create_model(
            lgb_params,
            alpha,
        )

        model.fit(
            X_tune_train,
            y_tune_train,
            categorical_feature=CATEGORICAL_FEATURES,
        )

        prediction = model.predict(
            X_tune_valid
        )

        # P50 is optimized for MAE.
        if quantile_name == "p50":

            score = mean_absolute_error(
                y_tune_valid,
                prediction,
            )

            metric_name = "MAE"

        # P10 and P90 are optimized for pinball loss.
        else:

            score = mean_pinball_loss(
                y_tune_valid,
                prediction,
                alpha=alpha,
            )

            metric_name = "Pinball"

        print(
            f"{quantile_name.upper()} "
            f"{metric_name}: "
            f"{score:.6f}"
        )

        tuning_results.append({
            "configuration": config_name,
            "quantile": quantile_name,
            "alpha": alpha,
            "metric": metric_name,
            "score": float(score),
        })

        # Save best model for this quantile.
        if score < best_scores[quantile_name]:

            best_scores[quantile_name] = score

            best_models[quantile_name] = {
                "model": model,
                "config": config.copy(),
                "score": float(score),
                "metric": metric_name,
            }

            print(
                f"NEW BEST {quantile_name.upper()} "
                f"model."
            )


# ============================================================
# 13. SAVE TUNING RESULTS
# ============================================================

tuning_results_df = pd.DataFrame(
    tuning_results
)

tuning_results_path = (
    OUTPUT_DIR
    / "los_hyperparameter_tuning_results.csv"
)

tuning_results_df.to_csv(
    tuning_results_path,
    index=False,
)

print_section("BEST TUNING RESULTS")

for quantile_name in [
    "p10",
    "p50",
    "p90",
]:

    best = best_models[quantile_name]

    print(
        f"\n{quantile_name.upper()}"
    )

    print(
        f"Metric: {best['metric']}"
    )

    print(
        f"Score: {best['score']:.6f}"
    )

    print(
        f"Configuration: "
        f"{best['config']['name']}"
    )

    print(
        f"num_leaves: "
        f"{best['config']['num_leaves']}"
    )

    print(
        f"max_depth: "
        f"{best['config']['max_depth']}"
    )

    print(
        f"learning_rate: "
        f"{best['config']['learning_rate']}"
    )

    print(
        f"n_estimators: "
        f"{best['config']['n_estimators']}"
    )

    print(
        f"min_child_samples: "
        f"{best['config']['min_child_samples']}"
    )


# ============================================================
# 14. RETRAIN BEST MODELS ON ALL 70K TRAINING ROWS
# ============================================================

print_section(
    "RETRAINING BEST MODELS ON FULL 70K TRAINING SET"
)

X_train = prepare_features(
    train_df
)

y_train = train_df[TARGET].values

X_calibration = prepare_features(
    calibration_df
)

y_calibration = calibration_df[TARGET].values

X_test = prepare_features(
    test_df
)

y_test = test_df[TARGET].values


final_models = {}

for quantile_name, alpha in QUANTILES.items():

    best = best_models[quantile_name]

    config = best["config"]

    lgb_params = {
        key: value
        for key, value in config.items()
        if key != "name"
    }

    print(
        f"\nRetraining "
        f"{quantile_name.upper()} "
        f"with {config['name']}..."
    )

    final_model = create_model(
        lgb_params,
        alpha,
    )

    final_model.fit(
        X_train,
        y_train,
        categorical_feature=CATEGORICAL_FEATURES,
    )

    final_models[quantile_name] = final_model

    model_path = (
        MODEL_DIR
        / f"los_lightgbm_tuned_{quantile_name}.joblib"
    )

    joblib.dump(
        final_model,
        model_path,
    )

    print(
        f"Saved: {model_path.name}"
    )


# ============================================================
# 15. RAW QUANTILE PREDICTIONS
# ============================================================

print_section(
    "RAW QUANTILE PREDICTIONS"
)

p10_cal = final_models["p10"].predict(
    X_calibration
)

p50_cal = final_models["p50"].predict(
    X_calibration
)

p90_cal = final_models["p90"].predict(
    X_calibration
)


p10_test = final_models["p10"].predict(
    X_test
)

p50_test = final_models["p50"].predict(
    X_test
)

p90_test = final_models["p90"].predict(
    X_test
)


# ============================================================
# 16. CHECK RAW QUANTILE ORDERING
# ============================================================

raw_ordering_cal = (
    (p10_cal <= p50_cal)
    & (p50_cal <= p90_cal)
)

raw_ordering_test = (
    (p10_test <= p50_test)
    & (p50_test <= p90_test)
)

print(
    f"Calibration raw ordering validity: "
    f"{np.mean(raw_ordering_cal) * 100:.3f}%"
)

print(
    f"Test raw ordering validity: "
    f"{np.mean(raw_ordering_test) * 100:.3f}%"
)


# ============================================================
# 17. RAW P50 TEST METRICS
# ============================================================

raw_point_metrics = evaluate_point_prediction(
    y_test,
    p50_test,
)

raw_p10_pinball = mean_pinball_loss(
    y_test,
    p10_test,
    alpha=0.10,
)

raw_p90_pinball = mean_pinball_loss(
    y_test,
    p90_test,
    alpha=0.90,
)

print_section(
    "RAW TUNED MODEL - TEST METRICS"
)

print(
    f"P50 MAE: "
    f"{raw_point_metrics['MAE_hours']:.3f} hours"
)

print(
    f"P50 RMSE: "
    f"{raw_point_metrics['RMSE_hours']:.3f} hours"
)

print(
    f"P50 R²: "
    f"{raw_point_metrics['R2']:.4f}"
)

print(
    f"P10 Pinball Loss: "
    f"{raw_p10_pinball:.3f}"
)

print(
    f"P90 Pinball Loss: "
    f"{raw_p90_pinball:.3f}"
)


# ============================================================
# 18. MAPIE CONFORMALIZED QUANTILE REGRESSION
# ============================================================

print_section(
    "MAPIE CONFORMAL CALIBRATION"
)

print(
    "Creating MAPIE CQR model using:"
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


# MAPIE expects the quantile estimators in:
#
# [lower_quantile, upper_quantile, median_quantile]
#
# For an 80% interval:
# lower = 0.10
# upper = 0.90
# median = 0.50

mapie_model = ConformalizedQuantileRegressor(
    estimator=[
        final_models["p10"],
        final_models["p90"],
        final_models["p50"],
    ],
    confidence_level=0.80,
    prefit=True,
)


print(
    "\nCalibrating MAPIE on the 15,000-row "
    "calibration dataset..."
)

mapie_model.conformalize(
    X_calibration,
    y_calibration,
)

print(
    "MAPIE calibration completed."
)


# ============================================================
# 19. FINAL MAPIE TEST PREDICTIONS
# ============================================================

print_section(
    "FINAL MAPIE TEST PREDICTIONS"
)

mapie_result = mapie_model.predict_interval(
    X_test
)

# MAPIE 1.5.0 returns:
# (y_pred, y_interval)
#
# y_pred:
#   median prediction
#
# y_interval:
#   lower and upper conformal bounds

mapie_prediction, mapie_interval = mapie_result

# MAPIE interval shape is expected to be:
# (n_samples, 2, 1)

mapie_interval = np.asarray(
    mapie_interval
)

if mapie_interval.ndim == 3:
    lower = mapie_interval[:, 0, 0]
    upper = mapie_interval[:, 1, 0]

elif mapie_interval.ndim == 2:
    lower = mapie_interval[:, 0]
    upper = mapie_interval[:, 1]

else:
    raise ValueError(
        f"Unexpected MAPIE interval shape: "
        f"{mapie_interval.shape}"
    )

# Use the P50 LightGBM prediction as the final
# point prediction.
median_prediction = p50_test


# ============================================================
# 20. FINAL METRICS
# ============================================================

final_point_metrics = evaluate_point_prediction(
    y_test,
    median_prediction,
)

final_interval_metrics = evaluate_interval(
    y_test,
    lower,
    upper,
)

print_section(
    "FINAL TUNED MODEL - TEST RESULTS"
)

print(
    f"P50 MAE: "
    f"{final_point_metrics['MAE_hours']:.3f} hours"
)

print(
    f"P50 RMSE: "
    f"{final_point_metrics['RMSE_hours']:.3f} hours"
)

print(
    f"R²: "
    f"{final_point_metrics['R2']:.4f}"
)

print(
    f"Median Absolute Error: "
    f"{final_point_metrics['Median_Absolute_Error_hours']:.3f} hours"
)

print(
    f"Mean Prediction Error: "
    f"{final_point_metrics['Mean_Prediction_Error_hours']:.3f} hours"
)

print(
    f"Underprediction: "
    f"{final_point_metrics['Underprediction_percent']:.3f}%"
)

print(
    f"Overprediction: "
    f"{final_point_metrics['Overprediction_percent']:.3f}%"
)

print(
    f"80% Coverage: "
    f"{final_interval_metrics['Coverage_percent']:.3f}%"
)

print(
    f"Mean Interval Width: "
    f"{final_interval_metrics['Mean_Interval_Width_hours']:.3f} hours"
)

print(
    f"Median Interval Width: "
    f"{final_interval_metrics['Median_Interval_Width_hours']:.3f} hours"
)

print(
    f"90th Percentile Interval Width: "
    f"{final_interval_metrics['P90_Interval_Width_hours']:.3f} hours"
)

print(
    f"Interval Ordering Validity: "
    f"{final_interval_metrics['Interval_Ordering_Validity_percent']:.3f}%"
)


# ============================================================
# 21. LOS RANGE ANALYSIS
# ============================================================

print_section(
    "PERFORMANCE BY ACTUAL LOS RANGE"
)

los_range_results = evaluate_los_ranges(
    y_test,
    median_prediction,
    lower,
    upper,
)

print(
    los_range_results.to_string(
        index=False
    )
)

los_range_path = (
    OUTPUT_DIR
    / "los_tuned_by_los_range.csv"
)

los_range_results.to_csv(
    los_range_path,
    index=False,
)


# ============================================================
# 22. TRIAGE ACUITY ANALYSIS
# ============================================================

print_section(
    "PERFORMANCE BY TRIAGE ACUITY"
)

triage_results = evaluate_triage_groups(
    y_test,
    median_prediction,
    lower,
    upper,
    test_df["triage_acuity"].values,
)

print(
    triage_results.to_string(
        index=False
    )
)

triage_path = (
    OUTPUT_DIR
    / "los_tuned_by_triage_acuity.csv"
)

triage_results.to_csv(
    triage_path,
    index=False,
)


# ============================================================
# 23. SAVE TEST PREDICTIONS
# ============================================================

print_section(
    "SAVING TEST PREDICTIONS"
)

prediction_output = pd.DataFrame({
    "stay_id": test_df["stay_id"].values,
    "patient_id": test_df["patient_id"].values,
    "arrival_time": test_df["arrival_time"].values,
    "triage_acuity": test_df["triage_acuity"].values,
    "actual_los_hours": y_test,
    "predicted_los_p50_hours": median_prediction,
    "predicted_los_p10_raw_hours": p10_test,
    "predicted_los_p90_raw_hours": p90_test,
    "conformal_lower_hours": lower,
    "conformal_upper_hours": upper,
    "prediction_error_hours": (
        median_prediction - y_test
    ),
})

prediction_path = (
    OUTPUT_DIR
    / "los_predictions_test_tuned.csv"
)

prediction_output.to_csv(
    prediction_path,
    index=False,
)

print(
    f"Saved: {prediction_path.name}"
)

print(
    f"Rows saved: "
    f"{len(prediction_output):,}"
)


# ============================================================
# 24. SAVE MODEL CONFIGURATION
# ============================================================

print_section(
    "SAVING MODEL CONFIGURATION"
)

model_configuration = {
    "model_type": (
        "LightGBM Quantile Regression "
        "+ MAPIE CQR"
    ),
    "target": TARGET,
    "features": FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "train_size": TRAIN_SIZE,
    "calibration_size": CALIBRATION_SIZE,
    "test_size": TEST_SIZE,
    "internal_tuning_validation_size": (
        TUNING_VALIDATION_SIZE
    ),
    "quantiles": QUANTILES,
    "mapie_confidence_level": 0.80,
    "random_state": RANDOM_STATE,
    "best_configurations": {
        quantile: best_models[quantile]["config"]
        for quantile in QUANTILES
    },
    "best_tuning_scores": {
        quantile: best_models[quantile]["score"]
        for quantile in QUANTILES
    },
    "final_test_metrics": {
        **final_point_metrics,
        **final_interval_metrics,
    },
}

config_path = (
    OUTPUT_DIR
    / "los_tuned_model_configuration.json"
)

with open(
    config_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        model_configuration,
        f,
        indent=4,
    )

print(
    f"Saved: {config_path.name}"
)


# ============================================================
# 25. SAVE MAPIE MODEL
# ============================================================

mapie_model_path = (
    MODEL_DIR
    / "los_mapie_cqr_tuned.joblib"
)

joblib.dump(
    mapie_model,
    mapie_model_path,
)

print(
    f"Saved: {mapie_model_path.name}"
)


# ============================================================
# 26. SAVE SUMMARY
# ============================================================

summary = {
    "model": "Tuned LightGBM Quantile Regression + MAPIE CQR",
    "test_rows": int(len(test_df)),
    "MAE_hours": final_point_metrics[
        "MAE_hours"
    ],
    "RMSE_hours": final_point_metrics[
        "RMSE_hours"
    ],
    "R2": final_point_metrics[
        "R2"
    ],
    "Median_Absolute_Error_hours":
        final_point_metrics[
            "Median_Absolute_Error_hours"
        ],
    "Mean_Prediction_Error_hours":
        final_point_metrics[
            "Mean_Prediction_Error_hours"
        ],
    "Underprediction_percent":
        final_point_metrics[
            "Underprediction_percent"
        ],
    "Overprediction_percent":
        final_point_metrics[
            "Overprediction_percent"
        ],
    "Coverage_percent":
        final_interval_metrics[
            "Coverage_percent"
        ],
    "Mean_Interval_Width_hours":
        final_interval_metrics[
            "Mean_Interval_Width_hours"
        ],
    "Median_Interval_Width_hours":
        final_interval_metrics[
            "Median_Interval_Width_hours"
        ],
    "P90_Interval_Width_hours":
        final_interval_metrics[
            "P90_Interval_Width_hours"
        ],
    "Interval_Ordering_Validity_percent":
        final_interval_metrics[
            "Interval_Ordering_Validity_percent"
        ],
}

summary_path = (
    OUTPUT_DIR
    / "los_tuned_metrics.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=4,
    )

print(
    f"Saved: {summary_path.name}"
)


# ============================================================
# 27. FINAL COMPLETION MESSAGE
# ============================================================

print_section(
    "TUNED LOS MODEL COMPLETE"
)

print(
    "The tuned LOS model has been trained successfully."
)

print(
    "\nArchitecture:"
)

print(
    "LightGBM P10/P50/P90 "
    "-> MAPIE Conformalized Quantile Regression"
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
    "\nImportant:"
)

print(
    "The original baseline model and its outputs "
    "were not overwritten."
)

print(
    "\nMain output:"
)

print(
    f"{prediction_path}"
)

print(
    "\nMain model:"
)

print(
    f"{mapie_model_path}"
)

print(
    "\nFinal metrics:"
)

print(
    f"MAE       = "
    f"{final_point_metrics['MAE_hours']:.3f} h"
)

print(
    f"RMSE      = "
    f"{final_point_metrics['RMSE_hours']:.3f} h"
)

print(
    f"R²        = "
    f"{final_point_metrics['R2']:.4f}"
)

print(
    f"Coverage   = "
    f"{final_interval_metrics['Coverage_percent']:.3f}%"
)

print(
    f"Mean Width = "
    f"{final_interval_metrics['Mean_Interval_Width_hours']:.3f} h"
)

print(
    "\nNext step: compare these results with the "
    "original baseline before deciding whether to "
    "keep the tuned model."
)

print(
    "\nDONE."
)