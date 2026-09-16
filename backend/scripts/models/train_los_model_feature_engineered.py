# ============================================================
# KAIZEN - LOS PREDICTION MODEL #7
# Feature-Engineered Tail-Aware LightGBM + MAPIE CQR
#
# Model #7 changes from Model #6:
# 1. Adds clinically derived arrival-time features
# 2. Adds cyclical arrival-hour features
# 3. Keeps Model #6 LightGBM parameters
# 4. Keeps Model #6 strong_p50_p90_tail weighting
# 5. Keeps P10 / P50 / P90 quantile regression
# 6. Keeps MAPIE 80% conformal calibration
# 7. Keeps the exact chronological 70k/15k/15k split
# 8. Corrects quantile crossing before MAPIE
#
# Target:
#     los_hours
#
# IMPORTANT:
# Only information available at arrival/triage is used
# as a prediction feature.
# ============================================================


import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from lightgbm import LGBMRegressor
from mapie.regression import ConformalizedQuantileRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "synthetic_patient_stays_100k.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


RANDOM_STATE = 42

TRAIN_SIZE = 70000
CALIBRATION_SIZE = 15000
TEST_SIZE = 15000

CONFIDENCE_LEVEL = 0.80

QUANTILES = [0.10, 0.50, 0.90]


# ============================================================
# 2. ORIGINAL ARRIVAL/Triage FEATURES
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
    "triage_acuity"
]

CATEGORICAL_FEATURES = [
    "gender",
    "chief_complaint"
]


# ============================================================
# 3. MODEL #7 DERIVED FEATURES
# ============================================================

DERIVED_FEATURES = [
    "shock_index",
    "pulse_pressure",
    "mean_arterial_pressure",
    "oxygen_deficit",
    "temperature_deviation",
    "arrival_hour_sin",
    "arrival_hour_cos"
]


ALL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + DERIVED_FEATURES
)


TARGET = "los_hours"


# ============================================================
# 4. MODEL #6 TAIL-WEIGHTING SCHEME
# ============================================================

WEIGHT_SCHEME_NAME = "strong_p50_p90_tail"

WEIGHT_TABLE = {
    "P10": {
        "0_24": 1.00,
        "24_72": 1.05,
        "72_168": 1.20,
        "gt_168": 1.35
    },

    "P50": {
        "0_24": 1.00,
        "24_72": 1.25,
        "72_168": 1.80,
        "gt_168": 2.50
    },

    "P90": {
        "0_24": 1.00,
        "24_72": 1.30,
        "72_168": 2.20,
        "gt_168": 4.00
    }
}


# ============================================================
# 5. LIGHTGBM PARAMETERS
# ============================================================

LGBM_PARAMS = {
    "objective": "quantile",
    "num_leaves": 31,
    "max_depth": 7,
    "learning_rate": 0.03,
    "n_estimators": 800,
    "min_child_samples": 100,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "verbosity": -1
}


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def get_los_range_weights(y, quantile_name):
    """
    Create tail-aware sample weights according to actual LOS.
    """

    y = np.asarray(y)

    weights = np.ones(len(y), dtype=float)

    weights[y < 24] = WEIGHT_TABLE[quantile_name]["0_24"]

    weights[
        (y >= 24) &
        (y < 72)
    ] = WEIGHT_TABLE[quantile_name]["24_72"]

    weights[
        (y >= 72) &
        (y <= 168)
    ] = WEIGHT_TABLE[quantile_name]["72_168"]

    weights[y > 168] = WEIGHT_TABLE[quantile_name]["gt_168"]

    return weights


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    abs_error = np.abs(
        np.asarray(y_true) -
        np.asarray(y_pred)
    )

    median_abs_error = np.median(abs_error)

    prediction_error = (
        np.asarray(y_pred) -
        np.asarray(y_true)
    )

    mean_error = np.mean(prediction_error)

    underprediction_pct = (
        np.mean(prediction_error < 0) * 100
    )

    overprediction_pct = (
        np.mean(prediction_error > 0) * 100
    )

    return {
        "MAE_hours": float(mae),
        "RMSE_hours": float(rmse),
        "R2": float(r2),
        "Median_Absolute_Error_hours": float(
            median_abs_error
        ),
        "Mean_Prediction_Error_hours": float(
            mean_error
        ),
        "Underprediction_pct": float(
            underprediction_pct
        ),
        "Overprediction_pct": float(
            overprediction_pct
        )
    }


def pinball_loss(y_true, y_pred, quantile):
    """
    Quantile / pinball loss.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    error = y_true - y_pred

    loss = np.maximum(
        quantile * error,
        (quantile - 1) * error
    )

    return float(np.mean(loss))


def fix_quantile_crossing(p10, p50, p90):
    """
    Enforce:

        P10 <= P50 <= P90

    by sorting predictions row-by-row.
    """

    stacked = np.column_stack([
        p10,
        p50,
        p90
    ])

    sorted_values = np.sort(
        stacked,
        axis=1
    )

    return (
        sorted_values[:, 0],
        sorted_values[:, 1],
        sorted_values[:, 2]
    )


def interval_coverage(y_true, lower, upper):
    """
    Calculate empirical prediction interval coverage.
    """

    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    covered = (
        (y_true >= lower) &
        (y_true <= upper)
    )

    return float(
        np.mean(covered) * 100
    )


def interval_width(lower, upper):
    return np.asarray(upper) - np.asarray(lower)


def get_range_label(value):
    if value < 24:
        return "0-24h"
    elif value < 72:
        return "24-72h"
    elif value <= 168:
        return "72-168h"
    else:
        return ">168h"


def calculate_range_metrics(
    y_true,
    y_pred,
    lower,
    upper
):

    df = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
        "lower": lower,
        "upper": upper
    })

    df["range"] = df["actual"].apply(
        get_range_label
    )

    results = []

    order = [
        "0-24h",
        "24-72h",
        "72-168h",
        ">168h"
    ]

    for range_name in order:

        group = df[
            df["range"] == range_name
        ]

        if len(group) == 0:
            continue

        actual = group["actual"].values
        predicted = group["predicted"].values

        error = predicted - actual

        coverage = interval_coverage(
            actual,
            group["lower"].values,
            group["upper"].values
        )

        results.append({
            "LOS_Range": range_name,
            "N": len(group),
            "Actual_Mean_hours": float(
                np.mean(actual)
            ),
            "Predicted_Mean_hours": float(
                np.mean(predicted)
            ),
            "MAE_hours": float(
                mean_absolute_error(
                    actual,
                    predicted
                )
            ),
            "RMSE_hours": float(
                np.sqrt(
                    mean_squared_error(
                        actual,
                        predicted
                    )
                )
            ),
            "Bias_hours": float(
                np.mean(error)
            ),
            "Underprediction_pct": float(
                np.mean(error < 0) * 100
            ),
            "Coverage_pct": float(coverage),
            "Mean_Interval_Width_hours": float(
                np.mean(
                    group["upper"].values -
                    group["lower"].values
                )
            )
        })

    return pd.DataFrame(results)


def calculate_threshold_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    thresholds = [
        72,
        168,
        240,
        300
    ]

    results = []

    for threshold in thresholds:

        mask = y_true > threshold

        if np.sum(mask) == 0:
            continue

        actual = y_true[mask]
        predicted = y_pred[mask]

        error = predicted - actual

        results.append({
            "Threshold": f">{threshold}h",
            "N": int(np.sum(mask)),
            "Actual_Mean_hours": float(
                np.mean(actual)
            ),
            "Predicted_Mean_hours": float(
                np.mean(predicted)
            ),
            "MAE_hours": float(
                mean_absolute_error(
                    actual,
                    predicted
                )
            ),
            "RMSE_hours": float(
                np.sqrt(
                    mean_squared_error(
                        actual,
                        predicted
                    )
                )
            ),
            "Bias_hours": float(
                np.mean(error)
            ),
            "Underprediction_pct": float(
                np.mean(error < 0) * 100
            )
        })

    return pd.DataFrame(results)


def calculate_quantiles(y):
    return {
        "P50": float(np.percentile(y, 50)),
        "P90": float(np.percentile(y, 90)),
        "P95": float(np.percentile(y, 95)),
        "P99": float(np.percentile(y, 99)),
        "Max": float(np.max(y))
    }


# ============================================================
# 7. START
# ============================================================

print_section(
    "KAIZEN - LOS PREDICTION MODEL #7"
)

print("Feature-Engineered Tail-Aware")
print("LightGBM Quantile Regression + MAPIE CQR")

print()
print("Working directory:")
print(BASE_DIR)

print()
print("Dataset:")
print(DATA_PATH)


# ============================================================
# 8. LOAD DATASET
# ============================================================

print_section(
    "STEP 1 - LOAD DATASET"
)

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset shape: {df.shape}"
)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ============================================================
# 9. BASIC VALIDATION
# ============================================================

print_section(
    "STEP 2 - DATA VALIDATION"
)

required_columns = list(
    set(
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [
            "arrival_time",
            TARGET
        ]
    )
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        + str(missing_columns)
    )

print("All required columns present.")


missing_target = df[TARGET].isna().sum()

print(
    f"Missing LOS values: {missing_target}"
)

if missing_target > 0:

    raise ValueError(
        "Target contains missing values."
    )


negative_los = (
    df[TARGET] < 0
).sum()

print(
    f"Negative LOS values: {negative_los}"
)

if negative_los > 0:

    raise ValueError(
        "Negative LOS values detected."
    )


duplicate_stays = (
    df["stay_id"].duplicated().sum()
)

print(
    f"Duplicate stay IDs: {duplicate_stays}"
)

if duplicate_stays > 0:

    raise ValueError(
        "Duplicate stay IDs detected."
    )


# ============================================================
# 10. SORT CHRONOLOGICALLY
# ============================================================

print_section(
    "STEP 3 - CHRONOLOGICAL ORDERING"
)

df["arrival_time"] = pd.to_datetime(
    df["arrival_time"],
    errors="coerce"
)

if df["arrival_time"].isna().any():

    raise ValueError(
        "Invalid arrival_time values detected."
    )

df = df.sort_values(
    "arrival_time"
).reset_index(drop=True)

print(
    "Dataset sorted by arrival_time."
)

print(
    "First arrival:",
    df["arrival_time"].min()
)

print(
    "Last arrival:",
    df["arrival_time"].max()
)


# ============================================================
# 11. FEATURE ENGINEERING
# ============================================================

print_section(
    "STEP 4 - FEATURE ENGINEERING"
)

# ------------------------------------------------------------
# 11.1 Shock Index
# ------------------------------------------------------------

df["shock_index"] = (
    df["heart_rate"] /
    df["sbp"].replace(0, np.nan)
)

# ------------------------------------------------------------
# 11.2 Pulse Pressure
# ------------------------------------------------------------

df["pulse_pressure"] = (
    df["sbp"] -
    df["dbp"]
)

# ------------------------------------------------------------
# 11.3 Mean Arterial Pressure
# ------------------------------------------------------------

df["mean_arterial_pressure"] = (
    df["dbp"] +
    (
        df["sbp"] -
        df["dbp"]
    ) / 3.0
)

# ------------------------------------------------------------
# 11.4 Oxygen Deficit
# ------------------------------------------------------------

df["oxygen_deficit"] = (
    100.0 -
    df["o2_sat"]
)

# ------------------------------------------------------------
# 11.5 Temperature Deviation
# ------------------------------------------------------------

df["temperature_deviation"] = (
    np.abs(
        df["temp_c"] - 37.0
    )
)

# ------------------------------------------------------------
# 11.6 Arrival Hour
# ------------------------------------------------------------

df["arrival_hour"] = (
    df["arrival_time"].dt.hour
)

# ------------------------------------------------------------
# 11.7 Cyclical Arrival Hour
# ------------------------------------------------------------

df["arrival_hour_sin"] = np.sin(
    2 * np.pi *
    df["arrival_hour"] / 24.0
)

df["arrival_hour_cos"] = np.cos(
    2 * np.pi *
    df["arrival_hour"] / 24.0
)


# ============================================================
# 12. DERIVED FEATURE VALIDATION
# ============================================================

print()
print("Derived features created:")

for feature in DERIVED_FEATURES:

    print(
        f"  {feature}"
    )

derived_missing = (
    df[DERIVED_FEATURES]
    .isna()
    .sum()
)

if derived_missing.sum() > 0:

    print()
    print(
        "Missing derived feature values:"
    )

    print(
        derived_missing
    )

    raise ValueError(
        "Derived features contain missing values."
    )

print()
print(
    "All derived features contain valid values."
)


# ============================================================
# 13. CATEGORICAL CONVERSION
# ============================================================

for column in CATEGORICAL_FEATURES:

    df[column] = (
        df[column]
        .astype(str)
        .astype("category")
    )


# ============================================================
# 14. SELECT FEATURES
# ============================================================

X = df[ALL_FEATURES].copy()

y = df[TARGET].astype(float).copy()

print()
print(
    f"Total model features: {len(ALL_FEATURES)}"
)

print()
print("Features:")

for feature in ALL_FEATURES:

    print(
        f"  - {feature}"
    )


# ============================================================
# 15. CHRONOLOGICAL TRAIN / CALIBRATION / TEST SPLIT
# ============================================================

print_section(
    "STEP 5 - CHRONOLOGICAL DATA SPLIT"
)

expected_total = (
    TRAIN_SIZE
    + CALIBRATION_SIZE
    + TEST_SIZE
)

if len(df) < expected_total:

    raise ValueError(
        f"Dataset has only {len(df)} rows, "
        f"but {expected_total} are required."
    )

train_end = TRAIN_SIZE

calibration_end = (
    TRAIN_SIZE +
    CALIBRATION_SIZE
)

X_train = X.iloc[
    :train_end
].copy()

y_train = y.iloc[
    :train_end
].copy()

X_calibration = X.iloc[
    train_end:calibration_end
].copy()

y_calibration = y.iloc[
    train_end:calibration_end
].copy()

X_test = X.iloc[
    calibration_end:
].copy()

y_test = y.iloc[
    calibration_end:
].copy()

test_ids = df.iloc[
    calibration_end:
]["stay_id"].values

test_acuity = df.iloc[
    calibration_end:
]["triage_acuity"].values


print(
    f"Training rows:     {len(X_train):,}"
)

print(
    f"Calibration rows:  {len(X_calibration):,}"
)

print(
    f"Test rows:         {len(X_test):,}"
)

print()

print(
    "Training period:",
    df.iloc[0]["arrival_time"],
    "to",
    df.iloc[train_end - 1]["arrival_time"]
)

print(
    "Calibration period:",
    df.iloc[train_end]["arrival_time"],
    "to",
    df.iloc[calibration_end - 1]["arrival_time"]
)

print(
    "Test period:",
    df.iloc[calibration_end]["arrival_time"],
    "to",
    df.iloc[-1]["arrival_time"]
)


# ============================================================
# 16. PRINT LOS DISTRIBUTION
# ============================================================

print_section(
    "STEP 6 - LOS DISTRIBUTION"
)

print(
    f"Training median LOS: "
    f"{np.median(y_train):.2f}h"
)

print(
    f"Calibration median LOS: "
    f"{np.median(y_calibration):.2f}h"
)

print(
    f"Test median LOS: "
    f"{np.median(y_test):.2f}h"
)

print()

print(
    f"Test P90 LOS: "
    f"{np.percentile(y_test, 90):.2f}h"
)

print(
    f"Test P95 LOS: "
    f"{np.percentile(y_test, 95):.2f}h"
)

print(
    f"Test P99 LOS: "
    f"{np.percentile(y_test, 99):.2f}h"
)

print(
    f"Test maximum LOS: "
    f"{np.max(y_test):.2f}h"
)


# ============================================================
# 17. TAIL WEIGHT SUMMARY
# ============================================================

print_section(
    "STEP 7 - TAIL-AWARE QUANTILE WEIGHTS"
)

print(
    f"Weight scheme: {WEIGHT_SCHEME_NAME}"
)

for quantile_name in [
    "P10",
    "P50",
    "P90"
]:

    print()
    print(
        quantile_name,
        "weights:"
    )

    for range_name, weight in (
        WEIGHT_TABLE[
            quantile_name
        ].items()
    ):

        print(
            f"  {range_name}: {weight}"
        )


# ============================================================
# 18. TRAIN RAW QUANTILE MODELS
# ============================================================

print_section(
    "STEP 8 - TRAIN LIGHTGBM QUANTILE MODELS"
)

models = {}

raw_predictions_train = {}
raw_predictions_calibration = {}
raw_predictions_test = {}


for quantile, quantile_name in zip(
    QUANTILES,
    ["P10", "P50", "P90"]
):

    print()
    print(
        f"Training {quantile_name} "
        f"(quantile={quantile})..."
    )

    params = LGBM_PARAMS.copy()

    params["alpha"] = quantile

    model = LGBMRegressor(
        **params
    )

    sample_weights = (
        get_los_range_weights(
            y_train.values,
            quantile_name
        )
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        categorical_feature=CATEGORICAL_FEATURES
    )

    models[quantile_name] = model

    raw_predictions_train[
        quantile_name
    ] = model.predict(
        X_train
    )

    raw_predictions_calibration[
        quantile_name
    ] = model.predict(
        X_calibration
    )

    raw_predictions_test[
        quantile_name
    ] = model.predict(
        X_test
    )

    print(
        f"{quantile_name} training complete."
    )


# ============================================================
# 19. RAW QUANTILE ORDERING
# ============================================================

print_section(
    "STEP 9 - RAW QUANTILE ORDERING CHECK"
)

raw_calibration = np.column_stack([
    raw_predictions_calibration["P10"],
    raw_predictions_calibration["P50"],
    raw_predictions_calibration["P90"]
])

raw_test = np.column_stack([
    raw_predictions_test["P10"],
    raw_predictions_test["P50"],
    raw_predictions_test["P90"]
])


calibration_ordered = (
    (
        raw_calibration[:, 0]
        <= raw_calibration[:, 1]
    )
    &
    (
        raw_calibration[:, 1]
        <= raw_calibration[:, 2]
    )
)

test_ordered = (
    (
        raw_test[:, 0]
        <= raw_test[:, 1]
    )
    &
    (
        raw_test[:, 1]
        <= raw_test[:, 2]
    )
)

calibration_ordering_pct = (
    np.mean(calibration_ordered)
    * 100
)

test_ordering_pct = (
    np.mean(test_ordered)
    * 100
)

print(
    f"Raw calibration ordering: "
    f"{calibration_ordering_pct:.3f}%"
)

print(
    f"Raw test ordering: "
    f"{test_ordering_pct:.3f}%"
)


# ============================================================
# 20. QUANTILE CROSSING CORRECTION
# ============================================================

print_section(
    "STEP 10 - QUANTILE CROSSING CORRECTION"
)

cal_p10, cal_p50, cal_p90 = (
    fix_quantile_crossing(
        raw_predictions_calibration["P10"],
        raw_predictions_calibration["P50"],
        raw_predictions_calibration["P90"]
    )
)

test_p10, test_p50, test_p90 = (
    fix_quantile_crossing(
        raw_predictions_test["P10"],
        raw_predictions_test["P50"],
        raw_predictions_test["P90"]
    )
)

train_p10, train_p50, train_p90 = (
    fix_quantile_crossing(
        raw_predictions_train["P10"],
        raw_predictions_train["P50"],
        raw_predictions_train["P90"]
    )
)


calibration_corrected_ordered = (
    (cal_p10 <= cal_p50)
    &
    (cal_p50 <= cal_p90)
)

test_corrected_ordered = (
    (test_p10 <= test_p50)
    &
    (test_p50 <= test_p90)
)

print(
    "Corrected calibration ordering: "
    f"{np.mean(calibration_corrected_ordered) * 100:.3f}%"
)

print(
    "Corrected test ordering: "
    f"{np.mean(test_corrected_ordered) * 100:.3f}%"
)


# ============================================================
# 21. RAW TEST METRICS
# ============================================================

print_section(
    "STEP 11 - RAW TEST METRICS"
)

raw_metrics = calculate_metrics(
    y_test.values,
    test_p50
)

print(
    f"P50 MAE:  "
    f"{raw_metrics['MAE_hours']:.3f} h"
)

print(
    f"P50 RMSE: "
    f"{raw_metrics['RMSE_hours']:.3f} h"
)

print(
    f"R²:        "
    f"{raw_metrics['R2']:.4f}"
)

print()

print(
    f"P10 Pinball Loss: "
    f"{pinball_loss(y_test.values, test_p10, 0.10):.4f}"
)

print(
    f"P90 Pinball Loss: "
    f"{pinball_loss(y_test.values, test_p90, 0.90):.4f}"
)


# ============================================================
# 22. MAPIE CONFORMALIZED QUANTILE REGRESSION
# ============================================================

print_section(
    "STEP 12 - MAPIE CONFORMAL CALIBRATION"
)

print(
    "Confidence level:",
    CONFIDENCE_LEVEL
)

print(
    "Using P10 / P50 / P90 quantile estimators."
)


# ------------------------------------------------------------
# MAPIE requires the three fitted quantile estimators in:
#
# [lower, upper, median]
#
# order when prefit=True.
# ------------------------------------------------------------

mapie_estimators = [
    models["P10"],
    models["P90"],
    models["P50"]
]


mapie_model = ConformalizedQuantileRegressor(
    estimator=mapie_estimators,
    confidence_level=CONFIDENCE_LEVEL,
    prefit=True
)


print(
    "Running conformal calibration..."
)

mapie_model.conformalize(
    X_calibration,
    y_calibration
)

print(
    "Conformal calibration complete."
)


# ============================================================
# 23. MAPIE PREDICTION
# ============================================================

print_section(
    "STEP 13 - GENERATE CONFORMAL TEST PREDICTIONS"
)

mapie_result = mapie_model.predict_interval(
    X_test
)


# ------------------------------------------------------------
# MAPIE 1.5.x returns:
#
# (y_pred, y_interval)
#
# y_interval normally has shape:
#
# (n_samples, 2, 1)
#
# But we handle both possible shapes safely.
# ------------------------------------------------------------

if isinstance(mapie_result, tuple):

    mapie_point_prediction = (
        mapie_result[0]
    )

    mapie_interval = (
        mapie_result[1]
    )

else:

    raise RuntimeError(
        "Unexpected MAPIE predict_interval "
        "return type."
    )


mapie_point_prediction = np.asarray(
    mapie_point_prediction
).reshape(-1)


mapie_interval = np.asarray(
    mapie_interval
)


print(
    "MAPIE point prediction shape:",
    mapie_point_prediction.shape
)

print(
    "MAPIE interval shape:",
    mapie_interval.shape
)


# ------------------------------------------------------------
# Extract lower / upper interval safely
# ------------------------------------------------------------

if mapie_interval.ndim == 3:

    conformal_lower = (
        mapie_interval[:, 0, 0]
    )

    conformal_upper = (
        mapie_interval[:, 1, 0]
    )

elif mapie_interval.ndim == 2:

    conformal_lower = (
        mapie_interval[:, 0]
    )

    conformal_upper = (
        mapie_interval[:, 1]
    )

else:

    raise RuntimeError(
        "Unexpected MAPIE interval dimensions: "
        f"{mapie_interval.ndim}"
    )


# ============================================================
# 24. ENFORCE NON-NEGATIVE LOS INTERVAL
# ============================================================

conformal_lower = np.maximum(
    conformal_lower,
    0
)

conformal_upper = np.maximum(
    conformal_upper,
    0
)


# ============================================================
# 25. FINAL POINT PREDICTION
# ============================================================

# MAPIE's central prediction corresponds to the
# median quantile model.

final_p50 = np.maximum(
    mapie_point_prediction,
    0
)


# ============================================================
# 26. FINAL METRICS
# ============================================================

print_section(
    "STEP 14 - FINAL MODEL #7 METRICS"
)

final_metrics = calculate_metrics(
    y_test.values,
    final_p50
)

for key, value in final_metrics.items():

    if key == "R2":

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value:.3f}"
        )


# ============================================================
# 27. CONFORMAL COVERAGE
# ============================================================

coverage = interval_coverage(
    y_test.values,
    conformal_lower,
    conformal_upper
)

widths = interval_width(
    conformal_lower,
    conformal_upper
)

mean_width = np.mean(widths)

median_width = np.median(widths)

p90_width = np.percentile(
    widths,
    90
)

ordering_validity = np.mean(
    (
        conformal_lower
        <= final_p50
    )
    &
    (
        final_p50
        <= conformal_upper
    )
) * 100


lower_below_zero_before_clip = (
    np.mean(
        mapie_interval[:, 0, 0]
        < 0
    ) * 100
    if mapie_interval.ndim == 3
    else
    np.mean(
        mapie_interval[:, 0]
        < 0
    ) * 100
)


print()
print(
    f"80% Interval Coverage: "
    f"{coverage:.3f}%"
)

print(
    f"Mean Interval Width: "
    f"{mean_width:.3f} h"
)

print(
    f"Median Interval Width: "
    f"{median_width:.3f} h"
)

print(
    f"90th Percentile Width: "
    f"{p90_width:.3f} h"
)

print(
    f"Interval Ordering Validity: "
    f"{ordering_validity:.3f}%"
)

print(
    f"Intervals Originally Below Zero: "
    f"{lower_below_zero_before_clip:.3f}%"
)


# ============================================================
# 28. LOS RANGE PERFORMANCE
# ============================================================

print_section(
    "STEP 15 - PERFORMANCE BY ACTUAL LOS RANGE"
)

range_metrics = calculate_range_metrics(
    y_test.values,
    final_p50,
    conformal_lower,
    conformal_upper
)

print(
    range_metrics.to_string(
        index=False
    )
)


# ============================================================
# 29. EXTREME TAIL PERFORMANCE
# ============================================================

print_section(
    "STEP 16 - EXTREME TAIL PERFORMANCE"
)

threshold_metrics = calculate_threshold_metrics(
    y_test.values,
    final_p50
)

print(
    threshold_metrics.to_string(
        index=False
    )
)


# ============================================================
# 30. PERFORMANCE BY TRIAGE ACUITY
# ============================================================

print_section(
    "STEP 17 - PERFORMANCE BY TRIAGE ACUITY"
)

triage_results = []

for acuity in sorted(
    np.unique(test_acuity)
):

    mask = (
        test_acuity == acuity
    )

    actual = y_test.values[mask]

    predicted = final_p50[mask]

    lower = conformal_lower[mask]

    upper = conformal_upper[mask]

    if len(actual) == 0:
        continue

    triage_results.append({

        "Triage_Acuity": int(acuity),

        "N": int(len(actual)),

        "Actual_Median_hours": float(
            np.median(actual)
        ),

        "Predicted_Median_hours": float(
            np.median(predicted)
        ),

        "Actual_Mean_hours": float(
            np.mean(actual)
        ),

        "Predicted_Mean_hours": float(
            np.mean(predicted)
        ),

        "MAE_hours": float(
            mean_absolute_error(
                actual,
                predicted
            )
        ),

        "RMSE_hours": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted
                )
            )
        ),

        "Bias_hours": float(
            np.mean(
                predicted - actual
            )
        ),

        "Coverage_pct": float(
            interval_coverage(
                actual,
                lower,
                upper
            )
        ),

        "Mean_Interval_Width_hours": float(
            np.mean(
                upper - lower
            )
        )
    })


triage_df = pd.DataFrame(
    triage_results
)

print(
    triage_df.to_string(
        index=False
    )
)


# ============================================================
# 31. ACTUAL VS PREDICTED DISTRIBUTION
# ============================================================

print_section(
    "STEP 18 - ACTUAL VS PREDICTED DISTRIBUTION"
)

actual_distribution = (
    calculate_quantiles(
        y_test.values
    )
)

predicted_distribution = (
    calculate_quantiles(
        final_p50
    )
)

distribution_df = pd.DataFrame({
    "Statistic": [
        "P50",
        "P90",
        "P95",
        "P99",
        "Max"
    ],

    "Actual_LOS_hours": [
        actual_distribution["P50"],
        actual_distribution["P90"],
        actual_distribution["P95"],
        actual_distribution["P99"],
        actual_distribution["Max"]
    ],

    "Predicted_LOS_hours": [
        predicted_distribution["P50"],
        predicted_distribution["P90"],
        predicted_distribution["P95"],
        predicted_distribution["P99"],
        predicted_distribution["Max"]
    ]
})

distribution_df["Predicted_to_Actual_Ratio"] = (
    distribution_df[
        "Predicted_LOS_hours"
    ]
    /
    distribution_df[
        "Actual_LOS_hours"
    ]
)

print(
    distribution_df.to_string(
        index=False
    )
)


# ============================================================
# 32. LONGEST ACTUAL STAYS
# ============================================================

print_section(
    "STEP 19 - LONGEST ACTUAL STAYS"
)

longest_df = pd.DataFrame({

    "stay_id": test_ids,

    "actual_los_hours": y_test.values,

    "predicted_los_hours": final_p50,

    "conformal_lower_hours": conformal_lower,

    "conformal_upper_hours": conformal_upper

})

longest_df["absolute_error_hours"] = (
    np.abs(
        longest_df["actual_los_hours"]
        -
        longest_df["predicted_los_hours"]
    )
)

longest_df = longest_df.sort_values(
    "actual_los_hours",
    ascending=False
)

print(
    longest_df.head(30).to_string(
        index=False
    )
)


# ============================================================
# 33. EXTREME UNDERPREDICTION CHECK
# ============================================================

print_section(
    "STEP 20 - EXTREME UNDERPREDICTION CHECK"
)

tail_mask = (
    y_test.values > 168
)

tail_actual = (
    y_test.values[tail_mask]
)

tail_predicted = (
    final_p50[tail_mask]
)

if len(tail_actual) > 0:

    ratio = (
        tail_predicted /
        tail_actual
    )

    severe_underprediction = (
        ratio < 0.50
    )

    print(
        f">168h cases: "
        f"{len(tail_actual):,}"
    )

    print(
        f"Predicted at least 2x too low: "
        f"{np.sum(severe_underprediction):,}"
        f" "
        f"({np.mean(severe_underprediction) * 100:.3f}%)"
    )

    print(
        f"Actual mean: "
        f"{np.mean(tail_actual):.3f}h"
    )

    print(
        f"Predicted mean: "
        f"{np.mean(tail_predicted):.3f}h"
    )

else:

    print(
        "No >168h cases found."
    )


# ============================================================
# 34. PREDICTION ERROR ANALYSIS
# ============================================================

print_section(
    "STEP 21 - PREDICTION ERROR ANALYSIS"
)

error_df = pd.DataFrame({

    "stay_id": test_ids,

    "actual_los_hours": y_test.values,

    "predicted_los_hours": final_p50,

    "conformal_lower_hours": conformal_lower,

    "conformal_upper_hours": conformal_upper,

    "prediction_error_hours": (
        final_p50 -
        y_test.values
    ),

    "absolute_error_hours": np.abs(
        final_p50 -
        y_test.values
    ),

    "triage_acuity": test_acuity
})

error_df = error_df.sort_values(
    "absolute_error_hours",
    ascending=False
)


# ============================================================
# 35. SAVE PREDICTIONS
# ============================================================

print_section(
    "STEP 22 - SAVE TEST PREDICTIONS"
)

prediction_output = pd.DataFrame({

    "stay_id": test_ids,

    "actual_los_hours": y_test.values,

    "raw_p10_hours": raw_predictions_test["P10"],

    "raw_p50_hours": raw_predictions_test["P50"],

    "raw_p90_hours": raw_predictions_test["P90"],

    "corrected_p10_hours": test_p10,

    "corrected_p50_hours": test_p50,

    "corrected_p90_hours": test_p90,

    "conformal_lower_hours": conformal_lower,

    "conformal_p50_hours": final_p50,

    "conformal_upper_hours": conformal_upper,

    "prediction_error_hours": (
        final_p50 -
        y_test.values
    ),

    "absolute_error_hours": np.abs(
        final_p50 -
        y_test.values
    ),

    "triage_acuity": test_acuity
})


prediction_path = os.path.join(
    OUTPUT_DIR,
    "los_predictions_model7_test.csv"
)

prediction_output.to_csv(
    prediction_path,
    index=False
)

print(
    "Saved:"
)

print(
    prediction_path
)


# ============================================================
# 36. SAVE DIAGNOSTIC TABLES
# ============================================================

print_section(
    "STEP 23 - SAVE DIAGNOSTIC TABLES"
)

range_path = os.path.join(
    OUTPUT_DIR,
    "los_model7_by_los_range.csv"
)

range_metrics.to_csv(
    range_path,
    index=False
)

print(
    "Saved:",
    range_path
)


threshold_path = os.path.join(
    OUTPUT_DIR,
    "los_model7_extreme_tail.csv"
)

threshold_metrics.to_csv(
    threshold_path,
    index=False
)

print(
    "Saved:",
    threshold_path
)


triage_path = os.path.join(
    OUTPUT_DIR,
    "los_model7_by_triage_acuity.csv"
)

triage_df.to_csv(
    triage_path,
    index=False
)

print(
    "Saved:",
    triage_path
)


distribution_path = os.path.join(
    OUTPUT_DIR,
    "los_model7_distribution_comparison.csv"
)

distribution_df.to_csv(
    distribution_path,
    index=False
)

print(
    "Saved:",
    distribution_path
)


errors_path = os.path.join(
    OUTPUT_DIR,
    "los_model7_prediction_errors.csv"
)

error_df.to_csv(
    errors_path,
    index=False
)

print(
    "Saved:",
    errors_path
)


# ============================================================
# 37. SAVE MODEL OBJECTS
# ============================================================

print_section(
    "STEP 24 - SAVE TRAINED MODELS"
)

for quantile_name, model in models.items():

    model_path = os.path.join(
        MODEL_DIR,
        f"los_model7_{quantile_name.lower()}.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "Saved:",
        model_path
    )


mapie_path = os.path.join(
    MODEL_DIR,
    "los_model7_mapie_cqr.joblib"
)

joblib.dump(
    mapie_model,
    mapie_path
)

print(
    "Saved:",
    mapie_path
)


# ============================================================
# 38. SAVE FEATURE INFORMATION
# ============================================================

feature_info = {

    "model": "LOS Prediction Model #7",

    "architecture": (
        "Feature-Engineered Tail-Aware "
        "LightGBM Quantile Regression + "
        "MAPIE CQR"
    ),

    "target": TARGET,

    "features": ALL_FEATURES,

    "original_numeric_features": (
        NUMERIC_FEATURES
    ),

    "categorical_features": (
        CATEGORICAL_FEATURES
    ),

    "derived_features": (
        DERIVED_FEATURES
    ),

    "quantiles": QUANTILES,

    "confidence_level": CONFIDENCE_LEVEL,

    "weight_scheme": WEIGHT_SCHEME_NAME,

    "weight_table": WEIGHT_TABLE,

    "lightgbm_parameters": LGBM_PARAMS,

    "split": {
        "train": TRAIN_SIZE,
        "calibration": CALIBRATION_SIZE,
        "test": TEST_SIZE
    }
}


feature_info_path = os.path.join(
    OUTPUT_DIR,
    "los_model7_configuration.json"
)

with open(
    feature_info_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        feature_info,
        f,
        indent=4
    )

print(
    "Saved:",
    feature_info_path
)


# ============================================================
# 39. SAVE FINAL METRICS
# ============================================================

final_report = {

    "model": "LOS Prediction Model #7",

    "architecture": (
        "Feature-Engineered Tail-Aware "
        "LightGBM + MAPIE CQR"
    ),

    "dataset_rows": int(len(df)),

    "train_rows": TRAIN_SIZE,

    "calibration_rows": CALIBRATION_SIZE,

    "test_rows": TEST_SIZE,

    "features": ALL_FEATURES,

    "raw_quantile_ordering": {
        "calibration_pct": float(
            calibration_ordering_pct
        ),
        "test_pct": float(
            test_ordering_pct
        )
    },

    "corrected_quantile_ordering": {
        "calibration_pct": float(
            np.mean(
                calibration_corrected_ordered
            ) * 100
        ),
        "test_pct": float(
            np.mean(
                test_corrected_ordered
            ) * 100
        )
    },

    "final_metrics": final_metrics,

    "p10_pinball_loss": (
        pinball_loss(
            y_test.values,
            test_p10,
            0.10
        )
    ),

    "p90_pinball_loss": (
        pinball_loss(
            y_test.values,
            test_p90,
            0.90
        )
    ),

    "mapie": {

        "confidence_level": (
            CONFIDENCE_LEVEL
        ),

        "coverage_pct": float(
            coverage
        ),

        "mean_interval_width_hours": float(
            mean_width
        ),

        "median_interval_width_hours": float(
            median_width
        ),

        "p90_interval_width_hours": float(
            p90_width
        ),

        "ordering_validity_pct": float(
            ordering_validity
        ),

        "lower_below_zero_before_clip_pct": float(
            lower_below_zero_before_clip
        )
    },

    "actual_distribution": (
        actual_distribution
    ),

    "predicted_distribution": (
        predicted_distribution
    ),

    "weight_scheme": WEIGHT_SCHEME_NAME,

    "lightgbm_parameters": LGBM_PARAMS
}


report_path = os.path.join(
    OUTPUT_DIR,
    "los_model7_final_report.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_report,
        f,
        indent=4
    )

print(
    "Saved:",
    report_path
)


# ============================================================
# 40. FINAL SUMMARY
# ============================================================

print_section(
    "MODEL #7 COMPLETE"
)

print()
print(
    "FINAL MODEL #7 RESULTS"
)

print(
    f"MAE:               "
    f"{final_metrics['MAE_hours']:.3f} h"
)

print(
    f"RMSE:              "
    f"{final_metrics['RMSE_hours']:.3f} h"
)

print(
    f"R²:                "
    f"{final_metrics['R2']:.4f}"
)

print(
    f"80% Coverage:      "
    f"{coverage:.3f}%"
)

print(
    f"Mean Interval:     "
    f"{mean_width:.3f} h"
)

print(
    f"Interval Ordering: "
    f"{ordering_validity:.3f}%"
)

print()

print(
    "Important:"
)

if len(threshold_metrics) > 0:

    tail_168 = threshold_metrics[
        threshold_metrics["Threshold"] == ">168h"
    ]

    if len(tail_168) > 0:

        row = tail_168.iloc[0]

        print(
            f">168h MAE:          "
            f"{row['MAE_hours']:.3f} h"
        )

        print(
            f">168h Bias:         "
            f"{row['Bias_hours']:.3f} h"
        )

        print(
            f">168h Underpred.:   "
            f"{row['Underprediction_pct']:.3f}%"
        )


print()
print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "=" * 75
)

print(
    "MODEL #7 TRAINING FINISHED SUCCESSFULLY"
)

print(
    "=" * 75
)