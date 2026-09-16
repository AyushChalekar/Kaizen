import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from mapie.regression import ConformalizedQuantileRegressor

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"C:\Users\suraj\OneDrive\Kaizen Files\LOS_Prediction"

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "synthetic_patient_stays_100k.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TRAIN_SIZE = 70000
CALIBRATION_SIZE = 15000
TEST_SIZE = 15000

CONFIDENCE_LEVEL = 0.80

LONG_STAY_THRESHOLD = 168.0


# ============================================================
# 3. FEATURES
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
    "severity_score"
]

CATEGORICAL_FEATURES = [
    "gender",
    "chief_complaint"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "los_hours"


# ============================================================
# 4. LOAD DATA
# ============================================================

print("=" * 70)
print("FINAL LOS PREDICTION MODEL")
print("=" * 70)

print("\n[1/10] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 5. BASIC VALIDATION
# ============================================================

print("\n[2/10] Validating dataset...")

required_columns = [
    col for col in FEATURES
    if col != "severity_score"
] + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

missing_values = df[required_columns].isna().sum().sum()

duplicate_stays = df["stay_id"].duplicated().sum()

negative_los = (df[TARGET] < 0).sum()

print(f"Missing values: {missing_values}")
print(f"Duplicate stay IDs: {duplicate_stays}")
print(f"Negative LOS values: {negative_los}")

if missing_values > 0:
    raise ValueError("Dataset contains missing values.")

if duplicate_stays > 0:
    raise ValueError("Duplicate stay IDs detected.")

if negative_los > 0:
    raise ValueError("Negative LOS values detected.")


# ============================================================
# 6. CALCULATE SEVERITY SCORE
# ============================================================

print("\n[3/10] Calculating causal severity score...")


def calculate_severity_score(data):

    heart_rate_severity = np.clip(
        (data["heart_rate"] - 80) / 60,
        0,
        1
    )

    blood_pressure_severity = np.clip(
        (120 - data["sbp"]) / 60,
        0,
        1
    )

    respiratory_severity = np.clip(
        (data["resp_rate"] - 18) / 20,
        0,
        1
    )

    oxygen_severity = np.clip(
        (95 - data["o2_sat"]) / 25,
        0,
        1
    )

    temperature_severity = np.clip(
        np.abs(data["temp_c"] - 37) / 4,
        0,
        1
    )

    comorbidity_severity = np.clip(
        data["charlson_index"] / 10,
        0,
        1
    )

    esi_severity = (
        (6 - data["triage_acuity"]) / 5
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

    return np.clip(severity, 0, 1)


df["severity_score"] = calculate_severity_score(df)

print(
    f"Severity range: "
    f"{df['severity_score'].min():.4f} - "
    f"{df['severity_score'].max():.4f}"
)


# ============================================================
# 7. CHRONOLOGICAL SORT
# ============================================================

print("\n[4/10] Sorting chronologically...")

df["arrival_time"] = pd.to_datetime(
    df["arrival_time"]
)

df = df.sort_values(
    "arrival_time"
).reset_index(drop=True)


# ============================================================
# 8. CHRONOLOGICAL TRAIN / CALIBRATION / TEST SPLIT
# ============================================================

print("\n[5/10] Creating chronological split...")

train_end = TRAIN_SIZE

calibration_end = (
    TRAIN_SIZE +
    CALIBRATION_SIZE
)

train_df = df.iloc[:train_end].copy()

calibration_df = df.iloc[
    train_end:calibration_end
].copy()

test_df = df.iloc[
    calibration_end:
].copy()

print(f"Training rows:      {len(train_df)}")
print(f"Calibration rows:   {len(calibration_df)}")
print(f"Test rows:          {len(test_df)}")


# ============================================================
# 9. PREPARE X / Y
# ============================================================

X_train = train_df[FEATURES].copy()
y_train = train_df[TARGET].copy()

X_calibration = calibration_df[FEATURES].copy()
y_calibration = calibration_df[TARGET].copy()

X_test = test_df[FEATURES].copy()
y_test = test_df[TARGET].copy()


# ============================================================
# 10. PREPROCESSING
# ============================================================

print("\n[6/10] Preparing preprocessing...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            NUMERIC_FEATURES
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            CATEGORICAL_FEATURES
        )
    ]
)

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_calibration_processed = preprocessor.transform(
    X_calibration
)

X_test_processed = preprocessor.transform(
    X_test
)

print(
    f"Processed feature count: "
    f"{X_train_processed.shape[1]}"
)


# ============================================================
# 11. LIGHTGBM QUANTILE MODELS
# ============================================================

print("\n[7/10] Training LightGBM quantile models...")


def create_quantile_model(alpha):

    return LGBMRegressor(
        objective="quantile",
        alpha=alpha,

        n_estimators=800,
        learning_rate=0.03,

        num_leaves=31,
        max_depth=7,

        min_child_samples=100,

        subsample=0.9,
        colsample_bytree=0.9,

        reg_alpha=0.1,
        reg_lambda=0.1,

        random_state=RANDOM_STATE,
        n_jobs=-1,

        verbosity=-1
    )


# P10
print("Training P10 model...")

model_p10 = create_quantile_model(0.10)

model_p10.fit(
    X_train_processed,
    y_train
)


# P50
print("Training P50 model...")

model_p50 = create_quantile_model(0.50)

model_p50.fit(
    X_train_processed,
    y_train
)


# P90
print("Training P90 model...")

model_p90 = create_quantile_model(0.90)

model_p90.fit(
    X_train_processed,
    y_train
)

print("All quantile models trained.")


# ============================================================
# 12. RAW QUANTILE PREDICTIONS
# ============================================================

print("\n[8/10] Generating quantile predictions...")

cal_p10 = model_p10.predict(
    X_calibration_processed
)

cal_p50 = model_p50.predict(
    X_calibration_processed
)

cal_p90 = model_p90.predict(
    X_calibration_processed
)

test_p10 = model_p10.predict(
    X_test_processed
)

test_p50 = model_p50.predict(
    X_test_processed
)

test_p90 = model_p90.predict(
    X_test_processed
)


# ============================================================
# 13. QUANTILE ORDERING
# ============================================================

raw_cal_ordering = np.mean(
    (cal_p10 <= cal_p50) &
    (cal_p50 <= cal_p90)
)

raw_test_ordering = np.mean(
    (test_p10 <= test_p50) &
    (test_p50 <= test_p90)
)

print(
    f"Raw calibration ordering: "
    f"{raw_cal_ordering * 100:.3f}%"
)

print(
    f"Raw test ordering: "
    f"{raw_test_ordering * 100:.3f}%"
)


# ============================================================
# 14. MAPIE CONFORMAL QUANTILE REGRESSION
# ============================================================

print("\n[9/10] Applying MAPIE conformal calibration...")

# MAPIE expects:
# [lower quantile, upper quantile, median]

mapie_cqr = ConformalizedQuantileRegressor(
    estimator=[
        model_p10,
        model_p90,
        model_p50
    ],
    confidence_level=CONFIDENCE_LEVEL,
    prefit=True
)

mapie_cqr.conformalize(
    X_calibration_processed,
    y_calibration.to_numpy()
)

print("MAPIE conformalization completed.")


# ============================================================
# 15. FINAL TEST PREDICTION + INTERVAL
# ============================================================

final_p50, interval = mapie_cqr.predict_interval(
    X_test_processed
)

# MAPIE returns:
# (n_samples, 2, 1)

lower_bound = interval[:, 0, 0]
upper_bound = interval[:, 1, 0]

final_p50 = np.asarray(final_p50).reshape(-1)

# ------------------------------------------------------------
# IMPORTANT:
# Point prediction = RAW P50
#
# MAPIE is used for uncertainty interval.
# We do NOT use the previous central adjustment.
# ------------------------------------------------------------

final_prediction = test_p50.copy()


# ============================================================
# 16. CLEAN INTERVAL ORDERING
# ============================================================

# Make sure lower <= upper.

lower_bound_clean = np.minimum(
    lower_bound,
    upper_bound
)

upper_bound_clean = np.maximum(
    lower_bound,
    upper_bound
)

# LOS cannot be negative.

lower_bound_clean = np.maximum(
    lower_bound_clean,
    0
)

upper_bound_clean = np.maximum(
    upper_bound_clean,
    0
)


final_interval_ordering = np.mean(
    lower_bound_clean <= upper_bound_clean
)

print(
    f"Final interval ordering: "
    f"{final_interval_ordering * 100:.3f}%"
)


# ============================================================
# 17. OVERALL METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    final_prediction
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        final_prediction
    )
)

r2 = r2_score(
    y_test,
    final_prediction
)

# Calculate empirical coverage manually
y_test_array = y_test.to_numpy()

coverage = np.mean(
    (y_test_array >= lower_bound_clean) &
    (y_test_array <= upper_bound_clean)
)

# Calculate mean interval width manually
mean_width = np.mean(
    upper_bound_clean - lower_bound_clean
)

median_width = np.median(
    upper_bound_clean -
    lower_bound_clean
)

p90_width = np.percentile(
    upper_bound_clean -
    lower_bound_clean,
    90
)

prediction_error = (
    final_prediction -
    y_test.to_numpy()
)

mean_prediction_error = np.mean(
    prediction_error
)

underprediction_rate = np.mean(
    final_prediction < y_test.to_numpy()
)


# ============================================================
# 18. LONG-STAY METRICS
# ============================================================

long_mask = (
    y_test.to_numpy()
    > LONG_STAY_THRESHOLD
)

long_actual = y_test.to_numpy()[long_mask]

long_predicted = final_prediction[long_mask]

long_lower = lower_bound_clean[long_mask]

long_upper = upper_bound_clean[long_mask]

long_mae = mean_absolute_error(
    long_actual,
    long_predicted
)

long_rmse = np.sqrt(
    mean_squared_error(
        long_actual,
        long_predicted
    )
)

long_bias = np.mean(
    long_predicted -
    long_actual
)

long_coverage = np.mean(
    (long_actual >= long_lower) &
    (long_actual <= long_upper)
)

# ============================================================
# 19. LOS RANGE ANALYSIS
# ============================================================

def range_metrics(
    actual,
    predicted,
    lower,
    upper,
    low,
    high
):

    mask = (
        (actual >= low) &
        (actual < high)
    )

    if mask.sum() == 0:
        return None

    return {
        "count": int(mask.sum()),

        "actual_mean": float(
            np.mean(actual[mask])
        ),

        "predicted_mean": float(
            np.mean(predicted[mask])
        ),

        "mae": float(
            mean_absolute_error(
                actual[mask],
                predicted[mask]
            )
        ),

       "coverage": float(
    np.mean(
        (actual[mask] >= lower[mask]) &
        (actual[mask] <= upper[mask])
    )
),

        "mean_width": float(
            np.mean(
                upper[mask] -
                lower[mask]
            )
        )
    }


actual = y_test.to_numpy()

range_results = {

    "0_24h": range_metrics(
        actual,
        final_prediction,
        lower_bound_clean,
        upper_bound_clean,
        0,
        24
    ),

    "24_72h": range_metrics(
        actual,
        final_prediction,
        lower_bound_clean,
        upper_bound_clean,
        24,
        72
    ),

    "72_168h": range_metrics(
        actual,
        final_prediction,
        lower_bound_clean,
        upper_bound_clean,
        72,
        168
    ),

    "over_168h": range_metrics(
        actual,
        final_prediction,
        lower_bound_clean,
        upper_bound_clean,
        168,
        np.inf
    )
}


# ============================================================
# 20. PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL LOS MODEL RESULTS")
print("=" * 70)

print(f"\nPoint Prediction Metrics")
print(f"P50 MAE:              {mae:.3f} hours")
print(f"P50 RMSE:             {rmse:.3f} hours")
print(f"R²:                   {r2:.4f}")

print(f"\nPrediction Interval Metrics")
print(
    f"80% Coverage:         "
    f"{coverage * 100:.3f}%"
)

print(
    f"Mean Interval Width:  "
    f"{mean_width:.3f} hours"
)

print(
    f"Median Interval Width:"
    f" {median_width:.3f} hours"
)

print(
    f"P90 Interval Width:   "
    f"{p90_width:.3f} hours"
)

print(
    f"Interval Ordering:    "
    f"{final_interval_ordering * 100:.3f}%"
)

print(
    f"Mean Prediction Error:"
    f" {mean_prediction_error:.3f} hours"
)

print(
    f"Underprediction Rate: "
    f"{underprediction_rate * 100:.3f}%"
)


print(f"\nLong Stay (>168h)")
print(
    f"Cases:                "
    f"{long_mask.sum()}"
)

print(
    f"MAE:                  "
    f"{long_mae:.3f} hours"
)

print(
    f"RMSE:                 "
    f"{long_rmse:.3f} hours"
)

print(
    f"Bias:                 "
    f"{long_bias:.3f} hours"
)

print(
    f"Coverage:             "
    f"{long_coverage * 100:.3f}%"
)


print("\nLOS Range Performance")

for name, result in range_results.items():

    if result is None:
        continue

    print(f"\n{name}")

    print(
        f"  Cases:             "
        f"{result['count']}"
    )

    print(
        f"  Actual mean:       "
        f"{result['actual_mean']:.3f} h"
    )

    print(
        f"  Predicted mean:    "
        f"{result['predicted_mean']:.3f} h"
    )

    print(
        f"  MAE:               "
        f"{result['mae']:.3f} h"
    )

    print(
        f"  Coverage:          "
        f"{result['coverage'] * 100:.3f}%"
    )

    print(
        f"  Mean width:        "
        f"{result['mean_width']:.3f} h"
    )


# ============================================================
# 21. SAVE MODELS
# ============================================================

print("\n[10/10] Saving model artifacts...")


joblib.dump(
    model_p10,
    os.path.join(
        MODEL_DIR,
        "los_final_p10.joblib"
    )
)

joblib.dump(
    model_p50,
    os.path.join(
        MODEL_DIR,
        "los_final_p50.joblib"
    )
)

joblib.dump(
    model_p90,
    os.path.join(
        MODEL_DIR,
        "los_final_p90.joblib"
    )
)

joblib.dump(
    preprocessor,
    os.path.join(
        MODEL_DIR,
        "los_final_preprocessor.joblib"
    )
)

joblib.dump(
    mapie_cqr,
    os.path.join(
        MODEL_DIR,
        "los_final_mapie_cqr.joblib"
    )
)


# ============================================================
# 22. SAVE TEST PREDICTIONS
# ============================================================

predictions_df = test_df[
    [
        "stay_id",
        "patient_id",
        "arrival_time",
        "triage_acuity",
        "age",
        "charlson_index"
    ]
].copy()

predictions_df["actual_los_hours"] = (
    y_test.to_numpy()
)

predictions_df["predicted_los_hours"] = (
    final_prediction
)

predictions_df["los_lower_bound_80"] = (
    lower_bound_clean
)

predictions_df["los_upper_bound_80"] = (
    upper_bound_clean
)

predictions_df["prediction_error_hours"] = (
    final_prediction -
    y_test.to_numpy()
)

predictions_df["actual_long_stay"] = (
    y_test.to_numpy()
    > LONG_STAY_THRESHOLD
)

predictions_df["predicted_long_stay"] = (
    final_prediction
    > LONG_STAY_THRESHOLD
)


predictions_path = os.path.join(
    OUTPUT_DIR,
    "los_final_test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# 23. SAVE METRICS REPORT
# ============================================================

metrics_report = {

    "model": "Final LOS Model",

    "architecture":
        "Causal-Feature-Enhanced "
        "LightGBM Quantile Regression "
        "+ MAPIE CQR",

    "target": TARGET,

    "features": FEATURES,

    "train_size": len(train_df),

    "calibration_size":
        len(calibration_df),

    "test_size": len(test_df),

    "quantiles": [
        0.10,
        0.50,
        0.90
    ],

    "confidence_level":
        CONFIDENCE_LEVEL,

    "point_prediction":
        "Raw P50",

    "central_adjustment":
        "None",

    "metrics": {

        "mae_hours":
            float(mae),

        "rmse_hours":
            float(rmse),

        "r2":
            float(r2),

        "coverage_80":
            float(coverage),

        "mean_interval_width_hours":
            float(mean_width),

        "median_interval_width_hours":
            float(median_width),

        "p90_interval_width_hours":
            float(p90_width),

        "interval_ordering":
            float(final_interval_ordering),

        "mean_prediction_error_hours":
            float(mean_prediction_error),

        "underprediction_rate":
            float(underprediction_rate)
    },

    "long_stay_metrics": {

        "threshold_hours":
            LONG_STAY_THRESHOLD,

        "cases":
            int(long_mask.sum()),

        "mae_hours":
            float(long_mae),

        "rmse_hours":
            float(long_rmse),

        "bias_hours":
            float(long_bias),

        "coverage":
            float(long_coverage)
    },

    "range_metrics":
        range_results
}


metrics_path = os.path.join(
    OUTPUT_DIR,
    "los_final_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics_report,
        f,
        indent=4
    )


# ============================================================
# 24. COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("FINAL LOS MODEL COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nSaved model files:")

print(
    os.path.join(
        MODEL_DIR,
        "los_final_p10.joblib"
    )
)

print(
    os.path.join(
        MODEL_DIR,
        "los_final_p50.joblib"
    )
)

print(
    os.path.join(
        MODEL_DIR,
        "los_final_p90.joblib"
    )
)

print(
    os.path.join(
        MODEL_DIR,
        "los_final_preprocessor.joblib"
    )
)

print(
    os.path.join(
        MODEL_DIR,
        "los_final_mapie_cqr.joblib"
    )
)

print("\nSaved outputs:")

print(predictions_path)

print(metrics_path)

print("\nDone.")