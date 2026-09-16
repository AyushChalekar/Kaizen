# rval.py
# ==============================================================================
# KAIZEN HEALTHCARE - PREDICTIVE RESOURCE OPTIMIZATION SYSTEM
# END-TO-END LOS BENCHMARK & EVALUATION PIPELINE
#
# Dataset: synthetic_patient_stays_100k_fixed.csv
# Split: Chronological 70,000 Train / 15,000 Calibration / 15,000 Test
#
# Evaluates All 8 Model Variations:
#   1. Model 1: Baseline LightGBM Quantile Regression + MAPIE CQR
#   2. Model 1B: Log-Transformed Target log1p(LOS) + MAPIE CQR
#   3. Model 2: Hyperparameter Tuned LightGBM + MAPIE CQR
#   4. Model 3: Tail-Weighted LightGBM + MAPIE CQR
#   5. Model 6: Quantile-Specific Tail-Weighted LightGBM + MAPIE CQR
#   6. Model 7: Feature-Engineered Tail-Aware LightGBM + MAPIE CQR
#   7. Model 9: Two-Stage Mixture (Classifier + Dual Quantiles) + MAPIE CQR
#   8. Model Final: Causal Severity Enhanced LightGBM + MAPIE CQR
#
# Outputs:
#   - Individual test prediction CSVs in outputs/
#   - outputs/model_benchmark_comparison.csv
#   - outputs/model_benchmark_comparison.json
#   - Terminal Comparative Performance Leaderboard
# ==============================================================================

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from mapie.regression import ConformalizedQuantileRegressor

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION AND DIRECTORY RESOLUTION
# ==============================================================================

RANDOM_STATE = 42
TRAIN_SIZE = 70000
CALIBRATION_SIZE = 15000
TEST_SIZE = 15000
CONFIDENCE_LEVEL = 0.80
LONG_STAY_THRESHOLD = 168.0

SCRIPT_DIR = Path(__file__).resolve().parent          # kaizen/backend/scripts
BACKEND_DIR = SCRIPT_DIR.parent                      # kaizen/backend
PROJECT_ROOT = SCRIPT_DIR.parent.parent               # kaizen

# Resolved Path objects
DATA_PATH = PROJECT_ROOT / "data"
OUTPUT_DIR = BACKEND_DIR / "outputs"
MODEL_DIR = BACKEND_DIR / "models"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LOS MODEL COMPREHENSIVE BENCHMARK PIPELINE")
print("=" * 80)
print(f"Dataset path : {DATA_PATH}")
print(f"Outputs path : {OUTPUT_DIR}")


# ==============================================================================
# 2. HELPER CLASSES FOR QUANTILE ORDERING AND TWO-STAGE BLENDING
# ==============================================================================

class QuantileOrderingWrapper(BaseEstimator, RegressorMixin):
    """Enforces P10 <= P50 <= P90 row-wise to eliminate quantile crossing."""
    def __init__(self, p10_model=None, p50_model=None, p90_model=None, role="median"):
        self.p10_model = p10_model
        self.p50_model = p50_model
        self.p90_model = p90_model
        self.role = role
        self.is_fitted_ = True

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        p10 = np.asarray(self.p10_model.predict(X))
        p50 = np.asarray(self.p50_model.predict(X))
        p90 = np.asarray(self.p90_model.predict(X))
        stacked = np.column_stack([p10, p50, p90])
        sorted_preds = np.sort(stacked, axis=1)
        if self.role == "lower":
            return sorted_preds[:, 0]
        elif self.role == "median":
            return sorted_preds[:, 1]
        elif self.role == "upper":
            return sorted_preds[:, 2]
        raise ValueError("role must be 'lower', 'median', or 'upper'")


class TwoStageQuantileModel(BaseEstimator, RegressorMixin):
    """Blends normal-stay and long-stay quantiles via classifier probability."""
    def __init__(self, classifier=None, normal_model=None, long_model=None):
        self.classifier = classifier
        self.normal_model = normal_model
        self.long_model = long_model
        self.is_fitted_ = True

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        prob_long = self.classifier.predict_proba(X)[:, 1]
        pred_normal = self.normal_model.predict(X)
        pred_long = self.long_model.predict(X)
        return (1.0 - prob_long) * pred_normal + prob_long * pred_long


# ==============================================================================
# 3. METRIC COMPUTATION ENGINE
# ==============================================================================

def compute_metrics(actual, predicted, lower, upper, model_name):
    """Standardized metrics computation across point, interval, and tail."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    clean_lower = np.maximum(np.minimum(lower, upper), 0.0)
    clean_upper = np.maximum(np.maximum(lower, upper), 0.0)

    error = predicted - actual
    abs_error = np.abs(error)
    widths = clean_upper - clean_lower

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r2 = float(r2_score(actual, predicted))
    median_ae = float(np.median(abs_error))
    mean_bias = float(np.mean(error))
    underpred_pct = float(np.mean(predicted < actual) * 100.0)

    coverage_pct = float(np.mean((actual >= clean_lower) & (actual <= clean_upper)) * 100.0)
    mean_width = float(np.mean(widths))
    median_width = float(np.median(widths))
    ordering_pct = float(np.mean(clean_lower <= clean_upper) * 100.0)

    # Extreme long-stay analysis (> 168h)
    tail_mask = actual > LONG_STAY_THRESHOLD
    tail_count = int(np.sum(tail_mask))

    if tail_count > 0:
        tail_actual = actual[tail_mask]
        tail_pred = predicted[tail_mask]
        tail_mae = float(mean_absolute_error(tail_actual, tail_pred))
        tail_bias = float(np.mean(tail_pred - tail_actual))
        tail_coverage = float(
            np.mean((tail_actual >= clean_lower[tail_mask]) & (tail_actual <= clean_upper[tail_mask])) * 100.0
        )
        tail_2x_underpred = float(
            np.mean((tail_actual / np.clip(tail_pred, 0.01, None)) >= 2.0) * 100.0
        )
    else:
        tail_mae = 0.0
        tail_bias = 0.0
        tail_coverage = 0.0
        tail_2x_underpred = 0.0

    return {
        "Model": model_name,
        "MAE_h": mae,
        "RMSE_h": rmse,
        "R2": r2,
        "Median_AE_h": median_ae,
        "Bias_h": mean_bias,
        "Underpred_pct": underpred_pct,
        "Coverage_80_pct": coverage_pct,
        "Mean_Width_h": mean_width,
        "Median_Width_h": median_width,
        "Ordering_pct": ordering_pct,
        "Tail_Count": tail_count,
        "Tail_MAE_h": tail_mae,
        "Tail_Bias_h": tail_bias,
        "Tail_Coverage_pct": tail_coverage,
        "Tail_2x_Underpred_pct": tail_2x_underpred,
    }


# ==============================================================================
# 4. DATA PREPARATION AND CAUSAL FEATURE ENGINEERING
# ==============================================================================

print("\n[1/10] Loading and preparing dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Total records loaded: {len(df):,}")

# Parse arrival time and normalize arrival hour
df["arrival_time"] = pd.to_datetime(df["arrival_time"])
df = df.sort_values("arrival_time").reset_index(drop=True)
df["arrival_hour"] = df["arrival_time"].dt.hour

# Derived clinical indices
df["shock_index"] = df["heart_rate"] / df["sbp"].replace(0, np.nan)
df["pulse_pressure"] = df["sbp"] - df["dbp"]
df["mean_arterial_pressure"] = df["dbp"] + (df["sbp"] - df["dbp"]) / 3.0
df["oxygen_deficit"] = 100.0 - df["o2_sat"]
df["temperature_deviation"] = np.abs(df["temp_c"] - 37.0)
df["arrival_hour_sin"] = np.sin(2 * np.pi * df["arrival_hour"] / 24.0)
df["arrival_hour_cos"] = np.cos(2 * np.pi * df["arrival_hour"] / 24.0)

# Causal severity score calculation
hr_sev = np.clip((df["heart_rate"] - 80.0) / 60.0, 0.0, 1.0)
bp_sev = np.clip((120.0 - df["sbp"]) / 60.0, 0.0, 1.0)
rr_sev = np.clip((df["resp_rate"] - 18.0) / 20.0, 0.0, 1.0)
o2_sev = np.clip((95.0 - df["o2_sat"]) / 25.0, 0.0, 1.0)
temp_sev = np.clip(np.abs(df["temp_c"] - 37.0) / 4.0, 0.0, 1.0)
comorb_sev = np.clip(df["charlson_index"] / 10.0, 0.0, 1.0)
esi_sev = (6.0 - df["triage_acuity"]) / 5.0

df["severity_score"] = np.clip(
    0.25 * esi_sev
    + 0.15 * hr_sev
    + 0.20 * bp_sev
    + 0.15 * rr_sev
    + 0.15 * o2_sev
    + 0.05 * temp_sev
    + 0.05 * comorb_sev,
    0.0,
    1.0,
)

if "requires_ventilation" in df.columns:
    df["requires_ventilation"] = df["requires_ventilation"].astype(int)
else:
    df["requires_ventilation"] = 0

# Chronological 70k / 15k / 15k split
train_df = df.iloc[:TRAIN_SIZE].copy()
cal_df = df.iloc[TRAIN_SIZE : TRAIN_SIZE + CALIBRATION_SIZE].copy()
test_df = df.iloc[TRAIN_SIZE + CALIBRATION_SIZE : TRAIN_SIZE + CALIBRATION_SIZE + TEST_SIZE].copy()

y_train = train_df["los_hours"].values
y_cal = cal_df["los_hours"].values
y_test = test_df["los_hours"].values

print(f"Train split       : {len(train_df):,} rows")
print(f"Calibration split : {len(cal_df):,} rows")
print(f"Test split        : {len(test_df):,} rows")

# Feature definition matrices
BASE_NUMERIC = [
    "age", "charlson_index", "heart_rate", "sbp", "dbp",
    "o2_sat", "resp_rate", "temp_c", "triage_acuity"
]
CATEGORICAL = ["gender", "chief_complaint"]
BASE_FEATURES = BASE_NUMERIC + CATEGORICAL

DERIVED_NUMERIC = BASE_NUMERIC + [
    "shock_index", "pulse_pressure", "mean_arterial_pressure",
    "oxygen_deficit", "temperature_deviation", "arrival_hour_sin", "arrival_hour_cos"
]
FEAT_ENG_FEATURES = DERIVED_NUMERIC + CATEGORICAL

CAUSAL_NUMERIC = BASE_NUMERIC + ["requires_ventilation", "severity_score"]
CAUSAL_FEATURES = CAUSAL_NUMERIC + CATEGORICAL


def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )


benchmark_results = []

# Shared hyperparameters
m2_params = {
    "objective": "quantile",
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
    "n_jobs": -1,
    "verbosity": -1,
}

m1_params = {
    "objective": "quantile",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


# ==============================================================================
# 5. MODEL 1: BASELINE LIGHTGBM + MAPIE CQR
# ==============================================================================

print("\n[2/10] Training Model 1: Baseline...")
prep_m1 = build_preprocessor(BASE_NUMERIC, CATEGORICAL)
X_train_m1 = prep_m1.fit_transform(train_df[BASE_FEATURES])
X_cal_m1 = prep_m1.transform(cal_df[BASE_FEATURES])
X_test_m1 = prep_m1.transform(test_df[BASE_FEATURES])

m1_p10 = LGBMRegressor(alpha=0.10, **m1_params).fit(X_train_m1, y_train)
m1_p50 = LGBMRegressor(alpha=0.50, **m1_params).fit(X_train_m1, y_train)
m1_p90 = LGBMRegressor(alpha=0.90, **m1_params).fit(X_train_m1, y_train)

m1_mapie = ConformalizedQuantileRegressor(
    estimator=[m1_p10, m1_p90, m1_p50],
    confidence_level=CONFIDENCE_LEVEL,
    prefit=True,
)
m1_mapie.conformalize(X_cal_m1, y_cal)
pred_m1, interval_m1 = m1_mapie.predict_interval(X_test_m1)

p50_m1 = np.maximum(pred_m1.reshape(-1), 0.0)
low_m1 = np.maximum(interval_m1[:, 0, 0], 0.0)
upp_m1 = np.maximum(interval_m1[:, 1, 0], 0.0)
low_m1 = np.minimum(low_m1, p50_m1)
upp_m1 = np.maximum(upp_m1, p50_m1)

res_m1 = compute_metrics(y_test, p50_m1, low_m1, upp_m1, "Model 1: Baseline")
benchmark_results.append(res_m1)

out_m1 = test_df[["stay_id", "patient_id", "arrival_time", "triage_acuity"]].copy()
out_m1["actual_los_hours"] = y_test
out_m1["predicted_los_hours"] = p50_m1
out_m1["los_lower_80"] = low_m1
out_m1["los_upper_80"] = upp_m1
out_m1.to_csv(OUTPUT_DIR / "los_predictions_test.csv", index=False)


# ==============================================================================
# 6. MODEL 1B: LOG-TRANSFORMED TARGET
# ==============================================================================

print("\n[3/10] Training Model 1B: Log-Transformed Target...")
y_train_log = np.log1p(y_train)
y_cal_log = np.log1p(y_cal)

m1b_p10 = LGBMRegressor(alpha=0.10, **m1_params).fit(X_train_m1, y_train_log)
m1b_p50 = LGBMRegressor(alpha=0.50, **m1_params).fit(X_train_m1, y_train_log)
m1b_p90 = LGBMRegressor(alpha=0.90, **m1_params).fit(X_train_m1, y_train_log)

m1b_mapie = ConformalizedQuantileRegressor(
    estimator=[m1b_p10, m1b_p90, m1b_p50],
    confidence_level=CONFIDENCE_LEVEL,
    prefit=True,
)
m1b_mapie.conformalize(X_cal_m1, y_cal_log)
pred_log, interval_log = m1b_mapie.predict_interval(X_test_m1)

p50_m1b = np.maximum(np.expm1(pred_log.reshape(-1)), 0.0)
low_m1b = np.maximum(np.expm1(interval_log[:, 0, 0]), 0.0)
upp_m1b = np.maximum(np.expm1(interval_log[:, 1, 0]), 0.0)
low_m1b = np.minimum(low_m1b, p50_m1b)
upp_m1b = np.maximum(upp_m1b, p50_m1b)

res_m1b = compute_metrics(y_test, p50_m1b, low_m1b, upp_m1b, "Model 1B: Log-Target")
benchmark_results.append(res_m1b)

out_m1b = test_df[["stay_id", "patient_id", "arrival_time", "triage_acuity"]].copy()
out_m1b["actual_los_hours"] = y_test
out_m1b["predicted_los_hours"] = p50_m1b
out_m1b["los_lower_80"] = low_m1b
out_m1b["los_upper_80"] = upp_m1b
out_m1b.to_csv(OUTPUT_DIR / "los_predictions_test_log.csv", index=False)


# ==============================================================================
# 7. MODEL 2: TUNED HYPERPARAMETERS
# ==============================================================================

print("\n[4/10] Training Model 2: Tuned Hyperparameters...")
m2_p10 = LGBMRegressor(alpha=0.10, **m2_params).fit(X_train_m1, y_train)
m2_p50 = LGBMRegressor(alpha=0.50, **m2_params).fit(X_train_m1, y_train)
m2_p90 = LGBMRegressor(alpha=0.90, **m2_params).fit(X_train_m1, y_train)

m2_mapie = ConformalizedQuantileRegressor(
    estimator=[m2_p10, m2_p90, m2_p50],
    confidence_level=CONFIDENCE_LEVEL,
    prefit=True,
)
m2_mapie.conformalize(X_cal_m1, y_cal)
pred_m2, interval_m2 = m2_mapie.predict_interval(X_test_m1)

p50_m2 = np.maximum(m2_p50.predict(X_test_m1), 0.0)
low_m2 = np.maximum(interval_m2[:, 0, 0], 0.0)
upp_m2 = np.maximum(interval_m2[:, 1, 0], 0.0)
low_m2 = np.minimum(low_m2, p50_m2)
upp_m2 = np.maximum(upp_m2, p50_m2)

res_m2 = compute_metrics(y_test, p50_m2, low_m2, upp_m2, "Model 2: Tuned")
benchmark_results.append(res_m2)

out_m2 = test_df[["stay_id", "patient_id", "arrival_time", "triage_acuity"]].copy()
out_m2["actual_los_hours"] = y_test
out_m2["predicted_los_p50_hours"] = p50_m2
out_m2["conformal_lower_hours"] = low_m2
out_m2["conformal_upper_hours"] = upp_m2
out_m2.to_csv(OUTPUT_DIR / "los_predictions_test_tuned.csv", index=False)


# ==============================================================================
# 8. MODEL 3: TAIL-WEIGHTED LIGHTGBM
# ==============================================================================

print("\n[5/10] Training Model 3: Tail-Weighted LightGBM...")
weights_m3 = np.ones(len(y_train), dtype=float)
weights_m3[(y_train >= 24) & (y_train < 72)] = 1.25
weights_m3[(y_train >= 72) & (y_train < 168)] = 1.85
weights_m3[y_train >= 168] = 2.75

m3_p10 = LGBMRegressor(alpha=0.10, **m2_params).fit(X_train_m1, y_train, sample_weight=weights_m3)
m3_p50 = LGBMRegressor(alpha=0.50, **m2_params).fit(X_train_m1, y_train, sample_weight=weights_m3)
m3_p90 = LGBMRegressor(alpha=0.90, **m2_params).fit(X_train_m1, y_train, sample_weight=weights_m3)

m3_mapie = ConformalizedQuantileRegressor(
    estimator=[m3_p10, m3_p90, m3_p50],
    confidence_level=CONFIDENCE_LEVEL,
    prefit=True,
)
m3_mapie.conformalize(X_cal_m1, y_cal)
pred_m3, interval_m3 = m3_mapie.predict_interval(X_test_m1)

p50_m3 = np.maximum(m3_p50.predict(X_test_m1), 0.0)
low_m3 = np.maximum(interval_m3[:, 0, 0], 0.0)
upp_m3 = np.maximum(interval_m3[:, 1, 0], 0.0)
low_m3 = np.minimum(low_m3, p50_m3)
upp_m3 = np.maximum(upp_m3, p50_m3)

res_m3 = compute_metrics(y_test, p50_m3, low_m3, upp_m3, "Model 3: Tail-Weighted")
benchmark_results.append(res_m3)

out_m3 = test_df[["stay_id", "patient_id", "arrival_time", "triage_acuity"]].copy()
out_m3["actual_los_hours"] = y_test
out_m3["predicted_los_p50_hours"] = p50_m3
out_m3["conformal_lower_hours"] = low_m3
out_m3["conformal_upper_hours"] = upp_m3
out_m3.to_csv(OUTPUT_DIR / "los_predictions_test_tail_weighted.csv", index=False)


# ==============================================================================
# 9. MODEL 6: QUANTILE-SPECIFIC TAIL-WEIGHTED LIGHTGBM
# ==============================================================================

print("\n[6/10] Training Model 6: Quantile-Specific Tail-Weighted...")
w_p10 = np.ones(len(y_train))
w_p10[(y_train >= 24) & (y_train < 72)] = 1.05
w_p10[(y_train >= 72) & (y_train <= 168)] = 1.20
w_p10[y_train > 168] = 1.35

w_p50 = np.ones(len(y_train))
w_p50[(y_train >= 24) & (y_train < 72)] = 1.25
w_p50[(y_train >= 72) & (y_train <= 168)] = 1.80
w_p50[y_train > 168] = 2.50

w_p90 = np.ones(len(y_train))
w_p90[(y_train >= 24) & (y_train < 72)] = 1.30
w_p90[(y_train >= 72) & (y_train <= 168)] = 2.20
w_p90[y_train > 168] = 4.00

m6_p10 = LGBMRegressor(alpha=0.10, **m2_params).fit(X_train_m1, y_train, sample_weight=w_p10)
m6_p50 = LGBMRegressor(alpha=0.50, **m2_params).fit(X_train_m1, y_train, sample_weight=w_p50)
m6_p90 = LGBMRegressor(alpha=0.90, **m2_params).fit(X_train_m1, y_train, sample_weight=w_p90)

wrap_m6_p10 = QuantileOrderingWrapper(m6_p10, m6_p50, m6_p90, role="lower")
wrap_m6_p50 = QuantileOrderingWrapper(m6_p10, m6_p50, m6_p90, role="median")
wrap_m6_p90 = QuantileOrderingWrapper(m6_p10, m6_p50, m6_p90, role="upper")

m6_mapie = ConformalizedQuantileRegressor(
    estimator=[wrap_m6_p10, wrap_m6_p90, wrap_m6_p50],
    confidence_level=CONFIDENCE_LEVEL,
    prefit=True,
)
m6_mapie.conformalize(X_cal_m1, y_cal)
pred_m6, interval_m6 = m6_mapie.predict_interval(X_test_m1)

p50_m6 = np.maximum(wrap_m6_p50.predict(X_test_m1), 0.0)
low_m6 = np.maximum(interval_m6[:, 0, 0], 0.0)
upp_m6 = np.maximum(interval_m6[:, 1, 0], 0.0)
low_m6 = np.minimum(low_m6, p50_m6)
upp_m6 = np.maximum(upp_m6, p50_m6)

res_m6 = compute_metrics(y_test, p50_m6, low_m6, upp_m6, "Model 6: Quantile-Weighted")
benchmark_results.append(res_m6)

out_m6 = test_df[["stay_id", "patient_id", "arrival_time", "triage_acuity"]].copy()
out_m6["actual_los_hours"] = y_test
out_m6["corrected_p50_hours"] = p50_m6
out_m6["conformal_lower_hours"] = low_m6
out_m6["conformal_upper_hours"] = upp_m6
out_m6.to_csv(OUTPUT_DIR / "los_predictions_test_quantile_specific.csv", index=False)


# ==============================================================================
# 10. MODEL 7: FEATURE-ENGINEERED TAIL-AWARE LIGHTGBM
# ==============================================================================

print("\n[7/10] Training Model 7: Feature-Engineered...")
prep_m7 = build_preprocessor(DERIVED_NUMERIC, CATEGORICAL)
X_train_m7 = prep_m7.fit_transform(train_df[FEAT_ENG_FEATURES])
X_cal_m7 = prep_m7.transform(cal_df[FEAT_ENG_FEATURES])
X_test_m7 = prep_m7.transform(test_df[FEAT_ENG_FEATURES])

m7_p10 = LGBMRegressor(alpha=0.10, **m2_params).fit(X_train_m7, y_train, sample_weight=w_p10)
m7_p50 = LGBMRegressor(alpha=0.50, **m2_params).fit(X_train_m7, y_train, sample_weight=w_p50)
m7_p90 = LGBMRegressor(alpha=0.90, **m2_params).fit(X_train_m7, y_train, sample_weight=w_p90)

wrap_m7_p10 = QuantileOrderingWrapper(m7_p10, m7_p50, m7_p90, role="lower")
wrap_m7_p50 = QuantileOrderingWrapper(m7_p10, m7_p50, m7_p90, role="median")
wrap_m7_p90 = QuantileOrderingWrapper(m7_p10, m7_p50, m7_p90, role="upper")

m7_mapie = ConformalizedQuantileRegressor(
    estimator=[wrap_m7_p10, wrap_m7_p90, wrap_m7_p50],
    confidence_level=CONFIDENCE_LEVEL,
    prefit=True,
)
m7_mapie.conformalize(X_cal_m7, y_cal)
pred_m7, interval_m7 = m7_mapie.predict_interval(X_test_m7)

p50_m7 = np.maximum(wrap_m7_p50.predict(X_test_m7), 0.0)
low_m7 = np.maximum(interval_m7[:, 0, 0], 0.0)
upp_m7 = np.maximum(interval_m7[:, 1, 0], 0.0)
low_m7 = np.minimum(low_m7, p50_m7)
upp_m7 = np.maximum(upp_m7, p50_m7)

res_m7 = compute_metrics(y_test, p50_m7, low_m7, upp_m7, "Model 7: Feature-Engineered")
benchmark_results.append(res_m7)

out_m7 = test_df[["stay_id", "patient_id", "arrival_time", "triage_acuity"]].copy()
out_m7["actual_los_hours"] = y_test
out_m7["conformal_p50_hours"] = p50_m7
out_m7["conformal_lower_hours"] = low_m7
out_m7["conformal_upper_hours"] = upp_m7
out_m7.to_csv(OUTPUT_DIR / "los_predictions_model7_test.csv", index=False)


# ==============================================================================
# 11. MODEL 9: TAIL-AWARE TWO-STAGE MIXTURE MODEL
# ==============================================================================

print("\n[8/10] Training Model 9: Two-Stage Mixture...")
prep_m9 = build_preprocessor(CAUSAL_NUMERIC, CATEGORICAL)
X_train_m9 = prep_m9.fit_transform(train_df[CAUSAL_FEATURES])
X_cal_m9 = prep_m9.transform(cal_df[CAUSAL_FEATURES])
X_test_m9 = prep_m9.transform(test_df[CAUSAL_FEATURES])

y_train_long = (y_train > LONG_STAY_THRESHOLD).astype(int)
pos_cnt = int(y_train_long.sum())
neg_cnt = len(y_train_long) - pos_cnt
scale_pos = neg_cnt / max(pos_cnt, 1)

clf = LGBMClassifier(
    objective="binary", n_estimators=800, num_leaves=31, max_depth=7,
    learning_rate=0.03, min_child_samples=100, subsample=0.85,
    colsample_bytree=0.85, reg_alpha=0.10, reg_lambda=1.00,
    scale_pos_weight=scale_pos, random_state=RANDOM_STATE,
    n_jobs=-1, verbosity=-1,
).fit(X_train_m9, y_train_long)

norm_mask = y_train <= LONG_STAY_THRESHOLD
long_mask = y_train > LONG_STAY_THRESHOLD

norm_p10 = LGBMRegressor(alpha=0.10, **m2_params).fit(X_train_m9[norm_mask], y_train[norm_mask])
norm_p50 = LGBMRegressor(alpha=0.50, **m2_params).fit(X_train_m9[norm_mask], y_train[norm_mask])
norm_p90 = LGBMRegressor(alpha=0.90, **m2_params).fit(X_train_m9[norm_mask], y_train[norm_mask])

long_p10 = LGBMRegressor(alpha=0.10, **m2_params).fit(X_train_m9[long_mask], y_train[long_mask])
long_p50 = LGBMRegressor(alpha=0.50, **m2_params).fit(X_train_m9[long_mask], y_train[long_mask])
long_p90 = LGBMRegressor(alpha=0.90, **m2_params).fit(X_train_m9[long_mask], y_train[long_mask])

two_p10 = TwoStageQuantileModel(clf, norm_p10, long_p10)
two_p50 = TwoStageQuantileModel(clf, norm_p50, long_p50)
two_p90 = TwoStageQuantileModel(clf, norm_p90, long_p90)

wrap_m9_p10 = QuantileOrderingWrapper(two_p10, two_p50, two_p90, role="lower")
wrap_m9_p50 = QuantileOrderingWrapper(two_p10, two_p50, two_p90, role="median")
wrap_m9_p90 = QuantileOrderingWrapper(two_p10, two_p50, two_p90, role="upper")

m9_mapie = ConformalizedQuantileRegressor(
    estimator=[wrap_m9_p10, wrap_m9_p90, wrap_m9_p50],
    confidence_level=CONFIDENCE_LEVEL,
    prefit=True,
)
m9_mapie.conformalize(X_cal_m9, y_cal)
pred_m9, interval_m9 = m9_mapie.predict_interval(X_test_m9)

p50_m9 = np.maximum(wrap_m9_p50.predict(X_test_m9), 0.0)
low_m9 = np.maximum(interval_m9[:, 0, 0], 0.0)
upp_m9 = np.maximum(interval_m9[:, 1, 0], 0.0)
low_m9 = np.minimum(low_m9, p50_m9)
upp_m9 = np.maximum(upp_m9, p50_m9)

res_m9 = compute_metrics(y_test, p50_m9, low_m9, upp_m9, "Model 9: Two-Stage")
benchmark_results.append(res_m9)

out_m9 = test_df[["stay_id", "patient_id", "arrival_time", "triage_acuity"]].copy()
out_m9["actual_los_hours"] = y_test
out_m9["predicted_los_hours"] = p50_m9
out_m9["los_lower_80"] = low_m9
out_m9["los_upper_80"] = upp_m9
out_m9.to_csv(OUTPUT_DIR / "los_predictions_model9_test.csv", index=False)


# ==============================================================================
# 12. MODEL FINAL: CAUSAL SEVERITY ENHANCED
# ==============================================================================

print("\n[9/10] Training Model Final: Causal Severity Enhanced...")
fin_p10 = LGBMRegressor(alpha=0.10, **m2_params).fit(X_train_m9, y_train)
fin_p50 = LGBMRegressor(alpha=0.50, **m2_params).fit(X_train_m9, y_train)
fin_p90 = LGBMRegressor(alpha=0.90, **m2_params).fit(X_train_m9, y_train)

fin_mapie = ConformalizedQuantileRegressor(
    estimator=[fin_p10, fin_p90, fin_p50],
    confidence_level=CONFIDENCE_LEVEL,
    prefit=True,
)
fin_mapie.conformalize(X_cal_m9, y_cal)
pred_fin, interval_fin = fin_mapie.predict_interval(X_test_m9)

p50_fin = np.maximum(fin_p50.predict(X_test_m9), 0.0)
low_fin = np.maximum(interval_fin[:, 0, 0], 0.0)
upp_fin = np.maximum(interval_fin[:, 1, 0], 0.0)
low_fin = np.minimum(low_fin, p50_fin)
upp_fin = np.maximum(upp_fin, p50_fin)

res_fin = compute_metrics(y_test, p50_fin, low_fin, upp_fin, "Model Final: Causal Severity")
benchmark_results.append(res_fin)

out_fin = test_df[["stay_id", "patient_id", "arrival_time", "triage_acuity"]].copy()
out_fin["actual_los_hours"] = y_test
out_fin["predicted_los_hours"] = p50_fin
out_fin["los_lower_bound_80"] = low_fin
out_fin["los_upper_bound_80"] = upp_fin
out_fin.to_csv(OUTPUT_DIR / "los_final_test_predictions.csv", index=False)


# ==============================================================================
# 13. GENERATE COMPARATIVE BENCHMARK LEADERBOARD
# ==============================================================================

print("\n[10/10] Generating master benchmark report...")
bench_df = pd.DataFrame(benchmark_results)
bench_df = bench_df.sort_values(by="MAE_h", ascending=True).reset_index(drop=True)

csv_report = OUTPUT_DIR / "model_benchmark_comparison.csv"
json_report = OUTPUT_DIR / "model_benchmark_comparison.json"
bench_df.to_csv(csv_report, index=False)

with open(json_report, "w", encoding="utf-8") as f:
    json.dump(bench_df.to_dict(orient="records"), f, indent=4)

print("\n" + "=" * 115)
print(f"{'LOS MODEL COMPARATIVE BENCHMARK (TEST SET N=15,000)':^115}")
print("=" * 115)

header = (
    f"{'Model':<30} | {'MAE (h)':>8} | {'RMSE (h)':>8} | {'R2':>8} | "
    f"{'Bias (h)':>8} | {'Cov 80%':>10} | {'Width (h)':>12} | "
    f"{'Tail MAE':>11} | {'Tail Bias':>11} | {'Tail Cov%':>12}"
)
print(header)
print("-" * 115)

for _, row in bench_df.iterrows():
    line = (
        f"{row['Model']:<30} | {row['MAE_h']:>8.3f} | {row['RMSE_h']:>8.3f} | "
        f"{row['R2']:>8.4f} | {row['Bias_h']:>+8.3f} | {row['Coverage_80_pct']:>9.2f}% | "
        f"{row['Mean_Width_h']:>12.2f} | {row['Tail_MAE_h']:>11.2f} | "
        f"{row['Tail_Bias_h']:>+11.2f} | {row['Tail_Coverage_pct']:>11.2f}%"
    )
    print(line)

print("=" * 115)
print(f"\nSaved CSV report  : {csv_report}")
print(f"Saved JSON report : {json_report}")
print("All 8 models evaluated and saved to outputs/.")