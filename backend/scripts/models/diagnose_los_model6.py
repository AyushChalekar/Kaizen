"""
MODEL #6 - COMPLETE LOS DIAGNOSTIC
==================================

Diagnoses the existing:
Quantile-specific tail-weighted LightGBM P10/P50/P90
+ MAPIE CQR

IMPORTANT:
- Does NOT retrain the model.
- Does NOT modify Model #6.
- Uses the existing Model #6 predictions.
- Checks algorithm behaviour, parameters indirectly,
  feature signal, tail behaviour, bias, calibration,
  and prediction compression.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path.cwd()

DATA_PATH = ROOT / "data" / "synthetic_patient_stays_100k.csv"

OUTPUT_DIR = ROOT / "outputs"

PRED_PATH = (
    OUTPUT_DIR /
    "los_predictions_test_quantile_specific.csv"
)

CONFIG_PATH = (
    OUTPUT_DIR /
    "los_quantile_specific_configuration.json"
)

METRICS_PATH = (
    OUTPUT_DIR /
    "los_quantile_specific_metrics.json"
)

OUTPUT_DIR.mkdir(exist_ok=True)


print("=" * 78)
print("MODEL #6 - COMPLETE LOS DIAGNOSTIC")
print("=" * 78)

print(f"Working directory : {ROOT}")
print(f"Dataset           : {DATA_PATH}")
print(f"Predictions       : {PRED_PATH}")


# ============================================================
# 2. CHECK FILES
# ============================================================

print("\n" + "=" * 78)
print("1. CHECKING REQUIRED FILES")
print("=" * 78)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

if not PRED_PATH.exists():
    raise FileNotFoundError(
        f"Model #6 prediction file not found:\n{PRED_PATH}"
    )

print("Dataset file       : PASS")
print("Prediction file    : PASS")

if CONFIG_PATH.exists():
    print("Configuration file : PASS")
else:
    print("Configuration file : NOT FOUND")

if METRICS_PATH.exists():
    print("Metrics file       : PASS")
else:
    print("Metrics file       : NOT FOUND")


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 78)
print("2. LOADING DATA")
print("=" * 78)

df = pd.read_csv(DATA_PATH)

pred = pd.read_csv(PRED_PATH)

print(f"Original dataset : {df.shape}")
print(f"Prediction file  : {pred.shape}")


# ============================================================
# 4. SHOW PREDICTION COLUMNS
# ============================================================

print("\n" + "=" * 78)
print("3. MODEL OUTPUT COLUMNS")
print("=" * 78)

print(list(pred.columns))


# ============================================================
# 5. FIND IMPORTANT COLUMNS
# ============================================================

def find_column(frame, candidates):

    lower_map = {
        str(c).lower(): c
        for c in frame.columns
    }

    # Exact match
    for candidate in candidates:

        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    # Partial match
    for column in frame.columns:

        column_lower = str(column).lower()

        for candidate in candidates:

            if candidate.lower() in column_lower:
                return column

    return None


ID_COL = find_column(
    pred,
    ["stay_id"]
)

ACTUAL_COL = find_column(
    pred,
    [
        "actual_los_hours",
        "actual_los",
        "los_hours",
        "actual"
    ]
)

PRED_COL = find_column(
    pred,
    [
        "predicted_los_hours",
        "predicted_los",
        "p50_prediction",
        "p50",
        "prediction"
    ]
)

LOWER_COL = find_column(
    pred,
    [
        "lower_bound_hours",
        "lower_bound",
        "prediction_lower",
        "lower"
    ]
)

UPPER_COL = find_column(
    pred,
    [
        "upper_bound_hours",
        "upper_bound",
        "prediction_upper",
        "upper"
    ]
)


print(f"ID column         : {ID_COL}")
print(f"Actual LOS column : {ACTUAL_COL}")
print(f"P50 column        : {PRED_COL}")
print(f"Lower interval    : {LOWER_COL}")
print(f"Upper interval    : {UPPER_COL}")


# ============================================================
# 6. MERGE ORIGINAL FEATURES
# ============================================================

print("\n" + "=" * 78)
print("4. MERGING ORIGINAL ARRIVAL-TIME FEATURES")
print("=" * 78)

feature_columns = [
    "stay_id",
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
    "arrival_hour"
]

available_features = [
    c for c in feature_columns
    if c in df.columns
]

source_features = df[available_features].copy()

# Prevent duplicate columns
duplicate_features = [
    c for c in available_features
    if c != "stay_id" and c in pred.columns
]

pred_clean = pred.drop(
    columns=duplicate_features,
    errors="ignore"
)

merged = pred_clean.merge(
    source_features,
    on="stay_id",
    how="left"
)

print(f"Merged rows       : {len(merged):,}")
print(
    f"Features available: "
    f"{len(available_features)}"
)


# ============================================================
# 7. PREPARE TARGET/PREDICTION
# ============================================================

if ACTUAL_COL is None:

    if "los_hours" in merged.columns:
        ACTUAL_COL = "los_hours"

    else:
        raise ValueError(
            "Could not find actual LOS column."
        )


if PRED_COL is None:

    numeric_columns = (
        merged
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    excluded = {
        "stay_id",
        "los_hours",
        "actual_los_hours"
    }

    candidates = [
        c for c in numeric_columns
        if c not in excluded
    ]

    preferred = [
        c for c in candidates
        if (
            "pred" in c.lower()
            or "p50" in c.lower()
            or "median" in c.lower()
        )
    ]

    if preferred:
        PRED_COL = preferred[0]

    elif candidates:
        PRED_COL = candidates[0]

    else:
        raise ValueError(
            "Could not identify P50 prediction column."
        )


y = pd.to_numeric(
    merged[ACTUAL_COL],
    errors="coerce"
)

p50 = pd.to_numeric(
    merged[PRED_COL],
    errors="coerce"
)

valid = (
    y.notna()
    & p50.notna()
)

merged = merged.loc[valid].copy()

y = y.loc[valid]

p50 = p50.loc[valid]

print(f"Valid rows        : {len(merged):,}")


# ============================================================
# 8. CREATE RESIDUALS
# ============================================================

merged["actual_los"] = y.values

merged["predicted_los"] = p50.values

merged["error"] = (
    merged["predicted_los"]
    - merged["actual_los"]
)

merged["absolute_error"] = (
    merged["error"].abs()
)

merged["squared_error"] = (
    merged["error"] ** 2
)

merged["underprediction"] = (
    merged["predicted_los"]
    < merged["actual_los"]
)


# ============================================================
# 9. LOS RANGE
# ============================================================

merged["los_range"] = pd.cut(
    merged["actual_los"],
    bins=[
        -np.inf,
        24,
        72,
        168,
        np.inf
    ],
    labels=[
        "0-24h",
        "24-72h",
        "72-168h",
        ">168h"
    ]
)


# ============================================================
# 10. OVERALL PERFORMANCE
# ============================================================

print("\n" + "=" * 78)
print("5. OVERALL MODEL PERFORMANCE")
print("=" * 78)

mae = mean_absolute_error(
    merged["actual_los"],
    merged["predicted_los"]
)

rmse = np.sqrt(
    mean_squared_error(
        merged["actual_los"],
        merged["predicted_los"]
    )
)

r2 = r2_score(
    merged["actual_los"],
    merged["predicted_los"]
)

bias = merged["error"].mean()

median_abs_error = (
    merged["absolute_error"].median()
)

underprediction_pct = (
    merged["underprediction"].mean()
    * 100
)

overprediction_pct = (
    100
    - underprediction_pct
)

print(f"MAE                  : {mae:.3f} h")
print(f"RMSE                 : {rmse:.3f} h")
print(f"R²                   : {r2:.4f}")
print(
    f"Median absolute error: "
    f"{median_abs_error:.3f} h"
)
print(
    f"Mean prediction error: "
    f"{bias:.3f} h"
)
print(
    f"Underprediction      : "
    f"{underprediction_pct:.3f}%"
)
print(
    f"Overprediction       : "
    f"{overprediction_pct:.3f}%"
)


# ============================================================
# 11. LOS RANGE PERFORMANCE
# ============================================================

print("\n" + "=" * 78)
print("6. PERFORMANCE BY ACTUAL LOS RANGE")
print("=" * 78)

range_results = []

for range_name, group in merged.groupby(
    "los_range",
    observed=False
):

    if len(group) == 0:
        continue

    actual = group["actual_los"]

    predicted = group["predicted_los"]

    range_results.append({

        "LOS_Range":
            str(range_name),

        "Count":
            len(group),

        "Actual_Mean_h":
            actual.mean(),

        "Predicted_Mean_h":
            predicted.mean(),

        "Actual_Median_h":
            actual.median(),

        "Predicted_Median_h":
            predicted.median(),

        "MAE_h":
            mean_absolute_error(
                actual,
                predicted
            ),

        "RMSE_h":
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted
                )
            ),

        "Bias_h":
            group["error"].mean(),

        "Underprediction_pct":
            group["underprediction"].mean()
            * 100
    })

range_df = pd.DataFrame(
    range_results
)

print(
    range_df.to_string(index=False)
)

range_df.to_csv(
    OUTPUT_DIR /
    "los_model6_diagnostic_by_los_range.csv",
    index=False
)


# ============================================================
# 12. EXTREME TAIL
# ============================================================

print("\n" + "=" * 78)
print("7. EXTREME LONG-STAY ANALYSIS")
print("=" * 78)

tail_results = []

for threshold in [72, 168, 240, 300]:

    group = merged[
        merged["actual_los"]
        > threshold
    ]

    if len(group) == 0:
        continue

    tail_results.append({

        "Actual_LOS_Above_h":
            threshold,

        "Count":
            len(group),

        "Actual_Mean_h":
            group["actual_los"].mean(),

        "Predicted_Mean_h":
            group["predicted_los"].mean(),

        "Actual_Median_h":
            group["actual_los"].median(),

        "Predicted_Median_h":
            group["predicted_los"].median(),

        "MAE_h":
            mean_absolute_error(
                group["actual_los"],
                group["predicted_los"]
            ),

        "RMSE_h":
            np.sqrt(
                mean_squared_error(
                    group["actual_los"],
                    group["predicted_los"]
                )
            ),

        "Bias_h":
            group["error"].mean(),

        "Underprediction_pct":
            group["underprediction"].mean()
            * 100
    })

tail_df = pd.DataFrame(
    tail_results
)

print(
    tail_df.to_string(index=False)
)

tail_df.to_csv(
    OUTPUT_DIR /
    "los_model6_diagnostic_extreme_tail.csv",
    index=False
)


# ============================================================
# 13. PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 78)
print("8. ACTUAL VS PREDICTED DISTRIBUTION")
print("=" * 78)

distribution = pd.DataFrame({

    "Statistic": [
        "Minimum",
        "Median",
        "90th percentile",
        "95th percentile",
        "99th percentile",
        "Maximum"
    ],

    "Actual_LOS_h": [
        y.min(),
        y.median(),
        y.quantile(0.90),
        y.quantile(0.95),
        y.quantile(0.99),
        y.max()
    ],

    "Predicted_P50_h": [
        p50.min(),
        p50.median(),
        p50.quantile(0.90),
        p50.quantile(0.95),
        p50.quantile(0.99),
        p50.max()
    ]
})

print(
    distribution.to_string(index=False)
)

distribution.to_csv(
    OUTPUT_DIR /
    "los_model6_diagnostic_distribution.csv",
    index=False
)


# ============================================================
# 14. PREDICTION COMPRESSION
# ============================================================

actual_p99 = y.quantile(0.99)

predicted_p99 = p50.quantile(0.99)

actual_max = y.max()

predicted_max = p50.max()

print("\n" + "=" * 78)
print("9. PREDICTION COMPRESSION CHECK")
print("=" * 78)

print(
    f"Actual P99      : "
    f"{actual_p99:.3f} h"
)

print(
    f"Predicted P99   : "
    f"{predicted_p99:.3f} h"
)

print(
    f"Actual maximum  : "
    f"{actual_max:.3f} h"
)

print(
    f"Predicted max   : "
    f"{predicted_max:.3f} h"
)

print(
    f"P99 prediction ratio: "
    f"{predicted_p99 / actual_p99:.3f}"
)

print(
    f"Maximum prediction ratio: "
    f"{predicted_max / actual_max:.3f}"
)


# ============================================================
# 15. TOP 30 LONGEST STAYS
# ============================================================

print("\n" + "=" * 78)
print("10. TOP 30 LONGEST ACTUAL STAYS")
print("=" * 78)

columns_for_top = [
    "stay_id",
    "actual_los",
    "predicted_los",
    "error",
    "absolute_error",
    "age",
    "charlson_index",
    "triage_acuity",
    "heart_rate",
    "sbp",
    "dbp",
    "o2_sat",
    "resp_rate",
    "temp_c",
    "chief_complaint"
]

columns_for_top = [
    c
    for c in columns_for_top
    if c in merged.columns
]

top_long_stays = (
    merged
    .nlargest(30, "actual_los")
    [columns_for_top]
)

print(
    top_long_stays.to_string(
        index=False
    )
)

top_long_stays.to_csv(
    OUTPUT_DIR /
    "los_model6_diagnostic_top_30_long_stays.csv",
    index=False
)


# ============================================================
# 16. SEVERE UNDERPREDICTION
# ============================================================

print("\n" + "=" * 78)
print("11. SEVERE UNDERPREDICTION")
print("=" * 78)

merged["prediction_ratio"] = (
    merged["predicted_los"]
    / merged["actual_los"].clip(lower=0.01)
)

merged["actual_to_predicted_ratio"] = (
    merged["actual_los"]
    / merged["predicted_los"].clip(lower=0.01)
)

severe = merged[
    (merged["actual_los"] > 168)
    &
    (merged["actual_to_predicted_ratio"] >= 2)
]

tail_total = merged[
    merged["actual_los"] > 168
]

print(
    f"Total >168h cases: "
    f"{len(tail_total):,}"
)

print(
    f">168h cases predicted "
    f"at least 2x too low: "
    f"{len(severe):,}"
)

if len(tail_total) > 0:

    severe_pct = (
        len(severe)
        / len(tail_total)
        * 100
    )

    print(
        f"Percentage: "
        f"{severe_pct:.3f}%"
    )


severe_output_columns = [
    "stay_id",
    "actual_los",
    "predicted_los",
    "error",
    "actual_to_predicted_ratio",
    "age",
    "charlson_index",
    "triage_acuity",
    "heart_rate",
    "sbp",
    "dbp",
    "o2_sat",
    "resp_rate",
    "temp_c",
    "chief_complaint"
]

severe_output_columns = [
    c
    for c in severe_output_columns
    if c in severe.columns
]

severe_output = (
    severe
    .sort_values(
        "actual_to_predicted_ratio",
        ascending=False
    )
    [severe_output_columns]
)

severe_output.to_csv(
    OUTPUT_DIR /
    "los_model6_diagnostic_severe_underprediction.csv",
    index=False
)


# ============================================================
# 17. TRIAGE ACUITY
# ============================================================

print("\n" + "=" * 78)
print("12. PERFORMANCE BY TRIAGE ACUITY")
print("=" * 78)

if "triage_acuity" in merged.columns:

    acuity_results = []

    for acuity, group in merged.groupby(
        "triage_acuity"
    ):

        acuity_results.append({

            "Triage_Acuity":
                acuity,

            "Count":
                len(group),

            "Actual_Mean_h":
                group["actual_los"].mean(),

            "Predicted_Mean_h":
                group["predicted_los"].mean(),

            "Actual_Median_h":
                group["actual_los"].median(),

            "Predicted_Median_h":
                group["predicted_los"].median(),

            "MAE_h":
                mean_absolute_error(
                    group["actual_los"],
                    group["predicted_los"]
                ),

            "RMSE_h":
                np.sqrt(
                    mean_squared_error(
                        group["actual_los"],
                        group["predicted_los"]
                    )
                ),

            "Bias_h":
                group["error"].mean(),

            "Underprediction_pct":
                group["underprediction"].mean()
                * 100
        })

    acuity_df = pd.DataFrame(
        acuity_results
    )

    print(
        acuity_df.to_string(
            index=False
        )
    )

    acuity_df.to_csv(
        OUTPUT_DIR /
        "los_model6_diagnostic_by_triage.csv",
        index=False
    )


# ============================================================
# 18. NUMERIC FEATURE ERROR CORRELATION
# ============================================================

print("\n" + "=" * 78)
print("13. FEATURE vs MODEL ERROR")
print("=" * 78)

numeric_features = [
    "age",
    "charlson_index",
    "heart_rate",
    "sbp",
    "dbp",
    "o2_sat",
    "resp_rate",
    "temp_c",
    "triage_acuity",
    "arrival_hour"
]

feature_results = []

for feature in numeric_features:

    if feature not in merged.columns:
        continue

    x = pd.to_numeric(
        merged[feature],
        errors="coerce"
    )

    mask = (
        x.notna()
        & merged["actual_los"].notna()
        & merged["error"].notna()
    )

    if mask.sum() < 20:
        continue

    feature_results.append({

        "Feature":
            feature,

        "Correlation_with_actual_LOS":
            x[mask].corr(
                merged.loc[
                    mask,
                    "actual_los"
                ]
            ),

        "Correlation_with_error":
            x[mask].corr(
                merged.loc[
                    mask,
                    "error"
                ]
            ),

        "Correlation_with_absolute_error":
            x[mask].corr(
                merged.loc[
                    mask,
                    "absolute_error"
                ]
            )
    })

feature_df = pd.DataFrame(
    feature_results
)

if len(feature_df):

    feature_df["Absolute_LOS_Correlation"] = (
        feature_df[
            "Correlation_with_actual_LOS"
        ].abs()
    )

    feature_df = (
        feature_df
        .sort_values(
            "Absolute_LOS_Correlation",
            ascending=False
        )
        .drop(
            columns=[
                "Absolute_LOS_Correlation"
            ]
        )
    )

    print(
        feature_df.to_string(
            index=False
        )
    )

    feature_df.to_csv(
        OUTPUT_DIR /
        "los_model6_diagnostic_feature_correlations.csv",
        index=False
    )


# ============================================================
# 19. DERIVED FEATURE SIGNAL
# ============================================================

print("\n" + "=" * 78)
print("14. DERIVED CLINICAL FEATURE SIGNAL")
print("=" * 78)

derived = pd.DataFrame(
    index=merged.index
)

if {
    "heart_rate",
    "sbp"
}.issubset(merged.columns):

    hr = pd.to_numeric(
        merged["heart_rate"],
        errors="coerce"
    )

    sbp = pd.to_numeric(
        merged["sbp"],
        errors="coerce"
    )

    derived["shock_index"] = (
        hr /
        sbp.clip(lower=1)
    )


if {
    "sbp",
    "dbp"
}.issubset(merged.columns):

    sbp = pd.to_numeric(
        merged["sbp"],
        errors="coerce"
    )

    dbp = pd.to_numeric(
        merged["dbp"],
        errors="coerce"
    )

    derived["pulse_pressure"] = (
        sbp - dbp
    )

    derived[
        "mean_arterial_pressure"
    ] = (
        dbp
        + (sbp - dbp) / 3
    )


if "o2_sat" in merged.columns:

    o2 = pd.to_numeric(
        merged["o2_sat"],
        errors="coerce"
    )

    derived["oxygen_deficit"] = (
        100 - o2
    )


if "temp_c" in merged.columns:

    temp = pd.to_numeric(
        merged["temp_c"],
        errors="coerce"
    )

    derived[
        "temperature_deviation"
    ] = (
        temp - 37
    ).abs()


derived_results = []

for feature in derived.columns:

    mask = derived[feature].notna()

    derived_results.append({

        "Derived_Feature":
            feature,

        "Correlation_with_LOS":
            derived.loc[
                mask,
                feature
            ].corr(
                merged.loc[
                    mask,
                    "actual_los"
                ]
            ),

        "Correlation_with_Error":
            derived.loc[
                mask,
                feature
            ].corr(
                merged.loc[
                    mask,
                    "error"
                ]
            ),

        "Correlation_with_Absolute_Error":
            derived.loc[
                mask,
                feature
            ].corr(
                merged.loc[
                    mask,
                    "absolute_error"
                ]
            )
    })

derived_df = pd.DataFrame(
    derived_results
)

if len(derived_df):

    print(
        derived_df.to_string(
            index=False
        )
    )

    derived_df.to_csv(
        OUTPUT_DIR /
        "los_model6_diagnostic_derived_features.csv",
        index=False
    )


# ============================================================
# 20. CATEGORICAL FEATURE ANALYSIS
# ============================================================

print("\n" + "=" * 78)
print("15. CATEGORICAL FEATURE PERFORMANCE")
print("=" * 78)

categorical_features = [
    "gender",
    "chief_complaint"
]

categorical_results = []

for feature in categorical_features:

    if feature not in merged.columns:
        continue

    for value, group in merged.groupby(
        feature,
        dropna=False
    ):

        if len(group) < 20:
            continue

        categorical_results.append({

            "Feature":
                feature,

            "Category":
                str(value),

            "Count":
                len(group),

            "Actual_Mean_h":
                group["actual_los"].mean(),

            "Predicted_Mean_h":
                group["predicted_los"].mean(),

            "MAE_h":
                mean_absolute_error(
                    group["actual_los"],
                    group["predicted_los"]
                ),

            "Bias_h":
                group["error"].mean(),

            "Underprediction_pct":
                group["underprediction"].mean()
                * 100,

            "LongStay_gt168_pct":
                (
                    group["actual_los"] > 168
                ).mean()
                * 100
        })

categorical_df = pd.DataFrame(
    categorical_results
)

if len(categorical_df):

    print(
        categorical_df
        .sort_values(
            "MAE_h",
            ascending=False
        )
        .to_string(index=False)
    )

    categorical_df.to_csv(
        OUTPUT_DIR /
        "los_model6_diagnostic_categorical.csv",
        index=False
    )


# ============================================================
# 21. ARRIVAL HOUR
# ============================================================

print("\n" + "=" * 78)
print("16. ARRIVAL-HOUR PERFORMANCE")
print("=" * 78)

if "arrival_hour" in merged.columns:

    hour_results = []

    for hour, group in merged.groupby(
        "arrival_hour"
    ):

        if len(group) < 20:
            continue

        hour_results.append({

            "Arrival_Hour":
                int(hour),

            "Count":
                len(group),

            "Actual_Mean_h":
                group["actual_los"].mean(),

            "Predicted_Mean_h":
                group["predicted_los"].mean(),

            "MAE_h":
                mean_absolute_error(
                    group["actual_los"],
                    group["predicted_los"]
                ),

            "Bias_h":
                group["error"].mean(),

            "Underprediction_pct":
                group["underprediction"].mean()
                * 100,

            "LongStay_gt168_pct":
                (
                    group["actual_los"] > 168
                ).mean()
                * 100
        })

    hour_df = pd.DataFrame(
        hour_results
    )

    print(
        hour_df.to_string(
            index=False
        )
    )

    hour_df.to_csv(
        OUTPUT_DIR /
        "los_model6_diagnostic_arrival_hour.csv",
        index=False
    )


# ============================================================
# 22. PREDICTION CALIBRATION
# ============================================================

print("\n" + "=" * 78)
print("17. PREDICTION CALIBRATION")
print("=" * 78)

# Ten approximately equal-sized prediction groups
merged["prediction_decile"] = (
    pd.qcut(
        merged["predicted_los"].rank(
            method="first"
        ),
        q=10,
        labels=False
    ) + 1
)

calibration_results = []

for decile, group in merged.groupby(
    "prediction_decile"
):

    calibration_results.append({

        "Prediction_Decile":
            int(decile),

        "Count":
            len(group),

        "Mean_Predicted_h":
            group["predicted_los"].mean(),

        "Mean_Actual_h":
            group["actual_los"].mean(),

        "Median_Actual_h":
            group["actual_los"].median(),

        "MAE_h":
            mean_absolute_error(
                group["actual_los"],
                group["predicted_los"]
            ),

        "Bias_h":
            group["error"].mean(),

        "Actual_gt168_pct":
            (
                group["actual_los"] > 168
            ).mean()
            * 100
    })

calibration_df = pd.DataFrame(
    calibration_results
)

print(
    calibration_df.to_string(
        index=False
    )
)

calibration_df.to_csv(
    OUTPUT_DIR /
    "los_model6_diagnostic_prediction_calibration.csv",
    index=False
)


# ============================================================
# 23. INTERVAL DIAGNOSTICS
# ============================================================

print("\n" + "=" * 78)
print("18. MAPIE INTERVAL DIAGNOSTICS")
print("=" * 78)

interval_summary = {}

if (
    LOWER_COL is not None
    and UPPER_COL is not None
):

    lower = pd.to_numeric(
        merged[LOWER_COL],
        errors="coerce"
    )

    upper = pd.to_numeric(
        merged[UPPER_COL],
        errors="coerce"
    )

    mask = (
        lower.notna()
        & upper.notna()
    )

    if mask.any():

        interval_width = (
            upper[mask]
            - lower[mask]
        )

        contains = (
            (y[mask] >= lower[mask])
            &
            (y[mask] <= upper[mask])
        )

        ordering = (
            lower[mask]
            <= upper[mask]
        )

        interval_summary = {

            "coverage_pct":
                contains.mean() * 100,

            "ordering_validity_pct":
                ordering.mean() * 100,

            "mean_width_h":
                interval_width.mean(),

            "median_width_h":
                interval_width.median(),

            "p90_width_h":
                interval_width.quantile(0.90),

            "lower_below_zero_pct":
                (lower[mask] < 0).mean() * 100
        }

        for key, value in interval_summary.items():

            print(
                f"{key}: "
                f"{value:.4f}"
            )

else:

    print(
        "Could not identify interval "
        "columns in prediction file."
    )


# ============================================================
# 24. MODEL CONFIGURATION
# ============================================================

print("\n" + "=" * 78)
print("19. MODEL CONFIGURATION")
print("=" * 78)

configuration = {}

if CONFIG_PATH.exists():

    try:

        with open(
            CONFIG_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            configuration = json.load(f)

        print(
            json.dumps(
                configuration,
                indent=2
            )
        )

    except Exception as e:

        print(
            f"Could not read configuration: "
            f"{e}"
        )

else:

    print(
        "Configuration file unavailable."
    )


# ============================================================
# 25. AUTOMATED DIAGNOSIS
# ============================================================

print("\n" + "=" * 78)
print("20. AUTOMATED ROOT-CAUSE DIAGNOSIS")
print("=" * 78)

diagnosis = []

tail = merged[
    merged["actual_los"] > 168
]

if r2 < 0.20:

    diagnosis.append(
        "MODEL SIGNAL IS WEAK: R² is below 0.20."
    )

elif r2 < 0.35:

    diagnosis.append(
        "MODEL SIGNAL IS MODERATE: "
        "the model captures useful LOS patterns, "
        "but substantial variance remains unexplained."
    )

else:

    diagnosis.append(
        "MODEL SIGNAL IS RELATIVELY STRONG."
    )


if len(tail) > 0:

    tail_bias = tail["error"].mean()

    tail_mae = mean_absolute_error(
        tail["actual_los"],
        tail["predicted_los"]
    )

    if tail_bias < -50:

        diagnosis.append(
            f"SEVERE LONG-TAIL UNDERPREDICTION: "
            f">168h bias = {tail_bias:.2f}h "
            f"and MAE = {tail_mae:.2f}h."
        )

    elif tail_bias < -20:

        diagnosis.append(
            f"LONG-TAIL UNDERPREDICTION: "
            f">168h bias = {tail_bias:.2f}h."
        )

    else:

        diagnosis.append(
            "Long-tail bias is not severe."
        )


if (
    predicted_p99
    < actual_p99 * 0.60
):

    diagnosis.append(
        "PREDICTION COMPRESSION: "
        "the model is strongly pulling extreme LOS "
        "predictions toward the centre."
    )


if (
    abs(bias) > 10
):

    diagnosis.append(
        f"OVERALL BIAS EXISTS: "
        f"mean error = {bias:.2f}h."
    )

else:

    diagnosis.append(
        "Overall bias is relatively small."
    )


if interval_summary:

    cov = interval_summary[
        "coverage_pct"
    ]

    if abs(cov - 80) <= 2:

        diagnosis.append(
            f"MAPIE COVERAGE IS GOOD: "
            f"{cov:.2f}% is close to the 80% target."
        )

    else:

        diagnosis.append(
            f"MAPIE COVERAGE NEEDS REVIEW: "
            f"{cov:.2f}% vs 80% target."
        )


if len(feature_df):

    strongest = feature_df.iloc[0]

    diagnosis.append(
        f"Strongest individual numeric "
        f"LOS association: "
        f"{strongest['Feature']} "
        f"(correlation = "
        f"{strongest['Correlation_with_actual_LOS']:.3f})."
    )


for i, item in enumerate(
    diagnosis,
    start=1
):

    print(
        f"{i}. {item}"
    )


# ============================================================
# 26. FINAL RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 78)
print("21. RECOMMENDED NEXT ACTION")
print("=" * 78)

recommendations = []

if (
    len(tail) > 0
    and tail["error"].mean() < -50
):

    recommendations.append(
        "PRIMARY PROBLEM = long-stay prediction. "
        "Do not simply increase n_estimators."
    )


if r2 < 0.35:

    recommendations.append(
        "Check whether additional non-leakage "
        "arrival-time/clinical derived features "
        "can provide additional signal."
    )


if (
    predicted_p99
    < actual_p99 * 0.60
):

    recommendations.append(
        "Prediction compression is significant. "
        "A tail-aware modeling strategy should be considered."
    )


if interval_summary:

    if abs(
        interval_summary["coverage_pct"] - 80
    ) <= 2:

        recommendations.append(
            "MAPIE coverage is already close to target. "
            "Do not change conformal confidence merely "
            "to fix the point prediction."
        )


recommendations.append(
    "Next model experiment should keep the same "
    "chronological test set and compare against Model #6."
)

recommendations.append(
    "Only one major modeling change should be introduced "
    "at a time so we know what actually improves performance."
)

for i, item in enumerate(
    recommendations,
    start=1
):

    print(
        f"{i}. {item}"
    )


# ============================================================
# 27. SAVE FINAL REPORT
# ============================================================

report = {

    "model":
        "Model #6 - Quantile-specific "
        "tail-weighted LightGBM + MAPIE CQR",

    "rows":
        int(len(merged)),

    "overall": {

        "mae_hours":
            float(mae),

        "rmse_hours":
            float(rmse),

        "r2":
            float(r2),

        "median_absolute_error_hours":
            float(median_abs_error),

        "mean_prediction_error_hours":
            float(bias),

        "underprediction_pct":
            float(underprediction_pct),

        "overprediction_pct":
            float(overprediction_pct)
    },

    "interval":
        interval_summary,

    "tail_gt_168h": {

        "count":
            int(len(tail)),

        "actual_mean_hours":
            float(
                tail["actual_los"].mean()
            )
            if len(tail)
            else None,

        "predicted_mean_hours":
            float(
                tail["predicted_los"].mean()
            )
            if len(tail)
            else None,

        "mae_hours":
            float(
                mean_absolute_error(
                    tail["actual_los"],
                    tail["predicted_los"]
                )
            )
            if len(tail)
            else None,

        "bias_hours":
            float(
                tail["error"].mean()
            )
            if len(tail)
            else None
    },

    "diagnosis":
        diagnosis,

    "recommendations":
        recommendations
}

report_path = (
    OUTPUT_DIR /
    "los_model6_complete_diagnostic_report.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


print(
    f"\nSaved final report:\n"
    f"{report_path}"
)

print("\n" + "=" * 78)
print("MODEL #6 DIAGNOSTIC COMPLETE")
print("=" * 78)

print(
    "No model was retrained."
)

print(
    "No model files were modified."
)

print(
    "Use this diagnostic before building Model #7."
)