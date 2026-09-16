# ============================================================
# Predictive Resource Optimization System for Healthcare
# LOS MODEL — DIAGNOSTIC & VALIDATION ANALYSIS
#
# Purpose:
#   Evaluate the already-trained LOS model without retraining.
#
# Input:
#   outputs/los_predictions_test.csv
#
# The script evaluates:
#
#   1. Overall prediction accuracy
#   2. Prediction bias
#   3. Prediction interval coverage
#   4. Prediction interval width
#   5. Performance by triage acuity
#   6. Performance by LOS range
#   7. Long-stay prediction performance
#   8. Underprediction / overprediction
#   9. Interval validity
#  10. Diagnostic plots
#
# IMPORTANT:
#   This script does NOT modify the trained models.
#   It only analyzes the saved test predictions.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURATION
# ============================================================

CONFIDENCE_LEVEL = 0.80

TARGET_COVERAGE = CONFIDENCE_LEVEL

RANDOM_STATE = 42


# ============================================================
# 3. PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 4. INPUT FILE
# ============================================================

PREDICTION_PATH = (
    OUTPUT_DIR
    /
    "los_predictions_test.csv"
)


# ============================================================
# 5. OUTPUT FILES
# ============================================================

OVERALL_METRICS_PATH = (
    OUTPUT_DIR
    /
    "los_diagnostic_metrics.json"
)

ACUITY_RESULTS_PATH = (
    OUTPUT_DIR
    /
    "los_diagnostic_by_triage_acuity.csv"
)

LOS_RANGE_RESULTS_PATH = (
    OUTPUT_DIR
    /
    "los_diagnostic_by_los_range.csv"
)

ERROR_RESULTS_PATH = (
    OUTPUT_DIR
    /
    "los_prediction_error_analysis.csv"
)

PLOT_ACTUAL_PREDICTED = (
    OUTPUT_DIR
    /
    "los_actual_vs_predicted.png"
)

PLOT_ERROR_DISTRIBUTION = (
    OUTPUT_DIR
    /
    "los_prediction_error_distribution.png"
)

PLOT_COVERAGE_ACUITY = (
    OUTPUT_DIR
    /
    "los_coverage_by_triage_acuity.png"
)

PLOT_MAE_RANGE = (
    OUTPUT_DIR
    /
    "los_mae_by_los_range.png"
)


# ============================================================
# 6. REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "stay_id",
    "patient_id",
    "arrival_time",
    "triage_acuity",
    "actual_los_hours",
    "predicted_los_hours",
    "los_lower_80",
    "los_upper_80",
    "interval_width_hours",
    "covered_by_80_interval",
]


# ============================================================
# 7. HEADER
# ============================================================

print("=" * 80)
print("PREDICTIVE RESOURCE OPTIMIZATION SYSTEM FOR HEALTHCARE")
print("LOS MODEL — DIAGNOSTIC & VALIDATION ANALYSIS")
print("=" * 80)


# ============================================================
# 8. CHECK INPUT FILE
# ============================================================

print("\n" + "-" * 80)
print("[1/10] Checking prediction file")
print("-" * 80)


print(
    f"Expected file:\n{PREDICTION_PATH}"
)


if not PREDICTION_PATH.exists():

    raise FileNotFoundError(
        "\nPrediction file not found.\n\n"
        f"Expected:\n{PREDICTION_PATH}\n\n"
        "First run train_los_model.py successfully."
    )


# ============================================================
# 9. LOAD PREDICTIONS
# ============================================================

df = pd.read_csv(
    PREDICTION_PATH
)


print(
    f"\nPrediction file loaded successfully."
)

print(
    f"Rows    : {len(df):,}"
)

print(
    f"Columns : {len(df.columns)}"
)


# ============================================================
# 10. VALIDATE PREDICTION FILE
# ============================================================

print("\n" + "-" * 80)
print("[2/10] Validating prediction data")
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
        +
        "\n".join(
            f"  - {column}"
            for column in missing_columns
        )
    )


print(
    "✓ All required prediction columns are present"
)


# ------------------------------------------------------------
# Empty dataset
# ------------------------------------------------------------

if len(df) == 0:

    raise ValueError(
        "Prediction file contains zero records."
    )


print(
    "✓ Prediction dataset is not empty"
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
        "\nMissing values:"
    )

    print(
        missing_values
    )

    raise ValueError(
        "Missing values detected in prediction file."
    )


print(
    "✓ No missing values"
)


# ------------------------------------------------------------
# Duplicate stay IDs
# ------------------------------------------------------------

duplicate_stays = (
    df["stay_id"]
    .duplicated()
    .sum()
)


if duplicate_stays > 0:

    raise ValueError(
        f"{duplicate_stays:,} duplicate stay_id values found."
    )


print(
    "✓ No duplicate stay IDs"
)


# ============================================================
# 11. NUMERIC VALIDATION
# ============================================================

print("\n" + "-" * 80)
print("[3/10] Validating predictions")
print("-" * 80)


numeric_columns = [
    "actual_los_hours",
    "predicted_los_hours",
    "los_lower_80",
    "los_upper_80",
    "interval_width_hours",
]


for column in numeric_columns:

    if not pd.api.types.is_numeric_dtype(
        df[column]
    ):

        raise ValueError(
            f"{column} must be numeric."
        )


    if not np.isfinite(
        df[column].to_numpy()
    ).all():

        raise ValueError(
            f"{column} contains NaN or infinite values."
        )


print(
    "✓ Prediction columns are numeric and finite"
)


# ------------------------------------------------------------
# Actual LOS must be positive
# ------------------------------------------------------------

if (
    df["actual_los_hours"]
    <= 0
).any():

    raise ValueError(
        "Actual LOS contains zero or negative values."
    )


print(
    "✓ Actual LOS values are positive"
)


# ============================================================
# 12. INTERVAL VALIDATION
# ============================================================

print("\n" + "-" * 80)
print("[4/10] Validating prediction intervals")
print("-" * 80)


actual = (
    df["actual_los_hours"]
    .to_numpy(
        dtype=float
    )
)


predicted = (
    df["predicted_los_hours"]
    .to_numpy(
        dtype=float
    )
)


lower = (
    df["los_lower_80"]
    .to_numpy(
        dtype=float
    )
)


upper = (
    df["los_upper_80"]
    .to_numpy(
        dtype=float
    )
)


width = (
    df["interval_width_hours"]
    .to_numpy(
        dtype=float
    )
)


# ------------------------------------------------------------
# Check lower <= predicted <= upper
# ------------------------------------------------------------

valid_ordering = (
    (lower <= predicted)
    &
    (predicted <= upper)
)


ordering_rate = float(
    np.mean(
        valid_ordering
    )
)


print(
    f"Interval ordering validity : "
    f"{ordering_rate:.3%}"
)


if ordering_rate < 1.0:

    invalid_count = int(
        np.sum(
            ~valid_ordering
        )
    )

    print(
        f"⚠ Invalid interval ordering "
        f"found in {invalid_count:,} records."
    )

else:

    print(
        "✓ Every interval contains the point prediction"
    )


# ------------------------------------------------------------
# Check negative lower bounds
# ------------------------------------------------------------

negative_lower_count = int(
    np.sum(
        lower < 0
    )
)


if negative_lower_count > 0:

    print(
        f"⚠ Negative lower bounds: "
        f"{negative_lower_count:,}"
    )

else:

    print(
        "✓ No negative lower bounds"
    )


# ------------------------------------------------------------
# Check interval width
# ------------------------------------------------------------

calculated_width = (
    upper
    -
    lower
)


width_difference = np.abs(
    calculated_width
    -
    width
)


if (
    width_difference > 0.001
).any():

    raise ValueError(
        "Some interval_width_hours values "
        "do not match upper - lower."
    )


print(
    "✓ Interval widths are valid"
)


# ============================================================
# 13. CALCULATE PREDICTION ERRORS
# ============================================================

print("\n" + "-" * 80)
print("[5/10] Calculating prediction errors")
print("-" * 80)


# Error:
#
# positive = model overpredicts
# negative = model underpredicts


error = (
    predicted
    -
    actual
)


absolute_error = np.abs(
    error
)


squared_error = (
    error
    ** 2
)


df["prediction_error_hours"] = (
    error
)


df["absolute_error_hours"] = (
    absolute_error
)


df["squared_error_hours"] = (
    squared_error
)


df["underpredicted"] = (
    predicted
    <
    actual
)


df["overpredicted"] = (
    predicted
    >
    actual
)


df["exact_prediction"] = (
    predicted
    ==
    actual
)


# ------------------------------------------------------------
# Coverage
# ------------------------------------------------------------

df["covered_by_80_interval"] = (
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


# ============================================================
# 14. OVERALL METRICS
# ============================================================

mae = float(
    mean_absolute_error(
        actual,
        predicted,
    )
)


rmse = float(
    np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )
)


r2 = float(
    r2_score(
        actual,
        predicted,
    )
)


median_absolute_error_value = float(
    median_absolute_error(
        actual,
        predicted,
    )
)


mean_error = float(
    np.mean(
        error
    )
)


median_error = float(
    np.median(
        error
    )
)


mean_absolute_error_value = float(
    np.mean(
        absolute_error
    )
)


underprediction_rate = float(
    np.mean(
        predicted < actual
    )
)


overprediction_rate = float(
    np.mean(
        predicted > actual
    )
)


coverage = float(
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


mean_interval_width = float(
    np.mean(
        width
    )
)


median_interval_width = float(
    np.median(
        width
    )
)


p90_interval_width = float(
    np.percentile(
        width,
        90,
    )
)


# ============================================================
# 15. PRINT OVERALL RESULTS
# ============================================================

print("\nOverall LOS model performance:")
print(
    f"  Test records              : {len(df):,}"
)

print(
    f"  MAE                       : {mae:.3f} hours"
)

print(
    f"  RMSE                      : {rmse:.3f} hours"
)

print(
    f"  R²                        : {r2:.4f}"
)

print(
    f"  Median Absolute Error     : "
    f"{median_absolute_error_value:.3f} hours"
)

print(
    f"  Mean Prediction Error     : "
    f"{mean_error:.3f} hours"
)

print(
    f"  Median Prediction Error   : "
    f"{median_error:.3f} hours"
)

print(
    f"  Underprediction rate      : "
    f"{underprediction_rate:.3%}"
)

print(
    f"  Overprediction rate       : "
    f"{overprediction_rate:.3%}"
)

print(
    f"  80% Interval Coverage     : "
    f"{coverage:.3%}"
)

print(
    f"  Target Coverage           : "
    f"{TARGET_COVERAGE:.3%}"
)

print(
    f"  Mean Interval Width       : "
    f"{mean_interval_width:.3f} hours"
)

print(
    f"  Median Interval Width     : "
    f"{median_interval_width:.3f} hours"
)

print(
    f"  90th % Interval Width     : "
    f"{p90_interval_width:.3f} hours"
)


# ============================================================
# 16. CHECK COVERAGE AGAINST TARGET
# ============================================================

coverage_difference = (
    coverage
    -
    TARGET_COVERAGE
)


print("\nCoverage assessment:")


print(
    f"  Coverage difference       : "
    f"{coverage_difference:+.3%}"
)


if abs(
    coverage_difference
) <= 0.02:

    print(
        "  ✓ Coverage is within ±2 percentage points "
        "of the 80% target."
    )

else:

    print(
        "  ⚠ Coverage is more than ±2 percentage points "
        "away from the target."
    )


# ============================================================
# 17. PERFORMANCE BY TRIAGE ACUITY
# ============================================================

print("\n" + "-" * 80)
print("[6/10] Evaluating performance by triage acuity")
print("-" * 80)


acuity_results = []


for acuity in sorted(
    df["triage_acuity"]
    .unique()
):

    group = df[
        df["triage_acuity"]
        ==
        acuity
    ]


    group_actual = (
        group[
            "actual_los_hours"
        ]
        .to_numpy(
            dtype=float
        )
    )


    group_predicted = (
        group[
            "predicted_los_hours"
        ]
        .to_numpy(
            dtype=float
        )
    )


    group_lower = (
        group[
            "los_lower_80"
        ]
        .to_numpy(
            dtype=float
        )
    )


    group_upper = (
        group[
            "los_upper_80"
        ]
        .to_numpy(
            dtype=float
        )
    )


    group_error = (
        group_predicted
        -
        group_actual
    )


    group_coverage = float(
        np.mean(
            (
                group_actual
                >=
                group_lower
            )
            &
            (
                group_actual
                <=
                group_upper
            )
        )
    )


    group_width = float(
        np.mean(
            group_upper
            -
            group_lower
        )
    )


    group_mae = float(
        mean_absolute_error(
            group_actual,
            group_predicted,
        )
    )


    group_rmse = float(
        np.sqrt(
            mean_squared_error(
                group_actual,
                group_predicted,
            )
        )
    )


    group_bias = float(
        np.mean(
            group_error
        )
    )


    group_underprediction = float(
        np.mean(
            group_predicted
            <
            group_actual
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
                        group_actual
                    )
                ),

            "actual_median_los_hours":
                float(
                    np.median(
                        group_actual
                    )
                ),

            "predicted_mean_los_hours":
                float(
                    np.mean(
                        group_predicted
                    )
                ),

            "predicted_median_los_hours":
                float(
                    np.median(
                        group_predicted
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
# 18. PERFORMANCE BY ACTUAL LOS RANGE
# ============================================================

print("\n" + "-" * 80)
print("[7/10] Evaluating performance by LOS range")
print("-" * 80)


# ------------------------------------------------------------
# Define clinically useful LOS ranges
# ------------------------------------------------------------

def classify_los(hours):

    if hours <= 24:

        return "0-24 hours"

    elif hours <= 72:

        return "24-72 hours"

    elif hours <= 168:

        return "72-168 hours"

    else:

        return ">168 hours"


df["los_range"] = (
    df["actual_los_hours"]
    .apply(
        classify_los
    )
)


los_range_order = [
    "0-24 hours",
    "24-72 hours",
    "72-168 hours",
    ">168 hours",
]


los_range_results = []


for los_range in los_range_order:

    group = df[
        df["los_range"]
        ==
        los_range
    ]


    if len(group) == 0:

        continue


    group_actual = (
        group[
            "actual_los_hours"
        ]
        .to_numpy(
            dtype=float
        )
    )


    group_predicted = (
        group[
            "predicted_los_hours"
        ]
        .to_numpy(
            dtype=float
        )
    )


    group_lower = (
        group[
            "los_lower_80"
        ]
        .to_numpy(
            dtype=float
        )
    )


    group_upper = (
        group[
            "los_upper_80"
        ]
        .to_numpy(
            dtype=float
        )
    )


    group_error = (
        group_predicted
        -
        group_actual
    )


    group_coverage = float(
        np.mean(
            (
                group_actual
                >=
                group_lower
            )
            &
            (
                group_actual
                <=
                group_upper
            )
        )
    )


    group_width = float(
        np.mean(
            group_upper
            -
            group_lower
        )
    )


    group_mae = float(
        mean_absolute_error(
            group_actual,
            group_predicted,
        )
    )


    group_rmse = float(
        np.sqrt(
            mean_squared_error(
                group_actual,
                group_predicted,
            )
        )
    )


    group_bias = float(
        np.mean(
            group_error
        )
    )


    group_underprediction = float(
        np.mean(
            group_predicted
            <
            group_actual
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
                        group_actual
                    )
                ),

            "actual_median_los_hours":
                float(
                    np.median(
                        group_actual
                    )
                ),

            "predicted_mean_los_hours":
                float(
                    np.mean(
                        group_predicted
                    )
                ),

            "predicted_median_los_hours":
                float(
                    np.median(
                        group_predicted
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
# 19. ERROR ANALYSIS
# ============================================================

print("\n" + "-" * 80)
print("[8/10] Performing detailed error analysis")
print("-" * 80)


# ------------------------------------------------------------
# Error percentiles
# ------------------------------------------------------------

error_percentiles = {
    "error_p01_hours":
        float(
            np.percentile(
                error,
                1,
            )
        ),

    "error_p05_hours":
        float(
            np.percentile(
                error,
                5,
            )
        ),

    "error_p25_hours":
        float(
            np.percentile(
                error,
                25,
            )
        ),

    "error_p50_hours":
        float(
            np.percentile(
                error,
                50,
            )
        ),

    "error_p75_hours":
        float(
            np.percentile(
                error,
                75,
            )
        ),

    "error_p95_hours":
        float(
            np.percentile(
                error,
                95,
            )
        ),

    "error_p99_hours":
        float(
            np.percentile(
                error,
                99,
            )
        ),
}


print(
    "\nPrediction error percentiles:"
)


for name, value in (
    error_percentiles.items()
):

    print(
        f"  {name:<25}: "
        f"{value:.3f} hours"
    )


# ------------------------------------------------------------
# Largest absolute errors
# ------------------------------------------------------------

largest_errors = (
    df[
        [
            "stay_id",
            "patient_id",
            "triage_acuity",
            "actual_los_hours",
            "predicted_los_hours",
            "los_lower_80",
            "los_upper_80",
            "prediction_error_hours",
            "absolute_error_hours",
            "covered_by_80_interval",
        ]
    ]
    .sort_values(
        by="absolute_error_hours",
        ascending=False,
    )
    .head(100)
)


largest_errors_path = (
    OUTPUT_DIR
    /
    "los_largest_prediction_errors.csv"
)


largest_errors.to_csv(
    largest_errors_path,
    index=False,
)


print(
    f"\n✓ Top 100 largest errors saved to:\n"
    f"{largest_errors_path}"
)


# ------------------------------------------------------------
# Underprediction severity
# ------------------------------------------------------------

underprediction_count = int(
    np.sum(
        predicted
        <
        actual
    )
)


overprediction_count = int(
    np.sum(
        predicted
        >
        actual
    )
)


print(
    "\nPrediction direction:"
)


print(
    f"  Underpredicted : "
    f"{underprediction_count:,} "
    f"({underprediction_rate:.3%})"
)


print(
    f"  Overpredicted  : "
    f"{overprediction_count:,} "
    f"({overprediction_rate:.3%})"
)


# ============================================================
# 20. SAVE DETAILED ERROR FILE
# ============================================================

error_analysis_columns = [
    "stay_id",
    "patient_id",
    "arrival_time",
    "triage_acuity",
    "actual_los_hours",
    "predicted_los_hours",
    "los_lower_80",
    "los_upper_80",
    "interval_width_hours",
    "covered_by_80_interval",
    "prediction_error_hours",
    "absolute_error_hours",
    "underpredicted",
    "overpredicted",
    "los_range",
]


error_analysis_df = (
    df[
        error_analysis_columns
    ]
    .copy()
)


error_analysis_df.to_csv(
    ERROR_RESULTS_PATH,
    index=False,
)


print(
    f"\n✓ Detailed error analysis saved to:\n"
    f"{ERROR_RESULTS_PATH}"
)


# ============================================================
# 21. CREATE DIAGNOSTIC PLOTS
# ============================================================

print("\n" + "-" * 80)
print("[9/10] Creating diagnostic plots")
print("-" * 80)


# ============================================================
# Plot 1 — Actual vs Predicted LOS
# ============================================================

plt.figure(
    figsize=(10, 7)
)


plt.scatter(
    actual,
    predicted,
    alpha=0.25,
    s=12,
)


max_value = max(
    np.max(actual),
    np.max(predicted),
)


plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--",
    linewidth=2,
)


plt.xlabel(
    "Actual LOS (hours)"
)

plt.ylabel(
    "Predicted LOS (hours)"
)

plt.title(
    "LOS Model — Actual vs Predicted"
)

plt.xlim(
    left=0
)

plt.ylim(
    bottom=0
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    PLOT_ACTUAL_PREDICTED,
    dpi=150,
)


plt.close()


print(
    f"✓ Created:\n"
    f"{PLOT_ACTUAL_PREDICTED}"
)


# ============================================================
# Plot 2 — Prediction Error Distribution
# ============================================================

plt.figure(
    figsize=(10, 7)
)


plt.hist(
    error,
    bins=60,
    alpha=0.75,
)


plt.axvline(
    0,
    linestyle="--",
    linewidth=2,
)


plt.xlabel(
    "Prediction Error (Predicted - Actual) [hours]"
)

plt.ylabel(
    "Number of Patients"
)

plt.title(
    "LOS Model — Prediction Error Distribution"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    PLOT_ERROR_DISTRIBUTION,
    dpi=150,
)


plt.close()


print(
    f"✓ Created:\n"
    f"{PLOT_ERROR_DISTRIBUTION}"
)


# ============================================================
# Plot 3 — Coverage by Triage Acuity
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.bar(
    acuity_results_df[
        "triage_acuity"
    ].astype(str),
    acuity_results_df[
        "coverage_80"
    ]
    * 100,
)


plt.axhline(
    TARGET_COVERAGE * 100,
    linestyle="--",
    linewidth=2,
)


plt.xlabel(
    "Triage Acuity (ESI)"
)

plt.ylabel(
    "80% Interval Coverage (%)"
)

plt.title(
    "LOS Model — Prediction Interval Coverage by Triage Acuity"
)

plt.ylim(
    0,
    100
)

plt.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()


plt.savefig(
    PLOT_COVERAGE_ACUITY,
    dpi=150,
)


plt.close()


print(
    f"✓ Created:\n"
    f"{PLOT_COVERAGE_ACUITY}"
)


# ============================================================
# Plot 4 — MAE by LOS Range
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.bar(
    los_range_results_df[
        "los_range"
    ],
    los_range_results_df[
        "mae_hours"
    ],
)


plt.xlabel(
    "Actual LOS Range"
)

plt.ylabel(
    "MAE (hours)"
)

plt.title(
    "LOS Model — MAE by Actual LOS Range"
)

plt.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()


plt.savefig(
    PLOT_MAE_RANGE,
    dpi=150,
)


plt.close()


print(
    f"✓ Created:\n"
    f"{PLOT_MAE_RANGE}"
)


# ============================================================
# 22. SAVE OVERALL METRICS
# ============================================================

print("\n" + "-" * 80)
print("[10/10] Saving diagnostic metrics")
print("-" * 80)


overall_metrics = {

    "model":
        "Length of Stay Prediction",

    "algorithm":
        "LightGBM Quantile Regression + MAPIE CQR",

    "confidence_level":
        CONFIDENCE_LEVEL,

    "target_coverage":
        TARGET_COVERAGE,

    "test_records":
        int(len(df)),

    "mae_hours":
        mae,

    "rmse_hours":
        rmse,

    "r2":
        r2,

    "median_absolute_error_hours":
        median_absolute_error_value,

    "mean_prediction_error_hours":
        mean_error,

    "median_prediction_error_hours":
        median_error,

    "underprediction_rate":
        underprediction_rate,

    "overprediction_rate":
        overprediction_rate,

    "interval_coverage":
        coverage,

    "coverage_difference_from_target":
        coverage_difference,

    "mean_interval_width_hours":
        mean_interval_width,

    "median_interval_width_hours":
        median_interval_width,

    "p90_interval_width_hours":
        p90_interval_width,

    "interval_ordering_validity":
        ordering_rate,

    "negative_lower_bound_count":
        negative_lower_count,

    "error_percentiles":
        error_percentiles,

    "interpretation": {
        "coverage_target_met":
            bool(
                abs(
                    coverage_difference
                )
                <=
                0.02
            ),

        "interval_validity":
            bool(
                ordering_rate
                ==
                1.0
            ),

        "model_underpredicts_more_often":
            bool(
                underprediction_rate
                >
                overprediction_rate
            ),
    },
}


with open(
    OVERALL_METRICS_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        overall_metrics,
        file,
        indent=4,
    )


print(
    f"✓ Diagnostic metrics saved to:\n"
    f"{OVERALL_METRICS_PATH}"
)


# ============================================================
# 23. FINAL DIAGNOSTIC SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("LOS MODEL DIAGNOSTIC SUMMARY")
print("=" * 80)


print(
    f"\nTest records          : "
    f"{len(df):,}"
)


print(
    f"MAE                   : "
    f"{mae:.3f} hours"
)


print(
    f"RMSE                  : "
    f"{rmse:.3f} hours"
)


print(
    f"R²                    : "
    f"{r2:.4f}"
)


print(
    f"80% Coverage          : "
    f"{coverage:.3%}"
)


print(
    f"Target Coverage       : "
    f"{TARGET_COVERAGE:.3%}"
)


print(
    f"Mean Interval Width   : "
    f"{mean_interval_width:.3f} hours"
)


print(
    f"Underprediction       : "
    f"{underprediction_rate:.3%}"
)


print(
    f"Overprediction        : "
    f"{overprediction_rate:.3%}"
)


print(
    f"Interval Validity     : "
    f"{ordering_rate:.3%}"
)


print("\nKey diagnostic outputs:")

print(
    f"  1. {OVERALL_METRICS_PATH.name}"
)

print(
    f"  2. {ACUITY_RESULTS_PATH.name}"
)

print(
    f"  3. {LOS_RANGE_RESULTS_PATH.name}"
)

print(
    f"  4. {ERROR_RESULTS_PATH.name}"
)

print(
    f"  5. {largest_errors_path.name}"
)

print(
    f"  6. {PLOT_ACTUAL_PREDICTED.name}"
)

print(
    f"  7. {PLOT_ERROR_DISTRIBUTION.name}"
)

print(
    f"  8. {PLOT_COVERAGE_ACUITY.name}"
)

print(
    f"  9. {PLOT_MAE_RANGE.name}"
)


print("\n" + "=" * 80)
print("LOS DIAGNOSTIC ANALYSIS COMPLETED")
print("=" * 80)

print(
    "\nNo model was retrained or modified."
)

print(
    "The analysis used the existing saved test predictions."
)