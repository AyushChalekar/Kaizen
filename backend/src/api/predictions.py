# backend/src/api/routers/predictions.py
"""FastAPI router for predictive routing and machine learning simulation endpoints.

Exposes endpoints for end-to-end simulation pipelines, demographic cohort synthesis,
clinical trajectory forecasting, shift-based staffing alignment, and real-time
resource utilization projections using Pydantic v2 schemas.
"""

# backend/src/api/predictions.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, Final, Optional, Union

from fastapi import APIRouter, Body, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from src.schemas.contracts import (
    CareUnitType,
    DispositionType,
    ShiftType,
    TIMESTAMP_FORMAT,
    _parse_strict_timestamp,
)
from src.schemas.prediction_contracts import (
    ClinicalTrajectoryPrediction,
    HourlyStaffingPrediction,
    PatientClinicalPrediction,
    ResourceUtilizationForecast,
)
from src.ml.model_pipeline import PredictiveModelPipeline

# ---------------------------------------------------------------------------
# Logging & Runtime Defaults
# ---------------------------------------------------------------------------
logger = logging.getLogger("digital_twin_predictions_api")

DEFAULT_START_DATETIME: Final[datetime] = datetime(2026, 9, 23, 0, 0, 0)
DEFAULT_START_TIMESTAMP: Final[str] = "2026-09-23 00:00:00"


# ---------------------------------------------------------------------------
# Request & Response Contracts
# ---------------------------------------------------------------------------
class PredictionRunRequest(BaseModel):
    """Payload configuring an end-to-end predictive pipeline execution."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    cohort_size: int = Field(
        default=10,
        ge=1,
        le=500,
        description="Number of simulated patient stays to synthesize and route.",
    )
    forecast_horizon_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Staffing and arrival forecast projection window in hours.",
    )
    start_datetime: Optional[Union[str, datetime]] = Field(
        default=None,
        description="Anchor datetime for simulations formatted as 'YYYY-MM-DD HH:MM:SS'.",
    )
    seed: Optional[int] = Field(
        default=42,
        description="Deterministic random number generator seed.",
    )


class PredictionRunResponse(BaseModel):
    """Envelope wrapping full predictive pipeline simulation outputs."""

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=False,
    )

    status: str = Field(default="success", description="Execution status message.")
    generated_at: str = Field(..., description="ISO 8601 timestamp of execution.")
    cohort_size: int = Field(..., description="Count of synthesized patient profiles.")
    forecast_horizon_hours: int = Field(..., description="Hours of staffing horizon projected.")
    patient_profiles: list[PatientClinicalPrediction] = Field(
        ..., description="Synthesized patient demographic and hemodynamic states."
    )
    trajectories: list[ClinicalTrajectoryPrediction] = Field(
        ..., description="Projected milestone timestamps and clinical routing pathways."
    )
    staffing_predictions: list[HourlyStaffingPrediction] = Field(
        ..., description="Hourly patient flow and shift-aligned staffing projections."
    )
    resource_forecast: ResourceUtilizationForecast = Field(
        ..., description="Projected bed utilizations and queue bottlenecks."
    )


class CohortSynthesisRequest(BaseModel):
    """Payload specifying cohort synthesis parameters."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    cohort_size: int = Field(
        default=10,
        ge=1,
        le=500,
        description="Number of patient profiles to synthesize.",
    )
    base_acuity: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="Target Emergency Severity Index (ESI 1-5) acuity level.",
    )


class TrajectoryForecastRequest(BaseModel):
    """Payload containing patients and baseline timing for trajectory projections."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    patients: list[PatientClinicalPrediction] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of clinical patient predictions to forecast routing for.",
    )
    base_arrival_time: Optional[Union[str, datetime]] = Field(
        default=None,
        description="Base arrival reference timestamp formatted as 'YYYY-MM-DD HH:MM:SS'.",
    )
    arrival_interval_minutes: float = Field(
        default=8.0,
        ge=0.0,
        le=120.0,
        description="Spacing interval in minutes between consecutive patient arrivals.",
    )


# ---------------------------------------------------------------------------
# Thread-Safe Pipeline Holder & Cache State
# ---------------------------------------------------------------------------
@dataclass
class PredictionCache:
    """Thread-safe holder for cached pipeline results and execution parameters."""

    results: dict[str, Any]
    request: PredictionRunRequest
    generated_at: str


_active_pipeline: Optional[PredictiveModelPipeline] = None
_active_pipeline_seed: Optional[int] = None
_latest_prediction_cache: Optional[PredictionCache] = None
_pipeline_lock = asyncio.Lock()


async def get_or_create_pipeline(seed: Optional[int] = 42) -> PredictiveModelPipeline:
    """Retrieve or lazily instantiate the singleton PredictiveModelPipeline instance."""
    global _active_pipeline, _active_pipeline_seed
    async with _pipeline_lock:
        if _active_pipeline is None or _active_pipeline_seed != seed:
            logger.info("Initializing PredictiveModelPipeline with seed=%s", seed)
            _active_pipeline = PredictiveModelPipeline(seed=seed)
            _active_pipeline_seed = seed
        return _active_pipeline


def _normalize_datetime(val: Optional[Union[str, datetime]]) -> datetime:
    """Normalize strings or datetime objects to a naive second-precision datetime."""
    if val is None:
        return DEFAULT_START_DATETIME
    if isinstance(val, datetime):
        return val.replace(microsecond=0)
    if isinstance(val, str):
        cleaned = val.strip().replace("T", " ")
        if len(cleaned) == 10:
            cleaned += " 00:00:00"
        elif len(cleaned) > 19:
            cleaned = cleaned[:19]
        try:
            return datetime.strptime(cleaned, TIMESTAMP_FORMAT)
        except ValueError as exc:
            try:
                return datetime.fromisoformat(val).replace(microsecond=0)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"Invalid timestamp format '{val}'. "
                        f"Expected format 'YYYY-MM-DD HH:MM:SS' or valid ISO 8601 string."
                    ),
                ) from exc
    return DEFAULT_START_DATETIME


# ---------------------------------------------------------------------------
# FastAPI Router Definition
# ---------------------------------------------------------------------------
router = APIRouter(
    prefix="/api/predictions",
    tags=["Predictions"],
)


# ---------------------------------------------------------------------------
# 1. End-to-End Pipeline Execution (POST /run)
# ---------------------------------------------------------------------------
@router.post(
    "/run",
    response_model=PredictionRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute End-to-End Predictive Simulation Pipeline",
)
async def run_predictive_pipeline(
    payload: PredictionRunRequest = Body(...),
) -> PredictionRunResponse:
    """Execute complete predictive simulation: synthesis, routing, staffing, and resources."""
    pipeline = await get_or_create_pipeline(seed=payload.seed)
    anchor_dt = _normalize_datetime(payload.start_datetime)

    try:
        # Offload synchronous pipeline computation to worker thread to maintain event loop responsiveness
        pipeline_output = await asyncio.to_thread(
            pipeline.run_pipeline,
            cohort_size=payload.cohort_size,
            start_time=anchor_dt,
            forecast_horizon_hours=payload.forecast_horizon_hours,
        )
    except Exception as exc:
        logger.error("Error executing predictive pipeline: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Predictive pipeline execution failure: {str(exc)}",
        ) from exc

    generated_at_iso = datetime.now(timezone.utc).isoformat()

    cache_entry = PredictionCache(
        results=pipeline_output,
        request=payload,
        generated_at=generated_at_iso,
    )
    global _latest_prediction_cache
    async with _pipeline_lock:
        _latest_prediction_cache = cache_entry

    return PredictionRunResponse(
        status="success",
        generated_at=generated_at_iso,
        cohort_size=payload.cohort_size,
        forecast_horizon_hours=payload.forecast_horizon_hours,
        patient_profiles=pipeline_output["patient_profiles"],
        trajectories=pipeline_output["trajectories"],
        staffing_predictions=pipeline_output["staffing_predictions"],
        resource_forecast=pipeline_output["resource_forecast"],
    )


# ---------------------------------------------------------------------------
# 2. Cohort Synthesis (POST /cohort)
# ---------------------------------------------------------------------------
@router.post(
    "/cohort",
    response_model=list[PatientClinicalPrediction],
    status_code=status.HTTP_200_OK,
    summary="Synthesize Clinical Patient Cohort",
)
async def synthesize_patient_cohort(
    payload: Optional[CohortSynthesisRequest] = None,
    cohort_size: Optional[int] = Query(
        default=None,
        ge=1,
        le=500,
        description="Override cohort size via query parameter.",
    ),
    base_acuity: Optional[int] = Query(
        default=None,
        ge=1,
        le=5,
        description="Override base triage acuity via query parameter.",
    ),
) -> list[PatientClinicalPrediction]:
    """Sample demographic profiles, comorbidity indices, complaints, and validated hemodynamics."""
    pipeline = await get_or_create_pipeline()

    # Resolve parameter precedence (Query parameter overrides payload, payload overrides defaults)
    effective_count = 10
    effective_acuity: Optional[int] = None

    if payload is not None:
        effective_count = payload.cohort_size
        effective_acuity = payload.base_acuity

    if cohort_size is not None:
        effective_count = cohort_size
    if base_acuity is not None:
        effective_acuity = base_acuity

    try:
        profiles = await asyncio.to_thread(
            pipeline.batch_sample_patient_clinicals,
            count=effective_count,
            base_acuity=effective_acuity,
        )
        return profiles
    except Exception as exc:
        logger.error("Error synthesizing clinical cohort: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clinical cohort synthesis failed: {str(exc)}",
        ) from exc


# ---------------------------------------------------------------------------
# 3. Trajectory Forecasting (POST /trajectories)
# ---------------------------------------------------------------------------
@router.post(
    "/trajectories",
    response_model=list[ClinicalTrajectoryPrediction],
    status_code=status.HTTP_200_OK,
    summary="Forecast Clinical Trajectories and Sequential Milestones",
)
async def forecast_patient_trajectories(
    payload: Union[TrajectoryForecastRequest, list[PatientClinicalPrediction]] = Body(...),
    base_arrival_time: Optional[str] = Query(
        default=None,
        description="Optional query anchor arrival timestamp 'YYYY-MM-DD HH:MM:SS'.",
    ),
) -> list[ClinicalTrajectoryPrediction]:
    """Anticipate patient movement, escalation risks, and monotonic timestamp projections."""
    pipeline = await get_or_create_pipeline()

    if isinstance(payload, TrajectoryForecastRequest):
        patients_list = payload.patients
        raw_base_time = payload.base_arrival_time or base_arrival_time
        interval_min = payload.arrival_interval_minutes
    else:
        patients_list = payload
        raw_base_time = base_arrival_time
        interval_min = 8.0

    if not patients_list:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Payload must include at least one PatientClinicalPrediction object.",
        )

    anchor_dt = _normalize_datetime(raw_base_time)

    try:
        trajectories = await asyncio.to_thread(
            pipeline.batch_predict_trajectories,
            patients=patients_list,
            base_arrival_time=anchor_dt,
            arrival_interval_minutes=interval_min,
        )
        return trajectories
    except Exception as exc:
        logger.error("Error projecting clinical trajectories: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clinical trajectory forecasting failed: {str(exc)}",
        ) from exc


# ---------------------------------------------------------------------------
# 4. Hourly Staffing Projection (GET /staffing)
# ---------------------------------------------------------------------------
@router.get(
    "/staffing",
    response_model=list[HourlyStaffingPrediction],
    status_code=status.HTTP_200_OK,
    summary="Predict Hourly Patient Flow and Shift-Aligned Staffing Demand",
)
async def get_hourly_staffing_projections(
    start_time: Optional[str] = Query(
        default=None,
        description="Anchor timestamp floored to hour ('YYYY-MM-DD HH:00:00').",
    ),
    horizon_hours: int = Query(
        default=24,
        ge=1,
        le=168,
        description="Projection horizon span in completed hours (1 to 168).",
    ),
) -> list[HourlyStaffingPrediction]:
    """Project shift-level operational requirements, flow rates, and rolling 4-hour forecasts."""
    pipeline = await get_or_create_pipeline()
    anchor_dt = _normalize_datetime(start_time).replace(minute=0, second=0, microsecond=0)

    try:
        predictions = await asyncio.to_thread(
            pipeline.predict_hourly_staffing,
            start_time=anchor_dt,
            horizon_hours=horizon_hours,
        )
        return predictions
    except Exception as exc:
        logger.error("Error projecting hourly staffing: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hourly staffing projection failed: {str(exc)}",
        ) from exc


# ---------------------------------------------------------------------------
# 5. Resource Utilization Forecast (GET /resources)
# ---------------------------------------------------------------------------
@router.get(
    "/resources",
    response_model=ResourceUtilizationForecast,
    status_code=status.HTTP_200_OK,
    summary="Forecast Bed Utilization Percentages and Queue Bottlenecks",
)
async def get_resource_utilization_forecast(
    active_ed_patients: Optional[int] = Query(
        default=None,
        ge=0,
        description="Current Emergency Department patient occupancy count.",
    ),
    active_ward_patients: Optional[int] = Query(
        default=None,
        ge=0,
        description="Current inpatient Ward patient occupancy count.",
    ),
    active_icu_patients: Optional[int] = Query(
        default=None,
        ge=0,
        description="Current Intensive Care Unit patient occupancy count.",
    ),
    waiting_queue_count: Optional[int] = Query(
        default=None,
        ge=0,
        description="Total patients currently awaiting rooming or transfer placement.",
    ),
    forecast_timestamp: Optional[str] = Query(
        default=None,
        description="Forecast timestamp snapshot formatted as 'YYYY-MM-DD HH:MM:SS'.",
    ),
) -> ResourceUtilizationForecast:
    """Forecast physical asset bottlenecks and bed utilization percentages across ED, Ward, and ICU."""
    pipeline = await get_or_create_pipeline()
    anchor_dt = _normalize_datetime(forecast_timestamp)

    try:
        forecast = await asyncio.to_thread(
            pipeline.predict_resource_utilization,
            trajectories=None,
            forecast_timestamp=anchor_dt,
            active_ed_patients=active_ed_patients,
            active_ward_patients=active_ward_patients,
            active_icu_patients=active_icu_patients,
            waiting_queue_count=waiting_queue_count,
        )
        return forecast
    except Exception as exc:
        logger.error("Error projecting resource utilization: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resource utilization forecast failed: {str(exc)}",
        ) from exc


__all__ = [
    "CohortSynthesisRequest",
    "PredictionCache",
    "PredictionRunRequest",
    "PredictionRunResponse",
    "TrajectoryForecastRequest",
    "get_or_create_pipeline",
    "router",
]