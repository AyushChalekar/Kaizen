# backend/src/main.py
"""Production FastAPI application for Digital Twin hospital simulation platform.

Orchestrates discrete-event simulation runs, maintains thread-safe in-memory
telemetry caches, and exposes validated endpoints for real-time frontend dashboards.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure flexible module resolution across varying invocation contexts
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
for search_path in (PROJECT_ROOT, BACKEND_SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

try:
    from backend.src.schemas.contracts import (
        BedTopologyContract,
        HourlyCensusContract,
        PatientStayContract,
    )
    from backend.src.simulation.engine import (
        HospitalSimulationEngine,
        SimulationConfig,
        SimulationResults,
    )
except ImportError:
    try:
        from schemas.contracts import (
            BedTopologyContract,
            HourlyCensusContract,
            PatientStayContract,
        )
        from simulation.engine import (
            HospitalSimulationEngine,
            SimulationConfig,
            SimulationResults,
        )
    except ImportError:
        from contracts import (
            BedTopologyContract,
            HourlyCensusContract,
            PatientStayContract,
        )
        from engine import (
            HospitalSimulationEngine,
            SimulationConfig,
            SimulationResults,
        )

# ---------------------------------------------------------------------------
# Logging & Runtime Defaults
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("digital_twin_api")

DEFAULT_START_DATETIME: Final[datetime] = datetime(2026, 9, 23, 0, 0, 0)


# ---------------------------------------------------------------------------
# Pydantic Request & Response Contracts
# ---------------------------------------------------------------------------
class SimulationRunRequest(BaseModel):
    """Payload to configure and execute a discrete-event simulation run."""

    duration_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Simulation duration span in hours (1 hour to 7 days).",
    )
    ed_capacity: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Emergency Department treatment bay capacity.",
    )
    ward_capacity: int = Field(
        default=150,
        ge=1,
        le=200,
        description="Inpatient medical-surgical ward bed capacity.",
    )
    icu_capacity: int = Field(
        default=30,
        ge=1,
        le=50,
        description="Intensive Care Unit bed capacity.",
    )
    seed: int = Field(
        default=42,
        description="Random number generator seed for deterministic reproducibility.",
    )
    start_datetime: datetime = Field(
        default_factory=lambda: DEFAULT_START_DATETIME,
        description="Anchor datetime representing simulation clock start.",
    )


class SimulationMetricsResponse(BaseModel):
    """Aggregated operational metrics for executive dashboards."""

    total_arrivals: int = Field(..., description="Total patient arrivals simulated.")
    total_admissions: int = Field(..., description="Total inpatient ward and ICU admissions.")
    total_discharges: int = Field(..., description="Total discharged patient encounters.")
    completed_stays_count: int = Field(..., description="Number of fully resolved stays.")
    peak_queue_length: int = Field(..., description="Maximum simultaneous waiting queue length.")
    average_los_hours: float = Field(..., description="Mean total length of stay in hours.")
    ed_utilization_pct: float = Field(..., description="Average ED bay utilization percentage.")
    ward_utilization_pct: float = Field(..., description="Average Ward bed utilization percentage.")
    icu_utilization_pct: float = Field(..., description="Average ICU bed utilization percentage.")


class SimulationDataResponse(BaseModel):
    """Envelope wrapping telemetry, configuration, bed topology, and patient stays."""

    status: str = Field(default="success", description="Response status message.")
    generated_at: str = Field(..., description="ISO 8601 timestamp of data generation.")
    config: SimulationRunRequest = Field(..., description="Parameters applied to this run.")
    metrics: SimulationMetricsResponse = Field(..., description="Summary operational metrics.")
    hourly_census: list[HourlyCensusContract] = Field(
        ..., description="Chronological hourly census snapshots."
    )
    bed_topology: list[BedTopologyContract] = Field(
        ..., description="Current physical bed assets and allocation statuses."
    )
    stays: list[PatientStayContract] = Field(
        default_factory=list,
        description="Detailed patient encounters, vitals, and trajectories.",
    )


# ---------------------------------------------------------------------------
# In-Memory Cache & Simulation State
# ---------------------------------------------------------------------------
@dataclass
class SimulationCache:
    """Thread-safe holder for the active simulation results and configuration."""

    results: SimulationResults
    config: SimulationRunRequest
    generated_at: str


_active_cache: Optional[SimulationCache] = None
_cache_lock = asyncio.Lock()


def _execute_simulation_sync(sim_config: SimulationConfig) -> SimulationResults:
    """Execute the simulation engine synchronously in a worker thread."""
    engine = HospitalSimulationEngine(sim_config)
    return engine.run()


def _build_response_payload(
    cache: SimulationCache, include_stays: bool = True
) -> SimulationDataResponse:
    metrics_model = SimulationMetricsResponse(**cache.results.metrics)
    return SimulationDataResponse(
        status="success",
        generated_at=cache.generated_at,
        config=cache.config,
        metrics=metrics_model,
        hourly_census=cache.results.hourly_census,  # These are now dicts, FastAPI will convert them
        bed_topology=cache.results.bed_topology,    # These are now dicts
        stays=cache.results.stays if include_stays else [], # These are now dicts
    )


async def _run_and_update_cache(
    request: SimulationRunRequest,
) -> SimulationDataResponse:
    """Instantiate config, execute simulation in worker thread, and swap cache."""
    try:
        sim_config = SimulationConfig(
            start_datetime=request.start_datetime,
            duration_hours=request.duration_hours,
            ed_capacity=request.ed_capacity,
            ward_capacity=request.ward_capacity,
            icu_capacity=request.icu_capacity,
            seed=request.seed,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid simulation configuration: {str(exc)}",
        ) from exc

    # Offload SimPy discrete-event execution to thread pool to preserve event loop
    results = await asyncio.to_thread(_execute_simulation_sync, sim_config)
    iso_timestamp = datetime.now(timezone.utc).isoformat()

    new_cache = SimulationCache(
        results=results,
        config=request,
        generated_at=iso_timestamp,
    )

    global _active_cache
    async with _cache_lock:
        _active_cache = new_cache

    logger.info(
        "Simulation successfully executed: %d hours, %d arrivals, %d stays.",
        request.duration_hours,
        results.metrics["total_arrivals"],
        len(results.stays),
    )
    return _build_response_payload(new_cache, include_stays=True)


# ---------------------------------------------------------------------------
# Application Lifespan Context Manager
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm application cache with baseline 24-hour simulation upon startup."""
    logger.info("Application startup: Warm cache with baseline 24-hour simulation...")
    baseline_request = SimulationRunRequest(
        duration_hours=24,
        ed_capacity=50,
        ward_capacity=150,
        icu_capacity=30,
        seed=42,
        start_datetime=DEFAULT_START_DATETIME,
    )
    await _run_and_update_cache(baseline_request)
    yield
    logger.info("Application shutdown: Cleaning up resources.")


# ---------------------------------------------------------------------------
# FastAPI Application & Middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Digital Twin Hospital Operations API",
    version="1.0.0",
    description=(
        "Production backend for discrete-event bed allocation simulation, "
        "hourly census telemetry, and clinical staffing synchronization."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def root_health_check() -> dict[str, Any]:
    """Health check returning API runtime state and active telemetry timestamp."""
    async with _cache_lock:
        active_ts = _active_cache.generated_at if _active_cache else None

    return {
        "status": "healthy",
        "service": "Digital Twin Hospital Operations API",
        "version": "1.0.0",
        "phase": "Production Simulation Telemetry",
        "active_simulation_timestamp": active_ts,
    }


@app.get(
    "/api/simulation/data",
    response_model=SimulationDataResponse,
    tags=["Simulation"],
)
async def get_simulation_data(
    include_stays: bool = Query(
        default=True,
        description="Whether to include granular patient stay records in the payload.",
    ),
) -> SimulationDataResponse:
    """Retrieve the cached simulation telemetry, census snapshots, and bed topology."""
    async with _cache_lock:
        if _active_cache is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Simulation cache is initializing. Please retry momentarily.",
            )
        return _build_response_payload(_active_cache, include_stays=include_stays)


@app.post(
    "/api/simulation/run",
    response_model=SimulationDataResponse,
    tags=["Simulation"],
)
async def run_simulation(
    payload: SimulationRunRequest,
) -> SimulationDataResponse:
    """Execute a parameterized discrete-event simulation and refresh telemetry cache."""
    return await _run_and_update_cache(payload)


@app.post(
    "/api/simulation/reset",
    response_model=SimulationDataResponse,
    tags=["Simulation"],
)
async def reset_simulation() -> SimulationDataResponse:
    """Reset the simulation environment to default baseline parameters (24h, seed 42)."""
    baseline_request = SimulationRunRequest(
        duration_hours=24,
        ed_capacity=50,
        ward_capacity=150,
        icu_capacity=30,
        seed=42,
        start_datetime=DEFAULT_START_DATETIME,
    )
    return await _run_and_update_cache(baseline_request)


# ---------------------------------------------------------------------------
# Direct Invocation Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )