# backend/tests/test_predictions.py
"""Automated intra-module integration verification suite for predictions module.

Validates inter-stage data flows, contract compatibility, stochastic engine
coupling, diurnal staffing alignment, and FastAPI router orchestration across
the prediction module components (schemas, ML pipeline, and API routers).
"""

# backend/tests/test_predictions.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Final

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.schemas.contracts import (
    CareUnitType,
    DispositionType,
    ShiftType,
    TIMESTAMP_FORMAT,
    MIN_PULSE_PRESSURE,
    _parse_strict_timestamp,
)
from src.schemas.prediction_contracts import (
    ClinicalTrajectoryPrediction,
    HourlyStaffingPrediction,
    PatientClinicalPrediction,
    PrimaryChiefComplaint,
    PRIMARY_CHIEF_COMPLAINTS,
    ResourceUtilizationForecast,
)
from src.ml.model_pipeline import (
    PredictiveModelPipeline,
    FallbackHospitalSimulationEngine,
    FallbackMIMICDistributionSampler,
    CHIEF_COMPLAINT_WEIGHTS,
    ESI_DISTRIBUTION_WEIGHTS,
)
from src.api.predictions import (
    router as predictions_router,
    PredictionCache,
    PredictionRunRequest,
    PredictionRunResponse,
    CohortSynthesisRequest,
    TrajectoryForecastRequest,
)

# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def baseline_datetime() -> datetime:
    """Standard anchor timestamp for deterministic predictive runs."""
    return datetime(2026, 9, 23, 0, 0, 0)


@pytest.fixture
def standard_pipeline() -> PredictiveModelPipeline:
    """Initialize a PredictiveModelPipeline with deterministic seed=42."""
    return PredictiveModelPipeline(
        seed=42,
        ed_capacity=50,
        ward_capacity=150,
        icu_capacity=25,
    )


@pytest.fixture
def test_api_client() -> TestClient:
    """FastAPI TestClient wrapping the prediction API router."""
    app = FastAPI(title="Prediction Subsystem Integration Test App")
    app.include_router(predictions_router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Test Group A: Inter-Stage Pipeline Integration & Data Hand-offs
# ---------------------------------------------------------------------------
class TestPipelineInterStageIntegration:
    """Validates data flow, identifier inheritance, and coupling between pipeline stages."""

    def test_cohort_to_trajectories_hand_off(
        self,
        standard_pipeline: PredictiveModelPipeline,
        baseline_datetime: datetime,
    ) -> None:
        """Verify output of patient synthesis seamlessly drives trajectory forecasting."""
        cohort_count = 20
        interval_minutes = 7.5

        # Stage 1: Synthesize cohort
        patients = standard_pipeline.batch_sample_patient_clinicals(count=cohort_count)
        assert len(patients) == cohort_count

        # Stage 2: Forecast trajectories using Stage 1 output
        trajectories = standard_pipeline.batch_predict_trajectories(
            patients=patients,
            base_arrival_time=baseline_datetime,
            arrival_interval_minutes=interval_minutes,
        )
        assert len(trajectories) == cohort_count

        for idx, (patient, trajectory) in enumerate(zip(patients, trajectories)):
            # Verify primary key and foreign key pass-through
            assert trajectory.patient_id == patient.patient_id
            assert trajectory.stay_id == patient.stay_id

            # Verify sequential staggered arrivals
            expected_arrival = baseline_datetime + timedelta(minutes=idx * interval_minutes)
            actual_arrival = _parse_strict_timestamp(trajectory.arrival_time, "arrival_time")
            assert actual_arrival == expected_arrival

            # Verify temporal monotonicity invariant
            dt_arr = actual_arrival
            dt_trg = _parse_strict_timestamp(trajectory.triage_start_time, "triage_start_time")
            dt_bed = _parse_strict_timestamp(trajectory.bed_assigned_time, "bed_assigned_time")
            dt_dis = _parse_strict_timestamp(trajectory.discharge_time, "discharge_time")
            assert dt_arr <= dt_trg <= dt_bed <= dt_dis

            # Verify LoS duration matches timestamps
            duration_hours = round((dt_dis - dt_arr).total_seconds() / 3600.0, 2)
            assert trajectory.predicted_los_hours == pytest.approx(duration_hours, abs=0.05)

    def test_trajectories_to_resource_forecast_aggregation(
        self,
        standard_pipeline: PredictiveModelPipeline,
        baseline_datetime: datetime,
    ) -> None:
        """Verify trajectories aggregate directly into resource utilization metrics."""
        patients = standard_pipeline.batch_sample_patient_clinicals(count=15)
        trajectories = standard_pipeline.batch_predict_trajectories(
            patients=patients,
            base_arrival_time=baseline_datetime,
        )

        # Feed trajectories into resource utilization stage
        forecast = standard_pipeline.predict_resource_utilization(
            trajectories=trajectories,
            forecast_timestamp=baseline_datetime,
        )

        assert isinstance(forecast, ResourceUtilizationForecast)
        assert forecast.forecast_timestamp == baseline_datetime.strftime(TIMESTAMP_FORMAT)

        # Length of stay in forecast must equal the average of individual trajectory LoS
        trajectory_los_list = [
            t.predicted_los_hours for t in trajectories if t.predicted_los_hours is not None
        ]
        expected_avg_los = round(sum(trajectory_los_list) / len(trajectory_los_list), 2)
        assert forecast.predicted_avg_los_hours == pytest.approx(expected_avg_los, abs=0.01)

        # Department breakdown mapping check
        breakdown = forecast.bed_utilization_breakdown
        assert breakdown[CareUnitType.ED_ONLY] == forecast.ed_utilization_pct
        assert breakdown[CareUnitType.WARD] == forecast.ward_utilization_pct
        assert breakdown[CareUnitType.ICU] == forecast.icu_utilization_pct

    def test_acuity_escalation_coupling_across_stages(
        self,
        standard_pipeline: PredictiveModelPipeline,
        baseline_datetime: datetime,
    ) -> None:
        """Validate clinical acuity translates to disposition and escalation outcomes."""
        sample_size = 40

        # High-acuity cohort (ESI 1 = Resuscitation)
        critical_patients = standard_pipeline.batch_sample_patient_clinicals(
            count=sample_size, base_acuity=1
        )
        critical_trajs = standard_pipeline.batch_predict_trajectories(
            patients=critical_patients, base_arrival_time=baseline_datetime
        )

        # Low-acuity cohort (ESI 5 = Non-urgent)
        minor_patients = standard_pipeline.batch_sample_patient_clinicals(
            count=sample_size, base_acuity=5
        )
        minor_trajs = standard_pipeline.batch_predict_trajectories(
            patients=minor_patients, base_arrival_time=baseline_datetime
        )

        critical_icu_or_ward_count = sum(
            1 for t in critical_trajs if t.predicted_disposition in (DispositionType.ICU, DispositionType.WARD)
        )
        minor_icu_or_ward_count = sum(
            1 for t in minor_trajs if t.predicted_disposition in (DispositionType.ICU, DispositionType.WARD)
        )
        critical_vent_count = sum(1 for t in critical_trajs if t.requires_ventilation == 1)
        minor_vent_count = sum(1 for t in minor_trajs if t.requires_ventilation == 1)

        # Clinical logic integration invariant: ESI 1 must demand higher admissions & ventilation
        assert critical_icu_or_ward_count > minor_icu_or_ward_count
        assert critical_vent_count >= minor_vent_count

    def test_run_pipeline_end_to_end_cohesion(
        self,
        standard_pipeline: PredictiveModelPipeline,
        baseline_datetime: datetime,
    ) -> None:
        """Assert run_pipeline orchestrates all stages with mutual data consistency."""
        cohort_size = 12
        horizon_hours = 24

        output = standard_pipeline.run_pipeline(
            cohort_size=cohort_size,
            start_time=baseline_datetime,
            forecast_horizon_hours=horizon_hours,
        )

        assert set(output.keys()) == {
            "patient_profiles",
            "trajectories",
            "staffing_predictions",
            "resource_forecast",
        }

        patients = output["patient_profiles"]
        trajectories = output["trajectories"]
        staffing = output["staffing_predictions"]
        forecast = output["resource_forecast"]

        assert len(patients) == cohort_size
        assert len(trajectories) == cohort_size
        assert len(staffing) == horizon_hours

        # Mutual ID verification across generated profiles and trajectories
        patient_id_map = {p.stay_id: p.patient_id for p in patients}
        for t in trajectories:
            assert t.stay_id in patient_id_map
            assert t.patient_id == patient_id_map[t.stay_id]

        # Forecast timestamp matches start_time string
        assert forecast.forecast_timestamp == baseline_datetime.strftime(TIMESTAMP_FORMAT)


# ---------------------------------------------------------------------------
# Test Group B: Simulation Engine & Sampler Subsystem Coupling
# ---------------------------------------------------------------------------
class TestEngineAndSamplerSubsystemIntegration:
    """Validates how the ML pipeline integrates with hospital state and distributions."""

    def test_engine_snapshot_integration_with_resource_forecaster(
        self,
        baseline_datetime: datetime,
    ) -> None:
        """Verify custom engine occupancy states drive resource forecasts when unoverridden."""
        custom_engine = FallbackHospitalSimulationEngine(
            ed_capacity=40,
            ward_capacity=100,
            icu_capacity=20,
        )
        custom_engine.ed_occupancy = 30
        custom_engine.ward_occupancy = 85
        custom_engine.icu_occupancy = 16
        custom_engine.ed_queue = 5
        custom_engine.ward_queue = 3
        custom_engine.icu_queue = 2

        pipeline = PredictiveModelPipeline(
            engine=custom_engine,
            ed_capacity=40,
            ward_capacity=100,
            icu_capacity=20,
            seed=42,
        )

        forecast = pipeline.predict_resource_utilization(
            forecast_timestamp=baseline_datetime,
        )

        assert forecast.ed_utilization_pct == pytest.approx(75.0)
        assert forecast.ward_utilization_pct == pytest.approx(85.0)
        assert forecast.icu_utilization_pct == pytest.approx(80.0)
        assert forecast.peak_queue_length == 10

    def test_pipeline_seed_reproducibility_and_divergence(
        self,
        baseline_datetime: datetime,
    ) -> None:
        """Assert identical seeds reproduce exact outputs, while differing seeds diverge."""
        pipeline_a1 = PredictiveModelPipeline(seed=12345)
        pipeline_a2 = PredictiveModelPipeline(seed=12345)
        pipeline_b = PredictiveModelPipeline(seed=99999)

        run_a1 = pipeline_a1.run_pipeline(cohort_size=8, start_time=baseline_datetime)
        run_a2 = pipeline_a2.run_pipeline(cohort_size=8, start_time=baseline_datetime)
        run_b = pipeline_b.run_pipeline(cohort_size=8, start_time=baseline_datetime)

        # Verify identical output on matching seeds
        dump_a1 = [p.model_dump() for p in run_a1["patient_profiles"]]
        dump_a2 = [p.model_dump() for p in run_a2["patient_profiles"]]
        dump_b = [p.model_dump() for p in run_b["patient_profiles"]]
        assert dump_a1 == dump_a2
        assert dump_a1 != dump_b

        # Verify trajectories identity
        traj_a1 = [t.model_dump() for t in run_a1["trajectories"]]
        traj_a2 = [t.model_dump() for t in run_a2["trajectories"]]
        assert traj_a1 == traj_a2

    def test_sampler_vital_bounds_and_pulse_pressure_preservation(
        self,
        standard_pipeline: PredictiveModelPipeline,
    ) -> None:
        """Ensure stochastic sampler vitals adhere to biological invariants in batch runs."""
        patients = standard_pipeline.batch_sample_patient_clinicals(count=100)

        for p in patients:
            assert isinstance(p, PatientClinicalPrediction)
            assert 18 <= p.age <= 95
            assert p.gender in ("Female", "Male")
            assert 0 <= p.charlson_index <= 10
            assert p.chief_complaint in PRIMARY_CHIEF_COMPLAINTS
            assert 1 <= p.triage_acuity <= 5

            # Biological invariant: Pulse pressure >= 15.0 mmHg
            pulse_pressure = p.sbp - p.dbp
            assert pulse_pressure >= MIN_PULSE_PRESSURE
            assert p.pulse_pressure == pulse_pressure


# ---------------------------------------------------------------------------
# Test Group C: Diurnal Staffing & Temporal Synchronization
# ---------------------------------------------------------------------------
# backend/tests/test_predictions.py

class TestDiurnalStaffingAndTemporalIntegration:
    """Validates staffing model integration with shifts, diurnal curves, and calendar days."""

    def test_predict_hourly_staffing_multiday_horizon(
        self,
        standard_pipeline: PredictiveModelPipeline,
        baseline_datetime: datetime,
    ) -> None:
        """Verify 48-hour projection preserves shift mapping, calendar days, and rolling sums."""
        horizon_hours = 48
        predictions = standard_pipeline.predict_hourly_staffing(
            start_time=baseline_datetime,
            horizon_hours=horizon_hours,
        )
        assert len(predictions) == horizon_hours

        for pred in predictions:
            assert isinstance(pred, HourlyStaffingPrediction)
            dt = _parse_strict_timestamp(pred.timestamp, "timestamp")

            # Invariant: Flooring to zero minutes and seconds
            assert dt.minute == 0
            assert dt.second == 0
            assert pred.hour == dt.hour

            # Shift hour boundary alignment
            if 7 <= pred.hour <= 14:
                assert pred.shift_id == ShiftType.DAY
                assert pred.baseline_nurses == 35
                assert pred.baseline_physicians == 10
            elif 15 <= pred.hour <= 22:
                assert pred.shift_id == ShiftType.EVENING
                assert pred.baseline_nurses == 30
                assert pred.baseline_physicians == 8
            else:
                assert pred.shift_id == ShiftType.NIGHT
                assert pred.baseline_nurses == 20
                assert pred.baseline_physicians == 5

            # Calendar day and weekend indicator alignment
            assert pred.day_of_week == dt.strftime("%A")
            assert pred.is_weekend == (1 if dt.weekday() in (5, 6) else 0)

            # Headcount variance metrics
            assert pred.nurse_variance == pred.predicted_active_nurses - pred.baseline_nurses
            assert pred.physician_variance == pred.predicted_active_physicians - pred.baseline_physicians

        # Validate rolling 4-hour forecast alignment for non-boundary slots
        for i in range(horizon_hours - 4):
            rolling_sum = sum(predictions[j].projected_arrivals for j in range(i + 1, i + 5))
            assert predictions[i].incoming_arrivals_next_4h == pytest.approx(rolling_sum, abs=0.01)
# ---------------------------------------------------------------------------
# Test Group D: FastAPI Router Intra-Module Integration
# ---------------------------------------------------------------------------
class TestApiRouterIntraModuleIntegration:
    """Validates FastAPI predictions router integration with pipeline and contracts."""

    def test_api_cohort_to_trajectories_roundtrip(
        self,
        test_api_client: TestClient,
    ) -> None:
        """Call /cohort endpoint, then pipe output directly into /trajectories endpoint."""
        # 1. Generate cohort via API
        cohort_resp = test_api_client.post(
            "/api/predictions/cohort",
            json={"cohort_size": 6, "base_acuity": 2},
        )
        assert cohort_resp.status_code == 200
        patients_json = cohort_resp.json()
        assert len(patients_json) == 6

        # Validate all returned items conform to PatientClinicalPrediction
        for p in patients_json:
            validated = PatientClinicalPrediction(**p)
            assert validated.triage_acuity == 2
            assert validated.sbp - validated.dbp >= MIN_PULSE_PRESSURE

        # 2. Pipe patients into /trajectories endpoint
        traj_payload = {
            "patients": patients_json,
            "base_arrival_time": "2026-09-23 08:00:00",
            "arrival_interval_minutes": 10.0,
        }
        traj_resp = test_api_client.post(
            "/api/predictions/trajectories",
            json=traj_payload,
        )
        assert traj_resp.status_code == 200
        trajs_json = traj_resp.json()
        assert len(trajs_json) == 6

        # Validate trajectory response schema and IDs
        for idx, t in enumerate(trajs_json):
            validated_traj = ClinicalTrajectoryPrediction(**t)
            assert validated_traj.patient_id == patients_json[idx]["patient_id"]
            assert validated_traj.stay_id == patients_json[idx]["stay_id"]

    def test_api_run_pipeline_orchestration(
        self,
        test_api_client: TestClient,
    ) -> None:
        """Verify POST /run executes full pipeline and returns validated PredictionRunResponse."""
        payload = {
            "cohort_size": 8,
            "forecast_horizon_hours": 12,
            "start_datetime": "2026-09-23 10:00:00",
            "seed": 42,
        }
        response = test_api_client.post("/api/predictions/run", json=payload)
        assert response.status_code == 200

        data = response.json()
        validated_response = PredictionRunResponse(**data)

        assert validated_response.status == "success"
        assert validated_response.cohort_size == 8
        assert validated_response.forecast_horizon_hours == 12
        assert len(validated_response.patient_profiles) == 8
        assert len(validated_response.trajectories) == 8
        assert len(validated_response.staffing_predictions) == 12
        assert isinstance(validated_response.resource_forecast, ResourceUtilizationForecast)

    def test_api_staffing_endpoint_query_parameters(
        self,
        test_api_client: TestClient,
    ) -> None:
        """Verify GET /staffing returns shift-aligned staffing projections."""
        response = test_api_client.get(
            "/api/predictions/staffing",
            params={
                "start_time": "2026-09-23 08:00:00",
                "horizon_hours": 16,
            },
        )
        assert response.status_code == 200
        items = response.json()
        assert len(items) == 16

        for item in items:
            pred = HourlyStaffingPrediction(**item)
            assert 0 <= pred.hour <= 23
            assert pred.predicted_active_nurses >= 15
            assert pred.predicted_active_physicians >= 3

    def test_api_resource_utilization_endpoint_with_occupancy_query(
        self,
        test_api_client: TestClient,
    ) -> None:
        """Verify GET /resources computes accurate utilization percentages from query params."""
        params = {
            "active_ed_patients": 25,
            "active_ward_patients": 105,
            "active_icu_patients": 20,
            "waiting_queue_count": 6,
            "forecast_timestamp": "2026-09-23 14:00:00",
        }
        response = test_api_client.get("/api/predictions/resources", params=params)
        assert response.status_code == 200

        forecast = ResourceUtilizationForecast(**response.json())
        assert forecast.ed_utilization_pct == pytest.approx(50.0)
        assert forecast.ward_utilization_pct == pytest.approx(70.0)
        assert forecast.icu_utilization_pct == pytest.approx(80.0)
        assert forecast.peak_queue_length == 6
        assert forecast.forecast_timestamp == "2026-09-23 14:00:00"

    def test_api_validation_error_handling(
        self,
        test_api_client: TestClient,
    ) -> None:
        """Verify router properly rejects schema violations with HTTP 422."""
        # 1. Invalid cohort size (exceeds le=500)
        resp_invalid_cohort = test_api_client.post(
            "/api/predictions/run",
            json={"cohort_size": 999},
        )
        assert resp_invalid_cohort.status_code == 422

        # 2. Extra forbidden fields in request
        resp_forbidden_field = test_api_client.post(
            "/api/predictions/run",
            json={"cohort_size": 10, "unexpected_key": "disallowed"},
        )
        assert resp_forbidden_field.status_code == 422

        # 3. Empty patient list in trajectory forecast request
        resp_empty_trajs = test_api_client.post(
            "/api/predictions/trajectories",
            json={"patients": []},
        )
        assert resp_empty_trajs.status_code == 422