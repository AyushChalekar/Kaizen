# backend/tests/test_simulation.py
"""Automated verification suite for hospital discrete-event simulation engine.

Validates deterministic seed reproducibility, clinical and temporal invariants,
hourly telemetry schema conformity, bed topology conservation, and configuration
boundaries for HospitalSimulationEngine and MIMICDistributionSampler.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import pytest

# Ensure repository root and backend source directories are discoverable on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
for p in (PROJECT_ROOT, BACKEND_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from schemas.contracts import (
    BedStatus,
    BedTopologyContract,
    CareUnitType,
    DispositionType,
    HourlyCensusContract,
    PatientStayContract,
    ShiftType,
)
from data_generator.mimic_distributions import (
    DistributionConfig,
    MIMICDistributionSampler,
)
from simulation.engine import (
    HospitalSimulationEngine,
    SimulationConfig,
    SimulationResults,
)

TIMESTAMP_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
HOURLY_FORMAT: Final[str] = "%Y-%m-%d %H:00:00"


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def base_datetime() -> datetime:
    """Standardized baseline simulation start timestamp."""
    return datetime(2026, 9, 23, 0, 0, 0)


@pytest.fixture
def standard_config_12h(base_datetime: datetime) -> SimulationConfig:
    """12-hour simulation configuration with deterministic seed=42."""
    return SimulationConfig(
        start_datetime=base_datetime,
        duration_hours=12,
        ed_capacity=50,
        ward_capacity=150,
        icu_capacity=30,
        seed=42,
    )


@pytest.fixture
def standard_results_12h(standard_config_12h: SimulationConfig) -> SimulationResults:
    """Execute and return simulation results for the 12-hour deterministic profile."""
    engine = HospitalSimulationEngine(standard_config_12h)
    return engine.run()


@pytest.fixture
def short_config_6h(base_datetime: datetime) -> SimulationConfig:
    """Short 6-hour simulation configuration for rapid assertion passes."""
    return SimulationConfig(
        start_datetime=base_datetime,
        duration_hours=6,
        ed_capacity=20,
        ward_capacity=60,
        icu_capacity=10,
        seed=101,
    )


# ---------------------------------------------------------------------------
# Test Group A: Determinism and Seed Reproducibility
# ---------------------------------------------------------------------------
def test_engine_determinism_identical_seeds(
    standard_config_12h: SimulationConfig,
) -> None:
    """Assert two independent simulation runs with identical seeds produce exact outputs."""
    engine_a = HospitalSimulationEngine(standard_config_12h)
    results_a = engine_a.run()

    engine_b = HospitalSimulationEngine(standard_config_12h)
    results_b = engine_b.run()

    # Verify identical counts
    assert len(results_a.stays) == len(results_b.stays)
    assert len(results_a.hourly_census) == len(results_b.hourly_census)
    assert len(results_a.bed_topology) == len(results_b.bed_topology)
    assert results_a.metrics == results_b.metrics

    # Verify identical patient stay models
    for stay_a, stay_b in zip(results_a.stays, results_b.stays):
        assert stay_a.model_dump() == stay_b.model_dump()

    # Verify identical hourly census models
    for census_a, census_b in zip(results_a.hourly_census, results_b.hourly_census):
        assert census_a.model_dump() == census_b.model_dump()

    # Verify identical bed topologies
    for bed_a, bed_b in zip(results_a.bed_topology, results_b.bed_topology):
        assert bed_a.model_dump() == bed_b.model_dump()


def test_engine_divergence_differing_seeds(base_datetime: datetime) -> None:
    """Assert two simulation runs with different seeds generate divergent metrics."""
    config_a = SimulationConfig(
        start_datetime=base_datetime,
        duration_hours=12,
        seed=42,
    )
    config_b = SimulationConfig(
        start_datetime=base_datetime,
        duration_hours=12,
        seed=999,
    )

    results_a = HospitalSimulationEngine(config_a).run()
    results_b = HospitalSimulationEngine(config_b).run()

    # Verify divergence in operational metrics and stays
    assert results_a.metrics != results_b.metrics
    dump_a = [s.model_dump() for s in results_a.stays]
    dump_b = [s.model_dump() for s in results_b.stays]
    assert dump_a != dump_b


# ---------------------------------------------------------------------------
# Test Group B: Patient Stay Invariants & Monotonicity
# ---------------------------------------------------------------------------
def test_patient_stay_invariants(standard_results_12h: SimulationResults) -> None:
    """Validate every completed patient stay strictly complies with clinical domain rules."""
    assert len(standard_results_12h.stays) > 0, "No stays completed in 12h run."

    for stay in standard_results_12h.stays:
        # Pydantic instance verification
        assert isinstance(stay, PatientStayContract)

        # Identifier bounds
        assert 1 <= stay.stay_id <= 100_000
        assert 100_001 <= stay.patient_id <= 200_000

        # Physiological pulse pressure invariant
        pulse_pressure = stay.sbp - stay.dbp
        assert pulse_pressure >= 15.0, (
            f"Pulse pressure invariant violated on stay_id={stay.stay_id}: "
            f"sbp={stay.sbp}, dbp={stay.dbp}, pp={pulse_pressure}"
        )

        # Monotonic timestamp ordering
        dt_arr = datetime.strptime(stay.arrival_time, TIMESTAMP_FORMAT)
        dt_trg = datetime.strptime(stay.triage_start_time, TIMESTAMP_FORMAT)
        dt_bed = datetime.strptime(stay.bed_assigned_time, TIMESTAMP_FORMAT)
        dt_dis = datetime.strptime(stay.discharge_time, TIMESTAMP_FORMAT)

        assert dt_arr <= dt_trg <= dt_bed <= dt_dis, (
            f"Temporal monotonicity violated on stay_id={stay.stay_id}: "
            f"arrival={stay.arrival_time}, triage={stay.triage_start_time}, "
            f"bed={stay.bed_assigned_time}, discharge={stay.discharge_time}"
        )

        # Relational key flooring verification
        expected_arrival_hour = dt_arr.strftime(HOURLY_FORMAT)
        assert stay.arrival_hour == expected_arrival_hour, (
            f"Relational key mismatch: arrival_hour={stay.arrival_hour} "
            f"!= expected={expected_arrival_hour}"
        )

        # Length of stay duration consistency
        calculated_los_hours = (dt_dis - dt_arr).total_seconds() / 3600.0
        assert stay.los_hours == pytest.approx(round(calculated_los_hours, 2), abs=0.02)
        assert 0.5 <= stay.los_hours <= 336.0


# ---------------------------------------------------------------------------
# Test Group C: Hourly Census Telemetry Invariants
# ---------------------------------------------------------------------------
def test_hourly_census_invariants(
    standard_results_12h: SimulationResults,
    standard_config_12h: SimulationConfig,
) -> None:
    """Validate completeness, chronological continuity, shift mapping, and staffing."""
    census_records = standard_results_12h.hourly_census
    assert len(census_records) == standard_config_12h.duration_hours

    sim_start = standard_config_12h.start_datetime

    for idx, census in enumerate(census_records):
        assert isinstance(census, HourlyCensusContract)

        # Chronological progression
        expected_dt = sim_start + timedelta(hours=idx + 1)
        expected_timestamp = expected_dt.strftime(HOURLY_FORMAT)

        assert census.timestamp == expected_timestamp
        assert census.hour == expected_dt.hour
        assert census.day_of_week == expected_dt.strftime("%A")
        assert census.is_weekend == (1 if expected_dt.weekday() in (5, 6) else 0)

        # Shift Mapping Verification
        # Day: 07:00 - 14:59 (hours 7 to 14)
        # Evening: 15:00 - 22:59 (hours 15 to 22)
        # Night: 23:00 - 06:59 (hours 23, 0 to 6)
        if 7 <= census.hour <= 14:
            assert census.shift_id == ShiftType.DAY
        elif 15 <= census.hour <= 22:
            assert census.shift_id == ShiftType.EVENING
        else:
            assert census.shift_id == ShiftType.NIGHT

        # Staffing Bounds
        assert 15 <= census.active_nurses <= 50
        assert 3 <= census.active_doctors <= 20

        # Occupancy capacities
        assert 0 <= census.ed_occupancy <= standard_config_12h.ed_capacity
        assert 0 <= census.ward_occupancy <= standard_config_12h.ward_capacity
        assert 0 <= census.icu_occupancy <= standard_config_12h.icu_capacity

        # Non-negative counters
        assert census.arrivals_count >= 0
        assert census.admissions_count >= 0
        assert census.discharges_count >= 0
        assert census.patients_in_queue >= 0
        assert census.incoming_arrivals_next_4h >= 0


# ---------------------------------------------------------------------------
# Test Group D: Bed Topology Conservation & Foreign Keys
# ---------------------------------------------------------------------------
def test_bed_topology_conservation_and_keys(
    standard_results_12h: SimulationResults,
    standard_config_12h: SimulationConfig,
) -> None:
    """Validate bed quantity conservation, department flags, and relational state."""
    beds = standard_results_12h.bed_topology
    expected_total_beds = (
        standard_config_12h.ed_capacity
        + standard_config_12h.ward_capacity
        + standard_config_12h.icu_capacity
    )
    assert len(beds) == expected_total_beds

    ed_count = sum(1 for b in beds if b.department == CareUnitType.ED_ONLY)
    ward_count = sum(1 for b in beds if b.department == CareUnitType.WARD)
    icu_count = sum(1 for b in beds if b.department == CareUnitType.ICU)

    assert ed_count == standard_config_12h.ed_capacity
    assert ward_count == standard_config_12h.ward_capacity
    assert icu_count == standard_config_12h.icu_capacity

    for bed in beds:
        assert isinstance(bed, BedTopologyContract)

        # Department-specific attributes
        if bed.department == CareUnitType.ED_ONLY:
            assert bed.bed_id.startswith("ED-BAY-")
        elif bed.department == CareUnitType.WARD:
            assert bed.bed_id.startswith("WARD-BED-")
        elif bed.department == CareUnitType.ICU:
            assert bed.bed_id.startswith("ICU-BED-")
            assert bed.has_ventilator is True
            assert bed.is_negative_pressure is True

        # Invariant: Occupied beds must retain a valid foreign key stay_id
        if bed.status == BedStatus.OCCUPIED:
            assert bed.current_stay_id is not None
            assert 1 <= bed.current_stay_id <= 100_000
        elif bed.status in {
            BedStatus.RESERVED,
            BedStatus.AWAITING_CLEANING,
            BedStatus.OUT_OF_SERVICE,
        }:
            assert bed.current_stay_id is None


# ---------------------------------------------------------------------------
# Test Group E: System Dynamics & Conservation
# ---------------------------------------------------------------------------
def test_system_dynamics_and_conservation(
    standard_results_12h: SimulationResults,
) -> None:
    """Verify high-level flow conservation and operational rate constraints."""
    metrics = standard_results_12h.metrics

    total_arr = metrics["total_arrivals"]
    total_adm = metrics["total_admissions"]
    total_dis = metrics["total_discharges"]
    completed = metrics["completed_stays_count"]

    assert total_arr >= completed
    assert total_arr >= total_dis
    assert completed == total_dis
    assert metrics["peak_queue_length"] >= 0

    assert 0.0 <= metrics["ed_utilization_pct"] <= 100.0
    assert 0.0 <= metrics["ward_utilization_pct"] <= 100.0
    assert 0.0 <= metrics["icu_utilization_pct"] <= 100.0
    assert metrics["average_los_hours"] >= 0.5


# ---------------------------------------------------------------------------
# Test Group F: Boundary and Configuration Validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ed,ward,icu,hours",
    [
        (0, 150, 30, 24),    # ed_capacity below minimum (1)
        (101, 150, 30, 24),  # ed_capacity exceeds maximum (100)
        (50, 0, 30, 24),     # ward_capacity below minimum (1)
        (50, 201, 30, 24),   # ward_capacity exceeds maximum (200)
        (50, 150, 0, 24),    # icu_capacity below minimum (1)
        (50, 150, 51, 24),   # icu_capacity exceeds maximum (50)
        (50, 150, 30, 0),    # duration_hours below minimum (1)
        (50, 150, 30, -4),   # duration_hours negative
    ],
)
def test_simulation_config_boundary_validation(
    ed: int, ward: int, icu: int, hours: int
) -> None:
    """Assert ValueError is raised when constructing invalid SimulationConfig instances."""
    with pytest.raises(ValueError):
        SimulationConfig(
            ed_capacity=ed,
            ward_capacity=ward,
            icu_capacity=icu,
            duration_hours=hours,
        )


def test_short_simulation_execution(short_config_6h: SimulationConfig) -> None:
    """Verify simulation executes to completion under constrained small-scale topologies."""
    engine = HospitalSimulationEngine(short_config_6h)
    results = engine.run()

    assert len(results.hourly_census) == 6
    assert len(results.bed_topology) == 20 + 60 + 10
    assert results.metrics["total_arrivals"] >= 0