# backend/src/simulation/engine.py
"""Discrete-event simulation engine for hospital operations and bed allocation.

Implements SimPy modeling of patient arrival flows, triage routing, resource
queuing, dynamic bed topology assignment, and hourly telemetry snapshots
synchronized with MIMIC-IV empirical distributions and Pydantic v2 domain contracts.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Final, Optional

import simpy

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

TIMESTAMP_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
HOURLY_FORMAT: Final[str] = "%Y-%m-%d %H:00:00"


# ---------------------------------------------------------------------------
# Configuration & Results Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SimulationConfig:
    """Configuration parameters for the hospital discrete-event simulation."""

    start_datetime: datetime = field(
        default_factory=lambda: datetime(2026, 9, 23, 0, 0, 0)
    )
    duration_hours: int = 24
    ed_capacity: int = 50
    ward_capacity: int = 150
    icu_capacity: int = 30
    seed: int = 42
    distribution_config: Optional[DistributionConfig] = None

    def __post_init__(self) -> None:
        if not (1 <= self.ed_capacity <= 100):
            raise ValueError(f"ed_capacity must be between 1 and 100, got {self.ed_capacity}")
        if not (1 <= self.ward_capacity <= 200):
            raise ValueError(f"ward_capacity must be between 1 and 200, got {self.ward_capacity}")
        if not (1 <= self.icu_capacity <= 50):
            raise ValueError(f"icu_capacity must be between 1 and 50, got {self.icu_capacity}")
        if self.duration_hours < 1:
            raise ValueError(f"duration_hours must be at least 1, got {self.duration_hours}")


@dataclass
class SimulationResults:
    stays: list[dict[str, Any]]
    hourly_census: list[dict[str, Any]]
    bed_topology: list[dict[str, Any]]
    metrics: dict[str, Any]


# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------
class HospitalSimulationEngine:
    """Discrete-event simulation engine managing clinical resources and telemetry."""

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self.config = config or SimulationConfig()
        self.rng = random.Random(self.config.seed)
        self.sampler = MIMICDistributionSampler(
            config=self.config.distribution_config,
            seed=self.config.seed,
        )

        # SimPy core environment and resource pools
        self.env = simpy.Environment()
        self.ed_beds = simpy.Resource(self.env, capacity=self.config.ed_capacity)
        self.ward_beds = simpy.Resource(self.env, capacity=self.config.ward_capacity)
        self.icu_beds = simpy.Resource(self.env, capacity=self.config.icu_capacity)

        # Bed topology tracking: bed_id -> BedTopologyContract
        self.bed_map: dict[str, BedTopologyContract] = {}
        self._initialize_bed_topology()

        # Telemetry storage
        self.completed_stays: list[PatientStayContract] = []
        self.hourly_census_records: list[HourlyCensusContract] = []

        # Identifier sequences
        self.current_stay_id = 1
        self.current_patient_id = 100_001

        # Hourly interval delta counters
        self.hourly_arrivals = 0
        self.hourly_admissions = 0
        self.hourly_discharges = 0

        # Cumulative counters for metrics
        self.total_arrivals = 0
        self.total_admissions = 0
        self.total_discharges = 0
        self.peak_queue_length = 0

    # -----------------------------------------------------------------------
    # Topology Initialization & State Synchronization
    # -----------------------------------------------------------------------
    def _initialize_bed_topology(self) -> None:
        """Instantiate physical bed contracts across ED, Ward, and ICU departments."""
        self.bed_map.clear()

        # 1. Emergency Department Bays (1 to ed_capacity)
        for i in range(1, self.config.ed_capacity + 1):
            bed_id = f"ED-BAY-{i:02d}"
            room_id = f"ED-ROOM-{(i - 1) // 4 + 1:02d}"
            self.bed_map[bed_id] = BedTopologyContract(
                bed_id=bed_id,
                department=CareUnitType.ED_ONLY,
                room_id=room_id,
                status=BedStatus.RESERVED,
                is_negative_pressure=False,
                has_ventilator=False,
                current_stay_id=None,
            )

        # 2. Inpatient Ward Beds (1 to ward_capacity)
        for i in range(1, self.config.ward_capacity + 1):
            bed_id = f"WARD-BED-{i:02d}"
            room_id = f"WARD-ROOM-{(i - 1) // 2 + 1:02d}"
            self.bed_map[bed_id] = BedTopologyContract(
                bed_id=bed_id,
                department=CareUnitType.WARD,
                room_id=room_id,
                status=BedStatus.RESERVED,
                is_negative_pressure=False,
                has_ventilator=False,
                current_stay_id=None,
            )

        # 3. Intensive Care Unit Beds (1 to icu_capacity)
        for i in range(1, self.config.icu_capacity + 1):
            bed_id = f"ICU-BED-{i:02d}"
            room_id = f"ICU-ROOM-{i:02d}"
            self.bed_map[bed_id] = BedTopologyContract(
                bed_id=bed_id,
                department=CareUnitType.ICU,
                room_id=room_id,
                status=BedStatus.RESERVED,
                is_negative_pressure=True,
                has_ventilator=True,
                current_stay_id=None,
            )

    def _assign_physical_bed(
        self, department: CareUnitType, stay_id: int
    ) -> BedTopologyContract:
        """Find an unassigned bed in the target department and bind the stay_id."""
        for bed_id, bed in self.bed_map.items():
            if bed.department == department and bed.status != BedStatus.OCCUPIED:
                updated_bed = bed.model_copy(
                    update={
                        "status": BedStatus.OCCUPIED,
                        "current_stay_id": stay_id,
                    }
                )
                self.bed_map[bed_id] = updated_bed
                return updated_bed

        raise RuntimeError(
            f"Physical bed exhaustion in department '{department.value}' for stay_id={stay_id}. "
            "Resource capacity and topology count are out of synchronization."
        )

    def _release_physical_bed(self, bed_id: str) -> None:
        """Release a physical bed back to available reserved status."""
        bed = self.bed_map.get(bed_id)
        if bed is not None:
            updated_bed = bed.model_copy(
                update={
                    "status": BedStatus.RESERVED,
                    "current_stay_id": None,
                }
            )
            self.bed_map[bed_id] = updated_bed

    # -----------------------------------------------------------------------
    # Patient Trajectory Lifecycle (SimPy Process)
    # -----------------------------------------------------------------------
    def _patient_process(self, stay_id: int, patient_id: int) -> simpy.events.Process:
        """Simulate the end-to-end clinical lifecycle for an individual patient."""
        # 1. Arrival Milestone
        t_arr_sec = self.env.now
        arrival_dt = self.config.start_datetime + timedelta(seconds=int(t_arr_sec))
        arrival_time_str = arrival_dt.strftime(TIMESTAMP_FORMAT)
        arrival_hour_str = arrival_dt.strftime(HOURLY_FORMAT)

        self.hourly_arrivals += 1
        self.total_arrivals += 1

        # 2. Demographic & Comorbidity Sampling
        cfg = self.config.distribution_config or DistributionConfig()
        age = int(
            round(
                self.sampler._sample_truncated_normal(
                    cfg.age_mean, cfg.age_std, cfg.age_min, cfg.age_max
                )
            )
        )
        gender = "Female" if self.rng.random() < cfg.prob_female else "Male"
        cci_lam = 1.2 + (age / 35.0)
        charlson_index = max(0, min(10, self.sampler._sample_poisson(cci_lam)))

        # 3. Triage Acuity & Chief Complaint
        acuity = self.rng.choices(
            [1, 2, 3, 4, 5],
            weights=cfg.esi_probabilities,
            k=1,
        )[0]
        complaint = self.rng.choices(
            cfg.chief_complaints,
            weights=cfg.chief_complaint_weights,
            k=1,
        )[0]

        # 4. Hemodynamics & Vitals
        vitals = self.sampler.sample_vitals(acuity, complaint)

        # 5. Trajectory & Disposition Determination
        p_icu, p_ward, p_ed = cfg.disposition_by_esi[acuity - 1]
        disposition = self.rng.choices(
            [DispositionType.ICU, DispositionType.WARD, DispositionType.ED_DISCHARGE],
            weights=[p_icu, p_ward, p_ed],
            k=1,
        )[0]

        # 6. Triage Milestones
        if acuity == 1:
            triage_delay = 0.0
            triage_duration = self.rng.uniform(60.0, 180.0)
        elif acuity == 2:
            triage_delay = self.rng.uniform(30.0, 180.0)
            triage_duration = self.rng.uniform(180.0, 420.0)
        else:
            triage_delay = self.rng.uniform(120.0, 600.0)
            triage_duration = self.rng.uniform(300.0, 900.0)

        if triage_delay > 0.0:
            yield self.env.timeout(triage_delay)

        t_trg_sec = self.env.now
        triage_start_dt = self.config.start_datetime + timedelta(seconds=int(t_trg_sec))
        triage_start_time_str = triage_start_dt.strftime(TIMESTAMP_FORMAT)

        yield self.env.timeout(triage_duration)

        # 7. Unit Routing Setup
        is_direct_icu = disposition == DispositionType.ICU and acuity == 1

        if is_direct_icu:
            initial_care_unit = CareUnitType.ICU
            icu_transfer_flag = 1
            primary_resource = self.icu_beds
            primary_dept = CareUnitType.ICU
            los_params = cfg.los_icu_params
        elif disposition == DispositionType.ED_DISCHARGE:
            initial_care_unit = CareUnitType.ED_ONLY
            icu_transfer_flag = 0
            primary_resource = self.ed_beds
            primary_dept = CareUnitType.ED_ONLY
            los_params = cfg.los_ed_params
        elif disposition == DispositionType.WARD:
            initial_care_unit = CareUnitType.WARD
            icu_transfer_flag = 1 if self.rng.random() < cfg.ward_icu_transfer_prob else 0
            primary_resource = self.ed_beds
            primary_dept = CareUnitType.ED_ONLY
            los_params = cfg.los_ward_params
        else:
            # ICU transfer via ED
            initial_care_unit = CareUnitType.ICU
            icu_transfer_flag = 1
            primary_resource = self.ed_beds
            primary_dept = CareUnitType.ED_ONLY
            los_params = cfg.los_icu_params

        # Ventilation Requirement
        if initial_care_unit == CareUnitType.ICU:
            vent_prob = 0.75 if (vitals["o2_sat"] < 88.0 and acuity <= 2) else 0.38
            requires_ventilation = 1 if self.rng.random() < vent_prob else 0
        else:
            requires_ventilation = 0

        # 8. Primary Bed Allocation
        bed_request = primary_resource.request()
        yield bed_request

        t_bed_sec = self.env.now
        bed_assigned_dt = self.config.start_datetime + timedelta(seconds=int(t_bed_sec))
        bed_assigned_time_str = bed_assigned_dt.strftime(TIMESTAMP_FORMAT)

        current_bed = self._assign_physical_bed(primary_dept, stay_id)
        current_resource = primary_resource
        current_request = bed_request

        if is_direct_icu:
            self.hourly_admissions += 1
            self.total_admissions += 1

        # 9. Clinical Care Delivery & Progression
        target_los_hours = self.sampler._sample_bounded_lognormal(*los_params)
        total_care_seconds = max(1800.0, target_los_hours * 3600.0)

        try:
            if disposition == DispositionType.ED_DISCHARGE:
                # Full treatment completed within ED
                yield self.env.timeout(total_care_seconds)

            elif disposition == DispositionType.WARD:
                # Initial ED stabilization before ward transfer
                ed_prep_sec = min(7200.0, total_care_seconds * 0.25)
                yield self.env.timeout(ed_prep_sec)

                # Request Inpatient Ward Bed
                ward_request = self.ward_beds.request()
                yield ward_request

                # Transition from ED to Ward
                self._release_physical_bed(current_bed.bed_id)
                current_resource.release(current_request)

                current_resource = self.ward_beds
                current_request = ward_request
                current_bed = self._assign_physical_bed(CareUnitType.WARD, stay_id)

                self.hourly_admissions += 1
                self.total_admissions += 1

                if icu_transfer_flag == 1:
                    # Deterioration and escalation to ICU
                    ward_portion = max(1800.0, (total_care_seconds - ed_prep_sec) * 0.40)
                    yield self.env.timeout(ward_portion)

                    icu_request = self.icu_beds.request()
                    yield icu_request

                    self._release_physical_bed(current_bed.bed_id)
                    current_resource.release(current_request)

                    current_resource = self.icu_beds
                    current_request = icu_request
                    current_bed = self._assign_physical_bed(CareUnitType.ICU, stay_id)

                    icu_portion = max(1800.0, total_care_seconds - ed_prep_sec - ward_portion)
                    yield self.env.timeout(icu_portion)
                else:
                    ward_portion = max(1800.0, total_care_seconds - ed_prep_sec)
                    yield self.env.timeout(ward_portion)

            elif disposition == DispositionType.ICU and not is_direct_icu:
                # ED stabilization followed by transfer to ICU
                ed_prep_sec = min(3600.0, total_care_seconds * 0.15)
                yield self.env.timeout(ed_prep_sec)

                icu_request = self.icu_beds.request()
                yield icu_request

                self._release_physical_bed(current_bed.bed_id)
                current_resource.release(current_request)

                current_resource = self.icu_beds
                current_request = icu_request
                current_bed = self._assign_physical_bed(CareUnitType.ICU, stay_id)

                self.hourly_admissions += 1
                self.total_admissions += 1

                icu_portion = max(1800.0, total_care_seconds - ed_prep_sec)
                yield self.env.timeout(icu_portion)

            else:
                # Direct ICU Stay
                yield self.env.timeout(total_care_seconds)

            # 10. Discharge Milestone & Contract Construction
            # Pre-discharge transition
            pending_bed = current_bed.model_copy(
                update={"status": BedStatus.PENDING_DISCHARGE}
            )
            self.bed_map[current_bed.bed_id] = pending_bed

            t_dis_sec = self.env.now
            discharge_dt = self.config.start_datetime + timedelta(seconds=int(t_dis_sec))

            # Ensure minimum stay of 1800 seconds (0.5 hours) for contract compliance
            if (discharge_dt - arrival_dt).total_seconds() < 1800.0:
                discharge_dt = arrival_dt + timedelta(seconds=1800)

            discharge_time_str = discharge_dt.strftime(TIMESTAMP_FORMAT)
            actual_los_hours = round(
                (discharge_dt - arrival_dt).total_seconds() / 3600.0, 2
            )
            actual_los_hours = max(0.5, min(336.0, actual_los_hours))

            stay_contract = PatientStayContract(
                stay_id=stay_id,
                patient_id=patient_id,
                age=age,
                gender=gender,
                charlson_index=charlson_index,
                heart_rate=vitals["heart_rate"],
                sbp=vitals["sbp"],
                dbp=vitals["dbp"],
                o2_sat=vitals["o2_sat"],
                resp_rate=vitals["resp_rate"],
                temp_c=vitals["temp_c"],
                triage_acuity=acuity,
                chief_complaint=complaint,
                arrival_time=arrival_time_str,
                triage_start_time=triage_start_time_str,
                bed_assigned_time=bed_assigned_time_str,
                discharge_time=discharge_time_str,
                arrival_hour=arrival_hour_str,
                los_hours=actual_los_hours,
                disposition=disposition,
                icu_transfer_flag=icu_transfer_flag,
                initial_care_unit=initial_care_unit,
                requires_ventilation=requires_ventilation,
            )
            self.completed_stays.append(stay_contract)

            self.hourly_discharges += 1
            self.total_discharges += 1

        finally:
            self._release_physical_bed(current_bed.bed_id)
            current_resource.release(current_request)

    # -----------------------------------------------------------------------
    # Diurnal Arrival Generator Process
    # -----------------------------------------------------------------------
    def _arrival_generator(self) -> simpy.events.Process:
        """Inject patient arrival events matching the diurnal MIMIC-IV curve."""
        total_simulation_seconds = self.config.duration_hours * 3600

        while self.env.now < total_simulation_seconds:
            current_dt = self.config.start_datetime + timedelta(
                seconds=int(self.env.now)
            )
            count = self.sampler.sample_arrival_count(current_dt)

            if count > 0:
                offsets = sorted(self.rng.uniform(0.0, 3600.0) for _ in range(count))
                last_offset = 0.0

                for offset in offsets:
                    dt_wait = offset - last_offset
                    if dt_wait > 0.0:
                        yield self.env.timeout(dt_wait)
                    last_offset = offset

                    # Enforce stay_id and patient_id contract bounds
                    if (
                        self.current_stay_id <= 100_000
                        and self.current_patient_id <= 200_000
                    ):
                        self.env.process(
                            self._patient_process(
                                self.current_stay_id, self.current_patient_id
                            )
                        )
                        self.current_stay_id += 1
                        self.current_patient_id += 1

                remaining_hour = 3600.0 - last_offset
                if remaining_hour > 0.0:
                    yield self.env.timeout(remaining_hour)
            else:
                yield self.env.timeout(3600.0)

    # -----------------------------------------------------------------------
    # Hourly Telemetry Monitor Process
    # -----------------------------------------------------------------------
    def _telemetry_monitor(self) -> simpy.events.Process:
        """Tick every 3600 seconds to construct and emit validated HourlyCensusContracts."""
        for _ in range(self.config.duration_hours):
            yield self.env.timeout(3600.0)

            current_dt = self.config.start_datetime + timedelta(
                seconds=int(self.env.now)
            )
            hour = current_dt.hour
            day_of_week = current_dt.strftime("%A")
            is_weekend = 1 if current_dt.weekday() in (5, 6) else 0

            # Shift Determination & Staffing Baselines
            if 7 <= hour <= 14:
                shift_id = ShiftType.DAY
                base_nurses, base_doctors = 35, 10
            elif 15 <= hour <= 22:
                shift_id = ShiftType.EVENING
                base_nurses, base_doctors = 30, 8
            else:
                shift_id = ShiftType.NIGHT
                base_nurses, base_doctors = 20, 5

            # Controlled Operational Variation within Contract Bounds
            active_nurses = max(
                15, min(50, base_nurses + self.rng.randint(-2, 2))
            )
            active_doctors = max(
                3, min(20, base_doctors + self.rng.randint(-1, 1))
            )

            # Real-time Occupancies & Queues
            ed_occ = min(100, self.ed_beds.count)
            ward_occ = min(200, self.ward_beds.count)
            icu_occ = min(50, self.icu_beds.count)

            queue_length = (
                len(self.ed_beds.queue)
                + len(self.ward_beds.queue)
                + len(self.icu_beds.queue)
            )
            self.peak_queue_length = max(self.peak_queue_length, queue_length)

            # Rolling 4-hour Incoming Arrival Expectation
            cfg = self.config.distribution_config or DistributionConfig()
            forecast_4h = 0
            for offset_h in range(1, 5):
                f_dt = current_dt + timedelta(hours=offset_h)
                lam = cfg.hourly_arrival_rates[f_dt.hour]
                if f_dt.weekday() in (5, 6):
                    lam *= cfg.weekend_rate_multiplier
                forecast_4h += int(round(lam))

            census = HourlyCensusContract(
                timestamp=current_dt.strftime(HOURLY_FORMAT),
                hour=hour,
                day_of_week=day_of_week,
                is_weekend=is_weekend,
                shift_id=shift_id,
                ed_occupancy=ed_occ,
                ward_occupancy=ward_occ,
                icu_occupancy=icu_occ,
                active_nurses=active_nurses,
                active_doctors=active_doctors,
                arrivals_count=self.hourly_arrivals,
                admissions_count=self.hourly_admissions,
                discharges_count=self.hourly_discharges,
                patients_in_queue=queue_length,
                incoming_arrivals_next_4h=forecast_4h,
            )
            self.hourly_census_records.append(census)

            # Reset delta counters for the next elapsed hour
            self.hourly_arrivals = 0
            self.hourly_admissions = 0
            self.hourly_discharges = 0

    # -----------------------------------------------------------------------
    # Main Execution Interface
    # -----------------------------------------------------------------------
    def run(self) -> SimulationResults:
        """Execute the discrete-event simulation and return validated telemetry."""
        # Register core SimPy daemon processes
        self.env.process(self._arrival_generator())
        monitor = self.env.process(self._telemetry_monitor())

        # Execute simulation clock until the telemetry monitor records all hours
        self.env.run(until=monitor)

        # Compute Operational Metrics
        total_ed_capacity = self.config.ed_capacity * self.config.duration_hours
        total_ward_capacity = self.config.ward_capacity * self.config.duration_hours
        total_icu_capacity = self.config.icu_capacity * self.config.duration_hours

        avg_ed_occ = (
            sum(c.ed_occupancy for c in self.hourly_census_records)
            / len(self.hourly_census_records)
            if self.hourly_census_records
            else 0.0
        )
        avg_ward_occ = (
            sum(c.ward_occupancy for c in self.hourly_census_records)
            / len(self.hourly_census_records)
            if self.hourly_census_records
            else 0.0
        )
        avg_icu_occ = (
            sum(c.icu_occupancy for c in self.hourly_census_records)
            / len(self.hourly_census_records)
            if self.hourly_census_records
            else 0.0
        )

        ed_util = (avg_ed_occ / self.config.ed_capacity) * 100.0
        ward_util = (avg_ward_occ / self.config.ward_capacity) * 100.0
        icu_util = (avg_icu_occ / self.config.icu_capacity) * 100.0

        avg_los = (
            sum(s.los_hours for s in self.completed_stays) / len(self.completed_stays)
            if self.completed_stays
            else 0.0
        )

        metrics: dict[str, Any] = {
            "total_arrivals": self.total_arrivals,
            "total_admissions": self.total_admissions,
            "total_discharges": self.total_discharges,
            "completed_stays_count": len(self.completed_stays),
            "peak_queue_length": self.peak_queue_length,
            "average_los_hours": round(avg_los, 2),
            "ed_utilization_pct": round(ed_util, 2),
            "ward_utilization_pct": round(ward_util, 2),
            "icu_utilization_pct": round(icu_util, 2),
        }

        return SimulationResults(
                    stays=[s.model_dump() for s in self.completed_stays],
                    hourly_census=[c.model_dump() for c in self.hourly_census_records],
                    bed_topology=[b.model_dump() for b in self.bed_map.values()],
                    metrics=metrics,
                )


# ---------------------------------------------------------------------------
# Self-Testing Block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Initializing 24-hour Digital Twin simulation (seed=42)...")
    config = SimulationConfig(
        start_datetime=datetime(2026, 9, 23, 0, 0, 0),
        duration_hours=24,
        ed_capacity=50,
        ward_capacity=150,
        icu_capacity=30,
        seed=42,
    )
    engine = HospitalSimulationEngine(config)
    results = engine.run()

    print("\n--- Simulation Execution Telemetry ---")
    print(f"Total Arrivals:        {results.metrics['total_arrivals']}")
    print(f"Total Admissions:      {results.metrics['total_admissions']}")
    print(f"Total Discharges:      {results.metrics['total_discharges']}")
    print(f"Completed Stays:       {results.metrics['completed_stays_count']}")
    print(f"Peak Queue Length:     {results.metrics['peak_queue_length']}")
    print(f"Avg LoS (Hours):       {results.metrics['average_los_hours']}")
    print(f"ED Utilization:        {results.metrics['ed_utilization_pct']}%")
    print(f"Ward Utilization:      {results.metrics['ward_utilization_pct']}%")
    print(f"ICU Utilization:       {results.metrics['icu_utilization_pct']}%")

    # 1. Assert Hourly Census Contract Invariants
    assert len(results.hourly_census) == 24, (
        f"Expected 24 hourly census records, got {len(results.hourly_census)}"
    )
    for census in results.hourly_census:
        assert isinstance(census, HourlyCensusContract)
        assert 0 <= census.hour <= 23
        assert 0 <= census.ed_occupancy <= 100
        assert 0 <= census.ward_occupancy <= 200
        assert 0 <= census.icu_occupancy <= 50

    print("\nPASS: All 24 HourlyCensusContract instances strictly validated.")

    # 2. Assert Patient Stay Contract Invariants
    assert len(results.stays) > 0, "No completed patient stays were recorded."
    for stay in results.stays:
        assert isinstance(stay, PatientStayContract)

        # Pulse pressure invariant
        assert stay.sbp - stay.dbp >= 15.0, (
            f"Pulse pressure violation on stay {stay.stay_id}: "
            f"sbp={stay.sbp}, dbp={stay.dbp}"
        )

        # Monotonicity invariant
        dt_arr = datetime.strptime(stay.arrival_time, TIMESTAMP_FORMAT)
        dt_trg = datetime.strptime(stay.triage_start_time, TIMESTAMP_FORMAT)
        dt_bed = datetime.strptime(stay.bed_assigned_time, TIMESTAMP_FORMAT)
        dt_dis = datetime.strptime(stay.discharge_time, TIMESTAMP_FORMAT)

        assert dt_arr <= dt_trg <= dt_bed <= dt_dis, (
            f"Temporal monotonicity violation on stay {stay.stay_id}: "
            f"{stay.arrival_time} <= {stay.triage_start_time} <= "
            f"{stay.bed_assigned_time} <= {stay.discharge_time}"
        )

        # Relational key flooring
        assert stay.arrival_hour == dt_arr.strftime(HOURLY_FORMAT), (
            f"Arrival hour mismatch: {stay.arrival_hour} vs {dt_arr}"
        )

    print(f"PASS: All {len(results.stays)} PatientStayContract instances strictly validated.")

    # 3. Assert Bed Topology Invariants
    total_beds = config.ed_capacity + config.ward_capacity + config.icu_capacity
    assert len(results.bed_topology) == total_beds, (
        f"Expected {total_beds} beds, got {len(results.bed_topology)}"
    )
    for bed in results.bed_topology:
        assert isinstance(bed, BedTopologyContract)
        if bed.status == BedStatus.OCCUPIED:
            assert bed.current_stay_id is not None
        elif bed.status in {BedStatus.AWAITING_CLEANING, BedStatus.OUT_OF_SERVICE}:
            assert bed.current_stay_id is None

    print(f"PASS: All {total_beds} BedTopologyContract instances validated in final state.")
    print("Simulation engine pipeline executed successfully.")