# backend/src/ml/model_pipeline.py
"""Predictive model pipeline for the Digital Twin hospital simulation platform.

Orchestrates machine learning simulation logic and predictive routing by ingesting
stochastic distributions from MIMICDistributionSampler and resource state logic from
HospitalSimulationEngine. Produces validated Pydantic v2 prediction models defined
in prediction_contracts.py while enforcing biological and temporal invariants.
"""

from datetime import datetime, timedelta
from enum import Enum
import math
import random
from typing import Any, ClassVar, Final, Sequence

from pydantic import ValidationError

from src.schemas.contracts import (
    CareUnitType,
    DispositionType,
    ShiftType,
    TIMESTAMP_FORMAT,
    EXPECTED_TS_LENGTH,
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
from src.data_generator.mimic_distributions import MIMICDistributionSampler
from src.simulation.engine import HospitalSimulationEngine

# ---------------------------------------------------------------------------
# External Simulation & Distribution Imports with Fallback Implementations
# ---------------------------------------------------------------------------
try:
    from backend.src.sim.mimic_distributions import MIMICDistributionSampler
except ImportError:
    try:
        from backend.src.ml.mimic_distributions import MIMICDistributionSampler
    except ImportError:
        try:
            from .mimic_distributions import MIMICDistributionSampler  # type: ignore
        except ImportError:
            MIMICDistributionSampler = None  # type: ignore

try:
    from backend.src.sim.engine import HospitalSimulationEngine
except ImportError:
    try:
        from backend.src.ml.engine import HospitalSimulationEngine
    except ImportError:
        try:
            from .engine import HospitalSimulationEngine  # type: ignore
        except ImportError:
            HospitalSimulationEngine = None  # type: ignore


# ---------------------------------------------------------------------------
# Empirical MIMIC-IV Distribution Reference Values
# ---------------------------------------------------------------------------
DEFAULT_ED_CAPACITY: Final[int] = 50
DEFAULT_WARD_CAPACITY: Final[int] = 150
DEFAULT_ICU_CAPACITY: Final[int] = 25

CHIEF_COMPLAINT_WEIGHTS: Final[dict[str, float]] = {
    "Chest Pain": 0.18,
    "Shortness of Breath": 0.16,
    "Abdominal Pain": 0.15,
    "Headache": 0.09,
    "Fever": 0.08,
    "Cough": 0.08,
    "Dizziness": 0.07,
    "Back Pain": 0.07,
    "Nausea/Vomiting": 0.06,
    "Weakness": 0.06,
}

ESI_DISTRIBUTION_WEIGHTS: Final[dict[int, float]] = {
    1: 0.02,
    2: 0.20,
    3: 0.52,
    4: 0.22,
    5: 0.04,
}

DISPOSITION_TRANSITION_MATRIX: Final[dict[int, dict[DispositionType, float]]] = {
    1: {DispositionType.ED_DISCHARGE: 0.05, DispositionType.WARD: 0.35, DispositionType.ICU: 0.60},
    2: {DispositionType.ED_DISCHARGE: 0.25, DispositionType.WARD: 0.55, DispositionType.ICU: 0.20},
    3: {DispositionType.ED_DISCHARGE: 0.65, DispositionType.WARD: 0.30, DispositionType.ICU: 0.05},
    4: {DispositionType.ED_DISCHARGE: 0.88, DispositionType.WARD: 0.11, DispositionType.ICU: 0.01},
    5: {DispositionType.ED_DISCHARGE: 0.98, DispositionType.WARD: 0.02, DispositionType.ICU: 0.00},
}

DIURNAL_HOURLY_WEIGHTS: Final[dict[int, float]] = {
    0: 0.50, 1: 0.40, 2: 0.32, 3: 0.26, 4: 0.28, 5: 0.38,
    6: 0.60, 7: 0.88, 8: 1.15, 9: 1.35, 10: 1.45, 11: 1.42,
    12: 1.38, 13: 1.32, 14: 1.28, 15: 1.22, 16: 1.18, 17: 1.12,
    18: 1.05, 19: 0.96, 20: 0.86, 21: 0.76, 22: 0.66, 23: 0.56,
}


# ---------------------------------------------------------------------------
# Fallback Sampler Implementation
# ---------------------------------------------------------------------------
class FallbackMIMICDistributionSampler:
    """Empirical parameter sampler modeling MIMIC-IV clinical emergency stays."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def sample_age(self) -> int:
        """Sample age truncated between 18 and 95."""
        val = int(round(self.rng.gauss(58.5, 17.5)))
        return max(18, min(95, val))

    def sample_gender(self) -> str:
        """Sample gender representation."""
        return "Female" if self.rng.random() < 0.51 else "Male"

    def sample_charlson(self) -> int:
        """Sample Charlson Comorbidity Index score."""
        weights = [0.35, 0.20, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.005]
        return self.rng.choices(range(11), weights=weights, k=1)[0]

    def sample_triage_acuity(self) -> int:
        """Sample Emergency Severity Index triage acuity."""
        levels = list(ESI_DISTRIBUTION_WEIGHTS.keys())
        weights = [ESI_DISTRIBUTION_WEIGHTS[lvl] for lvl in levels]
        return self.rng.choices(levels, weights=weights, k=1)[0]

    def sample_chief_complaint(self) -> str:
        """Sample one of 10 primary chief complaints using established empirical weights."""
        complaints = list(CHIEF_COMPLAINT_WEIGHTS.keys())
        weights = [CHIEF_COMPLAINT_WEIGHTS[c] for c in complaints]
        return self.rng.choices(complaints, weights=weights, k=1)[0]

    def sample_vitals(self, acuity: int, age: int, chief_complaint: str) -> dict[str, float]:
        """Sample physiologically coupled vital signs."""
        hr_means = {1: 106.0, 2: 92.0, 3: 82.0, 4: 76.0, 5: 72.0}
        hr = max(30.0, min(220.0, self.rng.gauss(hr_means[acuity], 14.0)))

        rr_means = {1: 26.0, 2: 21.0, 3: 18.0, 4: 16.0, 5: 15.0}
        rr = max(6.0, min(60.0, self.rng.gauss(rr_means[acuity], 3.0)))

        o2_means = {1: 91.5, 2: 94.5, 3: 97.2, 4: 98.4, 5: 99.0}
        o2 = max(60.0, min(100.0, self.rng.gauss(o2_means[acuity], 2.5)))

        if chief_complaint == "Fever":
            temp_mean = 38.6
        elif acuity <= 2:
            temp_mean = 37.2
        else:
            temp_mean = 36.8
        temp = max(33.0, min(43.0, self.rng.gauss(temp_mean, 0.5)))

        sbp_mean = 126.0 + (age - 50) * 0.30 + (5 - acuity) * 3.5
        sbp = max(60.0, min(240.0, self.rng.gauss(sbp_mean, 15.0)))

        dbp_mean = 78.0 + (age - 50) * 0.10 + (5 - acuity) * 1.5
        dbp = max(30.0, min(140.0, self.rng.gauss(dbp_mean, 10.0)))

        return {
            "heart_rate": round(hr, 1),
            "resp_rate": round(rr, 1),
            "o2_sat": round(o2, 1),
            "temp_c": round(temp, 1),
            "sbp": round(sbp, 1),
            "dbp": round(dbp, 1),
        }


# ---------------------------------------------------------------------------
# Fallback Simulation Engine State Logic
# ---------------------------------------------------------------------------
class FallbackHospitalSimulationEngine:
    """State logic tracker for bed topologies and queue capacities."""

    def __init__(
        self,
        ed_capacity: int = DEFAULT_ED_CAPACITY,
        ward_capacity: int = DEFAULT_WARD_CAPACITY,
        icu_capacity: int = DEFAULT_ICU_CAPACITY,
    ) -> None:
        self.ed_capacity = ed_capacity
        self.ward_capacity = ward_capacity
        self.icu_capacity = icu_capacity
        self.ed_occupancy = 0
        self.ward_occupancy = 0
        self.icu_occupancy = 0
        self.ed_queue = 0
        self.ward_queue = 0
        self.icu_queue = 0

    def get_snapshot(self) -> dict[str, int]:
        """Retrieve active census counts and queue state."""
        return {
            "ed_occupancy": self.ed_occupancy,
            "ward_occupancy": self.ward_occupancy,
            "icu_occupancy": self.icu_occupancy,
            "ed_queue": self.ed_queue,
            "ward_queue": self.ward_queue,
            "icu_queue": self.icu_queue,
            "total_queue": self.ed_queue + self.ward_queue + self.icu_queue,
        }


# ---------------------------------------------------------------------------
# Main Model Pipeline Orchestrator
# ---------------------------------------------------------------------------
class PredictiveModelPipeline:
    """Predictive routing and discrete-event simulation pipeline.

    Ingests empirical distributions and hospital topology state to generate
    Pydantic v2 prediction models validating biological, shift, and temporal invariants.
    """

    def __init__(
        self,
        sampler: Any = None,
        engine: Any = None,
        seed: int | None = None,
        ed_capacity: int = DEFAULT_ED_CAPACITY,
        ward_capacity: int = DEFAULT_WARD_CAPACITY,
        icu_capacity: int = DEFAULT_ICU_CAPACITY,
    ) -> None:
        self.rng = random.Random(seed)
        self.ed_capacity = ed_capacity
        self.ward_capacity = ward_capacity
        self.icu_capacity = icu_capacity

        # Configure stochastic sampler
        if sampler is not None:
            self.sampler = sampler
        elif MIMICDistributionSampler is not None:
            try:
                self.sampler = MIMICDistributionSampler(seed=seed)
            except Exception:
                self.sampler = MIMICDistributionSampler()
        else:
            self.sampler = FallbackMIMICDistributionSampler(seed=seed)

        # Configure resource tracking engine
        if engine is not None:
            self.engine = engine
        elif HospitalSimulationEngine is not None:
            try:
                self.engine = HospitalSimulationEngine(
                    ed_capacity=ed_capacity,
                    ward_capacity=ward_capacity,
                    icu_capacity=icu_capacity,
                )
            except Exception:
                self.engine = FallbackHospitalSimulationEngine(
                    ed_capacity=ed_capacity,
                    ward_capacity=ward_capacity,
                    icu_capacity=icu_capacity,
                )
        else:
            self.engine = FallbackHospitalSimulationEngine(
                ed_capacity=ed_capacity,
                ward_capacity=ward_capacity,
                icu_capacity=icu_capacity,
            )

    # -----------------------------------------------------------------------
    # 1. Demographic & Physiological Modeler
    # -----------------------------------------------------------------------
    def sample_patient_clinical(
        self,
        patient_id: int | None = None,
        stay_id: int | None = None,
        base_acuity: int | None = None,
        chief_complaint: str | None = None,
    ) -> PatientClinicalPrediction:
        """Synthesize demographic profiles, triage acuity, and coupled vitals.

        Strictly enforces the biological invariant that pulse pressure (SBP - DBP)
        is >= 15.0 mmHg before instantiating PatientClinicalPrediction.
        """
        # Demographic sampling
        if hasattr(self.sampler, "sample_age"):
            age = int(self.sampler.sample_age())
        else:
            age = max(18, min(95, int(round(self.rng.gauss(58.5, 17.5)))))

        if hasattr(self.sampler, "sample_gender"):
            gender = str(self.sampler.sample_gender())
        else:
            gender = "Female" if self.rng.random() < 0.51 else "Male"

        if hasattr(self.sampler, "sample_charlson"):
            charlson_index = int(self.sampler.sample_charlson())
        else:
            weights = [0.35, 0.20, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.005]
            charlson_index = self.rng.choices(range(11), weights=weights, k=1)[0]

        # Triage Acuity
        if base_acuity is not None and 1 <= base_acuity <= 5:
            triage_acuity = base_acuity
        elif hasattr(self.sampler, "sample_triage_acuity"):
            triage_acuity = int(self.sampler.sample_triage_acuity())
        else:
            levels = list(ESI_DISTRIBUTION_WEIGHTS.keys())
            weights = [ESI_DISTRIBUTION_WEIGHTS[lvl] for lvl in levels]
            triage_acuity = self.rng.choices(levels, weights=weights, k=1)[0]

        # Chief Complaint (One of 10 primary complaints)
        if chief_complaint is not None and chief_complaint in CHIEF_COMPLAINT_WEIGHTS:
            selected_complaint = chief_complaint
        elif hasattr(self.sampler, "sample_chief_complaint"):
            selected_complaint = str(self.sampler.sample_chief_complaint())
        else:
            complaints = list(CHIEF_COMPLAINT_WEIGHTS.keys())
            weights = [CHIEF_COMPLAINT_WEIGHTS[c] for c in complaints]
            selected_complaint = self.rng.choices(complaints, weights=weights, k=1)[0]

        # Sample coupled vitals
        if hasattr(self.sampler, "sample_vitals"):
            vitals = self.sampler.sample_vitals(
                acuity=triage_acuity,
                age=age,
                chief_complaint=selected_complaint,
            )
            hr = float(vitals.get("heart_rate", 80.0))
            rr = float(vitals.get("resp_rate", 18.0))
            o2 = float(vitals.get("o2_sat", 98.0))
            temp = float(vitals.get("temp_c", 37.0))
            sbp = float(vitals.get("sbp", 120.0))
            dbp = float(vitals.get("dbp", 80.0))
        else:
            fallback = FallbackMIMICDistributionSampler(seed=self.rng.randint(0, 1_000_000))
            vitals = fallback.sample_vitals(triage_acuity, age, selected_complaint)
            hr = vitals["heart_rate"]
            rr = vitals["resp_rate"]
            o2 = vitals["o2_sat"]
            temp = vitals["temp_c"]
            sbp = vitals["sbp"]
            dbp = vitals["dbp"]

        # Clamp vitals to domain contract boundaries
        hr = max(30.0, min(220.0, hr))
        rr = max(6.0, min(60.0, rr))
        o2 = max(60.0, min(100.0, o2))
        temp = max(33.0, min(43.0, temp))

        # Strict Biological Invariant Enforcement: Pulse Pressure (SBP - DBP) >= 15.0 mmHg
        dbp = max(30.0, min(140.0, dbp))
        min_required_sbp = max(60.0, dbp + MIN_PULSE_PRESSURE)
        if sbp < min_required_sbp:
            sbp = min_required_sbp
        if sbp > 240.0:
            sbp = 240.0
            dbp = min(dbp, sbp - MIN_PULSE_PRESSURE)
        if sbp - dbp < MIN_PULSE_PRESSURE:
            dbp = max(30.0, sbp - MIN_PULSE_PRESSURE)

        # Default identifiers if omitted
        assigned_patient_id = patient_id if patient_id is not None else self.rng.randint(100_001, 200_000)
        assigned_stay_id = stay_id if stay_id is not None else self.rng.randint(1, 100_000)

        return PatientClinicalPrediction(
            patient_id=assigned_patient_id,
            stay_id=assigned_stay_id,
            age=age,
            gender=gender,  # type: ignore[arg-type]
            charlson_index=charlson_index,
            chief_complaint=selected_complaint,
            triage_acuity=triage_acuity,
            heart_rate=round(hr, 1),
            resp_rate=round(rr, 1),
            temp_c=round(temp, 1),
            o2_sat=round(o2, 1),
            sbp=round(sbp, 1),
            dbp=round(dbp, 1),
        )

    def batch_sample_patient_clinicals(
        self,
        count: int,
        base_acuity: int | None = None,
    ) -> list[PatientClinicalPrediction]:
        """Batch-generate multiple validated PatientClinicalPrediction models."""
        results: list[PatientClinicalPrediction] = []
        for idx in range(count):
            p_id = 100_001 + (idx % 99_999)
            s_id = 1 + (idx % 99_999)
            pred = self.sample_patient_clinical(
                patient_id=p_id,
                stay_id=s_id,
                base_acuity=base_acuity,
            )
            results.append(pred)
        return results

    # -----------------------------------------------------------------------
    # 2. Trajectory & Patient Flow Forecaster
    # -----------------------------------------------------------------------
    def predict_clinical_trajectory(
        self,
        patient: PatientClinicalPrediction,
        arrival_time: datetime | str | None = None,
    ) -> ClinicalTrajectoryPrediction:
        """Predict patient flow milestones, disposition endpoint, and escalation risks.

        Strictly enforces temporal monotonicity:
        arrival_time <= triage_start_time <= bed_assigned_time <= discharge_time.
        """
        # Determine arrival datetime reference
        if arrival_time is None:
            dt_arr = datetime.now().replace(microsecond=0)
        elif isinstance(arrival_time, str):
            dt_arr = _parse_strict_timestamp(arrival_time, "arrival_time")
        else:
            dt_arr = arrival_time.replace(microsecond=0)

        # 1. ESI-conditional disposition weights
        trans_weights = DISPOSITION_TRANSITION_MATRIX.get(
            patient.triage_acuity,
            DISPOSITION_TRANSITION_MATRIX[3],
        )
        endpoints = [DispositionType.ED_DISCHARGE, DispositionType.WARD, DispositionType.ICU]
        probabilities = [trans_weights[ep] for ep in endpoints]
        disposition = self.rng.choices(endpoints, weights=probabilities, k=1)[0]

        # 2. Mechanical ventilation requirements
        if disposition == DispositionType.ICU:
            vent_prob = 0.35 if patient.triage_acuity == 1 else 0.18
        elif disposition == DispositionType.WARD:
            vent_prob = 0.05 if patient.triage_acuity <= 2 else 0.01
        else:
            vent_prob = 0.005 if patient.triage_acuity == 1 else 0.0
        requires_ventilation = 1 if self.rng.random() < vent_prob else 0

        # 3. Ward-to-ICU transfer probability
        if disposition == DispositionType.WARD:
            base_risk = 0.02 + 0.03 * (5 - patient.triage_acuity) + 0.015 * patient.charlson_index
            icu_transfer_probability = min(0.40, max(0.01, round(base_risk, 3)))
            icu_transfer_flag = 1 if self.rng.random() < icu_transfer_probability else 0
        elif disposition == DispositionType.ICU:
            icu_transfer_probability = 1.0
            icu_transfer_flag = 1
        else:
            icu_transfer_probability = 0.005 if patient.triage_acuity == 1 else 0.0
            icu_transfer_flag = 0

        # 4. Sequential milestone timestamps based on LogNormal parameters
        # Triage delay: median ~12 minutes
        triage_minutes = max(3.0, min(45.0, self.rng.lognormvariate(math.log(12.0), 0.35)))
        dt_trg = dt_arr + timedelta(seconds=int(round(triage_minutes * 60.0)))

        # Bed assignment delay: median ~30 minutes post-triage
        bed_minutes = max(5.0, min(120.0, self.rng.lognormvariate(math.log(30.0), 0.40)))
        dt_bed = dt_trg + timedelta(seconds=int(round(bed_minutes * 60.0)))

        # Unit-dependent bed stay duration
        if disposition == DispositionType.ED_DISCHARGE:
            stay_hours = max(0.5, min(24.0, self.rng.lognormvariate(math.log(3.5), 0.45)))
        elif disposition == DispositionType.WARD:
            stay_hours = max(8.0, min(240.0, self.rng.lognormvariate(math.log(48.0), 0.50)))
        else:  # ICU
            stay_hours = max(12.0, min(336.0, self.rng.lognormvariate(math.log(72.0), 0.55)))

        dt_dis = dt_bed + timedelta(seconds=int(round(stay_hours * 3600.0)))

        # Temporal monotonicity assertion & safeguarding
        if not (dt_arr <= dt_trg <= dt_bed <= dt_dis):
            dt_trg = max(dt_arr, dt_trg)
            dt_bed = max(dt_trg, dt_bed)
            dt_dis = max(dt_bed + timedelta(minutes=30), dt_dis)

        total_los_hours = round((dt_dis - dt_arr).total_seconds() / 3600.0, 2)

        return ClinicalTrajectoryPrediction(
            stay_id=patient.stay_id,
            patient_id=patient.patient_id,
            predicted_disposition=disposition,
            requires_ventilation=requires_ventilation,
            icu_transfer_probability=round(icu_transfer_probability, 3),
            icu_transfer_flag=icu_transfer_flag,
            arrival_time=dt_arr.strftime(TIMESTAMP_FORMAT),
            triage_start_time=dt_trg.strftime(TIMESTAMP_FORMAT),
            bed_assigned_time=dt_bed.strftime(TIMESTAMP_FORMAT),
            discharge_time=dt_dis.strftime(TIMESTAMP_FORMAT),
            predicted_los_hours=total_los_hours,
        )

    def batch_predict_trajectories(
        self,
        patients: Sequence[PatientClinicalPrediction],
        base_arrival_time: datetime | str | None = None,
        arrival_interval_minutes: float = 6.0,
    ) -> list[ClinicalTrajectoryPrediction]:
        """Batch-forecast sequential trajectories staggered by arrival spacing."""
        if base_arrival_time is None:
            current_dt = datetime.now().replace(microsecond=0)
        elif isinstance(base_arrival_time, str):
            current_dt = _parse_strict_timestamp(base_arrival_time, "base_arrival_time")
        else:
            current_dt = base_arrival_time.replace(microsecond=0)

        trajectories: list[ClinicalTrajectoryPrediction] = []
        for idx, p in enumerate(patients):
            staggered_arrival = current_dt + timedelta(minutes=idx * arrival_interval_minutes)
            traj = self.predict_clinical_trajectory(p, arrival_time=staggered_arrival)
            trajectories.append(traj)
        return trajectories

    # -----------------------------------------------------------------------
    # 3. Hourly Staffing & Diurnal Census Predictor
    # -----------------------------------------------------------------------
    def predict_hourly_staffing(
        self,
        start_time: datetime | str,
        horizon_hours: int = 24,
        base_arrival_rate: float = 12.0,
        base_admission_rate: float = 3.6,
        base_discharge_rate: float = 3.4,
        weekend_rate_multiplier: float = 0.92,
    ) -> list[HourlyStaffingPrediction]:
        """Project shift-level operational requirements, flow rates, and prescriptive staffing.

        Packages hourly metrics adhering to shift boundaries, calendar day alignment,
        and rolling 4-hour prescriptive arrival forecasts.
        """
        if isinstance(start_time, str):
            start_dt = _parse_strict_timestamp(start_time, "start_time")
        else:
            start_dt = start_time
        # Floor snapshot start to top of hour
        start_dt = start_dt.replace(minute=0, second=0, microsecond=0)

        # Precompute projected arrival rates for rolling 4-hour calculation
        total_slots = horizon_hours + 4
        hourly_projected_arrivals: list[float] = []

        for offset in range(total_slots):
            slot_dt = start_dt + timedelta(hours=offset)
            hr = slot_dt.hour
            is_wk = 1 if slot_dt.weekday() in (5, 6) else 0
            wk_multiplier = weekend_rate_multiplier if is_wk else 1.0
            diurnal_factor = DIURNAL_HOURLY_WEIGHTS.get(hr, 1.0)
            arr = round(base_arrival_rate * diurnal_factor * wk_multiplier, 2)
            hourly_projected_arrivals.append(arr)

        predictions: list[HourlyStaffingPrediction] = []

        for h in range(horizon_hours):
            current_dt = start_dt + timedelta(hours=h)
            hr = current_dt.hour
            day_name = current_dt.strftime("%A")
            is_wknd = 1 if current_dt.weekday() in (5, 6) else 0
            wk_multiplier = weekend_rate_multiplier if is_wknd else 1.0
            diurnal_factor = DIURNAL_HOURLY_WEIGHTS.get(hr, 1.0)

            # Flow rates
            arrivals = hourly_projected_arrivals[h]
            admissions = round(base_admission_rate * diurnal_factor * wk_multiplier, 2)

            # Discharges peak between 11:00 and 17:00
            discharge_factor = 1.55 if 11 <= hr <= 17 else (0.35 if hr >= 23 or hr <= 6 else 0.95)
            discharges = round(base_discharge_rate * discharge_factor * (0.85 if is_wknd else 1.0), 2)

            # Shift type classification matching contracts.py
            if 7 <= hr <= 14:
                shift = ShiftType.DAY
                base_nurses = 35
                base_doctors = 10
            elif 15 <= hr <= 22:
                shift = ShiftType.EVENING
                base_nurses = 30
                base_doctors = 8
            else:
                shift = ShiftType.NIGHT
                base_nurses = 20
                base_doctors = 5

            # Prescriptive staffing demand adjustment
            arrival_ratio = arrivals / max(base_arrival_rate, 1.0)
            staff_multiplier = 0.80 + 0.20 * arrival_ratio
            req_nurses = int(round(base_nurses * staff_multiplier))
            req_doctors = int(round(base_doctors * staff_multiplier))

            # Bound headcounts to domain contract boundaries
            active_nurses = max(15, min(50, req_nurses))
            active_doctors = max(3, min(20, req_doctors))

            # Rolling 4-hour prescriptive arrival forecast
            rolling_4h_sum = sum(hourly_projected_arrivals[h + 1 : h + 5])

            prediction = HourlyStaffingPrediction(
                timestamp=current_dt.strftime("%Y-%m-%d %H:00:00"),
                hour=hr,
                day_of_week=day_name,
                is_weekend=is_wknd,
                shift_id=shift,
                projected_arrivals=arrivals,
                projected_admissions=admissions,
                projected_discharges=discharges,
                predicted_active_nurses=active_nurses,
                predicted_active_physicians=active_doctors,
                incoming_arrivals_next_4h=round(rolling_4h_sum, 2),
            )
            predictions.append(prediction)

        return predictions

    # -----------------------------------------------------------------------
    # 4. Resource Utilization & Bottleneck Analyzer
    # -----------------------------------------------------------------------
    def predict_resource_utilization(
        self,
        trajectories: Sequence[ClinicalTrajectoryPrediction] | None = None,
        forecast_timestamp: datetime | str | None = None,
        active_ed_patients: int | None = None,
        active_ward_patients: int | None = None,
        active_icu_patients: int | None = None,
        waiting_queue_count: int | None = None,
    ) -> ResourceUtilizationForecast:
        """Forecast physical asset bottlenecks and bed utilization percentages.

        Mirrors engine telemetry monitor logic across ED, Ward, and ICU.
        """
        # Timestamp formatting
        if forecast_timestamp is None:
            ts_str = datetime.now().strftime(TIMESTAMP_FORMAT)
        elif isinstance(forecast_timestamp, str):
            ts_str = _parse_strict_timestamp(forecast_timestamp, "forecast_timestamp").strftime(
                TIMESTAMP_FORMAT
            )
        else:
            ts_str = forecast_timestamp.strftime(TIMESTAMP_FORMAT)

        # Length of stay calculations
        if trajectories and len(trajectories) > 0:
            los_values = [t.predicted_los_hours for t in trajectories if t.predicted_los_hours is not None]
            avg_los = round(sum(los_values) / len(los_values), 2) if los_values else 24.5
        else:
            # Empirical disposition-weighted average LoS
            avg_los = round(0.55 * 3.8 + 0.35 * 48.0 + 0.10 * 72.0, 2)

        # Pull occupancy from engine state or parameters
        engine_snapshot = {}
        if hasattr(self.engine, "get_snapshot"):
            engine_snapshot = self.engine.get_snapshot()

        ed_count = (
            active_ed_patients
            if active_ed_patients is not None
            else engine_snapshot.get("ed_occupancy", int(self.ed_capacity * 0.72))
        )
        ward_count = (
            active_ward_patients
            if active_ward_patients is not None
            else engine_snapshot.get("ward_occupancy", int(self.ward_capacity * 0.84))
        )
        icu_count = (
            active_icu_patients
            if active_icu_patients is not None
            else engine_snapshot.get("icu_occupancy", int(self.icu_capacity * 0.76))
        )

        # Unit bed utilization percentages
        ed_util_pct = min(100.0, max(0.0, round((ed_count / max(self.ed_capacity, 1)) * 100.0, 2)))
        ward_util_pct = min(100.0, max(0.0, round((ward_count / max(self.ward_capacity, 1)) * 100.0, 2)))
        icu_util_pct = min(100.0, max(0.0, round((icu_count / max(self.icu_capacity, 1)) * 100.0, 2)))

        # Forecast peak queue lengths for waiting patients
        if waiting_queue_count is not None:
            peak_queue = max(0, waiting_queue_count)
        elif "total_queue" in engine_snapshot:
            peak_queue = max(0, engine_snapshot["total_queue"])
        else:
            # Simulate combined bottleneck queues (ED waiting + Ward transfer delay + ICU boarding)
            ed_wait_q = int(round(max(0.0, ed_count - self.ed_capacity * 0.85)))
            ward_board_q = int(round(max(0.0, ward_count - self.ward_capacity * 0.90)))
            icu_board_q = int(round(max(0.0, icu_count - self.icu_capacity * 0.88)))
            peak_queue = ed_wait_q + ward_board_q + icu_board_q

        return ResourceUtilizationForecast(
            forecast_timestamp=ts_str,
            peak_queue_length=peak_queue,
            predicted_avg_los_hours=avg_los,
            ed_utilization_pct=ed_util_pct,
            ward_utilization_pct=ward_util_pct,
            icu_utilization_pct=icu_util_pct,
        )

    # -----------------------------------------------------------------------
    # End-to-End Orchestrator
    # -----------------------------------------------------------------------
    def run_pipeline(
        self,
        cohort_size: int = 10,
        start_time: datetime | str | None = None,
        forecast_horizon_hours: int = 24,
    ) -> dict[str, Any]:
        """Execute an end-to-end predictive simulation run.

        Returns validated Pydantic v2 prediction objects spanning:
        - patient_profiles: list[PatientClinicalPrediction]
        - trajectories: list[ClinicalTrajectoryPrediction]
        - staffing_predictions: list[HourlyStaffingPrediction]
        - resource_forecast: ResourceUtilizationForecast
        """
        if start_time is None:
            base_dt = datetime.now().replace(microsecond=0)
        elif isinstance(start_time, str):
            base_dt = _parse_strict_timestamp(start_time, "start_time")
        else:
            base_dt = start_time.replace(microsecond=0)

        # 1. Synthesize Demographic & Physiological Patient States
        patient_profiles = self.batch_sample_patient_clinicals(count=cohort_size)

        # 2. Forecast Patient Movement and Sequential Milestones
        trajectories = self.batch_predict_trajectories(
            patients=patient_profiles,
            base_arrival_time=base_dt,
            arrival_interval_minutes=8.0,
        )

        # 3. Predict Shift Staffing and Diurnal Census
        staffing_predictions = self.predict_hourly_staffing(
            start_time=base_dt,
            horizon_hours=forecast_horizon_hours,
        )

        # 4. Analyze Resource Utilization and Unit Bottlenecks
        resource_forecast = self.predict_resource_utilization(
            trajectories=trajectories,
            forecast_timestamp=base_dt,
        )

        return {
            "patient_profiles": patient_profiles,
            "trajectories": trajectories,
            "staffing_predictions": staffing_predictions,
            "resource_forecast": resource_forecast,
        }


__all__ = [
    "CHIEF_COMPLAINT_WEIGHTS",
    "DEFAULT_ED_CAPACITY",
    "DEFAULT_ICU_CAPACITY",
    "DEFAULT_WARD_CAPACITY",
    "DISPOSITION_TRANSITION_MATRIX",
    "DIURNAL_HOURLY_WEIGHTS",
    "ESI_DISTRIBUTION_WEIGHTS",
    "FallbackHospitalSimulationEngine",
    "FallbackMIMICDistributionSampler",
    "PredictiveModelPipeline",
]