# backend/src/data_generator/mimic_distributions.py
"""MIMIC-IV empirical distribution sampler and synthetic cohort generator.

Provides stochastic sampling primitives calibrated on MIMIC-IV emergency and
inpatient stays. Generates fully validated PatientStayContract and
HourlyCensusContract domain instances for discrete-event simulation (SimPy)
and prescriptive optimization engines.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Final
from schemas.contracts import (
    BedStatus,
    CareUnitType,
    DispositionType,
    HourlyCensusContract,
    PatientStayContract,
    ShiftType,
)

TIMESTAMP_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
HOURLY_FORMAT: Final[str] = "%Y-%m-%d %H:00:00"


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DistributionConfig:
    """Empirical parameters derived from MIMIC-IV for synthetic patient sampling."""

    # Hourly baseline arrival rates (hour 0 to 23)
    hourly_arrival_rates: tuple[float, ...] = (
        6.0,  # 00:00
        5.0,  # 01:00
        3.5,  # 02:00 (Trough)
        2.5,  # 03:00
        2.0,  # 04:00
        2.5,  # 05:00
        4.0,  # 06:00
        6.5,  # 07:00 (Ramp-up)
        8.0,  # 08:00
        9.5,  # 09:00
        10.5,  # 10:00
        11.5,  # 11:00
        13.0,  # 12:00 (Peak)
        14.5,  # 13:00
        15.5,  # 14:00
        15.0,  # 15:00
        14.0,  # 16:00
        13.5,  # 17:00
        12.0,  # 18:00
        10.0,  # 19:00 (Winding down)
        8.5,  # 20:00
        7.5,  # 21:00
        6.5,  # 22:00
        5.5,  # 23:00
    )
    weekend_rate_multiplier: float = 0.90

    # Demographics
    age_mean: float = 58.5
    age_std: float = 17.5
    age_min: float = 18.0
    age_max: float = 95.0
    prob_female: float = 0.52

    # Triage Acuity (ESI 1-5 categorical weights)
    esi_probabilities: tuple[float, ...] = (0.02, 0.20, 0.54, 0.20, 0.04)

    # Chief Complaints and empirical probabilities
    chief_complaints: tuple[str, ...] = (
        "Chest Pain",
        "Shortness of Breath / Dyspnea",
        "Abdominal Pain",
        "Altered Mental Status / Syncope",
        "Fall / Trauma",
        "Fever / Infection",
        "Headache / Neurological",
        "Gastrointestinal / Nausea / Vomiting",
        "Musculoskeletal Pain",
        "Other / General Malaise",
    )
    chief_complaint_weights: tuple[float, ...] = (
        0.14,
        0.12,
        0.11,
        0.08,
        0.08,
        0.07,
        0.06,
        0.06,
        0.05,
        0.23,
    )

    # Hemodynamics baseline distributions
    sbp_mean: float = 126.0
    sbp_std: float = 22.0
    sbp_min: float = 60.0
    sbp_max: float = 240.0

    pp_mean: float = 46.0
    pp_std: float = 12.0
    min_pulse_pressure: float = 15.0

    hr_mean: float = 82.0
    hr_std: float = 17.0
    hr_min: float = 30.0
    hr_max: float = 220.0

    o2_exp_scale: float = 1.6
    o2_min: float = 60.0
    o2_max: float = 100.0

    rr_mean: float = 18.0
    rr_std: float = 4.0
    rr_min: float = 6.0
    rr_max: float = 60.0

    temp_mean: float = 37.0
    temp_std: float = 0.6
    temp_min: float = 33.0
    temp_max: float = 43.0

    # Disposition probabilities conditional on ESI [ESI 1, 2, 3, 4, 5]
    # Each entry: (P(ICU), P(Ward), P(ED_Discharge))
    disposition_by_esi: tuple[tuple[float, float, float], ...] = (
        (0.70, 0.25, 0.05),  # ESI 1
        (0.22, 0.48, 0.30),  # ESI 2
        (0.03, 0.27, 0.70),  # ESI 3
        (0.00, 0.04, 0.96),  # ESI 4
        (0.00, 0.01, 0.99),  # ESI 5
    )

    # Ward deterioration to ICU probability
    ward_icu_transfer_prob: float = 0.06

    # Length of Stay (LoS) LogNormal parameters: (mu, sigma, min_h, max_h)
    los_ed_params: tuple[float, float, float, float] = (1.3, 0.5, 0.5, 24.0)
    los_ward_params: tuple[float, float, float, float] = (3.9, 0.7, 12.0, 240.0)
    los_icu_params: tuple[float, float, float, float] = (4.5, 0.8, 24.0, 336.0)


# ---------------------------------------------------------------------------
# Sampling Engine
# ---------------------------------------------------------------------------
class MIMICDistributionSampler:
    """Stochastic generator for synthetic healthcare entities calibrated on MIMIC-IV."""

    def __init__(
        self,
        config: DistributionConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or DistributionConfig()
        self._rng = random.Random(seed)

    def set_seed(self, seed: int) -> None:
        """Reset internal pseudo-random number generator state."""
        self._rng.seed(seed)

    # -----------------------------------------------------------------------
    # Low-level Stochastic Primitives
    # -----------------------------------------------------------------------
    def _sample_poisson(self, lam: float) -> int:
        """Sample from Poisson distribution using Knuth's algorithm."""
        if lam <= 0.0:
            return 0
        limit = math.exp(-lam)
        k = 0
        p = 1.0
        while p > limit:
            k += 1
            p *= self._rng.random()
        return k - 1

    def _sample_truncated_normal(
        self, mean: float, std: float, lower: float, upper: float
    ) -> float:
        """Sample from Normal distribution with hard truncation bounds."""
        for _ in range(50):
            val = self._rng.gauss(mean, std)
            if lower <= val <= upper:
                return val
        return max(lower, min(upper, self._rng.gauss(mean, std)))

    def _sample_bounded_lognormal(
        self, mu: float, sigma: float, lower: float, upper: float
    ) -> float:
        """Sample from LogNormal distribution clamped to [lower, upper]."""
        val = self._rng.lognormvariate(mu, sigma)
        return max(lower, min(upper, val))

    # -----------------------------------------------------------------------
    # Diurnal Arrival Process
    # -----------------------------------------------------------------------
    def sample_arrival_count(self, dt: datetime) -> int:
        """Sample hourly arrival count based on hour-of-day and day-of-week."""
        base_lambda = self.config.hourly_arrival_rates[dt.hour]
        if dt.weekday() in (5, 6):
            base_lambda *= self.config.weekend_rate_multiplier
        return self._sample_poisson(base_lambda)

    # -----------------------------------------------------------------------
    # Hemodynamics and Vitals
    # -----------------------------------------------------------------------
    def sample_vitals(self, acuity: int, complaint: str) -> dict[str, float]:
        """Generate physiologically coupled vitals satisfying clinical invariants."""
        cfg = self.config

        # 1. Systolic Blood Pressure (SBP)
        sbp_target_mean = cfg.sbp_mean
        if complaint == "Chest Pain":
            sbp_target_mean += 10.0
        elif acuity == 1 and self._rng.random() < 0.25:
            # Hypotensive shock state
            sbp_target_mean = 78.0

        sbp = self._sample_truncated_normal(
            sbp_target_mean, cfg.sbp_std, cfg.sbp_min, cfg.sbp_max
        )

        # 2. Diastolic Blood Pressure (DBP) & Pulse Pressure (PP)
        # Invariant: SBP >= DBP + 15.0, DBP in [30.0, 140.0]
        max_allowable_pp = sbp - 30.0
        min_allowable_pp = cfg.min_pulse_pressure

        if max_allowable_pp < min_allowable_pp:
            # If SBP was near absolute floor, adjust SBP upward to allow valid DBP
            sbp = 30.0 + min_allowable_pp
            max_allowable_pp = min_allowable_pp

        raw_pp = self._sample_truncated_normal(
            cfg.pp_mean, cfg.pp_std, min_allowable_pp, max(min_allowable_pp, max_allowable_pp)
        )
        pp = max(min_allowable_pp, min(max_allowable_pp, raw_pp))
        dbp = sbp - pp

        if dbp > 140.0:
            dbp = 140.0
            sbp = max(sbp, dbp + min_allowable_pp)
        elif dbp < 30.0:
            dbp = 30.0
            sbp = max(sbp, dbp + min_allowable_pp)

        # 3. Body Temperature
        temp_mean = cfg.temp_mean
        if complaint == "Fever / Infection":
            temp_mean = 38.6
        temp_c = self._sample_truncated_normal(
            temp_mean, cfg.temp_std, cfg.temp_min, cfg.temp_max
        )

        # 4. Heart Rate (HR)
        hr_mean = cfg.hr_mean
        if temp_c > 38.0:
            hr_mean += (temp_c - 37.0) * 8.0  # Tachycardia response to fever
        if acuity <= 2:
            hr_mean += 12.0
        if complaint in ("Chest Pain", "Shortness of Breath / Dyspnea"):
            hr_mean += 6.0
        hr = self._sample_truncated_normal(
            hr_mean, cfg.hr_std, cfg.hr_min, cfg.hr_max
        )

        # 5. O2 Saturation
        o2_sat_val = cfg.o2_max - self._rng.expovariate(1.0 / cfg.o2_exp_scale)
        if complaint == "Shortness of Breath / Dyspnea":
            o2_sat_val -= self._rng.uniform(4.0, 12.0)
        elif acuity <= 2:
            o2_sat_val -= self._rng.uniform(2.0, 8.0)
        o2_sat = max(cfg.o2_min, min(cfg.o2_max, o2_sat_val))

        # 6. Respiratory Rate (RR)
        rr_mean = cfg.rr_mean
        if o2_sat < 92.0:
            rr_mean += (92.0 - o2_sat) * 0.6  # Tachypnea compensatory reflex
        if acuity <= 2:
            rr_mean += 5.0
        resp_rate = self._sample_truncated_normal(
            rr_mean, cfg.rr_std, cfg.rr_min, cfg.rr_max
        )

        return {
            "sbp": round(sbp, 1),
            "dbp": round(dbp, 1),
            "heart_rate": round(hr, 1),
            "temp_c": round(temp_c, 2),
            "o2_sat": round(o2_sat, 1),
            "resp_rate": round(resp_rate, 1),
        }

    # -----------------------------------------------------------------------
    # Patient Trajectory & Stay Contract Generation
    # -----------------------------------------------------------------------
    def sample_patient_stay(
        self,
        stay_id: int,
        patient_id: int,
        arrival_time: datetime,
    ) -> PatientStayContract:
        """Sample a fully formed, invariant-validated patient encounter."""
        cfg = self.config

        # 1. Demographics
        age = int(round(self._sample_truncated_normal(
            cfg.age_mean, cfg.age_std, cfg.age_min, cfg.age_max
        )))
        gender = "Female" if self._rng.random() < cfg.prob_female else "Male"

        cci_lambda = 1.2 + (age / 35.0)
        charlson_index = max(0, min(10, self._sample_poisson(cci_lambda)))

        # 2. Triage & Complaint
        acuity = self._rng.choices(
            [1, 2, 3, 4, 5],
            weights=cfg.esi_probabilities,
            k=1,
        )[0]
        complaint = self._rng.choices(
            cfg.chief_complaints,
            weights=cfg.chief_complaint_weights,
            k=1,
        )[0]

        # 3. Vitals
        vitals = self.sample_vitals(acuity, complaint)

        # 4. Trajectory & Disposition
        # Disposition probabilities for the selected ESI acuity
        p_icu, p_ward, p_ed = cfg.disposition_by_esi[acuity - 1]
        disposition_choice = self._rng.choices(
            [DispositionType.ICU, DispositionType.WARD, DispositionType.ED_DISCHARGE],
            weights=[p_icu, p_ward, p_ed],
            k=1,
        )[0]

        if disposition_choice == DispositionType.ICU:
            initial_care_unit = CareUnitType.ICU
            icu_transfer_flag = 1
            los_params = cfg.los_icu_params
        elif disposition_choice == DispositionType.WARD:
            initial_care_unit = CareUnitType.WARD
            icu_transfer_flag = 1 if self._rng.random() < cfg.ward_icu_transfer_prob else 0
            los_params = cfg.los_ward_params
        else:
            initial_care_unit = CareUnitType.ED_ONLY
            icu_transfer_flag = 0
            los_params = cfg.los_ed_params

        # Ventilation requirement
        if initial_care_unit == CareUnitType.ICU:
            vent_prob = 0.75 if (vitals["o2_sat"] < 88.0 and acuity <= 2) else 0.38
            requires_ventilation = 1 if self._rng.random() < vent_prob else 0
        else:
            requires_ventilation = 0

        # 5. Length of Stay (LoS) & Milestone Progression
        los_hours = round(
            self._sample_bounded_lognormal(*los_params),
            2,
        )
        total_seconds = int(los_hours * 3600)

        # Sub-milestone latencies in seconds
        # delta_triage: prioritized by acuity (ESI 1 is prioritized immediately)
        if acuity == 1:
            triage_sec = self._rng.uniform(60, 180)
        elif acuity == 2:
            triage_sec = self._rng.uniform(180, 600)
        else:
            triage_sec = self._rng.uniform(300, 1500)

        # delta_bed_assignment
        if initial_care_unit == CareUnitType.ED_ONLY:
            bed_sec = triage_sec + self._rng.uniform(300, 1800)
        else:
            bed_sec = triage_sec + self._rng.expovariate(1.0 / 2700.0)

        # Defensive bounds ensuring strict monotonicity:
        # arrival_time <= triage_start_time <= bed_assigned_time <= discharge_time
        max_triage_sec = total_seconds * 0.30
        max_bed_sec = total_seconds * 0.80

        triage_sec = min(triage_sec, max_triage_sec)
        bed_sec = max(triage_sec, min(bed_sec, max_bed_sec))

        dt_triage = arrival_time + timedelta(seconds=int(triage_sec))
        dt_bed = arrival_time + timedelta(seconds=int(bed_sec))
        dt_discharge = arrival_time + timedelta(seconds=total_seconds)

        arrival_str = arrival_time.strftime(TIMESTAMP_FORMAT)
        triage_str = dt_triage.strftime(TIMESTAMP_FORMAT)
        bed_str = dt_bed.strftime(TIMESTAMP_FORMAT)
        discharge_str = dt_discharge.strftime(TIMESTAMP_FORMAT)
        arrival_hour_str = arrival_time.strftime(HOURLY_FORMAT)

        return PatientStayContract(
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
            arrival_time=arrival_str,
            triage_start_time=triage_str,
            bed_assigned_time=bed_str,
            discharge_time=discharge_str,
            arrival_hour=arrival_hour_str,
            los_hours=los_hours,
            disposition=disposition_choice,
            icu_transfer_flag=icu_transfer_flag,
            initial_care_unit=initial_care_unit,
            requires_ventilation=requires_ventilation,
        )

    # -----------------------------------------------------------------------
    # Hourly Batch Generation
    # -----------------------------------------------------------------------
    def sample_hourly_batch(
        self,
        arrival_hour: datetime,
        start_stay_id: int,
        start_patient_id: int,
    ) -> list[PatientStayContract]:
        """Generate all patient arrivals occurring within a designated clock hour."""
        floored_hour = arrival_hour.replace(minute=0, second=0, microsecond=0)
        arrival_count = self.sample_arrival_count(floored_hour)

        # Distribute arrivals across the 3,600 seconds of the hour
        arrival_offsets = sorted(
            self._rng.randint(0, 3599) for _ in range(arrival_count)
        )

        batch: list[PatientStayContract] = []
        for idx, offset in enumerate(arrival_offsets):
            arrival_dt = floored_hour + timedelta(seconds=offset)
            stay_contract = self.sample_patient_stay(
                stay_id=start_stay_id + idx,
                patient_id=start_patient_id + idx,
                arrival_time=arrival_dt,
            )
            batch.append(stay_contract)

        return batch

    # -----------------------------------------------------------------------
    # Hourly Census Contract Generation
    # -----------------------------------------------------------------------
    def sample_hourly_census(
        self,
        timestamp: datetime,
        ed_occupancy: int = 42,
        ward_occupancy: int = 135,
        icu_occupancy: int = 24,
        active_nurses: int | None = None,
        active_doctors: int | None = None,
        arrivals_count: int = 8,
        admissions_count: int = 3,
        discharges_count: int = 2,
        queue_length: int = 4,
        forecast_4h: int = 32,
    ) -> HourlyCensusContract:
        """Create a validated census snapshot with shift and staffing alignment."""
        hour = timestamp.hour

        # Determine shift from hour
        if 7 <= hour <= 14:
            shift_id = ShiftType.DAY
            default_nurses, default_doctors = 35, 10
        elif 15 <= hour <= 22:
            shift_id = ShiftType.EVENING
            default_nurses, default_doctors = 30, 8
        else:
            shift_id = ShiftType.NIGHT
            default_nurses, default_doctors = 20, 5

        nurses = active_nurses if active_nurses is not None else default_nurses
        doctors = active_doctors if active_doctors is not None else default_doctors

        day_of_week = timestamp.strftime("%A")
        is_weekend = 1 if timestamp.weekday() in (5, 6) else 0
        census_ts = timestamp.strftime(HOURLY_FORMAT)

        return HourlyCensusContract(
            timestamp=census_ts,
            hour=hour,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            shift_id=shift_id,
            ed_occupancy=max(0, min(100, ed_occupancy)),
            ward_occupancy=max(0, min(200, ward_occupancy)),
            icu_occupancy=max(0, min(50, icu_occupancy)),
            active_nurses=max(15, min(50, nurses)),
            active_doctors=max(3, min(20, doctors)),
            arrivals_count=max(0, arrivals_count),
            admissions_count=max(0, admissions_count),
            discharges_count=max(0, discharges_count),
            patients_in_queue=max(0, queue_length),
            incoming_arrivals_next_4h=max(0, forecast_4h),
        )


# ---------------------------------------------------------------------------
# Self-Verification Test Suite
# ---------------------------------------------------------------------------
def run_verification_suite(n_samples: int = 1000) -> None:
    """Execute standalone verification suite guaranteeing contract invariants."""
    print(f"Beginning contract verification across {n_samples:,} iterations...")

    # 1. Deterministic Reproducibility
    sampler_a = MIMICDistributionSampler(seed=42)
    sampler_b = MIMICDistributionSampler(seed=42)
    t0 = datetime(2026, 9, 23, 10, 15, 30)

    stay_a = sampler_a.sample_patient_stay(1, 100001, t0)
    stay_b = sampler_b.sample_patient_stay(1, 100001, t0)
    assert stay_a.model_dump() == stay_b.model_dump(), "Seed reproducibility check failed."
    print("PASS: test_deterministic_reproducibility")

    # 2. Large cohort invariant compliance
    sampler = MIMICDistributionSampler(seed=2026)
    base_time = datetime(2026, 9, 23, 0, 0, 0)

    for i in range(n_samples):
        dt = base_time + timedelta(minutes=i * 7)
        stay = sampler.sample_patient_stay(
            stay_id=(i % 99000) + 1,
            patient_id=100001 + (i % 99000),
            arrival_time=dt,
        )

        # Pulse pressure invariant
        assert stay.sbp - stay.dbp >= 15.0, (
            f"Pulse pressure violation on stay {stay.stay_id}: "
            f"sbp={stay.sbp}, dbp={stay.dbp}"
        )

        # Temporal monotonicity
        dt_arr = datetime.strptime(stay.arrival_time, TIMESTAMP_FORMAT)
        dt_trg = datetime.strptime(stay.triage_start_time, TIMESTAMP_FORMAT)
        dt_bed = datetime.strptime(stay.bed_assigned_time, TIMESTAMP_FORMAT)
        dt_dis = datetime.strptime(stay.discharge_time, TIMESTAMP_FORMAT)
        assert dt_arr <= dt_trg <= dt_bed <= dt_dis, (
            f"Monotonicity violation on stay {stay.stay_id}: "
            f"{stay.arrival_time} <= {stay.triage_start_time} <= "
            f"{stay.bed_assigned_time} <= {stay.discharge_time}"
        )

        # Arrival hour flooring
        assert stay.arrival_hour.endswith(":00:00"), (
            f"Arrival hour not floored: {stay.arrival_hour}"
        )
        assert stay.arrival_hour == dt.strftime("%Y-%m-%d %H:00:00"), (
            f"Arrival hour mismatch: {stay.arrival_hour} vs {dt}"
        )

    print(f"PASS: test_patient_contract_compliance ({n_samples:,} samples)")
    print(f"PASS: test_pulse_pressure_invariant ({n_samples:,} samples)")
    print(f"PASS: test_temporal_monotonicity ({n_samples:,} samples)")
    print(f"PASS: test_arrival_hour_flooring ({n_samples:,} samples)")

    # 3. Hourly batch and census alignment
    batch = sampler.sample_hourly_batch(
        arrival_hour=datetime(2026, 9, 23, 14, 0, 0),
        start_stay_id=5000,
        start_patient_id=105000,
    )
    assert all(isinstance(s, PatientStayContract) for s in batch)
    print(f"PASS: test_sample_hourly_batch (Generated {len(batch)} arrivals)")

    census = sampler.sample_hourly_census(datetime(2026, 9, 23, 14, 0, 0))
    assert census.shift_id == ShiftType.DAY
    assert census.is_weekend == 0
    assert census.day_of_week == "Wednesday"
    print("PASS: test_sample_hourly_census")
    print("All verification suite invariants passed successfully.")


if __name__ == "__main__":
    run_verification_suite(1000)