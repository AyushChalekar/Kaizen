# System Architecture & Forecasting Module Extraction
Generated on: Wed Sep 23 15:22:19 UTC 2026

## File: backend/src/schemas/prediction_contracts.py
```python
# backend/src/schemas/prediction_contracts.py
"""Prediction contracts for the Digital Twin hospital simulation platform.

Provides validated Pydantic v2 prediction models that mirror and validate against
the foundational domain contracts defined in contracts.py. Enforces strict temporal
monotonicity, shift categorization, and coupled physiological invariants.
"""

from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Final, Literal, Self
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Import Domain Primitives & Fallback Definitions
# ---------------------------------------------------------------------------
try:
    from src.schemas.contracts import (
        CareUnitType,
        DispositionType,
        ShiftType,
        TIMESTAMP_FORMAT,
        EXPECTED_TS_LENGTH,
        MIN_PULSE_PRESSURE,
        _parse_strict_timestamp,
    )
except ImportError:
    try:
        from .contracts import (
            CareUnitType,
            DispositionType,
            ShiftType,
            TIMESTAMP_FORMAT,
            EXPECTED_TS_LENGTH,
            MIN_PULSE_PRESSURE,
            _parse_strict_timestamp,
        )
    except ImportError:
        try:
            from contracts import (
                CareUnitType,
                DispositionType,
                ShiftType,
                TIMESTAMP_FORMAT,
                EXPECTED_TS_LENGTH,
                MIN_PULSE_PRESSURE,
                _parse_strict_timestamp,
            )
        except ImportError:
            TIMESTAMP_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
            EXPECTED_TS_LENGTH: Final[int] = 19
            MIN_PULSE_PRESSURE: Final[float] = 15.0

            def _parse_strict_timestamp(val: str, field_name: str) -> datetime:
                """Parse a datetime string strictly formatted as YYYY-MM-DD HH:MM:SS."""
                if not isinstance(val, str) or len(val) != EXPECTED_TS_LENGTH:
                    raise ValueError(
                        f"Field '{field_name}' must be a string of length {EXPECTED_TS_LENGTH} "
                        f"matching 'YYYY-MM-DD HH:MM:SS'. Received: {val!r}"
                    )
                try:
                    return datetime.strptime(val, TIMESTAMP_FORMAT)
                except ValueError as exc:
                    raise ValueError(
                        f"Field '{field_name}' failed datetime parsing with format "
                        f"'YYYY-MM-DD HH:MM:SS'. Received: {val!r}"
                    ) from exc

            class ShiftType(str, Enum):
                """Clinical working shifts within a 24-hour cycle."""

                DAY = "Day"
                EVENING = "Evening"
                NIGHT = "Night"

            class DispositionType(str, Enum):
                """Patient disposition endpoint upon departing emergency care."""

                ED_DISCHARGE = "ED_Discharge"
                WARD = "Ward"
                ICU = "ICU"

            class CareUnitType(str, Enum):
                """Hospital functional inpatient and emergency service units."""

                ED_ONLY = "ED_Only"
                WARD = "Ward"
                ICU = "ICU"


# ---------------------------------------------------------------------------
# Chief Complaints Constants & Enum
# ---------------------------------------------------------------------------
class PrimaryChiefComplaint(str, Enum):
    """Ten primary clinical chief complaints established in MIMIC-IV ED distributions."""

    CHEST_PAIN = "Chest Pain"
    SHORTNESS_OF_BREATH = "Shortness of Breath"
    ABDOMINAL_PAIN = "Abdominal Pain"
    HEADACHE = "Headache"
    FEVER = "Fever"
    COUGH = "Cough"
    DIZZINESS = "Dizziness"
    BACK_PAIN = "Back Pain"
    NAUSEA_VOMITING = "Nausea/Vomiting"
    WEAKNESS = "Weakness"


PRIMARY_CHIEF_COMPLAINTS: Final[tuple[str, ...]] = tuple(
    item.value for item in PrimaryChiefComplaint
)


# ---------------------------------------------------------------------------
# 1. Resource Utilization & Bottleneck Forecasts
# ---------------------------------------------------------------------------
class ResourceUtilizationForecast(BaseModel):
    """Forecasted hospital resource utilization, waiting queues, and unit occupancies."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
        populate_by_name=True,
    )

    forecast_timestamp: str | None = Field(
        default=None,
        description="Forecast projection snapshot timestamp ('YYYY-MM-DD HH:MM:SS').",
    )
    peak_queue_length: int = Field(
        ...,
        ge=0,
        alias="peak_queue_patients",
        description="Forecasted peak queue length for waiting patients.",
    )
    predicted_avg_los_hours: float = Field(
        ...,
        ge=0.0,
        le=336.0,
        alias="predicted_los_hours",
        description="Predicted average Length of Stay (LoS) measured in decimal hours.",
    )
    ed_utilization_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        alias="ed_bed_utilization_pct",
        description="Projected Emergency Department (ED_Only) bed utilization percentage.",
    )
    ward_utilization_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        alias="ward_bed_utilization_pct",
        description="Projected medical-surgical general ward (Ward) bed utilization percentage.",
    )
    icu_utilization_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        alias="icu_bed_utilization_pct",
        description="Projected Intensive Care Unit (ICU) bed utilization percentage.",
    )

    @field_validator("forecast_timestamp", mode="before")
    @classmethod
    def validate_timestamp_format(cls, val: Any) -> str | None:
        """Validate optional forecast timestamp adherence to YYYY-MM-DD HH:MM:SS."""
        if val is None:
            return None
        if not isinstance(val, str):
            raise ValueError("Field 'forecast_timestamp' must be a string timestamp.")
        _parse_strict_timestamp(val, "forecast_timestamp")
        return val.strip()

    @model_validator(mode="before")
    @classmethod
    def unpack_nested_utilization(cls, data: Any) -> Any:
        """Unpack nested bed utilization mappings if provided."""
        if isinstance(data, dict):
            breakdown = data.get("bed_utilization") or data.get("bed_utilization_breakdown")
            if isinstance(breakdown, dict):
                data = dict(data)
                for key, val in breakdown.items():
                    norm_key = str(key).upper().replace(" ", "_")
                    if "ED" in norm_key:
                        data.setdefault("ed_utilization_pct", val)
                    elif "WARD" in norm_key or "MED_SURG" in norm_key:
                        data.setdefault("ward_utilization_pct", val)
                    elif "ICU" in norm_key or "CCU" in norm_key:
                        data.setdefault("icu_utilization_pct", val)
        return data

    @property
    def bed_utilization_breakdown(self) -> dict[CareUnitType, float]:
        """Projected bed utilization percentages keyed by CareUnitType."""
        return {
            CareUnitType.ED_ONLY: self.ed_utilization_pct,
            CareUnitType.WARD: self.ward_utilization_pct,
            CareUnitType.ICU: self.icu_utilization_pct,
        }


# ---------------------------------------------------------------------------
# 2. Hourly Census & Staffing Alignment
# ---------------------------------------------------------------------------
class HourlyStaffingPrediction(BaseModel):
    """Hourly patient flow projections and shift-based clinical staffing requirements."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
        populate_by_name=True,
    )

    # Shift Baselines Lookup Table: Shift -> (baseline_nurses, baseline_doctors)
    SHIFT_BASELINES: ClassVar[dict[ShiftType, dict[str, int]]] = {
        ShiftType.DAY: {"nurses": 35, "doctors": 10},
        ShiftType.EVENING: {"nurses": 30, "doctors": 8},
        ShiftType.NIGHT: {"nurses": 20, "doctors": 5},
    }

    # Temporal & Diurnal Dimensions
    timestamp: str = Field(
        ...,
        description="Hourly forecast snapshot key floored to 'YYYY-MM-DD HH:00:00'.",
    )
    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of day (0 to 23).",
    )
    day_of_week: str = Field(
        ...,
        min_length=3,
        description="Day of week (e.g., 'Monday', 'Tuesday').",
    )
    is_weekend: int = Field(
        ...,
        ge=0,
        le=1,
        description="Weekend binary indicator (1 for Saturday/Sunday, 0 otherwise).",
    )
    shift_id: ShiftType = Field(
        ...,
        description="Operating shift category (Day, Evening, or Night).",
    )

    # Hourly Projected Flow Rates
    projected_arrivals: float = Field(
        ...,
        ge=0.0,
        alias="arrivals_count",
        description="Projected hourly patient arrivals adjusted for diurnal patterns.",
    )
    projected_admissions: float = Field(
        ...,
        ge=0.0,
        alias="admissions_count",
        description="Projected hourly patient admissions adjusted for diurnal patterns.",
    )
    projected_discharges: float = Field(
        ...,
        ge=0.0,
        alias="discharges_count",
        description="Projected hourly patient discharges adjusted for diurnal patterns.",
    )

    # Predicted Required Staffing Headcounts
    predicted_active_nurses: int = Field(
        ...,
        ge=0,
        alias="active_nurses",
        description="Predicted active registered and charge nurse headcount required.",
    )
    predicted_active_physicians: int = Field(
        ...,
        ge=0,
        alias="active_doctors",
        description="Predicted active physician headcount required to meet occupancy demands.",
    )

    # Rolling Horizon Metric
    incoming_arrivals_next_4h: float = Field(
        ...,
        ge=0.0,
        description="Prescriptive arrival forecast across the rolling 4-hour window.",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp_format(cls, val: Any) -> str:
        """Validate timestamp string adherence and zero-minute flooring."""
        if not isinstance(val, str):
            raise ValueError("Field 'timestamp' must be a valid string.")
        dt = _parse_strict_timestamp(val, "timestamp")
        if dt.minute != 0 or dt.second != 0:
            raise ValueError(
                f"Field 'timestamp' must be floored to zero minutes and seconds ('YYYY-MM-DD HH:00:00'). "
                f"Received: {val!r}"
            )
        return val.strip()

    @model_validator(mode="before")
    @classmethod
    def normalize_physician_alias(cls, data: Any) -> Any:
        """Normalize predicted_active_doctors alias to predicted_active_physicians."""
        if isinstance(data, dict):
            data = dict(data)
            if "predicted_active_doctors" in data and "predicted_active_physicians" not in data:
                data["predicted_active_physicians"] = data["predicted_active_doctors"]
        return data

    @model_validator(mode="after")
    def validate_shift_and_calendar_invariants(self) -> Self:
        """Enforce shift hour boundaries, diurnal alignment, and calendar synchronization."""
        dt = _parse_strict_timestamp(self.timestamp, "timestamp")

        # 1. Hour synchronization
        if dt.hour != self.hour:
            raise ValueError(
                f"Hour mismatch: 'timestamp' hour is {dt.hour:02d}, but 'hour' field is {self.hour:02d}."
            )

        # 2. Shift Mapping Verification
        # Day: 07:00 - 14:59 (hours 7 to 14)
        # Evening: 15:00 - 22:59 (hours 15 to 22)
        # Night: 23:00 - 06:59 (hours 23, 0 to 6)
        if 7 <= self.hour <= 14:
            expected_shift = ShiftType.DAY
        elif 15 <= self.hour <= 22:
            expected_shift = ShiftType.EVENING
        else:
            expected_shift = ShiftType.NIGHT

        if self.shift_id != expected_shift:
            raise ValueError(
                f"Invalid shift mapping: Hour {self.hour:02d} must map to {expected_shift.value} shift. "
                f"Received shift_id='{self.shift_id.value}'."
            )

        # 3. Calendar Day and Weekend Synchronization
        expected_day_name = dt.strftime("%A")
        if self.day_of_week.strip().lower() != expected_day_name.lower():
            raise ValueError(
                f"Calendar day mismatch: Timestamp {self.timestamp} corresponds to {expected_day_name}, "
                f"but 'day_of_week' was provided as '{self.day_of_week}'."
            )

        expected_is_weekend = 1 if dt.weekday() in (5, 6) else 0
        if self.is_weekend != expected_is_weekend:
            raise ValueError(
                f"Weekend indicator mismatch: For {expected_day_name}, 'is_weekend' must be "
                f"{expected_is_weekend}, but received {self.is_weekend}."
            )

        return self

    @property
    def baseline_nurses(self) -> int:
        """Scheduled baseline nurse count for the active shift."""
        return self.SHIFT_BASELINES[self.shift_id]["nurses"]

    @property
    def baseline_physicians(self) -> int:
        """Scheduled baseline physician count for the active shift."""
        return self.SHIFT_BASELINES[self.shift_id]["doctors"]

    @property
    def baseline_doctors(self) -> int:
        """Alias for baseline physician count."""
        return self.baseline_physicians

    @property
    def nurse_variance(self) -> float:
        """Variance between predicted nurse headcount and shift baseline."""
        return float(self.predicted_active_nurses - self.baseline_nurses)

    @property
    def physician_variance(self) -> float:
        """Variance between predicted physician headcount and shift baseline."""
        return float(self.predicted_active_physicians - self.baseline_physicians)


# ---------------------------------------------------------------------------
# 3. Patient Flow & Clinical Trajectories
# ---------------------------------------------------------------------------
class ClinicalTrajectoryPrediction(BaseModel):
    """Anticipated patient flow routing, escalation flags, and sequential milestones."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
        populate_by_name=True,
    )

    stay_id: int | None = Field(
        default=None,
        ge=1,
        le=100_000,
        description="Unique stay identifier across simulation state.",
    )
    patient_id: int | None = Field(
        default=None,
        ge=100_001,
        le=200_000,
        description="Master patient index identifier.",
    )

    # Disposition Routing
    predicted_disposition: DispositionType = Field(
        ...,
        alias="disposition",
        description="Anticipated routing destination endpoint (ED_Discharge, Ward, or ICU).",
    )

    # Clinical Escalation Flags & Risks
    requires_ventilation: int = Field(
        ...,
        ge=0,
        le=1,
        description="Indicator flag for sudden mechanical ventilation needs (0 or 1).",
    )
    icu_transfer_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of escalation and Ward-to-ICU transfer (0.0 to 1.0).",
    )
    icu_transfer_flag: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Indicator flag for ICU transfer escalation (0 or 1).",
    )

    # Sequential Timestamp Projections
    arrival_time: str = Field(
        ...,
        description="Projected arrival timestamp ('YYYY-MM-DD HH:MM:SS').",
    )
    triage_start_time: str = Field(
        ...,
        description="Projected triage assessment start ('YYYY-MM-DD HH:MM:SS').",
    )
    bed_assigned_time: str = Field(
        ...,
        description="Projected physical bed assignment timestamp ('YYYY-MM-DD HH:MM:SS').",
    )
    discharge_time: str = Field(
        ...,
        description="Projected departure disposition timestamp ('YYYY-MM-DD HH:MM:SS').",
    )
    predicted_los_hours: float | None = Field(
        default=None,
        ge=0.0,
        le=336.0,
        description="Predicted total length of stay in decimal hours.",
    )

    @field_validator("requires_ventilation", mode="before")
    @classmethod
    def normalize_ventilation_flag(cls, val: Any) -> int:
        """Normalize boolean or numeric 0/1 representation for mechanical ventilation."""
        if isinstance(val, bool):
            return 1 if val else 0
        if isinstance(val, (int, float)):
            int_val = int(val)
            if int_val in {0, 1}:
                return int_val
        if isinstance(val, str):
            token = val.strip().lower()
            if token in {"true", "1", "yes"}:
                return 1
            if token in {"false", "0", "no"}:
                return 0
        raise ValueError(
            f"Field 'requires_ventilation' must be a boolean or 0/1 indicator. Received: {val!r}"
        )

    @field_validator(
        "arrival_time",
        "triage_start_time",
        "bed_assigned_time",
        "discharge_time",
        mode="before",
    )
    @classmethod
    def validate_timestamp_format(cls, val: Any, info: Any) -> str:
        """Validate string adherence to YYYY-MM-DD HH:MM:SS."""
        if not isinstance(val, str):
            raise ValueError(f"Field '{info.field_name}' must be a string timestamp.")
        _parse_strict_timestamp(val, info.field_name)
        return val.strip()

    @model_validator(mode="after")
    def validate_temporal_monotonicity(self) -> Self:
        """Enforce strict temporal monotonicity across sequential projection milestones."""
        dt_arr = _parse_strict_timestamp(self.arrival_time, "arrival_time")
        dt_trg = _parse_strict_timestamp(self.triage_start_time, "triage_start_time")
        dt_bed = _parse_strict_timestamp(self.bed_assigned_time, "bed_assigned_time")
        dt_dis = _parse_strict_timestamp(self.discharge_time, "discharge_time")

        if not (dt_arr <= dt_trg <= dt_bed <= dt_dis):
            raise ValueError(
                "Temporal monotonicity violated: Enforce arrival_time <= "
                "triage_start_time <= bed_assigned_time <= discharge_time. Found sequence: "
                f"arrival='{self.arrival_time}', triage='{self.triage_start_time}', "
                f"bed_assigned='{self.bed_assigned_time}', discharge='{self.discharge_time}'."
            )

        if self.predicted_los_hours is None:
            calculated_los = (dt_dis - dt_arr).total_seconds() / 3600.0
            self.predicted_los_hours = round(calculated_los, 2)

        return self


# ---------------------------------------------------------------------------
# 4. Physiological & Demographic Modeling
# ---------------------------------------------------------------------------
class PatientClinicalPrediction(BaseModel):
    """Probabilistic demographic state, triage acuity, and coupled hemodynamics."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
        populate_by_name=True,
    )

    patient_id: int | None = Field(
        default=None,
        ge=100_001,
        le=200_000,
        description="Master patient index identifier.",
    )
    stay_id: int | None = Field(
        default=None,
        ge=1,
        le=100_000,
        description="Unique stay identifier.",
    )

    # Demographic Profile
    age: int = Field(
        ...,
        ge=18,
        le=95,
        description="Patient age in completed years (18 to 95).",
    )
    gender: Literal["Female", "Male"] = Field(
        ...,
        description="Normalized biological or administrative sex ('Female' or 'Male').",
    )
    charlson_index: int = Field(
        ...,
        ge=0,
        le=10,
        description="Charlson Comorbidity Index score (0 to 10).",
    )
    chief_complaint: str = Field(
        ...,
        min_length=1,
        description="Presenting clinical complaint, one of 10 primary complaints.",
    )

    # Acuity
    triage_acuity: int = Field(
        ...,
        ge=1,
        le=5,
        description="Forecasted Emergency Severity Index (ESI levels 1 to 5).",
    )

    # Predicted Physiologically Coupled Vitals
    heart_rate: float = Field(
        ...,
        ge=30.0,
        le=220.0,
        description="Predicted heart rate in beats per minute (30-220 bpm).",
    )
    resp_rate: float = Field(
        ...,
        ge=6.0,
        le=60.0,
        description="Predicted respiratory rate in breaths per minute (6-60 breaths/min).",
    )
    temp_c: float = Field(
        ...,
        ge=33.0,
        le=43.0,
        description="Predicted body temperature in degrees Celsius (33-43 C).",
    )
    o2_sat: float = Field(
        ...,
        ge=60.0,
        le=100.0,
        description="Predicted peripheral capillary oxygen saturation percentage (60-100%).",
    )
    sbp: float = Field(
        ...,
        ge=60.0,
        le=240.0,
        description="Predicted systolic blood pressure in mmHg (60-240 mmHg).",
    )
    dbp: float = Field(
        ...,
        ge=30.0,
        le=140.0,
        description="Predicted diastolic blood pressure in mmHg (30-140 mmHg).",
    )

    @field_validator("gender", mode="before")
    @classmethod
    def normalize_gender(cls, val: Any) -> str:
        """Normalize varied gender string representations to Female or Male."""
        if isinstance(val, str):
            token = val.strip().lower()
            if token in {"female", "f", "woman"}:
                return "Female"
            if token in {"male", "m", "man"}:
                return "Male"
        raise ValueError(
            f"Invalid gender value {val!r}. Expected representation of 'Female' or 'Male'."
        )

    @field_validator("chief_complaint", mode="before")
    @classmethod
    def normalize_chief_complaint(cls, val: Any) -> str:
        """Validate and normalize chief complaint against 10 primary complaints."""
        if not isinstance(val, str) or not val.strip():
            raise ValueError("Field 'chief_complaint' must be a non-empty string.")
        token = val.strip().lower()
        lookup = {c.value.lower(): c.value for c in PrimaryChiefComplaint}
        # Common clinical aliases and variations
        lookup.update({
            "nausea and vomiting": PrimaryChiefComplaint.NAUSEA_VOMITING.value,
            "nausea": PrimaryChiefComplaint.NAUSEA_VOMITING.value,
            "vomiting": PrimaryChiefComplaint.NAUSEA_VOMITING.value,
            "sob": PrimaryChiefComplaint.SHORTNESS_OF_BREATH.value,
            "dyspnea": PrimaryChiefComplaint.SHORTNESS_OF_BREATH.value,
            "cp": PrimaryChiefComplaint.CHEST_PAIN.value,
            "altered mental status": "Altered Mental Status",
            "fall": "Fall",
        })
        if token in lookup:
            return lookup[token]
        allowed = ", ".join(repr(c) for c in PRIMARY_CHIEF_COMPLAINTS)
        raise ValueError(
            f"Invalid chief_complaint {val!r}. Expected one of primary complaints: [{allowed}]."
        )

    @model_validator(mode="after")
    def validate_physiological_invariants(self) -> Self:
        """Enforce biological validity: pulse pressure (sbp - dbp) >= 15.0 mmHg."""
        pulse_pressure = self.sbp - self.dbp
        if pulse_pressure < MIN_PULSE_PRESSURE:
            raise ValueError(
                f"Physiological validity violated: Pulse pressure (sbp - dbp) must be >= "
                f"{MIN_PULSE_PRESSURE} mmHg. Calculated pulse pressure is {pulse_pressure:.2f} mmHg "
                f"(sbp={self.sbp:.1f}, dbp={self.dbp:.1f})."
            )
        return self

    @property
    def pulse_pressure(self) -> float:
        """Calculated pulse pressure (SBP - DBP) in mmHg."""
        return self.sbp - self.dbp


__all__ = [
    "CareUnitType",
    "ClinicalTrajectoryPrediction",
    "DispositionType",
    "HourlyStaffingPrediction",
    "PatientClinicalPrediction",
    "PrimaryChiefComplaint",
    "PRIMARY_CHIEF_COMPLAINTS",
    "ResourceUtilizationForecast",
    "ShiftType",
]```

## File: backend/src/ml/model_pipeline.py
```python
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
]```

## File: backend/tests/test_predictions.py
```python
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

from schemas.contracts import (
    CareUnitType,
    DispositionType,
    ShiftType,
    TIMESTAMP_FORMAT,
    MIN_PULSE_PRESSURE,
    _parse_strict_timestamp,
)
from schemas.prediction_contracts import (
    ClinicalTrajectoryPrediction,
    HourlyStaffingPrediction,
    PatientClinicalPrediction,
    PrimaryChiefComplaint,
    PRIMARY_CHIEF_COMPLAINTS,
    ResourceUtilizationForecast,
)
from ml.model_pipeline import (
    PredictiveModelPipeline,
    FallbackHospitalSimulationEngine,
    FallbackMIMICDistributionSampler,
    CHIEF_COMPLAINT_WEIGHTS,
    ESI_DISTRIBUTION_WEIGHTS,
)
from api.predictions import (
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
        assert resp_empty_trajs.status_code == 422```

## File: backend/src/api/predictions.py
```python
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
]```

## File: frontend/src/api/useForecastData.ts
```typescript
// frontend/src/api/useForecastData.ts
/**
 * React custom hook and TypeScript contracts for hospital digital twin simulation telemetry.
 * Connects to the FastAPI backend to fetch, execute, and reset discrete-event simulations.
 */

import { useState, useEffect, useCallback } from "react";

// ---------------------------------------------------------------------------
// Clinical & Operational Enumerations
// ---------------------------------------------------------------------------

export type ShiftType = "Day" | "Evening" | "Night";

export type DispositionType = "ED_Discharge" | "Ward" | "ICU";

export type CareUnitType = "ED_Only" | "Ward" | "ICU";

export type BedStatus =
  | "Occupied"
  | "Reserved"
  | "Pending Discharge"
  | "Awaiting Cleaning"
  | "Out of Service";

// ---------------------------------------------------------------------------
// 1. Configuration & Metrics Interfaces
// ---------------------------------------------------------------------------

export interface SimulationRunRequest {
  duration_hours?: number;
  ed_capacity?: number;
  ward_capacity?: number;
  icu_capacity?: number;
  seed?: number;
  start_datetime?: string;
}

export interface SimulationMetricsResponse {
  total_arrivals: number;
  total_admissions: number;
  total_discharges: number;
  completed_stays_count: number;
  peak_queue_length: number;
  average_los_hours: number;
  ed_utilization_pct: number;
  ward_utilization_pct: number;
  icu_utilization_pct: number;
}

// ---------------------------------------------------------------------------
// 2. Domain Contracts & Response Envelope
// ---------------------------------------------------------------------------

export interface HourlyCensusContract {
  timestamp: string;
  hour: number;
  day_of_week: string;
  is_weekend: number;
  shift_id: ShiftType;
  ed_occupancy: number;
  ward_occupancy: number;
  icu_occupancy: number;
  active_nurses: number;
  active_doctors: number;
  arrivals_count: number;
  admissions_count: number;
  discharges_count: number;
  patients_in_queue: number;
  incoming_arrivals_next_4h: number;
}

export interface BedTopologyContract {
  bed_id: string;
  department: CareUnitType;
  room_id: string;
  status: BedStatus;
  is_negative_pressure: boolean;
  has_ventilator: boolean;
  current_stay_id: number | null;
}

export interface PatientStayContract {
  stay_id: number;
  patient_id: number;
  age: number;
  gender: "Female" | "Male";
  charlson_index: number;
  heart_rate: number;
  sbp: number;
  dbp: number;
  o2_sat: number;
  resp_rate: number;
  temp_c: number;
  triage_acuity: number;
  chief_complaint: string;
  arrival_time: string;
  triage_start_time: string;
  bed_assigned_time: string;
  discharge_time: string;
  arrival_hour: string;
  los_hours: number;
  disposition: DispositionType;
  icu_transfer_flag: number;
  initial_care_unit: CareUnitType;
  requires_ventilation: number;
}

export interface SimulationDataResponse {
  status: string;
  generated_at: string;
  config: SimulationRunRequest;
  metrics: SimulationMetricsResponse;
  hourly_census: HourlyCensusContract[];
  bed_topology: BedTopologyContract[];
  stays: PatientStayContract[];
}

// ---------------------------------------------------------------------------
// Hook Return & Options Interfaces
// ---------------------------------------------------------------------------

export interface UseForecastDataOptions {
  baseUrl?: string;
  autoFetch?: boolean;
  initialIncludeStays?: boolean;
}

export interface UseForecastDataResult {
  data: SimulationDataResponse | null;
  isLoading: boolean;
  error: string | null;
  fetchSimulationData: (includeStays?: boolean) => Promise<SimulationDataResponse | null>;
  runSimulation: (config: SimulationRunRequest) => Promise<SimulationDataResponse | null>;
  resetSimulation: () => Promise<SimulationDataResponse | null>;
}

// ---------------------------------------------------------------------------
// Custom Hook: useForecastData
// ---------------------------------------------------------------------------

export function useForecastData(options?: UseForecastDataOptions): UseForecastDataResult {
  const baseUrl = options?.baseUrl ?? "";
  const autoFetch = options?.autoFetch ?? true;
  const initialIncludeStays = options?.initialIncludeStays ?? true;

  const [data, setData] = useState<SimulationDataResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * Helper to parse API error responses into clean messages.
   */
  const parseErrorMessage = async (response: Response): Promise<string> => {
    try {
      const errorJson = await response.json();
      if (errorJson && typeof errorJson.detail === "string") {
        return errorJson.detail;
      }
      if (errorJson && Array.isArray(errorJson.detail)) {
        return errorJson.detail.map((err: { msg?: string }) => err.msg || JSON.stringify(err)).join("; ");
      }
      return JSON.stringify(errorJson);
    } catch {
      return `HTTP error ${response.status}: ${response.statusText}`;
    }
  };

  /**
   * Fetch current simulation telemetry and records from GET /api/simulation/data
   */
  const fetchSimulationData = useCallback(
    async (includeStays: boolean = true): Promise<SimulationDataResponse | null> => {
      setIsLoading(true);
      setError(null);
      try {
        const url = `${baseUrl}/api/simulation/data?include_stays=${includeStays}`;
        const response = await fetch(url, {
          method: "GET",
          headers: {
            Accept: "application/json",
          },
        });

        if (!response.ok) {
          const detail = await parseErrorMessage(response);
          throw new Error(detail);
        }

        const payload: SimulationDataResponse = await response.json();
        setData(payload);
        return payload;
      } catch (err: unknown) {
        const errorMsg = err instanceof Error ? err.message : "Failed to fetch simulation data";
        setError(errorMsg);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [baseUrl]
  );

  /**
   * Execute a discrete-event simulation run using POST /api/simulation/run
   */
  const runSimulation = useCallback(
    async (config: SimulationRunRequest): Promise<SimulationDataResponse | null> => {
      setIsLoading(true);
      setError(null);
      try {
        const url = `${baseUrl}/api/simulation/run`;
        const response = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify(config),
        });

        if (!response.ok) {
          const detail = await parseErrorMessage(response);
          throw new Error(detail);
        }

        const payload: SimulationDataResponse = await response.json();
        setData(payload);
        return payload;
      } catch (err: unknown) {
        const errorMsg = err instanceof Error ? err.message : "Failed to execute simulation run";
        setError(errorMsg);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [baseUrl]
  );

  /**
   * Reset simulation state to default parameters via POST /api/simulation/reset
   */
  const resetSimulation = useCallback(async (): Promise<SimulationDataResponse | null> => {
    setIsLoading(true);
    setError(null);
    try {
      const url = `${baseUrl}/api/simulation/reset`;
      const response = await fetch(url, {
        method: "POST",
        headers: {
          Accept: "application/json",
        },
      });

      if (!response.ok) {
        const detail = await parseErrorMessage(response);
        throw new Error(detail);
      }

      const payload: SimulationDataResponse = await response.json();
      setData(payload);
      return payload;
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : "Failed to reset simulation";
      setError(errorMsg);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [baseUrl]);

  /**
   * Auto-fetch initial telemetry snapshot on mount
   */
  useEffect(() => {
    let isMounted = true;

    if (autoFetch) {
      fetchSimulationData(initialIncludeStays).then((res) => {
        if (!isMounted && res) {
          // Component unmounted during initial load
        }
      });
    }

    return () => {
      isMounted = false;
    };
  }, [autoFetch, fetchSimulationData, initialIncludeStays]);

  return {
    data,
    isLoading,
    error,
    fetchSimulationData,
    runSimulation,
    resetSimulation,
  };
}

export default useForecastData;```

## File: frontend/src/pages/ForecastDashboard.tsx
```typescript
// frontend/src/pages/ForecastDashboard.tsx
/**
 * Real-time operational command center and predictive intelligence dashboard.
 * Consumes telemetry and machine learning inferences from useForecastData,
 * enforcing biological and temporal invariants across five clinical UI zones.
 */

import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  useForecastData,
  type SimulationDataResponse,
  type PatientStayContract,
  type HourlyCensusContract,
  type ShiftType,
  type CareUnitType,
} from "../api/useForecastData";

// ---------------------------------------------------------------------------
// Constants & Baselines
// ---------------------------------------------------------------------------
const MIN_PULSE_PRESSURE = 15.0;

const HISTORICAL_TARGETS = {
  average_los: 24.0,
  ed_los: 4.2,
  ward_los: 48.0,
  icu_los: 72.0,
};

const SHIFT_BASELINES: Record<ShiftType, { nurses: number; doctors: number }> = {
  Day: { nurses: 35, doctors: 10 },
  Evening: { nurses: 30, doctors: 8 },
  Night: { nurses: 20, doctors: 5 },
};

// ---------------------------------------------------------------------------
// Helper Utilities
// ---------------------------------------------------------------------------
function getUtilizationColor(pct: number): { bg: string; text: string; label: string } {
  if (pct >= 90.0) {
    return { bg: "#dc2626", text: "#fef2f2", label: "Critical" };
  }
  if (pct >= 80.0) {
    return { bg: "#d97706", text: "#fffbeb", label: "Warning" };
  }
  return { bg: "#16a34a", text: "#f0fdf4", label: "Optimal" };
}

function getEsiBadgeStyle(acuity: number): { bg: string; text: string } {
  switch (acuity) {
    case 1:
      return { bg: "#dc2626", text: "#ffffff" }; // Resuscitation (Red)
    case 2:
      return { bg: "#ea580c", text: "#ffffff" }; // Emergent (Orange)
    case 3:
      return { bg: "#ca8a04", text: "#ffffff" }; // Urgent (Yellow)
    case 4:
      return { bg: "#2563eb", text: "#ffffff" }; // Less Urgent (Blue)
    case 5:
    default:
      return { bg: "#16a34a", text: "#ffffff" }; // Non-Urgent (Green)
  }
}

// ---------------------------------------------------------------------------
// 5. Asynchronous Data States: Skeletons & Latency Placeholders
// ---------------------------------------------------------------------------
interface SkeletonBlockProps {
  height?: string;
  width?: string;
  className?: string;
  style?: React.CSSProperties;
}

const SkeletonBlock: React.FC<SkeletonBlockProps> = ({
  height = "1.5rem",
  width = "100%",
  className = "",
  style,
}) => (
  <div
    className={`animate-pulse ${className}`}
    style={{
      height,
      width,
      backgroundColor: "#e2e8f0",
      borderRadius: "0.375rem",
      ...style,
    }}
  />
);

const DashboardSkeleton: React.FC = () => (
  <div style={{ padding: "1.5rem", maxWidth: "1600px", margin: "0 auto", color: "#1e293b" }}>
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "1.5rem" }}>
      <SkeletonBlock height="2.5rem" width="300px" />
      <SkeletonBlock height="2.5rem" width="220px" />
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem", marginBottom: "1.5rem" }}>
      <SkeletonBlock height="7rem" />
      <SkeletonBlock height="7rem" />
      <SkeletonBlock height="7rem" />
      <SkeletonBlock height="7rem" />
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem", marginBottom: "1.5rem" }}>
      <SkeletonBlock height="18rem" />
      <SkeletonBlock height="18rem" />
    </div>
    <SkeletonBlock height="20rem" style={{ marginBottom: "1.5rem" }} />
    <SkeletonBlock height="24rem" />
  </div>
);

// ---------------------------------------------------------------------------
// 1. Executive Prescriptive KPI Strip
// ---------------------------------------------------------------------------
interface KpiStripProps {
  metrics: SimulationDataResponse["metrics"];
  hourlyCensus: HourlyCensusContract[];
  stays: PatientStayContract[];
}

const ExecutiveKpiStrip: React.FC<KpiStripProps> = ({ metrics, hourlyCensus, stays }) => {
  // 1. Rolling 4-Hour Prescriptive Arrival Gauge
  const latestCensus = hourlyCensus[0];
  const projected4h = latestCensus?.incoming_arrivals_next_4h ?? 0;
  const currentArrivals = latestCensus?.arrivals_count ?? 0;
  const projectedHourlyAverage = projected4h / 4.0;
  const isUpwardTrend = projectedHourlyAverage >= currentArrivals;

  // 2. Departmental Length of Stay & Delta
  const edStays = stays.filter((s) => s.disposition === "ED_Discharge");
  const wardStays = stays.filter((s) => s.disposition === "Ward");

  const avgEdLos = edStays.length > 0
    ? edStays.reduce((acc, curr) => acc + curr.los_hours, 0) / edStays.length
    : HISTORICAL_TARGETS.ed_los;

  const avgWardLos = wardStays.length > 0
    ? wardStays.reduce((acc, curr) => acc + curr.los_hours, 0) / wardStays.length
    : HISTORICAL_TARGETS.ward_los;

  const overallLosDelta = metrics.average_los_hours - HISTORICAL_TARGETS.average_los;

  // 3. Peak Queue Bottleneck & Time-To-Peak
  let peakQueueCount = metrics.peak_queue_length;
  let peakQueueTime = "Current Shift";
  if (hourlyCensus.length > 0) {
    const peakRecord = hourlyCensus.reduce((prev, curr) =>
      curr.patients_in_queue > prev.patients_in_queue ? curr : prev
    );
    peakQueueCount = Math.max(peakQueueCount, peakRecord.patients_in_queue);
    peakQueueTime = `${peakRecord.hour.toString().padStart(2, "0")}:00 hrs`;
  }

  // 4. Active Escalations Badge (ICU transfer flagged or requires mechanical ventilation)
  const activeEscalationsCount = stays.filter(
    (s) => s.icu_transfer_flag === 1 || s.requires_ventilation === 1
  ).length;

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
        gap: "1rem",
        marginBottom: "1.5rem",
      }}
    >
      {/* 4h Prescriptive Arrivals */}
      <div
        style={{
          backgroundColor: "#ffffff",
          borderRadius: "0.5rem",
          padding: "1rem 1.25rem",
          border: "1px solid #e2e8f0",
          boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
        }}
      >
        <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "#64748b", textTransform: "uppercase" }}>
          Prescriptive Influx (4-Hour)
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem", marginTop: "0.25rem" }}>
          <span style={{ fontSize: "1.75rem", fontWeight: 700, color: "#0f172a" }}>
            +{Math.round(projected4h)}
          </span>
          <span
            style={{
              fontSize: "0.875rem",
              fontWeight: 600,
              color: isUpwardTrend ? "#dc2626" : "#16a34a",
            }}
          >
            {isUpwardTrend ? "[^] High Influx" : "[v] Stable Flow"}
          </span>
        </div>
        <div style={{ fontSize: "0.75rem", color: "#64748b", marginTop: "0.25rem" }}>
          Next 4h projected arrivals ({projectedHourlyAverage.toFixed(1)}/hr)
        </div>
      </div>

      {/* Average Length of Stay */}
      <div
        style={{
          backgroundColor: "#ffffff",
          borderRadius: "0.5rem",
          padding: "1rem 1.25rem",
          border: "1px solid #e2e8f0",
          boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
        }}
      >
        <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "#64748b", textTransform: "uppercase" }}>
          Projected Mean LoS
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem", marginTop: "0.25rem" }}>
          <span style={{ fontSize: "1.75rem", fontWeight: 700, color: "#0f172a" }}>
            {metrics.average_los_hours.toFixed(1)}h
          </span>
          <span
            style={{
              fontSize: "0.75rem",
              fontWeight: 600,
              padding: "0.15rem 0.4rem",
              borderRadius: "0.25rem",
              backgroundColor: overallLosDelta > 0 ? "#fef2f2" : "#f0fdf4",
              color: overallLosDelta > 0 ? "#b91c1c" : "#15803d",
            }}
          >
            {overallLosDelta >= 0 ? `+${overallLosDelta.toFixed(1)}h` : `${overallLosDelta.toFixed(1)}h`} vs Target
          </span>
        </div>
        <div style={{ fontSize: "0.75rem", color: "#64748b", marginTop: "0.25rem" }}>
          ED: {avgEdLos.toFixed(1)}h | Med-Surg: {avgWardLos.toFixed(1)}h
        </div>
      </div>

      {/* Peak Queue Bottleneck */}
      <div
        style={{
          backgroundColor: "#ffffff",
          borderRadius: "0.5rem",
          padding: "1rem 1.25rem",
          border: "1px solid #e2e8f0",
          boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
        }}
      >
        <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "#64748b", textTransform: "uppercase" }}>
          Bottleneck Peak Queue
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem", marginTop: "0.25rem" }}>
          <span style={{ fontSize: "1.75rem", fontWeight: 700, color: "#0f172a" }}>
            {peakQueueCount}
          </span>
          <span style={{ fontSize: "0.875rem", fontWeight: 500, color: "#475569" }}>
            patients waiting
          </span>
        </div>
        <div style={{ fontSize: "0.75rem", color: "#64748b", marginTop: "0.25rem" }}>
          Anticipated peak at {peakQueueTime}
        </div>
      </div>

      {/* Active Escalations Alert Badge */}
      <div
        style={{
          backgroundColor: activeEscalationsCount > 0 ? "#fef2f2" : "#ffffff",
          borderRadius: "0.5rem",
          padding: "1rem 1.25rem",
          border: activeEscalationsCount > 0 ? "1px solid #fecaca" : "1px solid #e2e8f0",
          boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
        }}
      >
        <div style={{ fontSize: "0.75rem", fontWeight: 600, color: activeEscalationsCount > 0 ? "#991b1b" : "#64748b", textTransform: "uppercase" }}>
          Escalation Watchlist
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem", marginTop: "0.25rem" }}>
          <span
            style={{
              fontSize: "1.75rem",
              fontWeight: 700,
              color: activeEscalationsCount > 0 ? "#dc2626" : "#0f172a",
            }}
          >
            {activeEscalationsCount}
          </span>
          <span
            style={{
              fontSize: "0.75rem",
              fontWeight: 600,
              padding: "0.15rem 0.4rem",
              borderRadius: "0.25rem",
              backgroundColor: activeEscalationsCount > 0 ? "#dc2626" : "#e2e8f0",
              color: activeEscalationsCount > 0 ? "#ffffff" : "#475569",
            }}
          >
            {activeEscalationsCount > 0 ? "CRITICAL ALERT" : "NORMAL"}
          </span>
        </div>
        <div style={{ fontSize: "0.75rem", color: activeEscalationsCount > 0 ? "#b91c1c" : "#64748b", marginTop: "0.25rem" }}>
          Predicted ICU transfer or invasive ventilation
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// 2. Resource Utilization & Bottleneck Analytics
// ---------------------------------------------------------------------------
interface ResourceAnalyticsProps {
  metrics: SimulationDataResponse["metrics"];
  hourlyCensus: HourlyCensusContract[];
}

const ResourceUtilizationSection: React.FC<ResourceAnalyticsProps> = ({ metrics, hourlyCensus }) => {
  const units: Array<{ name: string; pct: number; code: CareUnitType }> = [
    { name: "Emergency Department (ED)", pct: metrics.ed_utilization_pct, code: "ED_Only" },
    { name: "Med-Surg General Ward", pct: metrics.ward_utilization_pct, code: "Ward" },
    { name: "Intensive Care Unit (ICU)", pct: metrics.icu_utilization_pct, code: "ICU" },
  ];

  // Derive maximum arrivals or discharges for dual-axis chart normalization
  const maxVolume = useMemo(() => {
    if (hourlyCensus.length === 0) return 20;
    const maxVal = Math.max(
      ...hourlyCensus.map((h) => Math.max(h.arrivals_count, h.discharges_count, h.patients_in_queue))
    );
    return Math.max(maxVal, 10);
  }, [hourlyCensus]);

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1.6fr",
        gap: "1.5rem",
        marginBottom: "1.5rem",
      }}
    >
      {/* Bed Utilization Capacity Gauges */}
      <div
        style={{
          backgroundColor: "#ffffff",
          borderRadius: "0.5rem",
          padding: "1.25rem",
          border: "1px solid #e2e8f0",
        }}
      >
        <h3 style={{ fontSize: "1rem", fontWeight: 700, margin: "0 0 1rem 0", color: "#0f172a" }}>
          Department Bed Utilization Gauges
        </h3>
        <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
          {units.map((unit) => {
            const status = getUtilizationColor(unit.pct);
            return (
              <div key={unit.code}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.35rem" }}>
                  <span style={{ fontSize: "0.875rem", fontWeight: 600, color: "#334155" }}>
                    {unit.name}
                  </span>
                  <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                    <span
                      style={{
                        fontSize: "0.7rem",
                        fontWeight: 600,
                        padding: "0.1rem 0.4rem",
                        borderRadius: "0.25rem",
                        backgroundColor: status.bg,
                        color: status.text,
                      }}
                    >
                      {status.label}
                    </span>
                    <span style={{ fontSize: "0.875rem", fontWeight: 700, color: "#0f172a" }}>
                      {unit.pct.toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div
                  style={{
                    height: "0.75rem",
                    width: "100%",
                    backgroundColor: "#e2e8f0",
                    borderRadius: "0.375rem",
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width: `${Math.min(100, Math.max(0, unit.pct))}%`,
                      backgroundColor: status.bg,
                      transition: "width 0.4s ease-in-out",
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
        <div style={{ marginTop: "1.25rem", fontSize: "0.75rem", color: "#64748b" }}>
          Thresholds: Optimal (&lt;80%) | Warning (80-90%) | Critical Alert (&gt;90%)
        </div>
      </div>

      {/* Hourly Patient Flow Dual-Axis Chart Area */}
      <div
        style={{
          backgroundColor: "#ffffff",
          borderRadius: "0.5rem",
          padding: "1.25rem",
          border: "1px solid #e2e8f0",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
          <h3 style={{ fontSize: "1rem", fontWeight: 700, margin: 0, color: "#0f172a" }}>
            24h Projected Patient Influx vs. Discharges
          </h3>
          <div style={{ display: "flex", gap: "1rem", fontSize: "0.75rem" }}>
            <span style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
              <span style={{ width: "10px", height: "10px", backgroundColor: "#2563eb", borderRadius: "2px" }} />
              Arrivals
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
              <span style={{ width: "10px", height: "10px", backgroundColor: "#16a34a", borderRadius: "2px" }} />
              Discharges
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
              <span style={{ width: "10px", height: "10px", backgroundColor: "#ea580c", borderRadius: "2px" }} />
              Waiting Queue
            </span>
          </div>
        </div>

        {/* Visual Dual-Axis Timeline representation */}
        <div
          style={{
            height: "170px",
            display: "flex",
            alignItems: "flex-end",
            gap: "4px",
            paddingTop: "1rem",
            borderBottom: "1px solid #cbd5e1",
            position: "relative",
          }}
        >
          {hourlyCensus.slice(0, 24).map((h, idx) => {
            const arrHeight = (h.arrivals_count / maxVolume) * 130;
            const disHeight = (h.discharges_count / maxVolume) * 130;
            const qHeight = (h.patients_in_queue / maxVolume) * 130;
            const isShiftChange = h.hour === 7 || h.hour === 15 || h.hour === 23;

            return (
              <div
                key={idx}
                style={{
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  height: "100%",
                  justifyContent: "flex-end",
                  position: "relative",
                }}
                title={`Hour ${h.hour}:00 | Arr: ${h.arrivals_count} | Dis: ${h.discharges_count} | Queue: ${h.patients_in_queue}`}
              >
                {isShiftChange && (
                  <div
                    style={{
                      position: "absolute",
                      top: 0,
                      bottom: 0,
                      width: "1px",
                      backgroundColor: "#94a3b8",
                      borderStyle: "dashed",
                    }}
                  />
                )}
                <div style={{ display: "flex", alignItems: "flex-end", gap: "1px", width: "100%" }}>
                  <div style={{ width: "33%", height: `${arrHeight}px`, backgroundColor: "#2563eb" }} />
                  <div style={{ width: "33%", height: `${disHeight}px`, backgroundColor: "#16a34a" }} />
                  <div style={{ width: "33%", height: `${qHeight}px`, backgroundColor: "#ea580c" }} />
                </div>
              </div>
            );
          })}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: "0.5rem", fontSize: "0.7rem", color: "#64748b" }}>
          <span>00:00 (Night)</span>
          <span>07:00 (Day Shift Shiftover)</span>
          <span>15:00 (Evening Shiftover)</span>
          <span>23:00 (Night Shiftover)</span>
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// 3. Hourly Census & Shift-Staffing Matrix
// ---------------------------------------------------------------------------
interface ShiftStaffingProps {
  hourlyCensus: HourlyCensusContract[];
}

const ShiftStaffingMatrix: React.FC<ShiftStaffingProps> = ({ hourlyCensus }) => {
  const shiftGroups = useMemo(() => {
    const shifts: Record<ShiftType, HourlyCensusContract[]> = {
      Day: [],
      Evening: [],
      Night: [],
    };

    hourlyCensus.forEach((record) => {
      if (shifts[record.shift_id]) {
        shifts[record.shift_id].push(record);
      }
    });

    return shifts;
  }, [hourlyCensus]);

  const summary = (["Day", "Evening", "Night"] as ShiftType[]).map((shiftKey) => {
    const records = shiftGroups[shiftKey];
    const baseline = SHIFT_BASELINES[shiftKey];

    if (records.length === 0) {
      return {
        shift: shiftKey,
        meanCensus: 0,
        activeNurses: baseline.nurses,
        activeDoctors: baseline.doctors,
        targetNurses: baseline.nurses,
        targetDoctors: baseline.doctors,
        nurseGap: 0,
        safeRatioViolated: false,
      };
    }

    const totalCensus = records.reduce(
      (sum, r) => sum + r.ed_occupancy + r.ward_occupancy + r.icu_occupancy,
      0
    );
    const meanCensus = Math.round(totalCensus / records.length);

    const avgNurses = Math.round(
      records.reduce((sum, r) => sum + r.active_nurses, 0) / records.length
    );
    const avgDoctors = Math.round(
      records.reduce((sum, r) => sum + r.active_doctors, 0) / records.length
    );

    // Dynamic staffing target adjustment based on occupancy volume
    const targetNurses = baseline.nurses;
    const targetDoctors = baseline.doctors;
    const nurseGap = avgNurses - targetNurses;

    // Safe ratio check: if total hospital census to active nurses exceeds 7:1
    const safeRatioViolated = meanCensus / Math.max(1, avgNurses) > 6.5;

    return {
      shift: shiftKey,
      meanCensus,
      activeNurses: avgNurses,
      activeDoctors: avgDoctors,
      targetNurses,
      targetDoctors,
      nurseGap,
      safeRatioViolated,
    };
  });

  return (
    <div
      style={{
        backgroundColor: "#ffffff",
        borderRadius: "0.5rem",
        padding: "1.25rem",
        border: "1px solid #e2e8f0",
        marginBottom: "1.5rem",
      }}
    >
      <h3 style={{ fontSize: "1rem", fontWeight: 700, margin: "0 0 1rem 0", color: "#0f172a" }}>
        Shift-Staffing Alignment Matrix & Ratios
      </h3>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.875rem", textAlign: "left" }}>
          <thead>
            <tr style={{ borderBottom: "2px solid #e2e8f0", color: "#475569" }}>
              <th style={{ padding: "0.6rem" }}>Shift Window</th>
              <th style={{ padding: "0.6rem" }}>Hours (Active)</th>
              <th style={{ padding: "0.6rem" }}>Mean Census</th>
              <th style={{ padding: "0.6rem" }}>Target RN / Active</th>
              <th style={{ padding: "0.6rem" }}>Target MD / Active</th>
              <th style={{ padding: "0.6rem" }}>Staffing Delta / Status</th>
              <th style={{ padding: "0.6rem" }}>Ratio Safety Feedback</th>
            </tr>
          </thead>
          <tbody>
            {summary.map((row) => {
              const hoursWindow =
                row.shift === "Day"
                  ? "07:00 - 15:00"
                  : row.shift === "Evening"
                  ? "15:00 - 23:00"
                  : "23:00 - 07:00";

              return (
                <tr key={row.shift} style={{ borderBottom: "1px solid #f1f5f9" }}>
                  <td style={{ padding: "0.6rem", fontWeight: 600, color: "#1e293b" }}>{row.shift}</td>
                  <td style={{ padding: "0.6rem", color: "#64748b" }}>{hoursWindow}</td>
                  <td style={{ padding: "0.6rem", fontWeight: 600 }}>{row.meanCensus} pts</td>
                  <td style={{ padding: "0.6rem" }}>
                    {row.targetNurses} / <strong>{row.activeNurses} RNs</strong>
                  </td>
                  <td style={{ padding: "0.6rem" }}>
                    {row.targetDoctors} / <strong>{row.activeDoctors} MDs</strong>
                  </td>
                  <td style={{ padding: "0.6rem" }}>
                    <span
                      style={{
                        padding: "0.2rem 0.5rem",
                        borderRadius: "0.25rem",
                        fontSize: "0.75rem",
                        fontWeight: 600,
                        backgroundColor:
                          row.nurseGap < 0 ? "#fef2f2" : row.nurseGap > 0 ? "#f0fdf4" : "#f8fafc",
                        color:
                          row.nurseGap < 0 ? "#b91c1c" : row.nurseGap > 0 ? "#15803d" : "#475569",
                      }}
                    >
                      {row.nurseGap > 0
                        ? `+${row.nurseGap} RN (Overstaffed)`
                        : row.nurseGap < 0
                        ? `${row.nurseGap} RNs (Understaffed Alert)`
                        : "Balanced"}
                    </span>
                  </td>
                  <td style={{ padding: "0.6rem" }}>
                    {row.safeRatioViolated ? (
                      <span style={{ color: "#dc2626", fontWeight: 700, fontSize: "0.75rem" }}>
                        [!] Ratio Breach Warning (&gt;6.5:1)
                      </span>
                    ) : (
                      <span style={{ color: "#16a34a", fontWeight: 600, fontSize: "0.75rem" }}>
                        Safe Workforce Baselines Maintained
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// 4. Patient Flow Trajectories & Clinical Escalation Radar
// ---------------------------------------------------------------------------
interface ClinicalRadarProps {
  stays: PatientStayContract[];
}

const ClinicalEscalationRadar: React.FC<ClinicalRadarProps> = ({ stays }) => {
  // Disposition split calculation
  const dispositionBreakdown = useMemo(() => {
    if (stays.length === 0) {
      return { edDischarge: 0, wardAdmit: 0, icuAdmit: 0 };
    }
    const edDischarge = (stays.filter((s) => s.disposition === "ED_Discharge").length / stays.length) * 100;
    const wardAdmit = (stays.filter((s) => s.disposition === "Ward").length / stays.length) * 100;
    const icuAdmit = (stays.filter((s) => s.disposition === "ICU").length / stays.length) * 100;

    return {
      edDischarge: Math.round(edDischarge),
      wardAdmit: Math.round(wardAdmit),
      icuAdmit: Math.round(icuAdmit),
    };
  }, [stays]);

  // Prioritize high acuity, ICU transfer flags, or ventilation requirement
  const prioritizedWatchlist = useMemo(() => {
    return [...stays]
      .sort((a, b) => {
        // High risk sort: ventilation flag -> ICU transfer flag -> Acuity (lower is more acute)
        if (b.requires_ventilation !== a.requires_ventilation) {
          return b.requires_ventilation - a.requires_ventilation;
        }
        if (b.icu_transfer_flag !== a.icu_transfer_flag) {
          return b.icu_transfer_flag - a.icu_transfer_flag;
        }
        return a.triage_acuity - b.triage_acuity;
      })
      .slice(0, 8);
  }, [stays]);

  return (
    <div
      style={{
        backgroundColor: "#ffffff",
        borderRadius: "0.5rem",
        padding: "1.25rem",
        border: "1px solid #e2e8f0",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
        <h3 style={{ fontSize: "1rem", fontWeight: 700, margin: 0, color: "#0f172a" }}>
          Patient Flow Trajectories & Clinical Escalation Radar
        </h3>
        {/* Trajectory percentage split indicator */}
        <div style={{ display: "flex", gap: "1rem", fontSize: "0.8rem", fontWeight: 600 }}>
          <span style={{ color: "#16a34a" }}>ED Discharge: {dispositionBreakdown.edDischarge}%</span>
          <span style={{ color: "#2563eb" }}>Med-Surg Ward: {dispositionBreakdown.wardAdmit}%</span>
          <span style={{ color: "#dc2626" }}>Direct ICU: {dispositionBreakdown.icuAdmit}%</span>
        </div>
      </div>

      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.875rem", textAlign: "left" }}>
          <thead>
            <tr style={{ borderBottom: "2px solid #e2e8f0", color: "#475569" }}>
              <th style={{ padding: "0.6rem" }}>Stay / Patient</th>
              <th style={{ padding: "0.6rem" }}>Acuity (ESI)</th>
              <th style={{ padding: "0.6rem" }}>Comorbidity & Demographics</th>
              <th style={{ padding: "0.6rem" }}>Complaint / Path</th>
              <th style={{ padding: "0.6rem" }}>Escalation Flags</th>
              <th style={{ padding: "0.6rem" }}>Physiological Vitals Trajectory</th>
              <th style={{ padding: "0.6rem" }}>Invariant Verification</th>
            </tr>
          </thead>
          <tbody>
            {prioritizedWatchlist.map((stay) => {
              const esiStyle = getEsiBadgeStyle(stay.triage_acuity);
              const pulsePressure = stay.sbp - stay.dbp;
              const isPulsePressureValid = pulsePressure >= MIN_PULSE_PRESSURE;

              return (
                <tr
                  key={stay.stay_id}
                  style={{
                    borderBottom: "1px solid #f1f5f9",
                    backgroundColor: stay.icu_transfer_flag === 1 || stay.requires_ventilation === 1 ? "#fffbfb" : "#ffffff",
                  }}
                >
                  {/* Identifiers */}
                  <td style={{ padding: "0.6rem", fontWeight: 600, color: "#1e293b" }}>
                    <div>#{stay.stay_id}</div>
                    <div style={{ fontSize: "0.75rem", color: "#64748b" }}>PT-{stay.patient_id}</div>
                  </td>

                  {/* Triage Acuity */}
                  <td style={{ padding: "0.6rem" }}>
                    <span
                      style={{
                        padding: "0.2rem 0.6rem",
                        borderRadius: "0.25rem",
                        fontSize: "0.75rem",
                        fontWeight: 700,
                        backgroundColor: esiStyle.bg,
                        color: esiStyle.text,
                      }}
                    >
                      ESI Level {stay.triage_acuity}
                    </span>
                  </td>

                  {/* Comorbidity & Demographics */}
                  <td style={{ padding: "0.6rem", color: "#334155" }}>
                    <div>{stay.age}yo {stay.gender}</div>
                    <div style={{ fontSize: "0.75rem", color: "#64748b" }}>Charlson CCI: {stay.charlson_index}/10</div>
                  </td>

                  {/* Complaint & Disposition */}
                  <td style={{ padding: "0.6rem" }}>
                    <div style={{ fontWeight: 500 }}>{stay.chief_complaint}</div>
                    <div style={{ fontSize: "0.75rem", color: "#2563eb", fontWeight: 600 }}>
                      -&gt; {stay.disposition} ({stay.initial_care_unit})
                    </div>
                  </td>

                  {/* Escalation Risk Flags */}
                  <td style={{ padding: "0.6rem" }}>
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                      {stay.icu_transfer_flag === 1 && (
                        <span
                          style={{
                            backgroundColor: "#fef2f2",
                            color: "#b91c1c",
                            fontSize: "0.7rem",
                            fontWeight: 700,
                            padding: "0.15rem 0.4rem",
                            borderRadius: "0.25rem",
                            border: "1px solid #fecaca",
                          }}
                        >
                          Ward -&gt; ICU Transfer Triggered
                        </span>
                      )}
                      {stay.requires_ventilation === 1 && (
                        <span
                          style={{
                            backgroundColor: "#eff6ff",
                            color: "#1d4ed8",
                            fontSize: "0.7rem",
                            fontWeight: 700,
                            padding: "0.15rem 0.4rem",
                            borderRadius: "0.25rem",
                            border: "1px solid #bfdbfe",
                          }}
                        >
                          Mechanical Ventilation
                        </span>
                      )}
                      {stay.icu_transfer_flag === 0 && stay.requires_ventilation === 0 && (
                        <span style={{ fontSize: "0.75rem", color: "#64748b" }}>Stable Path</span>
                      )}
                    </div>
                  </td>

                  {/* Vitals Panel */}
                  <td style={{ padding: "0.6rem", fontSize: "0.75rem", color: "#334155" }}>
                    <div>HR: {stay.heart_rate} bpm | SpO2: {stay.o2_sat}%</div>
                    <div>BP: {stay.sbp}/{stay.dbp} mmHg | RR: {stay.resp_rate}</div>
                  </td>

                  {/* Invariant Verification */}
                  <td style={{ padding: "0.6rem" }}>
                    {isPulsePressureValid ? (
                      <span
                        style={{
                          backgroundColor: "#f0fdf4",
                          color: "#166534",
                          fontSize: "0.7rem",
                          fontWeight: 700,
                          padding: "0.15rem 0.4rem",
                          borderRadius: "0.25rem",
                          border: "1px solid #bbf7d0",
                          display: "inline-block",
                        }}
                      >
                        [PP Valid: {pulsePressure.toFixed(1)} mmHg]
                      </span>
                    ) : (
                      <span
                        style={{
                          backgroundColor: "#fef2f2",
                          color: "#991b1b",
                          fontSize: "0.7rem",
                          fontWeight: 700,
                          padding: "0.15rem 0.4rem",
                          borderRadius: "0.25rem",
                          border: "1px solid #f87171",
                          display: "inline-block",
                        }}
                      >
                        Sensor Invariant Failure: Plausibility Check Failed
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Main Dashboard Page Component
// ---------------------------------------------------------------------------
export const ForecastDashboard: React.FC = () => {
  const { data, isLoading, error, fetchSimulationData, runSimulation, resetSimulation } =
    useForecastData({ autoFetch: true });

  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);

  // Background polling heartbeat setup (every 20 seconds)
  useEffect(() => {
    const interval = setInterval(async () => {
      setIsRefreshing(true);
      await fetchSimulationData(true);
      setIsRefreshing(false);
    }, 20000);

    return () => clearInterval(interval);
  }, [fetchSimulationData]);

  const handleManualRefresh = useCallback(async () => {
    setIsRefreshing(true);
    await fetchSimulationData(true);
    setIsRefreshing(false);
  }, [fetchSimulationData]);

  const handleTriggerRun = useCallback(async () => {
    setIsRefreshing(true);
    await runSimulation({
      duration_hours: 24,
      ed_capacity: 50,
      ward_capacity: 150,
      icu_capacity: 30,
      seed: 42,
    });
    setIsRefreshing(false);
  }, [runSimulation]);

  // Loading skeleton on initial cold load
  if (isLoading && !data) {
    return <DashboardSkeleton />;
  }

  // Error Banner State
  if (error && !data) {
    return (
      <div style={{ padding: "2rem", maxWidth: "1200px", margin: "0 auto", textAlign: "center" }}>
        <div
          style={{
            backgroundColor: "#fef2f2",
            border: "1px solid #f87171",
            color: "#991b1b",
            padding: "1.5rem",
            borderRadius: "0.5rem",
          }}
        >
          <h2 style={{ fontSize: "1.25rem", fontWeight: 700, margin: "0 0 0.5rem 0" }}>
            Operational Telemetry Stream Disconnected
          </h2>
          <p style={{ margin: "0 0 1rem 0" }}>{error}</p>
          <button
            onClick={handleManualRefresh}
            style={{
              backgroundColor: "#dc2626",
              color: "#ffffff",
              border: "none",
              padding: "0.5rem 1.25rem",
              borderRadius: "0.375rem",
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            Retry Telemetry Ingestion
          </button>
        </div>
      </div>
    );
  }

  if (!data) {
    return null;
  }

  return (
    <div
      style={{
        backgroundColor: "#f8fafc",
        minHeight: "100vh",
        padding: "1.5rem",
        color: "#0f172a",
        fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
      }}
    >
      <div style={{ maxWidth: "1600px", margin: "0 auto" }}>
        {/* Header with background pulse indicator and controls */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "1.5rem",
            flexWrap: "wrap",
            gap: "1rem",
          }}
        >
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <h1 style={{ fontSize: "1.5rem", fontWeight: 800, margin: 0, color: "#0f172a" }}>
                Hospital Digital Twin - Forecast & Telemetry Command Center
              </h1>
              {/* Background Revalidation Pulse */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.35rem",
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  padding: "0.2rem 0.5rem",
                  borderRadius: "9999px",
                  backgroundColor: isRefreshing ? "#fef3c7" : "#f0fdf4",
                  color: isRefreshing ? "#b45309" : "#166534",
                }}
              >
                <span
                  style={{
                    width: "8px",
                    height: "8px",
                    borderRadius: "50%",
                    backgroundColor: isRefreshing ? "#d97706" : "#22c55e",
                  }}
                />
                {isRefreshing ? "Live Sync Active..." : "Telemetry Synchronized"}
              </div>
            </div>
            <div style={{ fontSize: "0.8rem", color: "#64748b", marginTop: "0.25rem" }}>
              Snapshot generated: {data.generated_at} | SimClock Anchor: 2026-09-23 00:00:00
            </div>
          </div>

          <div style={{ display: "flex", gap: "0.75rem" }}>
            <button
              onClick={handleManualRefresh}
              disabled={isRefreshing}
              style={{
                backgroundColor: "#ffffff",
                border: "1px solid #cbd5e1",
                padding: "0.5rem 1rem",
                borderRadius: "0.375rem",
                fontSize: "0.875rem",
                fontWeight: 600,
                color: "#334155",
                cursor: isRefreshing ? "not-allowed" : "pointer",
              }}
            >
              Refresh Data
            </button>
            <button
              onClick={handleTriggerRun}
              disabled={isRefreshing}
              style={{
                backgroundColor: "#2563eb",
                border: "none",
                padding: "0.5rem 1rem",
                borderRadius: "0.375rem",
                fontSize: "0.875rem",
                fontWeight: 600,
                color: "#ffffff",
                cursor: isRefreshing ? "not-allowed" : "pointer",
              }}
            >
              Execute ML Forecast Run
            </button>
            <button
              onClick={resetSimulation}
              disabled={isRefreshing}
              style={{
                backgroundColor: "#ffffff",
                border: "1px solid #cbd5e1",
                padding: "0.5rem 1rem",
                borderRadius: "0.375rem",
                fontSize: "0.875rem",
                fontWeight: 600,
                color: "#dc2626",
                cursor: isRefreshing ? "not-allowed" : "pointer",
              }}
            >
              Reset Baseline
            </button>
          </div>
        </div>

        {/* 1. Executive Prescriptive KPI Strip */}
        <ExecutiveKpiStrip
          metrics={data.metrics}
          hourlyCensus={data.hourly_census}
          stays={data.stays}
        />

        {/* 2. Resource Utilization & Bottleneck Analytics */}
        <ResourceUtilizationSection
          metrics={data.metrics}
          hourlyCensus={data.hourly_census}
        />

        {/* 3. Hourly Census & Shift-Staffing Matrix */}
        <ShiftStaffingMatrix hourlyCensus={data.hourly_census} />

        {/* 4. Patient Flow Trajectories & Clinical Escalation Radar */}
        <ClinicalEscalationRadar stays={data.stays} />
      </div>
    </div>
  );
};

export default ForecastDashboard;```

