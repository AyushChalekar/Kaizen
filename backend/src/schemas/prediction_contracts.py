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
        from src.contracts import (
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
            from src.contracts import (
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
]