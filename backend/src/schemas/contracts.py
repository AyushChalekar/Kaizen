# backend/src/schemas/contracts.py
"""Domain contracts for the Digital Twin hospital simulation platform.

Provides validated Pydantic v2 models synchronized across MIMIC-IV sampling
distributions, SimPy discrete-event simulation engines, prescriptive solvers,
and real-time streaming interfaces.
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
# Constants & Date-Time Formatting
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
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


class BedStatus(str, Enum):
    """Physical asset state for topology and allocation dispatching."""

    OCCUPIED = "Occupied"
    RESERVED = "Reserved"
    PENDING_DISCHARGE = "Pending Discharge"
    AWAITING_CLEANING = "Awaiting Cleaning"
    OUT_OF_SERVICE = "Out of Service"


class StaffRole(str, Enum):
    """Clinical practitioner role designations."""

    PHYSICIAN = "Physician"
    CHARGE_NURSE = "Charge Nurse"
    REGISTERED_NURSE = "Registered Nurse"
    NURSING_ASSISTANT = "Nursing Assistant"


# ---------------------------------------------------------------------------
# PatientStayContract
# ---------------------------------------------------------------------------
class PatientStayContract(BaseModel):
    """Contract governing individual patient trajectory, hemodynamics, and timing."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )

    # Identifiers
    stay_id: int = Field(
        ...,
        ge=1,
        le=100_000,
        description="Unique stay identifier across simulation state.",
    )
    patient_id: int = Field(
        ...,
        ge=100_001,
        le=200_000,
        description="Master patient index identifier.",
    )

    # Demographics
    age: int = Field(
        ...,
        ge=18,
        le=95,
        description="Patient age in completed years.",
    )
    gender: Literal["Female", "Male"] = Field(
        ...,
        description="Normalized biological or administrative sex.",
    )
    charlson_index: int = Field(
        ...,
        ge=0,
        le=10,
        description="Charlson Comorbidity Index score.",
    )

    # Hemodynamics & Vitals
    heart_rate: float = Field(
        ...,
        ge=30.0,
        le=220.0,
        description="Heart rate in beats per minute.",
    )
    sbp: float = Field(
        ...,
        ge=60.0,
        le=240.0,
        description="Systolic blood pressure in mmHg.",
    )
    dbp: float = Field(
        ...,
        ge=30.0,
        le=140.0,
        description="Diastolic blood pressure in mmHg.",
    )
    o2_sat: float = Field(
        ...,
        ge=60.0,
        le=100.0,
        description="Peripheral capillary oxygen saturation percentage.",
    )
    resp_rate: float = Field(
        ...,
        ge=6.0,
        le=60.0,
        description="Respiratory rate in breaths per minute.",
    )
    temp_c: float = Field(
        ...,
        ge=33.0,
        le=43.0,
        description="Body temperature in degrees Celsius.",
    )

    # Triage
    triage_acuity: int = Field(
        ...,
        ge=1,
        le=5,
        description="Emergency Severity Index (ESI) triage level (1 to 5).",
    )
    chief_complaint: str = Field(
        ...,
        min_length=1,
        description="Presenting clinical complaint.",
    )

    # Timestamps (string formatted as YYYY-MM-DD HH:MM:SS)
    arrival_time: str = Field(
        ...,
        description="Timestamp of arrival formatted as 'YYYY-MM-DD HH:MM:SS'.",
    )
    triage_start_time: str = Field(
        ...,
        description="Timestamp of triage assessment start formatted as 'YYYY-MM-DD HH:MM:SS'.",
    )
    bed_assigned_time: str = Field(
        ...,
        description="Timestamp of physical bed assignment formatted as 'YYYY-MM-DD HH:MM:SS'.",
    )
    discharge_time: str = Field(
        ...,
        description="Timestamp of final disposition departure formatted as 'YYYY-MM-DD HH:MM:SS'.",
    )
    arrival_hour: str = Field(
        ...,
        description="Relational partition key floored to 'YYYY-MM-DD HH:00:00'.",
    )

    # Outcomes & Trajectory
    los_hours: float = Field(
        ...,
        ge=0.5,
        le=336.0,
        description="Total duration of stay in decimal hours.",
    )
    disposition: DispositionType = Field(
        ...,
        description="Discharge disposition destination.",
    )
    icu_transfer_flag: int = Field(
        ...,
        ge=0,
        le=1,
        description="Indicator flag for ICU escalation (0 or 1).",
    )
    initial_care_unit: CareUnitType = Field(
        ...,
        description="Initial department placement unit.",
    )
    requires_ventilation: int = Field(
        ...,
        ge=0,
        le=1,
        description="Indicator flag for mechanical ventilation requirement (0 or 1).",
    )

    # Field Normalizers
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

    @field_validator("initial_care_unit", mode="before")
    @classmethod
    def normalize_initial_care_unit(cls, val: Any) -> Any:
        """Normalize unit representations like 'General_Ward' or 'Ward' to CareUnitType.WARD."""
        if isinstance(val, str):
            token = val.strip().replace(" ", "_").upper()
            if token in {"WARD", "GENERAL_WARD", "GENERALWARD", "MED_SURG"}:
                return CareUnitType.WARD
            if token in {"ED_ONLY", "ED", "EMERGENCY"}:
                return CareUnitType.ED_ONLY
            if token in {"ICU", "INTENSIVE_CARE_UNIT", "CCU"}:
                return CareUnitType.ICU
        return val

    @field_validator(
        "arrival_time",
        "triage_start_time",
        "bed_assigned_time",
        "discharge_time",
        "arrival_hour",
        mode="before",
    )
    @classmethod
    def validate_timestamp_format(cls, val: Any, info: Any) -> str:
        """Validate string adherence to YYYY-MM-DD HH:MM:SS."""
        if not isinstance(val, str):
            raise ValueError(f"Field '{info.field_name}' must be a string timestamp.")
        _parse_strict_timestamp(val, info.field_name)
        return val.strip()

    # Model Invariants
    @model_validator(mode="after")
    def validate_stay_invariants(self) -> Self:
        """Enforce temporal monotonicity, arrival_hour flooring, and hemodynamics."""
        # 1. Temporal Monotonicity
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

        # 2. Relational Key Flooring
        expected_arrival_hour = dt_arr.strftime("%Y-%m-%d %H:00:00")
        if self.arrival_hour != expected_arrival_hour:
            raise ValueError(
                f"Relational key mismatch: 'arrival_hour' must match 'arrival_time' floored "
                f"to the nearest zero-minute hour. Expected '{expected_arrival_hour}', "
                f"but received '{self.arrival_hour}'."
            )

        # 3. Physiological Pulse Pressure Validity
        pulse_pressure = self.sbp - self.dbp
        if pulse_pressure < MIN_PULSE_PRESSURE:
            raise ValueError(
                f"Physiological validity violated: Pulse pressure (sbp - dbp) must be >= "
                f"{MIN_PULSE_PRESSURE} mmHg. Calculated pulse pressure is {pulse_pressure:.2f} mmHg "
                f"(sbp={self.sbp:.1f}, dbp={self.dbp:.1f})."
            )
        return self


# ---------------------------------------------------------------------------
# HourlyCensusContract
# ---------------------------------------------------------------------------
class HourlyCensusContract(BaseModel):
    """Aggregated departmental census snapshot, staffing, and system counters."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )

    # Baseline Staffing Lookup Table: Shift -> (baseline_nurses, baseline_doctors)
    SHIFT_BASELINES: ClassVar[dict[ShiftType, dict[str, int]]] = {
        ShiftType.DAY: {"nurses": 35, "doctors": 10},
        ShiftType.EVENING: {"nurses": 30, "doctors": 8},
        ShiftType.NIGHT: {"nurses": 20, "doctors": 5},
    }

    # Dimensions
    timestamp: str = Field(
        ...,
        description="Hourly census snapshot key formatted as 'YYYY-MM-DD HH:00:00'.",
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

    # Unit Occupancies
    ed_occupancy: int = Field(
        ...,
        ge=0,
        le=100,
        description="Count of active patients residing in the Emergency Department.",
    )
    ward_occupancy: int = Field(
        ...,
        ge=0,
        le=200,
        description="Count of admitted patients occupying general inpatient beds.",
    )
    icu_occupancy: int = Field(
        ...,
        ge=0,
        le=50,
        description="Count of critical care patients occupying ICU beds.",
    )

    # Active Headcounts
    active_nurses: int = Field(
        ...,
        ge=15,
        le=50,
        description="Active registered and charge nursing headcount on duty.",
    )
    active_doctors: int = Field(
        ...,
        ge=3,
        le=20,
        description="Active attending and resident physician headcount on duty.",
    )

    # System Counters
    arrivals_count: int = Field(
        ...,
        ge=0,
        description="Patient arrivals registered during this hour.",
    )
    admissions_count: int = Field(
        ...,
        ge=0,
        description="Inpatient admissions executed during this hour.",
    )
    discharges_count: int = Field(
        ...,
        ge=0,
        description="Discharges completed during this hour.",
    )
    patients_in_queue: int = Field(
        ...,
        ge=0,
        description="Patients awaiting triage, rooming, or transfer placement.",
    )
    incoming_arrivals_next_4h: int = Field(
        ...,
        ge=0,
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

    @model_validator(mode="after")
    def validate_census_invariants(self) -> Self:
        """Enforce shift hour boundaries, calendar alignment, and staffing baselines."""
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
    def baseline_doctors(self) -> int:
        """Scheduled baseline physician count for the active shift."""
        return self.SHIFT_BASELINES[self.shift_id]["doctors"]

    @property
    def nurse_variance(self) -> int:
        """Variance between active nursing headcount and shift baseline."""
        return self.active_nurses - self.baseline_nurses

    @property
    def doctor_variance(self) -> int:
        """Variance between active physician headcount and shift baseline."""
        return self.active_doctors - self.baseline_doctors


# ---------------------------------------------------------------------------
# BedTopologyContract
# ---------------------------------------------------------------------------
class BedTopologyContract(BaseModel):
    """Physical infrastructure state for unit layouts and ventilator tracking."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )

    bed_id: str = Field(
        ...,
        min_length=1,
        description="Unique physical bed location code (e.g., 'ED-BAY-04', 'ICU-BED-12').",
    )
    department: CareUnitType = Field(
        ...,
        description="Hospital department housing this physical asset.",
    )
    room_id: str = Field(
        ...,
        min_length=1,
        description="Room or enclosure identifier.",
    )
    status: BedStatus = Field(
        ...,
        description="Current operational lifecycle status of the bed.",
    )
    is_negative_pressure: bool = Field(
        ...,
        description="Indicates airborne isolation (negative pressure) capability.",
    )
    has_ventilator: bool = Field(
        ...,
        description="Indicates dedicated invasive mechanical ventilator availability.",
    )
    current_stay_id: int | None = Field(
        default=None,
        ge=1,
        le=100_000,
        description="Foreign key referencing active PatientStayContract.stay_id, if occupied.",
    )

    @field_validator("department", mode="before")
    @classmethod
    def normalize_department(cls, val: Any) -> Any:
        """Normalize department strings to CareUnitType."""
        if isinstance(val, str):
            token = val.strip().replace(" ", "_").upper()
            if token in {"WARD", "GENERAL_WARD", "GENERALWARD", "MED_SURG"}:
                return CareUnitType.WARD
            if token in {"ED_ONLY", "ED", "EMERGENCY"}:
                return CareUnitType.ED_ONLY
            if token in {"ICU", "INTENSIVE_CARE_UNIT", "CCU"}:
                return CareUnitType.ICU
        return val

    @model_validator(mode="after")
    def validate_bed_invariants(self) -> Self:
        """Verify consistency between bed status and patient assignment."""
        if self.status == BedStatus.OCCUPIED and self.current_stay_id is None:
            raise ValueError(
                f"Bed topology invariant violated: Bed '{self.bed_id}' is marked as Occupied "
                "but has no 'current_stay_id' assigned."
            )
        if self.status in {BedStatus.AWAITING_CLEANING, BedStatus.OUT_OF_SERVICE} and self.current_stay_id is not None:
            raise ValueError(
                f"Bed topology invariant violated: Bed '{self.bed_id}' is marked as {self.status.value} "
                f"but retains an active 'current_stay_id' ({self.current_stay_id})."
            )
        return self