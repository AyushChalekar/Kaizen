# backend/src/schemas/contracts.py
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


class ShiftType(str, Enum):
    DAY = "Day"
    EVENING = "Evening"
    NIGHT = "Night"


class DispositionType(str, Enum):
    ED_DISCHARGE = "ED_Discharge"
    WARD = "Ward"
    ICU = "ICU"


class CareUnitType(str, Enum):
    ED_ONLY = "ED_Only"
    WARD = "Ward"
    ICU = "ICU"


class PatientStayContract(BaseModel):
    stay_id: int = Field(..., ge=1, le=100000, description="Primary key for encounter")
    patient_id: int = Field(..., ge=100001, le=200000, description="Master patient identifier")
    age: int = Field(..., ge=18, le=95, description="Patient age at triage")
    gender: str = Field(..., description="Gender (Female or Male)")
    charlson_index: int = Field(..., ge=0, le=10, description="Comorbidity score")

    heart_rate: float = Field(..., ge=30.0, le=220.0)
    sbp: float = Field(..., ge=60.0, le=240.0)
    dbp: float = Field(..., ge=30.0, le=140.0)
    o2_sat: float = Field(..., ge=60.0, le=100.0)
    resp_rate: float = Field(..., ge=6.0, le=60.0)
    temp_c: float = Field(..., ge=33.0, le=43.0)

    triage_acuity: int = Field(..., ge=1, le=5)
    chief_complaint: str = Field(..., min_length=1)

    arrival_time: str = Field(..., description="YYYY-MM-DD HH:MM:SS")
    arrival_hour: str = Field(..., description="YYYY-MM-DD HH:00:00 foreign key")
    triage_start_time: str = Field(..., description="YYYY-MM-DD HH:MM:SS")
    bed_assigned_time: str = Field(..., description="YYYY-MM-DD HH:MM:SS")
    discharge_time: str = Field(..., description="YYYY-MM-DD HH:MM:SS")

    los_hours: float = Field(..., ge=0.5, le=336.0)
    disposition: DispositionType = Field(...)
    icu_transfer_flag: int = Field(..., ge=0, le=1)
    initial_care_unit: CareUnitType = Field(...)
    requires_ventilation: int = Field(..., ge=0, le=1)

    @field_validator("gender", mode="before")
    @classmethod
    def normalize_gender(cls, v: str) -> str:
        val = str(v).strip().upper()
        if val in ["FEMALE", "F"]:
            return "Female"
        return "Male"

    @field_validator("initial_care_unit", mode="before")
    @classmethod
    def normalize_care_unit(cls, v: str) -> CareUnitType:
        val = str(v).strip()
        if val in ["General_Ward", "Ward", "WARD"]:
            return CareUnitType.WARD
        if val in ["ICU", "icu"]:
            return CareUnitType.ICU
        return CareUnitType.ED_ONLY

    @model_validator(mode="after")
    def validate_invariants(self):
        # 1. Mandatory Invariant: Temporal Monotonicity
        # arrival_time <= triage_start_time <= bed_assigned_time <= discharge_time
        dt_arr = datetime.strptime(self.arrival_time, "%Y-%m-%d %H:%M:%S")
        dt_tri = datetime.strptime(self.triage_start_time, "%Y-%m-%d %H:%M:%S")
        dt_bed = datetime.strptime(self.bed_assigned_time, "%Y-%m-%d %H:%M:%S")
        dt_dis = datetime.strptime(self.discharge_time, "%Y-%m-%d %H:%M:%S")

        if not (dt_arr <= dt_tri <= dt_bed <= dt_dis):
            raise ValueError(
                f"Violated temporal monotonicity: arr={dt_arr} <= tri={dt_tri} <= "
                f"bed={dt_bed} <= dis={dt_dis}"
            )

        # 2. Relational Key Monotonicity: arrival_hour floored to hour mark
        expected_hour = dt_arr.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:00:00")
        if self.arrival_hour != expected_hour:
            raise ValueError(
                f"arrival_hour ({self.arrival_hour}) does not match floored arrival_time ({expected_hour})"
            )

        # 3. Mandatory Invariant: Physiological Consistency (Pulse pressure >= 15 mmHg)
        if self.sbp < self.dbp + 15.0:
            raise ValueError(
                f"Violated physiological consistency: SBP ({self.sbp}) must be >= "
                f"DBP ({self.dbp}) + 15.0 mmHg"
            )

        return self


class HourlyCensusContract(BaseModel):
    timestamp: str = Field(..., description="YYYY-MM-DD HH:00:00")
    hour: int = Field(..., ge=0, le=23)
    day_of_week: str = Field(...)
    is_weekend: int = Field(..., ge=0, le=1)
    shift_id: ShiftType = Field(...)

    ed_occupancy: int = Field(..., ge=0, le=100)
    ward_occupancy: int = Field(..., ge=0, le=200)
    icu_occupancy: int = Field(..., ge=0, le=50)

    active_nurses: int = Field(..., ge=15, le=50)
    active_doctors: int = Field(..., ge=3, le=20)

    arrivals_count: int = Field(..., ge=0)
    admissions_count: int = Field(..., ge=0)
    discharges_count: int = Field(..., ge=0)
    patients_in_queue: int = Field(..., ge=0)

    incoming_arrivals_next_4h: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_shift_alignment(self):
        # Shift hour window validation
        if 7 <= self.hour < 15:
            expected_shift = ShiftType.DAY
            expected_nurses = 35
            expected_docs = 10
        elif 15 <= self.hour < 23:
            expected_shift = ShiftType.EVENING
            expected_nurses = 30
            expected_docs = 8
        else:
            expected_shift = ShiftType.NIGHT
            expected_nurses = 20
            expected_docs = 5

        if self.shift_id != expected_shift:
            raise ValueError(
                f"Hour {self.hour} misaligned with shift {self.shift_id} (expected {expected_shift})"
            )
        if self.active_nurses != expected_nurses:
            raise ValueError(
                f"Shift {self.shift_id} requires {expected_nurses} nurses, got {self.active_nurses}"
            )
        if self.active_doctors != expected_docs:
            raise ValueError(
                f"Shift {self.shift_id} requires {expected_docs} doctors, got {self.active_doctors}"
            )

        return self