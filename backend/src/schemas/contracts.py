from datetime import datetime
from enum import Enum
from typing import Optional
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
    stay_id: int = Field(..., ge=1, description="Primary key for encounter")
    patient_id: int = Field(..., ge=100000, description="Master patient identifier")
    age: int = Field(..., ge=18, le=105, description="Patient age at triage")
    gender: str = Field(..., description="Gender (F or M)")
    charlson_index: int = Field(..., ge=0, le=15, description="Comorbidity score")
    
    heart_rate: float = Field(..., ge=30.0, le=220.0)
    sbp: float = Field(..., ge=60.0, le=240.0)
    dbp: float = Field(..., ge=30.0, le=140.0)
    o2_sat: float = Field(..., ge=60.0, le=100.0)
    resp_rate: float = Field(..., ge=6.0, le=60.0)
    temp_c: float = Field(..., ge=33.0, le=43.0)
    
    triage_acuity: int = Field(..., ge=1, le=5)
    chief_complaint: str = Field(..., min_length=1)
    
    arrival_time: str = Field(...)
    arrival_hour: str = Field(...)
    triage_start_time: str = Field(...)
    bed_assigned_time: str = Field(...)
    discharge_time: str = Field(...)
    
    los_hours: float = Field(..., ge=0.1, le=500.0)
    disposition: DispositionType = Field(...)
    icu_transfer_flag: int = Field(..., ge=0, le=1)
    initial_care_unit: CareUnitType = Field(...)
    requires_ventilation: int = Field(..., ge=0, le=1)
    
    @field_validator("gender", mode="before")
    @classmethod
    def normalize_gender(cls, v: str) -> str:
        val = str(v).strip().upper()
        if val in ["F", "FEMALE"]:
            return "F"
        return "M"
    
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
    def validate_temporal_flow(self):
        try:
            arr = datetime.strptime(self.arrival_time, "%Y-%m-%d %H:%M:%S")
            dis = datetime.strptime(self.discharge_time, "%Y-%m-%d %H:%M:%S")
            if dis <= arr:
                raise ValueError("discharge_time must occur after arrival_time")
        except ValueError as e:
            # Pass on formatting errors to specific field validators
            pass
        
        if self.sbp <= self.dbp:
            raise ValueError(f"SBP ({self.sbp}) must be greater than DBP ({self.dbp})")
        return self


class HourlyCensusContract(BaseModel):
    timestamp: str = Field(...)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: str = Field(...)
    is_weekend: int = Field(..., ge=0, le=1)
    shift_id: ShiftType = Field(...)
    
    ed_occupancy: int = Field(..., ge=0)
    ward_occupancy: int = Field(..., ge=0)
    icu_occupancy: int = Field(..., ge=0)
    
    active_nurses: int = Field(..., ge=10, le=50)
    active_doctors: int = Field(..., ge=2, le=20)
    
    arrivals_count: int = Field(..., ge=0)
    admissions_count: int = Field(..., ge=0)
    discharges_count: int = Field(..., ge=0)
    patients_in_queue: int = Field(..., ge=0)
    
    incoming_arrivals_next_4h: int = Field(..., ge=0)