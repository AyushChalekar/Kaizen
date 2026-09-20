# backend/tests/test_simulation.py
import pytest
from datetime import datetime
from src.schemas.contracts import PatientStayContract, HourlyCensusContract
from src.simulation.engine import HospitalDigitalTwinEngine


def test_contracts_validation():
    valid_stay = {
        "stay_id": 1,
        "patient_id": 100001,
        "age": 45,
        "gender": "Female",
        "charlson_index": 2,
        "heart_rate": 88.0,
        "sbp": 120.0,
        "dbp": 80.0,
        "o2_sat": 98.0,
        "resp_rate": 16.0,
        "temp_c": 37.0,
        "triage_acuity": 3,
        "chief_complaint": "Chest Pain",
        "arrival_time": "2026-01-01 10:15:00",
        "arrival_hour": "2026-01-01 10:00:00",
        "triage_start_time": "2026-01-01 10:18:00",
        "bed_assigned_time": "2026-01-01 10:45:00",
        "discharge_time": "2026-01-01 15:00:00",
        "los_hours": 4.75,
        "disposition": "ED_Discharge",
        "icu_transfer_flag": 0,
        "initial_care_unit": "ED_Only",
        "requires_ventilation": 0
    }
    contract = PatientStayContract(**valid_stay)
    assert contract.gender == "Female"
    assert contract.initial_care_unit == "ED_Only"
    assert contract.sbp >= contract.dbp + 15.0


def test_pulse_pressure_invariant_failure():
    invalid_stay = {
        "stay_id": 2,
        "patient_id": 100002,
        "age": 50,
        "gender": "Male",
        "charlson_index": 1,
        "heart_rate": 80.0,
        "sbp": 100.0,
        "dbp": 90.0,  # Pulse pressure is only 10 mmHg (< 15 mmHg floor)
        "o2_sat": 98.0,
        "resp_rate": 18.0,
        "temp_c": 37.0,
        "triage_acuity": 3,
        "chief_complaint": "Abdominal",
        "arrival_time": "2026-01-01 10:00:00",
        "arrival_hour": "2026-01-01 10:00:00",
        "triage_start_time": "2026-01-01 10:05:00",
        "bed_assigned_time": "2026-01-01 10:15:00",
        "discharge_time": "2026-01-01 14:00:00",
        "los_hours": 4.0,
        "disposition": "ED_Discharge",
        "icu_transfer_flag": 0,
        "initial_care_unit": "ED_Only",
        "requires_ventilation": 0
    }
    with pytest.raises(ValueError, match="Violated physiological consistency"):
        PatientStayContract(**invalid_stay)


def test_simulation_run_and_relational_contracts():
    start_dt = datetime(2026, 1, 1, 0, 0, 0)
    twin = HospitalDigitalTwinEngine(start_date=start_dt, seed=42)

    df_stays, df_census = twin.simulate(total_hours=48, max_stays=200)

    assert len(df_census) == 48
    assert len(df_stays) > 0

    # 1. Validate relational key alignment (arrival_hour -> census timestamp)
    census_timestamps = set(df_census["timestamp"])
    for arr_hr in df_stays["arrival_hour"].unique():
        assert arr_hr in census_timestamps

    # 2. Validate contracts on actual generated records
    for _, row in df_stays.head(50).iterrows():
        contract = PatientStayContract(**row.to_dict())
        assert contract.sbp >= contract.dbp + 15.0

    for _, row in df_census.head(48).iterrows():
        HourlyCensusContract(**row.to_dict())

    # 3. Validate supervised incoming arrivals label
    for idx in range(len(df_census) - 4):
        actual_future_sum = int(df_census.loc[idx + 1:idx + 4, "arrivals_count"].sum())
        assert df_census.loc[idx, "incoming_arrivals_next_4h"] == actual_future_sum