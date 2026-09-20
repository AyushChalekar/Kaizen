# backend/src/simulation/engine.py
import pandas as pd
import simpy
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from ..data_generator.mimic_distributions import MimicIVEmpiricalEngine


class HospitalDigitalTwinEngine:
    def __init__(
        self,
        start_date: datetime,
        ed_capacity: int = 50,
        ward_capacity: int = 130,
        icu_capacity: int = 20,
        seed: int = 42
    ):
        self.env = simpy.Environment()
        self.start_date = start_date
        self.engine = MimicIVEmpiricalEngine(seed=seed)

        # Simulation resources
        self.ed_beds = simpy.PriorityResource(self.env, capacity=ed_capacity)
        self.ward_beds = simpy.PriorityResource(self.env, capacity=ward_capacity)
        self.icu_beds = simpy.PriorityResource(self.env, capacity=icu_capacity)

        # Simulation output collections
        self.completed_stays: List[Dict[str, Any]] = []
        self.hourly_census: List[Dict[str, Any]] = []

        # System counters
        self.stay_counter = 0
        self.active_patients = 0
        self.patients_in_queue = 0

        # Current hour event tallies
        self.current_hour_arrivals = 0
        self.current_hour_admissions = 0
        self.current_hour_discharges = 0

    def _get_current_dt(self) -> datetime:
        return self.start_date + timedelta(hours=self.env.now)

    def _determine_shift(self, hour: int) -> str:
        if 7 <= hour < 15:
            return "Day"
        elif 15 <= hour < 23:
            return "Evening"
        else:
            return "Night"

    def patient_lifecycle(self, patient_data: Dict[str, Any]):
        """
        Simulates individual patient flow through the hospital care continuum.
        Enforces proper bed release upon inpatient handoff and records exact timestamps.
        """
        self.active_patients += 1
        self.current_hour_arrivals += 1
        self.patients_in_queue += 1

        arrival_dt = self._get_current_dt()
        patient_data["arrival_time"] = arrival_dt.strftime("%Y-%m-%d %H:%M:%S")
        patient_data["arrival_hour"] = arrival_dt.strftime("%Y-%m-%d %H:00:00")

        # Triage phase begins immediately upon arrival
        triage_start_dt = self._get_current_dt()
        patient_data["triage_start_time"] = triage_start_dt.strftime("%Y-%m-%d %H:%M:%S")

        # Priority request for ED Bay based on triage acuity (ESI 1 is highest priority)
        with self.ed_beds.request(priority=patient_data["triage_acuity"]) as ed_req:
            yield ed_req
            self.patients_in_queue -= 1

            bed_assigned_dt = self._get_current_dt()
            patient_data["bed_assigned_time"] = bed_assigned_dt.strftime("%Y-%m-%d %H:%M:%S")

            if patient_data["initial_care_unit"] == "ED_Only":
                # ED only encounter: occupies ED bay for full duration
                yield self.env.timeout(patient_data["los_hours"])
            else:
                # Inpatient encounter: ED evaluation time (2 to 4 hours)
                ed_eval_time = min(patient_data["los_hours"] * 0.3, self.engine.rng.uniform(2.0, 4.0))
                yield self.env.timeout(ed_eval_time)

        # Inpatient Phase: ED Bed has been freed, patient transfers to Ward or ICU
        if patient_data["initial_care_unit"] != "ED_Only":
            self.current_hour_admissions += 1
            remaining_los = max(0.5, patient_data["los_hours"] - ed_eval_time)

            target_resource = (
                self.icu_beds
                if (patient_data["initial_care_unit"] == "ICU" or patient_data["icu_transfer_flag"] == 1)
                else self.ward_beds
            )

            with target_resource.request(priority=patient_data["triage_acuity"]) as admit_req:
                yield admit_req
                yield self.env.timeout(remaining_los)

        # Discharge Phase
        discharge_dt = self._get_current_dt()
        patient_data["discharge_time"] = discharge_dt.strftime("%Y-%m-%d %H:%M:%S")

        # Calculate realized LOS in hours (including queue wait times)
        realized_los = (discharge_dt - arrival_dt).total_seconds() / 3600.0
        patient_data["los_hours"] = round(realized_los, 2)

        self.completed_stays.append(patient_data)
        self.active_patients -= 1
        self.current_hour_discharges += 1

    def arrival_generator(self, max_stays: int):
        """
        Generates non-homogeneous Poisson arrivals with drift-free hourly scheduling.
        """
        while self.stay_counter < max_stays:
            current_hour_index = int(self.env.now)
            hour_start_dt = self.start_date + timedelta(hours=current_hour_index)

            arrivals_this_hour = self.engine.sample_hourly_arrivals(hour_start_dt)

            if arrivals_this_hour > 0:
                # Disperse arrivals evenly across the hour
                offsets = sorted(self.engine.rng.uniform(0.0, 0.999, size=arrivals_this_hour))
                last_offset = 0.0

                for offset in offsets:
                    if self.stay_counter >= max_stays:
                        break
                    delay = offset - last_offset
                    if delay > 0:
                        yield self.env.timeout(delay)
                    last_offset = offset

                    self.stay_counter += 1
                    arrival_dt = self._get_current_dt()
                    patient_data = self.engine.sample_encounter_features(self.stay_counter, arrival_dt)
                    self.env.process(self.patient_lifecycle(patient_data))

                # Step simulation forward to the exact completion of the hour
                remaining_in_hour = 1.0 - last_offset
                if remaining_in_hour > 0:
                    yield self.env.timeout(remaining_in_hour)
            else:
                yield self.env.timeout(1.0)

    def census_monitor(self):
        """
        Monitors facility occupancy every simulation hour and tracks conservation of flow.
        """
        while True:
            current_hour_index = int(self.env.now)
            current_dt = self.start_date + timedelta(hours=current_hour_index)

            shift = self._determine_shift(current_dt.hour)
            active_nurses = 35 if shift == "Day" else (30 if shift == "Evening" else 20)
            active_docs = 10 if shift == "Day" else (8 if shift == "Evening" else 5)

            self.hourly_census.append({
                "timestamp": current_dt.strftime("%Y-%m-%d %H:00:00"),
                "hour": current_dt.hour,
                "day_of_week": current_dt.strftime("%A"),
                "is_weekend": 1 if current_dt.weekday() >= 5 else 0,
                "shift_id": shift,
                "ed_occupancy": self.ed_beds.count,
                "ward_occupancy": self.ward_beds.count,
                "icu_occupancy": self.icu_beds.count,
                "active_nurses": active_nurses,
                "active_doctors": active_docs,
                "arrivals_count": self.current_hour_arrivals,
                "admissions_count": self.current_hour_admissions,
                "discharges_count": self.current_hour_discharges,
                "patients_in_queue": self.patients_in_queue,
                # Will be post-processed to reflect true supervised future arrivals
                "incoming_arrivals_next_4h": 0
            })

            # Reset hourly event delta accumulators
            self.current_hour_arrivals = 0
            self.current_hour_admissions = 0
            self.current_hour_discharges = 0

            yield self.env.timeout(1.0)

    def simulate(self, total_hours: int, max_stays: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.env.process(self.arrival_generator(max_stays))
        self.env.process(self.census_monitor())
        self.env.run(until=total_hours)

        df_stays = pd.DataFrame(self.completed_stays)
        df_census = pd.DataFrame(self.hourly_census)

        # Post-process incoming_arrivals_next_4h to match supervised ground truth:
        # incoming_arrivals_next_4h(t) = sum_{k=1}^{4} arrivals_count(t+k)
        if not df_census.empty:
            next_4h = (
                df_census["arrivals_count"].shift(-1).fillna(0)
                + df_census["arrivals_count"].shift(-2).fillna(0)
                + df_census["arrivals_count"].shift(-3).fillna(0)
                + df_census["arrivals_count"].shift(-4).fillna(0)
            ).astype(int)
            df_census["incoming_arrivals_next_4h"] = next_4h

        return df_stays, df_census