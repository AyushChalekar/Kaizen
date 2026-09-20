# backend/src/simulation/engine.py
import pandas as pd
import simpy
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from ..data_generator.mimic_distributions import MimicIVEmpiricalEngine

class HospitalDigitalTwinEngine:
    def __init__(self, start_date: datetime, ed_capacity: int = 50, ward_capacity: int = 130, icu_capacity: int = 20, seed: int = 42):
        self.env = simpy.Environment()
        self.start_date = start_date
        self.engine = MimicIVEmpiricalEngine(seed=seed)
        
        # Resources
        self.ed_beds = simpy.PriorityResource(self.env, capacity=ed_capacity)
        self.ward_beds = simpy.Resource(self.env, capacity=ward_capacity)
        self.icu_beds = simpy.Resource(self.env, capacity=icu_capacity)
        
        # State trackers
        self.completed_stays: List[Dict[str, Any]] = []
        self.hourly_census: List[Dict[str, Any]] = []
        self.stay_counter = 0
        self.active_patients = 0
        self.patients_in_queue = 0
        
        # Counters for the current hour
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

    def patient_lifecycle(self, arrival_delay: float):
        """Simulates a single patient's journey through the hospital."""
        yield self.env.timeout(arrival_delay)
        
        self.stay_counter += 1
        self.active_patients += 1
        self.current_hour_arrivals += 1
        self.patients_in_queue += 1
        
        stay_id = self.stay_counter
        arrival_dt = self._get_current_dt()
        
        # Generate patient features using MIMIC-IV empirical distributions
        patient_data = self.engine.sample_encounter_features(stay_id, arrival_dt)
        
        patient_data["arrival_time"] = arrival_dt.strftime("%Y-%m-%d %H:%M:%S")
        patient_data["arrival_hour"] = arrival_dt.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Triage / ED Phase
        triage_start_dt = self._get_current_dt()
        patient_data["triage_start_time"] = triage_start_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        with self.ed_beds.request(priority=patient_data["triage_acuity"]) as req:
            yield req
            self.patients_in_queue -= 1
            bed_assigned_dt = self._get_current_dt()
            patient_data["bed_assigned_time"] = bed_assigned_dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Simulate initial ED assessment time (e.g., 2-6 hours)
            ed_time = min(patient_data["los_hours"], self.engine.rng.uniform(2.0, 6.0))
            yield self.env.timeout(ed_time)
            
            # 2. Admission Phase (if applicable)
            if patient_data["initial_care_unit"] != "ED_Only":
                self.current_hour_admissions += 1
                remaining_los = patient_data["los_hours"] - ed_time
                
                target_resource = self.icu_beds if patient_data["initial_care_unit"] == "ICU" else self.ward_beds
                
                with target_resource.request() as admit_req:
                    yield admit_req
                    # Hold the inpatient bed for the remaining length of stay
                    yield self.env.timeout(remaining_los)
            
        # 3. Discharge Phase
        discharge_dt = self._get_current_dt()
        patient_data["discharge_time"] = discharge_dt.strftime("%Y-%m-%d %H:%M:%S")
        self.completed_stays.append(patient_data)
        self.active_patients -= 1
        self.current_hour_discharges += 1

    def arrival_generator(self, max_stays: int):
        """Generates patient arrivals based on NHPP."""
        while self.stay_counter < max_stays:
            current_dt = self._get_current_dt()
            # Calculate time to next hour
            next_hour = current_dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            hours_until_next = (next_hour - current_dt).total_seconds() / 3600.0
            
            arrivals_this_hour = self.engine.sample_hourly_arrivals(current_dt)
            
            if arrivals_this_hour > 0:
                # Distribute arrivals uniformly across the remaining time in the current hour
                interarrival_time = hours_until_next / arrivals_this_hour
                for _ in range(arrivals_this_hour):
                    if self.stay_counter >= max_stays:
                        break
                    self.env.process(self.patient_lifecycle(0))
                    yield self.env.timeout(interarrival_time)
            else:
                yield self.env.timeout(hours_until_next)

    def census_monitor(self):
        """Logs the hospital state every hour."""
        while True:
            current_dt = self._get_current_dt()
            
            # Calculate incoming arrivals for the next 4 hours
            future_dt = current_dt
            incoming_4h = 0
            for _ in range(4):
                incoming_4h += self.engine.sample_hourly_arrivals(future_dt)
                future_dt += timedelta(hours=1)
            
            # Determine active staff (simplified static assignment based on shift)
            shift = self._determine_shift(current_dt.hour)
            active_nurses = 35 if shift == "Day" else (30 if shift == "Evening" else 20)
            active_docs = 10 if shift == "Day" else (8 if shift == "Evening" else 5)
            
            self.hourly_census.append({
                "timestamp": current_dt.strftime("%Y-%m-%d %H:%M:%S"),
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
                "incoming_arrivals_next_4h": incoming_4h
            })
            
            # Reset hourly counters
            self.current_hour_arrivals = 0
            self.current_hour_admissions = 0
            self.current_hour_discharges = 0
            
            yield self.env.timeout(1.0) # Wait 1 simulation hour

    def simulate(self, total_hours: int, max_stays: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.env.process(self.arrival_generator(max_stays))
        self.env.process(self.census_monitor())
        self.env.run(until=total_hours)
        
        df_stays = pd.DataFrame(self.completed_stays)
        df_census = pd.DataFrame(self.hourly_census)
        
        return df_stays, df_census