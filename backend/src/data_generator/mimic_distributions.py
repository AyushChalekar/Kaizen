# backend/src/data_generator/mimic_distributions.py
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple

class MimicIVEmpiricalEngine:
    DIURNAL_WEIGHTS = [
        0.58, 0.44, 0.35, 0.28, 0.26, 0.32,
        0.50, 0.76, 1.04, 1.28, 1.44, 1.52,
        1.56, 1.58, 1.54, 1.50, 1.46, 1.38,
        1.26, 1.14, 1.00, 0.86, 0.72, 0.63
    ]
    
    DAY_FACTORS = {
        0: 1.10, 1: 1.04, 2: 1.01, 3: 0.99,
        4: 1.00, 5: 0.94, 6: 0.92
    }
    
    COMPLAINT_CHOICES = [
        ("Chest Pain", 0.16), ("Shortness of Breath", 0.14),
        ("Abdominal Pain", 0.15), ("Fever/Infection", 0.12),
        ("Fall/Trauma", 0.11), ("Altered Mental Status", 0.08),
        ("Headache/Neuro", 0.08), ("Laceration/Injury", 0.06),
        ("General Weakness", 0.10)
    ]
    
    def __init__(self, base_arrival_rate: float = 5.71, unique_patients: int = 75000, seed: int = 42):
        self.base_rate = base_arrival_rate
        self.rng = np.random.default_rng(seed)
        self.patient_pool = np.arange(100000, 100000 + unique_patients)
        self.patient_baselines = {
            pid: {
                "age_base": int(np.clip(self.rng.normal(51.2, 16.1), 18, 100)),
                "gender": "F" if self.rng.random() < 0.498 else "M"
            }
            for pid in self.patient_pool
        }
    
    def sample_hourly_arrivals(self, current_dt: datetime) -> int:
        diurnal = self.DIURNAL_WEIGHTS[current_dt.hour]
        day_factor = self.DAY_FACTORS[current_dt.weekday()]
        mean_lambda = self.base_rate * diurnal * day_factor
        r = 12.0
        p = r / (r + mean_lambda)
        return int(self.rng.negative_binomial(r, p))
    
    def sample_encounter_features(self, stay_id: int, arrival_dt: datetime) -> Dict[str, Any]:
        patient_id = int(self.rng.choice(self.patient_pool))
        patient_data = self.patient_baselines[patient_id]
        
        raw_age = patient_data["age_base"]
        age = 91 if raw_age > 89 else raw_age
        gender = patient_data["gender"]
        
        charlson_mean = max(0.2, (age - 25) / 14.0)
        charlson_index = int(np.clip(self.rng.poisson(charlson_mean), 0, 10))
        
        acuity_probs = [0.05, 0.20, 0.45, 0.25, 0.05]
        triage_acuity = int(self.rng.choice([1, 2, 3, 4, 5], p=acuity_probs))
        
        if triage_acuity <= 2:
            sbp = float(np.clip(self.rng.normal(105.0, 20.0), 70.0, 195.0))
            dbp = float(np.clip(self.rng.normal(65.0, 15.0), 40.0, sbp - 10.0))
            o2_sat = float(np.clip(self.rng.normal(92.0, 5.0), 70.0, 100.0))
        else:
            sbp = float(np.clip(self.rng.normal(122.0, 14.0), 90.0, 180.0))
            dbp = float(np.clip(self.rng.normal(74.0, 9.0), 50.0, sbp - 10.0))
            o2_sat = float(np.clip(self.rng.normal(97.0, 1.8), 90.0, 100.0))
            
        heart_rate = float(np.clip(self.rng.normal(94.1, 15.0), 45.0, 180.0))
        resp_rate = float(np.clip(self.rng.normal(20.0, 4.0), 10.0, 45.0))
        temp_c = float(np.clip(self.rng.normal(37.4, 0.8), 34.0, 41.0))
        
        c_names, c_weights = zip(*self.COMPLAINT_CHOICES)
        complaint = str(self.rng.choice(c_names, p=c_weights))
        
        if triage_acuity == 1:
            disp_weights = [0.05, 0.35, 0.60]
        elif triage_acuity == 2:
            disp_weights = [0.20, 0.55, 0.25]
        else:
            disp_weights = [0.75, 0.23, 0.02]
            
        disposition = str(self.rng.choice(["ED_Discharge", "Ward", "ICU"], p=disp_weights))
        
        initial_care_unit = disposition if disposition != "ED_Discharge" else "ED_Only"
        icu_transfer_flag = 0
        requires_ventilation = 0
        
        if disposition == "Ward" and self.rng.random() < 0.08:
            icu_transfer_flag = 1
            
        if disposition == "ICU" or icu_transfer_flag == 1:
            requires_ventilation = 1 if self.rng.random() < 0.25 else 0
            los_hours = float(np.clip(self.rng.lognormal(4.2, 0.8), 6.0, 336.0))
        elif disposition == "Ward":
            los_hours = float(np.clip(self.rng.lognormal(3.5, 0.7), 4.0, 240.0))
        else:
            los_hours = float(np.clip(self.rng.lognormal(1.3, 0.55), 0.5, 24.0))
            
        return {
            "stay_id": stay_id,
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "charlson_index": charlson_index,
            "heart_rate": round(heart_rate, 1),
            "sbp": round(sbp, 1),
            "dbp": round(dbp, 1),
            "o2_sat": round(o2_sat, 1),
            "resp_rate": round(resp_rate, 1),
            "temp_c": round(temp_c, 2),
            "triage_acuity": triage_acuity,
            "chief_complaint": complaint,
            "los_hours": round(los_hours, 2),
            "disposition": disposition,
            "icu_transfer_flag": icu_transfer_flag,
            "initial_care_unit": initial_care_unit,
            "requires_ventilation": requires_ventilation
        }