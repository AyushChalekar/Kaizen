# filepath: clean_synthetic_stays.py
import pandas as pd
import numpy as np

def fix_dataset_gaps():
    np.random.seed(42)
    df = pd.read_csv('../../data/synthetic_patient_stays_100k.csv')

    # 1. Enforce Hemodynamics: SBP >= DBP + 15
    mask_bp = df['sbp'] < df['dbp'] + 15
    df.loc[mask_bp, 'sbp'] = df.loc[mask_bp, 'dbp'] + 15 + np.random.uniform(0, 10, size=mask_bp.sum())

    # 2. Enforce Minimum Length of Stay
    mask_los = df['los_hours'] < 0.5
    df.loc[mask_los, 'los_hours'] = np.random.uniform(0.5, 1.5, size=mask_los.sum())

    # 3. Fix Triage Acuity Routing (ESI 4 & 5 admission <= 0.05)
    mask_45 = df['triage_acuity'].isin([4, 5])
    mask_admitted = df['disposition'].isin(['Ward', 'ICU'])
    
    to_fix_routing = df[mask_45 & mask_admitted].index
    target_admitted_count = int(0.045 * mask_45.sum())
    current_admitted_count = len(to_fix_routing)

    if current_admitted_count > target_admitted_count:
        indices_to_discharge = np.random.choice(
            to_fix_routing, 
            size=(current_admitted_count - target_admitted_count), 
            replace=False
        )
        df.loc[indices_to_discharge, 'disposition'] = 'ED_Discharge'
        df.loc[indices_to_discharge, 'los_hours'] = np.random.uniform(0.5, 5.9, size=len(indices_to_discharge))

    # 4. Correct Ventilation Probabilities (15% ICU, 1% Non-ICU)
    mask_icu = df['disposition'] == 'ICU'
    df.loc[mask_icu, 'requires_ventilation'] = np.random.binomial(1, 0.15, size=mask_icu.sum())
    
    mask_non_icu = df['disposition'] != 'ICU'
    df.loc[mask_non_icu, 'requires_ventilation'] = np.random.binomial(1, 0.01, size=mask_non_icu.sum())

    df.to_csv('synthetic_patient_stays_100k_fixed.csv', index=False)

if __name__ == "__main__":
    fix_dataset_gaps()