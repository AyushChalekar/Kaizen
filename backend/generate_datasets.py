# backend/generate_datasets.py
import argparse
import os
import pandas as pd
from datetime import datetime
from src.simulation.engine import HospitalDigitalTwinEngine
from src.schemas.contracts import PatientStayContract, HourlyCensusContract


def run_pipeline(output_dir: str, stays: int, hours: int, seed: int):
    print(f"Starting PROSH simulation pipeline ({stays} stays over {hours} hours)...")
    start_dt = datetime(2026, 1, 1, 0, 0, 0)

    twin = HospitalDigitalTwinEngine(
        start_date=start_dt,
        ed_capacity=50,
        ward_capacity=130,
        icu_capacity=20,
        seed=seed
    )

    df_stays, df_census = twin.simulate(total_hours=hours, max_stays=stays)

    print(f"Validating {len(df_stays)} patient encounters against Pydantic schema...")
    for _, row in df_stays.iterrows():
        PatientStayContract(**row.to_dict())

    print(f"Validating {len(df_census)} hourly census records against Pydantic schema...")
    for _, row in df_census.iterrows():
        HourlyCensusContract(**row.to_dict())

    os.makedirs(output_dir, exist_ok=True)
    stays_file = os.path.join(output_dir, "synthetic_patient_stays_100k_fixed.csv")
    census_file = os.path.join(output_dir, "synthetic_hourly_census.csv")

    df_stays.to_csv(stays_file, index=False)
    df_census.to_csv(census_file, index=False)

    print(f"SUCCESS: Exported {len(df_stays)} compliant stays to {stays_file}")
    print(f"SUCCESS: Exported {len(df_census)} compliant census records to {census_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PROSH Dataset Generator")
    parser.add_argument("--output-dir", type=str, default="./data")
    parser.add_argument("--stays", type=int, default=100000, help="Target completed stays")
    parser.add_argument("--hours", type=int, default=17520, help="Simulation hours (17520 = 2 years)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_pipeline(args.output_dir, args.stays, args.hours, args.seed)