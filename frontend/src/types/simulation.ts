// /home/adomin/kaizen/frontend/src/types/simulation.ts

/**
 * Enumeration for shift types in the simulation.
 */
export const ShiftType = {
  DAY: "Day",
  EVENING: "Evening",
  NIGHT: "Night"
} as const;
export type ShiftType = (typeof ShiftType)[keyof typeof ShiftType];

/**
 * Enumeration for disposition types of patient stays.
 */
export const DispositionType = {
  ED_DISCHARGE: "ED_Discharge",
  WARD: "Ward",
  ICU: "ICU"
} as const;
export type DispositionType = (typeof DispositionType)[keyof typeof DispositionType];

/**
 * Enumeration for types of care units.
 */
export const CareUnitType = {
  ED_ONLY: "ED_Only",
  WARD: "Ward",
  ICU: "ICU"
} as const;
export type CareUnitType = (typeof CareUnitType)[keyof typeof CareUnitType];

/**
 * Interface representing a patient stay contract.
 * Matches the backend schema exactly.
 */
export interface PatientStayContract {
  stay_id: number;
  patient_id: number;
  age: number;
  gender: "Female" | "Male";
  charlson_index: number;
  heart_rate: number;
  sbp: number;
  dbp: number;
  o2_sat: number;
  resp_rate: number;
  temp_c: number;
  triage_acuity: number;
  chief_complaint: string;
  arrival_time: string;
  arrival_hour: string;
  triage_start_time: string;
  bed_assigned_time: string;
  discharge_time: string;
  los_hours: number;
  disposition: DispositionType;
  icu_transfer_flag: 0 | 1;
  initial_care_unit: CareUnitType;
  requires_ventilation: 0 | 1;
}

/**
 * Interface representing hourly census data.
 * Matches the backend schema exactly.
 */
export interface HourlyCensusContract {
  timestamp: string;
  hour: number;
  day_of_week: string;
  is_weekend: 0 | 1;
  shift_id: ShiftType;
  ed_occupancy: number;
  ward_occupancy: number;
  icu_occupancy: number;
  active_nurses: number;
  active_doctors: number;
  arrivals_count: number;
  admissions_count: number;
  discharges_count: number;
  patients_in_queue: number;
  incoming_arrivals_next_4h: number;
}

/**
 * Interface representing the overall simulation data response.
 * Matches the backend API contract.
 */
export interface SimulationDataResponse {
  message?: string;
  data: {
    patient_stays?: PatientStayContract[];
    hourly_census?: HourlyCensusContract[];
  };
}