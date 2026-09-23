// frontend/src/api/useForecastData.ts
/**
 * React custom hook and TypeScript contracts for hospital digital twin simulation telemetry.
 * Connects to the FastAPI backend to fetch, execute, and reset discrete-event simulations.
 */

import { useState, useEffect, useCallback } from "react";

// ---------------------------------------------------------------------------
// Clinical & Operational Enumerations
// ---------------------------------------------------------------------------

export type ShiftType = "Day" | "Evening" | "Night";

export type DispositionType = "ED_Discharge" | "Ward" | "ICU";

export type CareUnitType = "ED_Only" | "Ward" | "ICU";

export type BedStatus =
  | "Occupied"
  | "Reserved"
  | "Pending Discharge"
  | "Awaiting Cleaning"
  | "Out of Service";

// ---------------------------------------------------------------------------
// 1. Configuration & Metrics Interfaces
// ---------------------------------------------------------------------------

export interface SimulationRunRequest {
  duration_hours?: number;
  ed_capacity?: number;
  ward_capacity?: number;
  icu_capacity?: number;
  seed?: number;
  start_datetime?: string;
}

export interface SimulationMetricsResponse {
  total_arrivals: number;
  total_admissions: number;
  total_discharges: number;
  completed_stays_count: number;
  peak_queue_length: number;
  average_los_hours: number;
  ed_utilization_pct: number;
  ward_utilization_pct: number;
  icu_utilization_pct: number;
}

// ---------------------------------------------------------------------------
// 2. Domain Contracts & Response Envelope
// ---------------------------------------------------------------------------

export interface HourlyCensusContract {
  timestamp: string;
  hour: number;
  day_of_week: string;
  is_weekend: number;
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

export interface BedTopologyContract {
  bed_id: string;
  department: CareUnitType;
  room_id: string;
  status: BedStatus;
  is_negative_pressure: boolean;
  has_ventilator: boolean;
  current_stay_id: number | null;
}

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
  triage_start_time: string;
  bed_assigned_time: string;
  discharge_time: string;
  arrival_hour: string;
  los_hours: number;
  disposition: DispositionType;
  icu_transfer_flag: number;
  initial_care_unit: CareUnitType;
  requires_ventilation: number;
}

export interface SimulationDataResponse {
  status: string;
  generated_at: string;
  config: SimulationRunRequest;
  metrics: SimulationMetricsResponse;
  hourly_census: HourlyCensusContract[];
  bed_topology: BedTopologyContract[];
  stays: PatientStayContract[];
}

// ---------------------------------------------------------------------------
// Hook Return & Options Interfaces
// ---------------------------------------------------------------------------

export interface UseForecastDataOptions {
  baseUrl?: string;
  autoFetch?: boolean;
  initialIncludeStays?: boolean;
}

export interface UseForecastDataResult {
  data: SimulationDataResponse | null;
  isLoading: boolean;
  error: string | null;
  fetchSimulationData: (includeStays?: boolean) => Promise<SimulationDataResponse | null>;
  runSimulation: (config: SimulationRunRequest) => Promise<SimulationDataResponse | null>;
  resetSimulation: () => Promise<SimulationDataResponse | null>;
}

// ---------------------------------------------------------------------------
// Custom Hook: useForecastData
// ---------------------------------------------------------------------------

export function useForecastData(options?: UseForecastDataOptions): UseForecastDataResult {
  const baseUrl = options?.baseUrl ?? "";
  const autoFetch = options?.autoFetch ?? true;
  const initialIncludeStays = options?.initialIncludeStays ?? true;

  const [data, setData] = useState<SimulationDataResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * Helper to parse API error responses into clean messages.
   */
  const parseErrorMessage = async (response: Response): Promise<string> => {
    try {
      const errorJson = await response.json();
      if (errorJson && typeof errorJson.detail === "string") {
        return errorJson.detail;
      }
      if (errorJson && Array.isArray(errorJson.detail)) {
        return errorJson.detail.map((err: { msg?: string }) => err.msg || JSON.stringify(err)).join("; ");
      }
      return JSON.stringify(errorJson);
    } catch {
      return `HTTP error ${response.status}: ${response.statusText}`;
    }
  };

  /**
   * Fetch current simulation telemetry and records from GET /api/simulation/data
   */
  const fetchSimulationData = useCallback(
    async (includeStays: boolean = true): Promise<SimulationDataResponse | null> => {
      setIsLoading(true);
      setError(null);
      try {
        const url = `${baseUrl}/api/simulation/data?include_stays=${includeStays}`;
        const response = await fetch(url, {
          method: "GET",
          headers: {
            Accept: "application/json",
          },
        });

        if (!response.ok) {
          const detail = await parseErrorMessage(response);
          throw new Error(detail);
        }

        const payload: SimulationDataResponse = await response.json();
        setData(payload);
        return payload;
      } catch (err: unknown) {
        const errorMsg = err instanceof Error ? err.message : "Failed to fetch simulation data";
        setError(errorMsg);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [baseUrl]
  );

  /**
   * Execute a discrete-event simulation run using POST /api/simulation/run
   */
  const runSimulation = useCallback(
    async (config: SimulationRunRequest): Promise<SimulationDataResponse | null> => {
      setIsLoading(true);
      setError(null);
      try {
        const url = `${baseUrl}/api/simulation/run`;
        const response = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify(config),
        });

        if (!response.ok) {
          const detail = await parseErrorMessage(response);
          throw new Error(detail);
        }

        const payload: SimulationDataResponse = await response.json();
        setData(payload);
        return payload;
      } catch (err: unknown) {
        const errorMsg = err instanceof Error ? err.message : "Failed to execute simulation run";
        setError(errorMsg);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [baseUrl]
  );

  /**
   * Reset simulation state to default parameters via POST /api/simulation/reset
   */
  const resetSimulation = useCallback(async (): Promise<SimulationDataResponse | null> => {
    setIsLoading(true);
    setError(null);
    try {
      const url = `${baseUrl}/api/simulation/reset`;
      const response = await fetch(url, {
        method: "POST",
        headers: {
          Accept: "application/json",
        },
      });

      if (!response.ok) {
        const detail = await parseErrorMessage(response);
        throw new Error(detail);
      }

      const payload: SimulationDataResponse = await response.json();
      setData(payload);
      return payload;
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : "Failed to reset simulation";
      setError(errorMsg);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [baseUrl]);

  /**
   * Auto-fetch initial telemetry snapshot on mount
   */
  useEffect(() => {
    let isMounted = true;

    if (autoFetch) {
      fetchSimulationData(initialIncludeStays).then((res) => {
        if (!isMounted && res) {
          // Component unmounted during initial load
        }
      });
    }

    return () => {
      isMounted = false;
    };
  }, [autoFetch, fetchSimulationData, initialIncludeStays]);

  return {
    data,
    isLoading,
    error,
    fetchSimulationData,
    runSimulation,
    resetSimulation,
  };
}

export default useForecastData;