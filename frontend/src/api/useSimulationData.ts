// frontend/src/api/useSimulationData.ts
/**
 * TanStack Query hooks and API client for the Digital Twin simulation platform.
 *
 * Provides strongly-typed React hooks to query cached telemetry, trigger
 * discrete-event simulation runs, and reset the simulation environment.
 */

import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationResult,
  type UseQueryResult,
} from "@tanstack/react-query";

// ---------------------------------------------------------------------------
// Base URL and HTTP Client Setup
// ---------------------------------------------------------------------------
const API_BASE_URL: string =
  (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_API_BASE_URL ||
  "http://localhost:8000";

export class ApiError extends Error {
  public readonly status: number;
  public readonly details?: unknown;

  constructor(message: string, status: number, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

async function requestJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    let errorDetail = response.statusText;
    try {
      const errJson = await response.json();
      errorDetail = errJson.detail || JSON.stringify(errJson);
    } catch {
      // Fallback to response statusText if JSON parsing fails
    }
    throw new ApiError(
      `API request failed with status ${response.status}: ${errorDetail}`,
      response.status,
      errorDetail
    );
  }

  return response.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Domain Enums and Contract Types (Mirrored from Backend Schemas)
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

export type StaffRole =
  | "Physician"
  | "Charge Nurse"
  | "Registered Nurse"
  | "Nursing Assistant";

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

export interface SimulationDataResponse {
  status: string;
  generated_at: string;
  config: Required<SimulationRunRequest>;
  metrics: SimulationMetricsResponse;
  hourly_census: HourlyCensusContract[];
  bed_topology: BedTopologyContract[];
  stays: PatientStayContract[];
}

export interface HealthCheckResponse {
  status: string;
  service: string;
  version: string;
  phase: string;
  active_simulation_timestamp: string | null;
}

// ---------------------------------------------------------------------------
// Query Keys Factory
// ---------------------------------------------------------------------------
export const simulationKeys = {
  all: ["simulation"] as const,
  health: () => [...simulationKeys.all, "health"] as const,
  data: (includeStays: boolean) =>
    [...simulationKeys.all, "data", { includeStays }] as const,
};

// ---------------------------------------------------------------------------
// API Direct Fetch Functions
// ---------------------------------------------------------------------------
export async function fetchHealthCheck(): Promise<HealthCheckResponse> {
  return requestJson<HealthCheckResponse>("/");
}

export async function fetchSimulationData(
  includeStays: boolean = true
): Promise<SimulationDataResponse> {
  const queryParam = `?include_stays=${includeStays ? "true" : "false"}`;
  return requestJson<SimulationDataResponse>(`/api/simulation/data${queryParam}`);
}

export async function runSimulation(
  payload: SimulationRunRequest
): Promise<SimulationDataResponse> {
  return requestJson<SimulationDataResponse>("/api/simulation/run", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function resetSimulation(): Promise<SimulationDataResponse> {
  return requestJson<SimulationDataResponse>("/api/simulation/reset", {
    method: "POST",
  });
}

// ---------------------------------------------------------------------------
// TanStack Query Hooks
// ---------------------------------------------------------------------------

export interface UseSimulationDataOptions {
  includeStays?: boolean;
  enabled?: boolean;
  refetchInterval?: number | false;
}

/**
 * Hook to retrieve the current simulation telemetry, bed topology, and patient stays.
 */
export function useSimulationData(
  options: UseSimulationDataOptions = {}
): UseQueryResult<SimulationDataResponse, ApiError> {
  const { includeStays = true, enabled = true, refetchInterval = false } = options;

  return useQuery<SimulationDataResponse, ApiError>({
    queryKey: simulationKeys.data(includeStays),
    queryFn: () => fetchSimulationData(includeStays),
    enabled,
    refetchInterval,
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
}

/**
 * Hook to check API health and active simulation timestamp.
 */
export function useSimulationHealth(): UseQueryResult<HealthCheckResponse, ApiError> {
  return useQuery<HealthCheckResponse, ApiError>({
    queryKey: simulationKeys.health(),
    queryFn: fetchHealthCheck,
    staleTime: 1000 * 10, // 10 seconds
  });
}

/**
 * Hook to execute a customized simulation scenario and atomically refresh caches.
 */
export function useRunSimulation(): UseMutationResult<
  SimulationDataResponse,
  ApiError,
  SimulationRunRequest
> {
  const queryClient = useQueryClient();

  return useMutation<SimulationDataResponse, ApiError, SimulationRunRequest>({
    mutationFn: runSimulation,
    onSuccess: (newData) => {
      // Optimistically seed queries with returned payload
      queryClient.setQueryData(simulationKeys.data(true), newData);
      queryClient.setQueryData(simulationKeys.data(false), {
        ...newData,
        stays: [],
      });
      // Invalidate health check to synchronize timestamp
      queryClient.invalidateQueries({ queryKey: simulationKeys.health() });
    },
  });
}

/**
 * Hook to reset simulation state to the 24-hour default baseline.
 */
export function useResetSimulation(): UseMutationResult<
  SimulationDataResponse,
  ApiError,
  void
> {
  const queryClient = useQueryClient();

  return useMutation<SimulationDataResponse, ApiError, void>({
    mutationFn: resetSimulation,
    onSuccess: (newData) => {
      queryClient.setQueryData(simulationKeys.data(true), newData);
      queryClient.setQueryData(simulationKeys.data(false), {
        ...newData,
        stays: [],
      });
      queryClient.invalidateQueries({ queryKey: simulationKeys.health() });
    },
  });
}