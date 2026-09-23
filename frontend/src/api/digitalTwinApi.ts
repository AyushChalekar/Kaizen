import type { SimulationDataResponse } from "../types/simulation";

/**
 * Fetches simulation data from the backend API.
 * @returns A promise that resolves to the simulation data response.
 * @throws Error if the request fails or the server is unreachable.
 */
export async function fetchSimulationData(): Promise<SimulationDataResponse> {
    try {
        const response = await fetch("http://localhost:8000/api/simulation/data");

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data: SimulationDataResponse = await response.json();
        return data;
    } catch (error) {
        console.error("Failed to fetch simulation data:", error);
        // Return a fallback response with empty data if the server is unreachable
        return {
            message: "Failed to fetch simulation data",
            data: {
                patient_stays: [],
                hourly_census: []
            }
        };
    }
}
