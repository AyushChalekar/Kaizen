// src/pages/SimulationView.tsx
import React, { useState, useEffect } from 'react';
import './SimulationView.css';

interface SimulationConfig {
  duration_hours: number;
  ed_capacity: number;
  ward_capacity: number;
  icu_capacity: number;
  seed: number;
}

interface SimulationMetrics {
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

interface PatientStay {
  stay_id: number;
  triage_acuity: number;
  chief_complaint: string;
  sbp: number;
  dbp: number;
  heart_rate: number;
  temp_c: number;
  arrival_time: string;
  triage_start_time: string;
  bed_assigned_time: string;
  discharge_time: string;
  initial_care_unit: string;
  disposition: string;
  icu_transfer_flag: number;
  los_hours: number;
}

interface HourlyCensus {
  hour: number;
  shift_id: string;
  active_nurses: number;
  active_doctors: number;
  arrivals_count: number;
  patients_in_queue: number;
  ed_occupancy: number;
  ward_occupancy: number;
  icu_occupancy: number;
}

interface SimulationData {
  generated_at: string;
  config: SimulationConfig;
  metrics: SimulationMetrics;
  stays: PatientStay[];
  hourly_census: HourlyCensus[];
}

export function SimulationView() {
  const [data, setData] = useState<SimulationData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSimulation = async (endpoint: string, method: string = 'GET', body?: any) => {
    setLoading(true);
    setError(null);
    try {
      const options: RequestInit = {
        method,
        headers: { 'Content-Type': 'application/json' },
      };
      if (body) options.body = JSON.stringify(body);

      const response = await fetch(`http://localhost:8000${endpoint}`, options);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      
      const jsonData = await response.json();
      setData(jsonData);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSimulation('/api/simulation/data?include_stays=true');
  }, []);

  const runStressTest = () => {
    fetchSimulation('/api/simulation/run', 'POST', {
      duration_hours: 24,
      ed_capacity: 30, // Bottlenecked ED
      ward_capacity: 100, // Reduced Ward
      icu_capacity: 15, // Reduced ICU
      seed: 99
    });
  };

  const runBaseline = () => {
    fetchSimulation('/api/simulation/reset', 'POST');
  };

  if (loading) return <div className="sim-container loading">Executing Discrete-Event Simulation...</div>;
  if (error) return <div className="sim-container error">Connection Error: {error}</div>;
  if (!data) return null;

  return (
    <div className="sim-container">
      <div className="sim-header">
        <div>
          <h1>Backend Capability Validation Dashboard</h1>
          <p className="timestamp">Generated: {new Date(data.generated_at).toLocaleString()}</p>
        </div>
        <div className="sim-actions">
          <button className="btn-baseline" onClick={runBaseline}>Run Baseline (24h)</button>
          <button className="btn-stress" onClick={runStressTest}>Run Resource Stress Test</button>
        </div>
      </div>

      {/* Active Conditions Panel */}
      <div className="config-panel">
        <h2>Active Simulation Conditions</h2>
        <div className="config-grid">
          <div className="config-item">
            <label>Duration</label>
            <span>{data.config.duration_hours} hours</span>
          </div>
          <div className="config-item">
            <label>ED Capacity</label>
            <span>{data.config.ed_capacity} bays</span>
          </div>
          <div className="config-item">
            <label>Ward Capacity</label>
            <span>{data.config.ward_capacity} beds</span>
          </div>
          <div className="config-item">
            <label>ICU Capacity</label>
            <span>{data.config.icu_capacity} beds</span>
          </div>
          <div className="config-item">
            <label>RNG Seed</label>
            <span>{data.config.seed}</span>
          </div>
        </div>
      </div>

      <div className="metrics-grid">
        <div className="metric-box">
          <label>Peak Queue Length</label>
          <span className={data.metrics.peak_queue_length > 10 ? 'text-danger' : 'text-success'}>
            {data.metrics.peak_queue_length}
          </span>
        </div>
        <div className="metric-box">
          <label>Completed Encounters</label>
          <span>{data.metrics.completed_stays_count}</span>
        </div>
        <div className="metric-box">
          <label>ED Utilization</label>
          <span className={data.metrics.ed_utilization_pct > 85 ? 'text-danger' : ''}>
            {data.metrics.ed_utilization_pct.toFixed(1)}%
          </span>
        </div>
        <div className="metric-box">
          <label>ICU Utilization</label>
          <span className={data.metrics.icu_utilization_pct > 85 ? 'text-danger' : ''}>
            {data.metrics.icu_utilization_pct.toFixed(1)}%
          </span>
        </div>
        <div className="metric-box">
          <label>Average LoS</label>
          <span>{data.metrics.average_los_hours.toFixed(1)} hrs</span>
        </div>
      </div>

      <div className="data-panels">
        {/* Capability 1 & 3: Systemic Bottlenecks & Diurnal Staffing */}
        <section className="data-panel">
          <h2>Diurnal Staffing & Systemic Bottlenecks</h2>
          <p className="panel-desc">Validating HourlyCensusContract alignment with shift mappings, capacity constraints, and queue formations[cite: 4].</p>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Hour</th>
                  <th>Shift</th>
                  <th>Staff (RN / MD)</th>
                  <th>Arrivals</th>
                  <th>Queue Size</th>
                  <th>ED Occ</th>
                  <th>Ward Occ</th>
                  <th>ICU Occ</th>
                </tr>
              </thead>
              <tbody>
                {data.hourly_census.map((census, idx) => (
                  <tr key={idx} className={census.patients_in_queue > 0 ? 'row-warning' : ''}>
                    <td>{census.hour.toString().padStart(2, '0')}:00</td>
                    <td>{census.shift_id}</td>
                    <td>{census.active_nurses} / {census.active_doctors}</td>
                    <td>{census.arrivals_count}</td>
                    <td className={census.patients_in_queue > 0 ? 'text-danger font-bold' : ''}>
                      {census.patients_in_queue}
                    </td>
                    <td>{census.ed_occupancy} / {data.config.ed_capacity}</td>
                    <td>{census.ward_occupancy} / {data.config.ward_capacity}</td>
                    <td>{census.icu_occupancy} / {data.config.icu_capacity}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <div className="grid-2-col">
          {/* Capability 2: Clinical Trajectories */}
          <section className="data-panel">
            <h2>Clinical Trajectories & Monotonicity</h2>
            <p className="panel-desc">Validating sequential timestamp logic (Arrival &le; Triage &le; Bed &le; Discharge) and patient flow[cite: 4].</p>
            <div className="table-wrapper max-h-500">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Timeline (Arr &rarr; Trg &rarr; Bed &rarr; Dis)</th>
                    <th>Routing Path</th>
                    <th>Valid Sequence</th>
                  </tr>
                </thead>
                <tbody>
                  {data.stays.slice(0, 50).map((stay) => {
                    const isMonotonic = 
                      stay.arrival_time <= stay.triage_start_time &&
                      stay.triage_start_time <= stay.bed_assigned_time &&
                      stay.bed_assigned_time <= stay.discharge_time;

                    return (
                      <tr key={stay.stay_id}>
                        <td>{stay.stay_id}</td>
                        <td className="text-xs">
                          {stay.arrival_time.split(' ')[1]} &rarr; {stay.triage_start_time.split(' ')[1]} &rarr; {stay.bed_assigned_time.split(' ')[1]} &rarr; {stay.discharge_time.split(' ')[1]}
                        </td>
                        <td>
                          {stay.initial_care_unit} &rarr; {stay.disposition}
                          {stay.icu_transfer_flag === 1 ? ' (ICU Esc)' : ''}
                        </td>
                        <td className={isMonotonic ? 'text-success font-bold' : 'text-danger font-bold'}>
                          {isMonotonic ? 'PASS' : 'FAIL'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>

          {/* Capability 4: Physiological Invariants */}
          <section className="data-panel">
            <h2>Physiological Invariants</h2>
            <p className="panel-desc">Validating stochastic sampling bounds. Minimum pulse pressure (SBP - DBP) must remain &ge; 15.0 mmHg[cite: 4].</p>
            <div className="table-wrapper max-h-500">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>ESI</th>
                    <th>SBP</th>
                    <th>DBP</th>
                    <th>Pulse Pressure</th>
                    <th>HR</th>
                    <th>O2 Sat</th>
                  </tr>
                </thead>
                <tbody>
                  {data.stays.slice(0, 50).map((stay) => {
                    const pulsePressure = (stay.sbp - stay.dbp).toFixed(1);
                    const isValid = (stay.sbp - stay.dbp) >= 15.0;

                    return (
                      <tr key={stay.stay_id}>
                        <td>{stay.stay_id}</td>
                        <td>{stay.triage_acuity}</td>
                        <td>{stay.sbp}</td>
                        <td>{stay.dbp}</td>
                        <td className={isValid ? 'text-success font-bold' : 'text-danger font-bold'}>
                          {pulsePressure}
                        </td>
                        <td>{stay.heart_rate}</td>
                        <td>{stay.temp_c}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}