// src/pages/SimulationView.tsx
import React, { useState, useEffect } from 'react';

interface SimulationConfig {
  duration_hours: number;
  ed_capacity: number;
  ward_capacity: number;
  icu_capacity: number;
  nurse_staffing_multiplier: number;
  doctor_staffing_multiplier: number;
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

interface ScenarioPreset {
  id: string;
  label: string;
  description: string;
  isBaseline?: boolean;
  payload: {
    duration_hours: number;
    ed_capacity: number;
    ward_capacity: number;
    icu_capacity: number;
    nurse_staffing_multiplier: number;
    doctor_staffing_multiplier: number;
    seed: number;
  };
}

const SCENARIOS: ScenarioPreset[] = [
  {
    id: 'baseline',
    label: 'Baseline — Standard 24h Operations',
    description: 'Full bed capacity (ED 50 / Ward 150 / ICU 30) and full staffing.',
    isBaseline: true,
    payload: { duration_hours: 24, ed_capacity: 50, ward_capacity: 150, icu_capacity: 30, nurse_staffing_multiplier: 1.0, doctor_staffing_multiplier: 1.0, seed: 42 },
  },
  {
    id: 'icu-bed-shortage',
    label: 'Test Case 1 — ICU Bed Shortage',
    description: 'ICU capacity cut to 8 beds (from 30). Ward, ED, and staffing unaffected.',
    payload: { duration_hours: 24, ed_capacity: 50, ward_capacity: 150, icu_capacity: 8, nurse_staffing_multiplier: 1.0, doctor_staffing_multiplier: 1.0, seed: 101 },
  },
  {
    id: 'ward-bed-shortage',
    label: 'Test Case 2 — General Ward Bed Shortage',
    description: 'Ward capacity cut to 50 beds (from 150). ICU, ED, and staffing unaffected.',
    payload: { duration_hours: 24, ed_capacity: 50, ward_capacity: 50, icu_capacity: 30, nurse_staffing_multiplier: 1.0, doctor_staffing_multiplier: 1.0, seed: 102 },
  },
  {
    id: 'doctor-shortage',
    label: 'Test Case 3 — Doctor Staffing Shortage',
    description: 'Active doctors per shift cut to ~30% of baseline. Bed capacity is unchanged, but reduced staffing slows down patient care, which can still cause bed occupancy and queueing to rise.',
    payload: { duration_hours: 24, ed_capacity: 50, ward_capacity: 150, icu_capacity: 30, nurse_staffing_multiplier: 1.0, doctor_staffing_multiplier: 0.3, seed: 103 },
  },
  {
    id: 'nurse-shortage',
    label: 'Test Case 4 — Nurse Staffing Shortage',
    description: 'Active nurses per shift cut to ~40% of baseline. Bed capacity is unchanged, but reduced staffing slows down patient care, which can still cause bed occupancy and queueing to rise.',
    payload: { duration_hours: 24, ed_capacity: 50, ward_capacity: 150, icu_capacity: 30, nurse_staffing_multiplier: 0.4, doctor_staffing_multiplier: 1.0, seed: 104 },
  },
  {
    id: 'compound-crisis',
    label: 'Test Case 5 — Compound Resource Crisis',
    description: 'Simultaneous bed AND staffing shortage across ED, Ward, ICU, doctors and nurses (mass-casualty style surge).',
    payload: { duration_hours: 24, ed_capacity: 25, ward_capacity: 80, icu_capacity: 10, nurse_staffing_multiplier: 0.5, doctor_staffing_multiplier: 0.5, seed: 105 },
  },
];

export function SimulationView() {
  const [data, setData] = useState<SimulationData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>('baseline');

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

  const runSelectedScenario = () => {
    const scenario = SCENARIOS.find((s) => s.id === selectedScenarioId);
    if (!scenario) return;
    if (scenario.isBaseline) {
      fetchSimulation('/api/simulation/reset', 'POST');
    } else {
      fetchSimulation('/api/simulation/run', 'POST', scenario.payload);
    }
  };

  if (loading) return <div className="sim-container loading">Executing Discrete-Event Simulation...</div>;
  if (error) return <div className="sim-container error">Connection Error: {error}</div>;
  if (!data) return null;

  return (
    <div className="sim-container">
      <style dangerouslySetInnerHTML={{ __html: `
/* src/pages/Simulation.css */
.sim-container {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  padding: 2rem;
  background-color: #0f172a;
  color: #e2e8f0;
  min-height: 100vh;
}

.sim-container.loading {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.25rem;
  color: #38bdf8;
}

.sim-container.error {
  color: #f87171;
  display: flex;
  align-items: center;
  justify-content: center;
}

.sim-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  border-bottom: 1px solid #334155;
  padding-bottom: 1rem;
  margin-bottom: 1.5rem;
}

.sim-header h1 {
  margin: 0 0 0.5rem 0;
  font-size: 1.75rem;
  color: #f8fafc;
}

.timestamp {
  margin: 0;
  font-size: 0.875rem;
  color: #94a3b8;
}

.sim-actions {
  display: flex;
  gap: 1rem;
}

.sim-actions button {
  color: white;
  border: none;
  padding: 0.6rem 1.2rem;
  border-radius: 4px;
  cursor: pointer;
  font-family: inherit;
  font-weight: 600;
  transition: opacity 0.2s;
}

.sim-actions button:hover {
  opacity: 0.9;
}

.scenario-select {
  background-color: #1e293b;
  color: #e2e8f0;
  border: 1px solid #475569;
  border-radius: 4px;
  padding: 0.55rem 0.8rem;
  font-family: inherit;
  font-size: 0.9rem;
  min-width: 320px;
}

.btn-run-scenario {
  background-color: #b91c1c;
}

.scenario-description {
  margin: 0.5rem 0 1.5rem 0;
  font-size: 0.85rem;
  color: #94a3b8;
  font-style: italic;
}

.config-panel {
  background-color: #1e293b;
  border: 1px solid #3b82f6;
  border-radius: 6px;
  padding: 1rem 1.5rem;
  margin-bottom: 1.5rem;
}

.config-panel h2 {
  margin: 0 0 1rem 0;
  font-size: 1rem;
  color: #60a5fa;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.config-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 2rem;
}

.config-item {
  display: flex;
  flex-direction: column;
}

.config-item label {
  font-size: 0.75rem;
  color: #94a3b8;
  margin-bottom: 0.25rem;
}

.config-item span {
  font-size: 1.125rem;
  font-weight: 600;
  color: #f8fafc;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.metric-box {
  background-color: #0f172a;
  border: 1px solid #334155;
  padding: 1.25rem;
  border-radius: 6px;
  display: flex;
  flex-direction: column;
}

.metric-box label {
  font-size: 0.75rem;
  color: #94a3b8;
  margin-bottom: 0.5rem;
  text-transform: uppercase;
}

.metric-box span {
  font-size: 1.75rem;
  font-weight: 700;
  color: #38bdf8;
}

.data-panels {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.grid-2-col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
}

.data-panel {
  background-color: #1e293b;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
}

.data-panel h2 {
  margin: 0 0 0.5rem 0;
  font-size: 1.125rem;
  color: #f1f5f9;
}

.panel-desc {
  color: #94a3b8;
  font-size: 0.875rem;
  margin-bottom: 1.25rem;
}

.table-wrapper {
  overflow-x: auto;
  border: 1px solid #334155;
  border-radius: 4px;
}

.max-h-500 {
  max-height: 500px;
  overflow-y: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  text-align: left;
  font-size: 0.875rem;
}

th, td {
  padding: 0.75rem;
  border-bottom: 1px solid #334155;
  white-space: nowrap;
}

th {
  background-color: #0f172a;
  color: #cbd5e1;
  font-weight: 600;
  position: sticky;
  top: 0;
  z-index: 10;
}

tr:hover {
  background-color: #334155;
}

.row-warning {
  background-color: rgba(245, 158, 11, 0.15);
}

.truncate {
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
}

.text-xs {
  font-size: 0.75rem;
}

.text-success { color: #4ade80 !important; }
.text-danger { color: #f87171 !important; }
.font-bold { font-weight: 700; }

/* Custom Scrollbar for inner tables */
.table-wrapper::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
.table-wrapper::-webkit-scrollbar-track {
  background: #0f172a; 
}
.table-wrapper::-webkit-scrollbar-thumb {
  background: #334155; 
  border-radius: 4px;
}
.table-wrapper::-webkit-scrollbar-thumb:hover {
  background: #475569; 
}
      ` }} />
      <div className="sim-header">
        <div>
          <h1>Backend Capability Validation Dashboard</h1>
          <p className="timestamp">Generated: {new Date(data.generated_at).toLocaleString()}</p>
        </div>
        <div className="sim-actions">
          <select
            className="scenario-select"
            value={selectedScenarioId}
            onChange={(e) => setSelectedScenarioId(e.target.value)}
            aria-label="Select simulation test case"
          >
            {SCENARIOS.map((s) => (
              <option key={s.id} value={s.id}>{s.label}</option>
            ))}
          </select>
          <button className="btn-run-scenario" onClick={runSelectedScenario}>
            Run Selected Test Case
          </button>
        </div>
        <p className="scenario-description">
          {SCENARIOS.find((s) => s.id === selectedScenarioId)?.description}
        </p>
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
            <label>Doctor Staffing</label>
            <span>{Math.round(data.config.doctor_staffing_multiplier * 100)}%</span>
          </div>
          <div className="config-item">
            <label>Nurse Staffing</label>
            <span>{Math.round(data.config.nurse_staffing_multiplier * 100)}%</span>
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