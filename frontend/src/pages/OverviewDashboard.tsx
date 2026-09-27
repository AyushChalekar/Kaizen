import React, { useEffect, useState } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import { Users, UserPlus, Clock } from 'lucide-react';
import './Dashboard.css';

interface HourlyCensus {
  hour: number;
  ed_occupancy: number;
  ward_occupancy: number;
  icu_occupancy: number;
  active_nurses: number;
  active_doctors: number;
  patients_in_queue: number;
  incoming_arrivals_next_4h: number;
}

interface SimulationConfig {
  ed_capacity: number;
  ward_capacity: number;
  icu_capacity: number;
}

interface SimulationMetrics {
  average_los_hours: number;
  peak_queue_length: number;
}

interface SimulationDataResponse {
  generated_at: string;
  config: SimulationConfig;
  metrics: SimulationMetrics;
  hourly_census: HourlyCensus[];
}

export function OverviewDashboard() {
  const [data, setData] = useState<SimulationDataResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchOverviewSource = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(
          'http://localhost:8000/api/simulation/data?include_stays=false'
        );
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const json = await response.json();
        setData(json);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchOverviewSource();
  }, []);

  // Occupancy status thresholds, shared by all three departments.
  const occupancyStatus = (pct: number): { label: string; cssClass: string } => {
    if (pct >= 90) return { label: 'Critical', cssClass: 'critical' };
    if (pct >= 70) return { label: 'High', cssClass: 'warning' };
    return { label: 'Normal', cssClass: 'normal' };
  };

  const latest = data && data.hourly_census.length > 0
    ? data.hourly_census[data.hourly_census.length - 1]
    : null;

  const edPct = latest && data ? Math.round((latest.ed_occupancy / data.config.ed_capacity) * 100) : 0;
  const wardPct = latest && data ? Math.round((latest.ward_occupancy / data.config.ward_capacity) * 100) : 0;
  const icuPct = latest && data ? Math.round((latest.icu_occupancy / data.config.icu_capacity) * 100) : 0;

  const edStatus = occupancyStatus(edPct);
  const wardStatus = occupancyStatus(wardPct);
  const icuStatus = occupancyStatus(icuPct);

  // Last 6 simulated hours for the trend chart (or fewer if the run is shorter).
  const trendData = data
    ? data.hourly_census.slice(-6).map((c) => ({
        time: `${c.hour.toString().padStart(2, '0')}:00`,
        ed: c.ed_occupancy,
        ward: c.ward_occupancy,
      }))
    : [];

  if (loading) return <div className="dashboard loading">Loading Overview...</div>;
  if (error) return <div className="dashboard error">Connection Error: {error}</div>;
  if (!data || !latest) return null;

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>Overview</h1>
        <p>
          Reflecting last simulation run &bull; Generated: {new Date(data.generated_at).toLocaleString()}
        </p>
      </div>

      {/* Occupancy Gauges Section */}
      <div className="dashboard-grid">
        <div className="card glass-card">
          <div className="card-header">
            <h3>ED Occupancy</h3>
            <span className={`status-indicator ${edStatus.cssClass}`}>{edStatus.label}</span>
          </div>
          <div className="progress-circle-container">
            <div className={`progress-circle ${edStatus.cssClass}-circle`}>
              <span className="percentage">{edPct}%</span>
            </div>
            <p>{latest.ed_occupancy} / {data.config.ed_capacity} Beds</p>
          </div>
        </div>

        <div className="card glass-card">
          <div className="card-header">
            <h3>Ward Occupancy</h3>
            <span className={`status-indicator ${wardStatus.cssClass}`}>{wardStatus.label}</span>
          </div>
          <div className="progress-circle-container">
            <div className={`progress-circle ${wardStatus.cssClass}-circle`}>
              <span className="percentage">{wardPct}%</span>
            </div>
            <p>{latest.ward_occupancy} / {data.config.ward_capacity} Beds</p>
          </div>
        </div>

        <div className="card glass-card">
          <div className="card-header">
            <h3>ICU Occupancy</h3>
            <span className={`status-indicator ${icuStatus.cssClass}`}>{icuStatus.label}</span>
          </div>
          <div className="progress-circle-container">
            <div className={`progress-circle ${icuStatus.cssClass}-circle`}>
              <span className="percentage">{icuPct}%</span>
            </div>
            <p>{latest.icu_occupancy} / {data.config.icu_capacity} Beds</p>
          </div>
        </div>
      </div>

      <div className="dashboard-grid-2">
        {/* Staffing & Queue */}
        <div className="stats-column">
          <div className="card stat-card">
            <div className="stat-icon bg-blue">
              <Users size={24} />
            </div>
            <div className="stat-content">
              <h4>Active Staff</h4>
              <p className="stat-value">{latest.active_nurses} <span className="stat-sub">Nurses</span> &bull; {latest.active_doctors} <span className="stat-sub">Doctors</span></p>
            </div>
          </div>

          <div className="card stat-card">
            <div className="stat-icon bg-amber">
              <Clock size={24} />
            </div>
            <div className="stat-content">
              <h4>Patients in Queue</h4>
              <p className="stat-value">{latest.patients_in_queue} <span className="stat-sub">Avg LOS: {data.metrics.average_los_hours.toFixed(1)}h</span></p>
            </div>
          </div>

          <div className="card stat-card">
            <div className="stat-icon bg-teal">
              <UserPlus size={24} />
            </div>
            <div className="stat-content">
              <h4>Expected Arrivals (4h)</h4>
              <p className="stat-value">~{latest.incoming_arrivals_next_4h} <span className="stat-sub">Patients</span></p>
            </div>
          </div>
        </div>

        {/* Chart */}
        <div className="card chart-card">
          <h3>Occupancy Trend (Last 6 Hours)</h3>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={trendData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorEd" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorWard" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc' }}
                  itemStyle={{ color: '#f8fafc' }}
                />
                <Area type="monotone" dataKey="ed" stroke="#f59e0b" fillOpacity={1} fill="url(#colorEd)" name="ED" />
                <Area type="monotone" dataKey="ward" stroke="#3b82f6" fillOpacity={1} fill="url(#colorWard)" name="Ward" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
