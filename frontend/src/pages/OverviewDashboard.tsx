import React from 'react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts';
import { Users, UserPlus, Clock } from 'lucide-react';
import './Dashboard.css';

const occupancyData = [
  { time: '08:00', ed: 45, ward: 180, icu: 15 },
  { time: '09:00', ed: 52, ward: 182, icu: 16 },
  { time: '10:00', ed: 68, ward: 185, icu: 16 },
  { time: '11:00', ed: 74, ward: 188, icu: 17 },
  { time: '12:00', ed: 82, ward: 190, icu: 18 },
  { time: '13:00', ed: 78, ward: 195, icu: 19 },
  { time: '14:00', ed: 85, ward: 198, icu: 19 },
];

export function OverviewDashboard() {
  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>Overview</h1>
        <p>Real-time hospital occupancy and staffing metrics.</p>
      </div>

      {/* Occupancy Gauges Section */}
      <div className="dashboard-grid">
        <div className="card glass-card">
          <div className="card-header">
            <h3>ED Occupancy</h3>
            <span className="status-indicator warning">High</span>
          </div>
          <div className="progress-circle-container">
            <div className="progress-circle warning-circle">
              <span className="percentage">85%</span>
            </div>
            <p>85 / 100 Beds</p>
          </div>
        </div>

        <div className="card glass-card">
          <div className="card-header">
            <h3>Ward Occupancy</h3>
            <span className="status-indicator normal">Normal</span>
          </div>
          <div className="progress-circle-container">
            <div className="progress-circle normal-circle">
              <span className="percentage">72%</span>
            </div>
            <p>198 / 275 Beds</p>
          </div>
        </div>

        <div className="card glass-card">
          <div className="card-header">
            <h3>ICU Occupancy</h3>
            <span className="status-indicator critical">Critical</span>
          </div>
          <div className="progress-circle-container">
            <div className="progress-circle critical-circle">
              <span className="percentage">95%</span>
            </div>
            <p>19 / 20 Beds</p>
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
              <p className="stat-value">42 <span className="stat-sub">Nurses</span> &bull; 18 <span className="stat-sub">Doctors</span></p>
            </div>
          </div>

          <div className="card stat-card">
            <div className="stat-icon bg-amber">
              <Clock size={24} />
            </div>
            <div className="stat-content">
              <h4>Patients in Queue</h4>
              <p className="stat-value">24 <span className="stat-sub">Avg Wait: 45m</span></p>
            </div>
          </div>

          <div className="card stat-card">
            <div className="stat-icon bg-teal">
              <UserPlus size={24} />
            </div>
            <div className="stat-content">
              <h4>Expected Arrivals (4h)</h4>
              <p className="stat-value">~35 <span className="stat-sub">Patients</span></p>
            </div>
          </div>
        </div>

        {/* Chart */}
        <div className="card chart-card">
          <h3>Occupancy Trend (Last 6 Hours)</h3>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={occupancyData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
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
