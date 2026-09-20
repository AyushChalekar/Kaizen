import React from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell 
} from 'recharts';
import { TrendingDown, TrendingUp, CheckCircle, ArrowRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import './Optimization.css';

const bedAllocationData = [
  { name: 'General', before: 180, after: 150 },
  { name: 'HDU', before: 20, after: 40 },
  { name: 'ICU', before: 20, after: 30 },
];

const staffAllocationData = [
  { name: 'ED Nurses', before: 12, after: 16 },
  { name: 'Ward Nurses', before: 25, after: 20 },
  { name: 'ICU Nurses', before: 5, after: 6 },
  { name: 'Doctors', before: 18, after: 18 },
];

export function OptimizationView() {
  const navigate = useNavigate();

  return (
    <div className="optimization-page">
      <div className="page-header">
        <div className="header-title">
          <h1>Optimization Engine</h1>
          <p>MILP/DRL recommended resource re-allocation.</p>
        </div>
        <div className="header-actions">
          <button className="btn-secondary" onClick={() => navigate('/explainability')}>
            View Explainability Log
          </button>
          <button className="btn-primary">
            Apply Recommendations <ArrowRight size={18} />
          </button>
        </div>
      </div>

      {/* KPI Improvements */}
      <div className="kpi-grid">
        <div className="kpi-card glass-card">
          <div className="kpi-icon positive">
            <TrendingDown size={24} />
          </div>
          <div className="kpi-info">
            <h4>Avg Queue Time</h4>
            <div className="kpi-values">
              <span className="value-old">45m</span>
              <span className="value-arrow">→</span>
              <span className="value-new positive">32m</span>
            </div>
            <p className="kpi-subtext">Improved by 28%</p>
          </div>
        </div>

        <div className="kpi-card glass-card">
          <div className="kpi-icon positive">
            <TrendingDown size={24} />
          </div>
          <div className="kpi-info">
            <h4>Est. Overtime Cost</h4>
            <div className="kpi-values">
              <span className="value-old">$4.2k</span>
              <span className="value-arrow">→</span>
              <span className="value-new positive">$3.5k</span>
            </div>
            <p className="kpi-subtext">Reduced by 16%</p>
          </div>
        </div>

        <div className="kpi-card glass-card">
          <div className="kpi-icon neutral">
            <CheckCircle size={24} />
          </div>
          <div className="kpi-info">
            <h4>Occupancy Balance</h4>
            <div className="kpi-values">
              <span className="value-old">72% / 95%</span>
              <span className="value-arrow">→</span>
              <span className="value-new neutral">81% / 85%</span>
            </div>
            <p className="kpi-subtext">Ward / ICU variance stabilized</p>
          </div>
        </div>
      </div>

      <div className="charts-grid">
        {/* Bed Allocation Chart */}
        <div className="chart-card glass-card">
          <h3>Bed Allocation Strategy</h3>
          <p className="chart-desc">Flexing capacity to handle projected ICU surge.</p>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={bedAllocationData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="name" stroke="#94a3b8" tickLine={false} axisLine={false} />
                <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} />
                <Tooltip 
                  cursor={{fill: 'rgba(255,255,255,0.05)'}}
                  contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc' }}
                />
                <Legend iconType="circle" wrapperStyle={{ paddingTop: '20px' }} />
                <Bar dataKey="before" name="Current Allocation" fill="#475569" radius={[4, 4, 0, 0]} />
                <Bar dataKey="after" name="Recommended Allocation" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Staff Allocation Chart */}
        <div className="chart-card glass-card">
          <h3>Staff Re-allocation Strategy</h3>
          <p className="chart-desc">Redistributing nursing staff to high-acuity areas.</p>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={staffAllocationData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="name" stroke="#94a3b8" tickLine={false} axisLine={false} />
                <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} />
                <Tooltip 
                  cursor={{fill: 'rgba(255,255,255,0.05)'}}
                  contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc' }}
                />
                <Legend iconType="circle" wrapperStyle={{ paddingTop: '20px' }} />
                <Bar dataKey="before" name="Current Allocation" fill="#475569" radius={[4, 4, 0, 0]} />
                <Bar dataKey="after" name="Recommended Allocation" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
