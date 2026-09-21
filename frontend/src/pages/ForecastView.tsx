import React from 'react';
import { 
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { Target, TrendingUp, AlertTriangle } from 'lucide-react';
import './Forecast.css';

const forecastData = [
  { time: '08:00', actual: 45, predicted: 48, band: [40, 55] },
  { time: '09:00', actual: 52, predicted: 50, band: [42, 58] },
  { time: '10:00', actual: 68, predicted: 65, band: [55, 75] },
  { time: '11:00', actual: 74, predicted: 72, band: [62, 82] },
  { time: '12:00', actual: null, predicted: 85, band: [75, 95] },
  { time: '13:00', actual: null, predicted: 90, band: [80, 100] },
  { time: '14:00', actual: null, predicted: 88, band: [78, 98] },
  { time: '15:00', actual: null, predicted: 82, band: [72, 92] },
  { time: '16:00', actual: null, predicted: 75, band: [65, 85] },
];

export function ForecastView() {
  return (
    <div className="forecast-page">
      <div className="page-header">
        <div className="header-title">
          <h1>Demand Forecast</h1>
          <p>Machine Learning predictions for patient arrivals and LOS.</p>
        </div>
      </div>

      <div className="forecast-summary">
        <div className="summary-card glass-card">
          <Target size={24} className="summary-icon blue" />
          <div className="summary-info">
            <h4>Prediction Accuracy (MAE)</h4>
            <span className="summary-value">4.2 Patients</span>
            <span className="summary-trend positive">Top 5% Model Performance</span>
          </div>
        </div>

        <div className="summary-card glass-card">
          <TrendingUp size={24} className="summary-icon amber" />
          <div className="summary-info">
            <h4>Peak Expected Surge</h4>
            <span className="summary-value">13:00</span>
            <span className="summary-trend">90 Patients (Median)</span>
          </div>
        </div>

        <div className="summary-card glass-card warning-state">
          <AlertTriangle size={24} className="summary-icon red" />
          <div className="summary-info">
            <h4>Upper Bound Alert</h4>
            <span className="summary-value">100 Patients</span>
            <span className="summary-trend negative">May exceed max ED capacity</span>
          </div>
        </div>
      </div>

      <div className="forecast-chart-container glass-card">
        <div className="chart-header">
          <h3>ED Arrivals Forecast (Next 8 Hours)</h3>
          <p>The shaded band represents the 95% confidence interval ($y_{lower}$ to $y_{upper}$)</p>
        </div>
        
        <div className="main-chart">
          <ResponsiveContainer width="100%" height={400}>
            <ComposedChart data={forecastData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="time" stroke="#94a3b8" tickLine={false} axisLine={false} />
              <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc' }}
                itemStyle={{ color: '#f8fafc' }}
              />
              <Legend wrapperStyle={{ paddingTop: '20px' }} />
              
              {/* Shaded Area for Confidence Interval */}
              <Area 
                type="monotone" 
                dataKey="band" 
                stroke="none" 
                fill="#3b82f6" 
                fillOpacity={0.15} 
                name="95% Confidence Interval" 
              />

              {/* To truly draw a band in Recharts without masking, we use an array dataKey [lower, upper]. Recharts 2.x supports this. Let's try [lower, upper] syntax. */}
              
              {/* Actual historical data */}
              <Line 
                type="monotone" 
                dataKey="actual" 
                stroke="#10b981" 
                strokeWidth={3} 
                dot={{ r: 4 }} 
                activeDot={{ r: 6 }} 
                name="Actual Arrivals" 
              />

              {/* Predicted median line */}
              <Line 
                type="monotone" 
                dataKey="predicted" 
                stroke="#3b82f6" 
                strokeWidth={3} 
                strokeDasharray="5 5" 
                dot={{ r: 4 }} 
                name="Predicted (Median)" 
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
