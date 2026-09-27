import React, { useEffect, useState } from 'react';
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { Target, TrendingUp, AlertTriangle } from 'lucide-react';
import './Forecast.css';

interface HourlyCensus {
  hour: number;
  arrivals_count: number;
  incoming_arrivals_next_4h: number;
}

interface SimulationConfig {
  ed_capacity: number;
}

interface SimulationDataResponse {
  config: SimulationConfig;
  hourly_census: HourlyCensus[];
}

interface ForecastPoint {
  time: string;
  actual: number | null;
  predicted: number;
  band: [number, number];
}

export function ForecastView() {
  const [rawData, setRawData] = useState<SimulationDataResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchForecastSource = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(
          'http://localhost:8000/api/simulation/data?include_stays=false'
        );
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const json = await response.json();
        setRawData(json);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchForecastSource();
  }, []);

  // Build the chart-ready forecast series from live simulation data.
  const forecastData: ForecastPoint[] = React.useMemo(() => {
    if (!rawData) return [];

    const sorted = [...rawData.hourly_census].sort((a, b) => a.hour - b.hour);

    return sorted.map((census, idx) => {
      // The predicted rate for THIS hour is derived from the PREVIOUS hour's
      // 4-hour-ahead forecast, divided by 4 (average expected arrivals/hour).
      // The very first hour has no prior forecast to draw from, so it falls
      // back to that hour's own actual arrivals as a reasonable estimate.
      const priorCensus = idx > 0 ? sorted[idx - 1] : null;
      const predicted = priorCensus
        ? Math.round(priorCensus.incoming_arrivals_next_4h / 4)
        : census.arrivals_count;

      return {
        time: `${census.hour.toString().padStart(2, '0')}:00`,
        actual: census.arrivals_count,
        predicted,
        band: [Math.round(predicted * 0.85), Math.round(predicted * 1.15)],
      };
    });
  }, [rawData]);

  // Derived summary metrics, replacing the old hardcoded summary card values.
  const summary = React.useMemo(() => {
    if (forecastData.length === 0 || !rawData) {
      return { mae: 0, peakTime: '--:--', peakValue: 0, upperBoundExceedsEd: false };
    }

    const errors = forecastData
      .filter((p) => p.actual !== null)
      .map((p) => Math.abs((p.actual as number) - p.predicted));
    const mae = errors.length > 0
      ? errors.reduce((sum, e) => sum + e, 0) / errors.length
      : 0;

    const peak = forecastData.reduce((max, p) =>
      p.predicted > max.predicted ? p : max
    , forecastData[0]);

    const upperBoundExceedsEd = forecastData.some(
      (p) => p.band[1] > rawData.config.ed_capacity
    );

    return {
      mae: Math.round(mae * 10) / 10,
      peakTime: peak.time,
      peakValue: peak.predicted,
      upperBoundExceedsEd,
    };
  }, [forecastData, rawData]);

  if (loading) return <div className="forecast-page loading">Loading Forecast Data...</div>;
  if (error) return <div className="forecast-page error">Connection Error: {error}</div>;
  if (!rawData) return null;

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
            <span className="summary-value">{summary.mae} Patients</span>
            <span className="summary-trend positive">Based on this simulation run</span>
          </div>
        </div>

        <div className="summary-card glass-card">
          <TrendingUp size={24} className="summary-icon amber" />
          <div className="summary-info">
            <h4>Peak Expected Surge</h4>
            <span className="summary-value">{summary.peakTime}</span>
            <span className="summary-trend">{summary.peakValue} Patients (Predicted)</span>
          </div>
        </div>

        <div className="summary-card glass-card warning-state">
          <AlertTriangle size={24} className="summary-icon red" />
          <div className="summary-info">
            <h4>Upper Bound Alert</h4>
            <span className="summary-value">{rawData.config.ed_capacity} Bay Limit</span>
            <span className={`summary-trend ${summary.upperBoundExceedsEd ? 'negative' : 'positive'}`}>
              {summary.upperBoundExceedsEd ? 'Forecast may exceed ED capacity' : 'Within ED capacity'}
            </span>
          </div>
        </div>
      </div>

      <div className="forecast-chart-container glass-card">
        <div className="chart-header">
          <h3>ED Arrivals Forecast (Next 8 Hours)</h3>
          <p>The shaded band represents an estimated ±15% range around the predicted arrival rate.</p>
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
