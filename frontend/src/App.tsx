import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Login } from './pages/Login';
import { OverviewDashboard } from './pages/OverviewDashboard';
import { OptimizationView } from './pages/OptimizationView';
import { ExplainabilityView } from './pages/ExplainabilityView';
import { SimulationView } from './pages/SimulationView';
import { ForecastDashboard } from './pages/ForecastDashboard';
import { QueueStaffView } from './pages/QueueStaffView';
import { SettingsView } from './pages/SettingsView';
import { AppLayout } from './components/layout/AppLayout';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/" element={<Navigate to="/login" replace />} />
        
        <Route path="/app" element={<AppLayout />}>
          <Route index element={<Navigate to="/app/dashboard" replace />} />
          <Route path="dashboard" element={<OverviewDashboard />} />
          <Route path="optimization" element={<OptimizationView />} />
          <Route path="explainability" element={<ExplainabilityView />} />
          <Route path="simulation" element={<SimulationView />} />
          <Route path="forecast" element={<ForecastDashboard />} />
          <Route path="queue" element={<QueueStaffView />} />
          <Route path="settings" element={<SettingsView />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
