import React from 'react';
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { HistoricalData } from './pages/HistoricalData';
import { Forecast } from './pages/Forecast';
import { ModelPerformance } from './pages/ModelPerformance';
import { DatasetManagement } from './pages/DatasetManagement';
import { Training } from './pages/Training';

export const App: React.FC = () => {
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="history" element={<HistoricalData />} />
          <Route path="forecast" element={<Forecast />} />
          <Route path="performance" element={<ModelPerformance />} />
          <Route path="dataset" element={<DatasetManagement />} />
          <Route path="training" element={<Training />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Route>
      </Routes>
    </HashRouter>
  );
};

export default App;
