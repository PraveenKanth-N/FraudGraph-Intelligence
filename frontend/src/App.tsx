import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppLayout } from './components/layout/AppLayout';
import { OverviewPage } from './pages/OverviewPage';
import { GraphExplorerPage } from './pages/GraphExplorerPage';
import { TransactionsPage } from './pages/TransactionsPage';
import { CudaLabPage } from './pages/CudaLabPage';
import { ExperimentManagerPage } from './pages/ExperimentManagerPage';
import { EvaluationCenterPage } from './pages/EvaluationCenterPage';
import { DatasetsPage } from './pages/DatasetsPage';
import { ReportsPage } from './pages/ReportsPage';
import { NotFoundPage } from './pages/NotFoundPage';
import './styles/design-tokens.css';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AppLayout />}>
          <Route index element={<OverviewPage />} />
          <Route path="overview" element={<Navigate to="/" replace />} />
          <Route path="explorer" element={<GraphExplorerPage />} />
          <Route path="transactions" element={<TransactionsPage />} />
          <Route path="cuda-lab" element={<CudaLabPage />} />
          <Route path="experiments" element={<ExperimentManagerPage />} />
          <Route path="evaluation" element={<EvaluationCenterPage />} />
          <Route path="datasets" element={<DatasetsPage />} />
          <Route path="reports" element={<ReportsPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default App;
