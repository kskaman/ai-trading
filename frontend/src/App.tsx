import { NavLink, Navigate, Route, Routes } from 'react-router-dom';
import StatusPage from './pages/StatusPage';
import PaperTradePage from './pages/PaperTradePage';
import ComingSoon from './pages/ComingSoon';

const PAGES = [
  { path: '/status', label: 'System status' },
  { path: '/data', label: 'Data health' },
  { path: '/features', label: 'Features' },
  { path: '/paper-trade', label: 'Paper trade' },
  { path: '/backtest', label: 'Backtest' },
];

export default function App() {
  return (
    <div className="layout">
      <header className="topbar">
        <span className="brand">AI Trading</span>
        <nav>
          {PAGES.map((page) => (
            <NavLink key={page.path} to={page.path}>
              {page.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Navigate to="/status" replace />} />
          <Route path="/status" element={<StatusPage />} />
          <Route path="/data" element={<ComingSoon title="Data health" phase={2} />} />
          <Route path="/features" element={<ComingSoon title="Features" phase={3} />} />
          <Route path="/paper-trade" element={<PaperTradePage />} />
          <Route path="/strategies" element={<Navigate to="/paper-trade" replace />} />
          <Route path="/backtest" element={<ComingSoon title="Backtest" phase={4} />} />
        </Routes>
      </main>
    </div>
  );
}
