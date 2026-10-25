import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { AlertOctagon, Home, ArrowLeft } from 'lucide-react';
import { CyberButton } from '../components/common/CyberButton';

export const NotFoundPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <div className="flex-1 flex items-center justify-center p-6 bg-[var(--surface-obsidian)]">
      <div className="max-w-md w-full p-8 border border-[var(--alert-critical)] bg-[var(--surface-graphite)] text-center shadow-2xl">
        <div className="w-12 h-12 mx-auto mb-4 border border-[var(--alert-critical)] bg-[rgba(255,59,48,0.1)] flex items-center justify-center">
          <AlertOctagon size={28} className="text-[var(--alert-critical)]" />
        </div>

        <div className="font-mono text-xs text-[var(--alert-critical)] uppercase tracking-wider font-semibold mb-1">
          404 // ROUTE OUT OF TOPOLOGY BOUNDS
        </div>

        <h2 className="font-serif text-2xl text-[var(--tertiary)] mb-2 font-normal">
          Unregistered View Path
        </h2>

        <p className="text-xs text-[var(--text-muted)] font-mono mb-4 leading-relaxed break-all">
          URI <span className="text-[var(--primary)]">{location.pathname}</span> is not an active view in the FraudGraph Intelligence platform sitemap.
        </p>

        <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)] text-[11px] font-mono text-[var(--metric-dim)] text-left mb-6 space-y-1">
          <div>AVAILABLE PLATFORM VIEWS:</div>
          <div className="text-[var(--text-muted)]">
            / (Overview) • /explorer • /transactions • /cuda-lab • /experiments • /evaluation • /datasets • /reports
          </div>
        </div>

        <div className="flex gap-3 justify-center">
          <CyberButton
            variant="secondary"
            size="md"
            onClick={() => navigate(-1)}
            icon={<ArrowLeft size={14} />}
          >
            PREVIOUS VIEW
          </CyberButton>
          <CyberButton
            variant="primary"
            size="md"
            onClick={() => navigate('/')}
            icon={<Home size={14} />}
          >
            PLATFORM OVERVIEW
          </CyberButton>
        </div>
      </div>
    </div>
  );
};
