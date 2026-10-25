import React from 'react';
import { AlertCircle, RefreshCw, Cpu, Database } from 'lucide-react';
import { CyberButton } from './CyberButton';

export interface StateViewProps {
  type: 'loading' | 'error' | 'empty' | 'unavailable';
  title?: string;
  message?: string;
  onRetry?: () => void;
  actionText?: string;
}

export const StateView: React.FC<StateViewProps> = ({
  type,
  title,
  message,
  onRetry,
  actionText = 'RETRY OPERATION',
}) => {
  if (type === 'loading') {
    return (
      <div className="p-12 border border-[var(--border-hairline)] bg-[var(--surface-graphite)] flex flex-col items-center justify-center text-center">
        <div className="w-8 h-8 border-2 border-[var(--border-hairline)] border-t-[var(--primary)] animate-spin mb-4" />
        <div className="font-mono text-xs uppercase tracking-wider text-[var(--primary)] font-semibold mb-1">
          {title || 'SYNCHRONIZING COMPUTATION MATRIX'}
        </div>
        <div className="text-xs text-[var(--text-muted)] max-w-sm">
          {message || 'Reading verified repository artifacts and hardware telemetry...'}
        </div>
      </div>
    );
  }

  if (type === 'error') {
    return (
      <div className="p-8 border border-[var(--alert-critical)] bg-[rgba(255,59,48,0.05)] flex flex-col items-center justify-center text-center">
        <AlertCircle size={28} className="text-[var(--alert-critical)] mb-3" />
        <div className="font-mono text-xs uppercase tracking-wider text-[var(--alert-critical)] font-semibold mb-1">
          {title || 'TELEMETRY EXCEPTION ENCOUNTERED'}
        </div>
        <div className="text-xs text-[var(--text-primary)] max-w-md mb-4 font-mono">
          {message || 'Failed to establish connection to the FastAPI backend or repository artifact.'}
        </div>
        {onRetry && (
          <CyberButton variant="danger" size="sm" onClick={onRetry} icon={<RefreshCw size={13} />}>
            {actionText}
          </CyberButton>
        )}
      </div>
    );
  }

  if (type === 'unavailable') {
    return (
      <div className="p-8 border border-[var(--border-hairline)] bg-[var(--surface-raised)] flex flex-col items-center justify-center text-center">
        <Cpu size={28} className="text-[var(--alert-warning)] mb-3" />
        <div className="font-mono text-xs uppercase tracking-wider text-[var(--alert-warning)] font-semibold mb-1">
          {title || 'HARDWARE ACCELERATION UNAVAILABLE (LOCAL HOST)'}
        </div>
        <div className="text-xs text-[var(--text-muted)] max-w-lg mb-2 leading-relaxed">
          {message ||
            'Local CUDA compilation is unavailable on this Windows platform. GPU kernel measurements are sourced strictly from historical Google Colab Tesla T4 benchmark sessions.'}
        </div>
        <div className="text-[11px] font-mono text-[var(--metric-dim)] border border-[var(--border-hairline)] px-2 py-1 mt-2">
          STATUS: SKIPPED_CUDA_UNAVAILABLE // COLAB WORKFLOW CONFIGURED
        </div>
      </div>
    );
  }

  // Empty State
  return (
    <div className="p-10 border border-[var(--border-hairline)] bg-[var(--surface-graphite)] flex flex-col items-center justify-center text-center">
      <Database size={28} className="text-[var(--metric-dim)] mb-3" />
      <div className="font-mono text-xs uppercase tracking-wider text-[var(--text-primary)] font-semibold mb-1">
        {title || 'NO RECORDS IDENTIFIED IN PARTITION'}
      </div>
      <div className="text-xs text-[var(--text-muted)] max-w-sm mb-4">
        {message || 'The requested query yielded zero topological records or matching transaction hashes.'}
      </div>
      {onRetry && (
        <CyberButton variant="secondary" size="sm" onClick={onRetry}>
          {actionText}
        </CyberButton>
      )}
    </div>
  );
};
