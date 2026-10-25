import React from 'react';
import { Cpu, Server, CheckCircle2, ShieldCheck, RefreshCw } from 'lucide-react';
import { Modal } from '../common/Modal';
import { CyberButton } from '../common/CyberButton';
import { StatusBadge } from '../common/StatusBadge';
import { HealthData } from '../../types/api';
import { TelemetryState } from '../../types/telemetry';

export interface SystemHealthDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  health: HealthData | null;
  isLoading: boolean;
  onRefresh: () => void;
  telemetryState?: TelemetryState;
  lastSuccessTime?: Date | null;
  error?: Error | null;
}

export const SystemHealthDrawer: React.FC<SystemHealthDrawerProps> = ({
  isOpen,
  onClose,
  health,
  isLoading,
  onRefresh,
  telemetryState,
  lastSuccessTime,
  error,
}) => {
  const isHealthy = health?.status === 'healthy';

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="System Health & Runtime Diagnostics"
      subtitle="FGI-CORE // VIT VELLORE CSI4007"
      maxWidth="lg"
      actions={
        <div className="flex items-center justify-between w-full">
          <span className="text-[11px] font-mono text-[var(--metric-dim)]">
            RUNTIME:{' '}
            {health?.system?.python_version
              ? `Python ${health.system.python_version} • AMD64`
              : isLoading
              ? 'PROBING RUNTIME...'
              : 'UNAVAILABLE'}
          </span>
          <div className="flex gap-2">
            <CyberButton
              variant="secondary"
              size="sm"
              onClick={onRefresh}
              disabled={isLoading}
              icon={<RefreshCw size={12} className={isLoading ? 'animate-spin' : ''} />}
            >
              PROBE RUNTIME
            </CyberButton>
            <CyberButton variant="primary" size="sm" onClick={onClose}>
              DISMISS
            </CyberButton>
          </div>
        </div>
      }
    >
      <div className="space-y-6">
        {/* Top Status Banner */}
        <div className="p-4 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)] flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div
              className={`w-3 h-3 ${
                telemetryState === 'DEGRADED'
                  ? 'bg-[var(--alert-warning)]'
                  : isHealthy
                  ? 'bg-[var(--secondary)]'
                  : 'bg-[var(--alert-critical)]'
              }`}
            />
            <div>
              <div className="font-mono text-xs uppercase text-[#FFFFFF] font-semibold">
                {telemetryState === 'DEGRADED'
                  ? 'TELEMETRY DEGRADED // RETAINING STALE PROBE'
                  : isHealthy
                  ? 'ALL CORE SUBSYSTEMS OPERATIONAL'
                  : 'SYSTEM OFFLINE / STANDALONE'}
              </div>
              <div className="text-xs text-[var(--text-muted)] font-mono">
                LAST PROBE: {health?.system?.timestamp || (lastSuccessTime ? lastSuccessTime.toISOString() : 'UNAVAILABLE')}
              </div>
              {error && (
                <div className="text-[10px] text-[var(--alert-warning)] font-mono mt-0.5 truncate max-w-md">
                  PROBE WARNING: {error.message}
                </div>
              )}
            </div>
          </div>
          <StatusBadge
            variant={
              telemetryState === 'DEGRADED'
                ? 'warning'
                : isHealthy
                ? 'success'
                : 'critical'
            }
          >
            {telemetryState === 'DEGRADED'
              ? 'DEGRADED'
              : isHealthy
              ? 'VERIFIED PASS'
              : 'DISCONNECTED'}
          </StatusBadge>
        </div>

        {/* Subsystems Breakdown Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Card 1: Compute Engine */}
          <div className="p-4 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)]">
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-[11px] text-[var(--text-muted)] uppercase">
                COMPUTE BACKEND
              </span>
              <Server size={14} className="text-[var(--primary)]" />
            </div>
            <div className="text-lg font-semibold text-[#FFFFFF] mb-1">
              {health?.compute_engine?.backend
                ? health.compute_engine.backend.toUpperCase()
                : isLoading
                ? 'PROBING...'
                : 'UNAVAILABLE'}
            </div>
            <div className="text-xs text-[var(--text-muted)] leading-relaxed">
              {health?.compute_engine
                ? 'CSR-based deterministic parallel neighbor aggregator. Verified NumPy vectorized execution.'
                : 'Compute engine diagnostics unreachable. Verify that FastAPI service is running.'}
            </div>
            <div className="mt-3 pt-2 border-t border-[var(--border-hairline)] flex justify-between text-[11px] font-mono">
              <span className="text-[var(--text-muted)]">NUMERICAL TOLERANCE:</span>
              <span className="text-[var(--secondary)]">rtol=1e-5, atol=1e-5</span>
            </div>
          </div>

          {/* Card 2: Hardware Accelerator */}
          <div className="p-4 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)]">
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-[11px] text-[var(--text-muted)] uppercase">
                LOCAL GPU DETECTED
              </span>
              <Cpu size={14} className="text-[var(--alert-warning)]" />
            </div>
            <div className="text-lg font-semibold text-[#FFFFFF] mb-1">
              {health?.compute_engine?.local_gpu_detected
                ? health.compute_engine.local_gpu_detected
                : isLoading
                ? 'PROBING...'
                : 'NOT DETECTED'}
            </div>
            <div className="text-xs text-[var(--text-muted)] leading-relaxed">
              {health?.compute_engine
                ? health.compute_engine.local_gpu_detected
                  ? `${
                      health.compute_engine.vram_free_mb !== null &&
                      health.compute_engine.vram_free_mb !== undefined
                        ? `${health.compute_engine.vram_free_mb} MB Free VRAM • `
                        : ''
                    }Local NVCC is unavailable; kernels compiled in Colab.`
                  : 'No local GPU accelerator detected. CPU execution mode active; CUDA benchmarks executed via Colab runner.'
                : 'Hardware accelerator diagnostics unreachable.'}
            </div>
            <div className="mt-3 pt-2 border-t border-[var(--border-hairline)] flex justify-between text-[11px] font-mono">
              <span className="text-[var(--text-muted)]">LOCAL COMPILER:</span>
              <span
                className={
                  health?.compute_engine?.cuda_available_locally
                    ? 'text-[var(--secondary)]'
                    : 'text-[var(--alert-warning)]'
                }
              >
                {health?.compute_engine?.cuda_compiler_status ||
                  (isLoading ? 'PROBING...' : 'UNAVAILABLE')}
              </span>
            </div>
          </div>

          {/* Card 3: Colab Remote Runner */}
          <div className="p-4 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)]">
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-[11px] text-[var(--text-muted)] uppercase">
                COLAB RUNNER ADAPTER
              </span>
              <CheckCircle2 size={14} className="text-[var(--secondary)]" />
            </div>
            <div className="text-lg font-semibold text-[#FFFFFF] mb-1">
              NVIDIA Tesla T4 (Phase 6 Archive)
            </div>
            <div className="text-xs text-[var(--text-muted)] leading-relaxed">
              Remote automated packaging, compilation, and ingestion pipeline enabled (`fgi_colab_workflow.ipynb`).
            </div>
            <div className="mt-3 pt-2 border-t border-[var(--border-hairline)] flex justify-between text-[11px] font-mono">
              <span className="text-[var(--text-muted)]">GPU VERIFICATION:</span>
              <span className="text-[var(--secondary)]">0.068 ms (Tier A Mean)</span>
            </div>
          </div>

          {/* Card 4: Quality & Test Verification */}
          <div className="p-4 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)]">
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-[11px] text-[var(--text-muted)] uppercase">
                TEST SUITE VERIFICATION
              </span>
              <ShieldCheck size={14} className="text-[var(--primary)]" />
            </div>
            <div className="text-sm font-semibold text-[#FFFFFF] mb-1 font-mono uppercase tracking-wider">
              Recorded Regression Suite
            </div>
            <div className="text-xs text-[var(--text-muted)] leading-relaxed">
              Static historical baseline from checked-in verification record (Phase 9B.1 audit). Covers CSR graphs, aggregators, CUDA Colab runners, ML evaluation, and REST APIs.
            </div>
            <div className="mt-3 pt-2 border-t border-[var(--border-hairline)] flex flex-col gap-1.5 text-[11px] font-mono">
              <div className="flex justify-between items-center">
                <span className="text-[var(--text-muted)]">BASELINE AUDIT:</span>
                <span className="text-[var(--secondary)]">146 Passed • 2 Skipped (148 Total)</span>
              </div>
              <div className="flex justify-between items-center text-[10px] text-[var(--metric-dim)]">
                <span>AUDIT RECORD:</span>
                <span>COMMIT d0ea85a // OCT 2026</span>
              </div>
              <div className="text-[10px] text-[var(--text-muted)] italic pt-1 border-t border-[var(--border-hairline)]/50">
                * Note: Live dynamic test runner execution API is planned for Phase 9D.
              </div>
            </div>
          </div>
        </div>

        {/* Academic Disclaimers Notice */}
        <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-raised)] text-xs text-[var(--text-muted)] leading-relaxed">
          <strong className="text-[var(--tertiary)] uppercase font-mono tracking-wider block mb-1">
            Academic Research Prototype Scope:
          </strong>
          This software is engineered exclusively for academic research within the VIT Vellore CSI4007 GPU Programming capstone.
          All benchmarks and evaluation results are derived from controlled synthetic graph fixtures and do not reflect live enterprise financial banking distributions.
        </div>
      </div>
    </Modal>
  );
};
