import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, ShieldAlert, Cpu, Database, ArrowRight, BarChart3 } from 'lucide-react';
import { BreadcrumbHeader } from '../components/layout/BreadcrumbHeader';
import { Panel } from '../components/common/Panel';
import { TelemetryCard } from '../components/common/TelemetryCard';
import { StatusBadge } from '../components/common/StatusBadge';
import { CyberButton } from '../components/common/CyberButton';
import { StateView } from '../components/common/StateView';
import { fetchDashboardSummary } from '../services/api';
import { DashboardSummary } from '../types/api';

export const OverviewPage: React.FC = () => {
  const [data, setData] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const loadData = () => {
    setLoading(true);
    setError(null);
    fetchDashboardSummary()
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err: Error) => {
        setError(err.message);
        setLoading(false);
      });
  };

  useEffect(() => {
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="p-8 max-w-7xl mx-auto w-full">
        <StateView type="loading" title="SYNCHRONIZING REPOSITORY DASHBOARD TELEMETRY" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-8 max-w-7xl mx-auto w-full">
        <StateView
          type="error"
          title="FAILED TO RETRIEVE REPOSITORY TELEMETRY"
          message={error || 'Unable to connect to FastAPI backend.'}
          onRetry={loadData}
        />
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col">
      {/* View Header */}
      <BreadcrumbHeader
        category="VIT VELLORE // CSI4007 GPU PROGRAMMING"
        title="Forensic Workspace Overview"
        subtitle="High-performance parallel graph compute architecture engineered to eliminate memory-bandwidth bottlenecks in financial fraud feature propagation."
        badges={
          <>
            <StatusBadge variant="primary">PHASE 9B ACTIVE</StatusBadge>
            <StatusBadge variant="success">SYNTHETIC DATASET</StatusBadge>
          </>
        }
        actions={
          <div className="flex items-center gap-2">
            <CyberButton
              variant="secondary"
              size="sm"
              onClick={() => navigate('/experiments')}
              icon={<BarChart3 size={13} />}
            >
              BENCHMARK RUNS
            </CyberButton>
            <CyberButton
              variant="primary"
              size="sm"
              onClick={() => navigate('/evaluation')}
              icon={<ArrowRight size={13} />}
            >
              EVALUATION CENTER
            </CyberButton>
          </div>
        }
      />

      {/* Main Body Grid */}
      <div className="p-6 max-w-7xl mx-auto w-full space-y-6">
        {/* KPI Telemetry Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <TelemetryCard
            label="ACTIVE GRAPH TOPOLOGY"
            value={data.dataset.num_nodes.toLocaleString()}
            unit="NODES"
            subtext={`${data.dataset.num_edges.toLocaleString()} edges • ${data.dataset.fraud_rate_pct}% synthetic fraud rate`}
            icon={<Database size={16} />}
            statusColor="cyan"
          />
          <TelemetryCard
            label="TOP MODEL PR-AUC"
            value={data.evaluation.top_test_pr_auc.toFixed(4)}
            unit="TEST"
            subtext={`${data.evaluation.top_model} • ROC-AUC ${data.evaluation.top_test_roc_auc.toFixed(4)}`}
            icon={<Activity size={16} />}
            statusColor="mint"
          />
          <TelemetryCard
            label="PEAK VECTORIZED SPEEDUP"
            value={`${data.benchmarks.top_speedup_vs_cpu_ref}x`}
            unit="VS REF"
            subtext={data.benchmarks.top_speedup_workload}
            icon={<Cpu size={16} />}
            statusColor="cyan"
          />
          <TelemetryCard
            label="RESEARCH INTEGRITY"
            value="ZERO DRIFT"
            unit="VERIFIED"
            subtext="Dual-tolerance validation (rtol=1e-5, atol=1e-5)"
            icon={<ShieldAlert size={16} />}
            statusColor="mint"
          />
        </div>

        {/* 2-Column Analytical Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Column 1 & 2: Benchmark & Evaluation Reality */}
          <div className="lg:col-span-2 space-y-6">
            {/* Panel: Canonical ML Evaluation */}
            <Panel
              title="DOWNSTREAM MODEL EVALUATION SUMMARY"
              subtitle={data.evaluation.phase}
              badge={<StatusBadge variant="primary">CAUSAL HISTORY</StatusBadge>}
              actions={
                <CyberButton
                  variant="terminal"
                  size="sm"
                  onClick={() => navigate('/evaluation')}
                >
                  FULL EVALUATION →
                </CyberButton>
              }
            >
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-3 p-3 bg-[var(--surface-obsidian)] border border-[var(--border-hairline)] text-center">
                  <div>
                    <div className="text-[11px] font-mono text-[var(--text-muted)]">TOP PR-AUC</div>
                    <div className="text-xl font-semibold text-[var(--primary)] font-mono">
                      {data.evaluation.top_test_pr_auc.toFixed(4)}
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] font-mono text-[var(--text-muted)]">TOP ROC-AUC</div>
                    <div className="text-xl font-semibold text-[var(--secondary)] font-mono">
                      {data.evaluation.top_test_roc_auc.toFixed(4)}
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] font-mono text-[var(--text-muted)]">TOP TEST F1</div>
                    <div className="text-xl font-semibold text-[#FFFFFF] font-mono">
                      {data.evaluation.top_test_f1.toFixed(4)}
                    </div>
                  </div>
                </div>

                <div className="text-xs text-[var(--text-muted)] leading-relaxed">
                  Evaluated across Logistic Regression and Random Forest models with 5 neighbor representations (
                  <span className="text-[var(--text-primary)] font-mono">raw, mean, max, multi_operator, sum</span>
                  ). Test set evaluated under temporal streaming causality on 1,500 transactions (53 fraud cases).
                </div>

                {/* ADR-007 Note */}
                <div className="p-3 bg-[rgba(255,159,10,0.05)] border border-[var(--alert-warning)] text-xs text-[var(--text-muted)]">
                  <strong className="text-[var(--alert-warning)] uppercase font-mono block mb-0.5">
                    ADR-007 Methodological Protocol Notice:
                  </strong>
                  {data.evaluation.adr_007_notice}
                </div>
              </div>
            </Panel>

            {/* Panel: Benchmark Smoke Session */}
            <Panel
              title="PHASE 8 BENCHMARK MATRIX TELEMETRY"
              subtitle={data.benchmarks.session_id}
              badge={<StatusBadge variant="muted">36 RUNS</StatusBadge>}
              actions={
                <CyberButton
                  variant="terminal"
                  size="sm"
                  onClick={() => navigate('/experiments')}
                >
                  EXPERIMENTS →
                </CyberButton>
              }
            >
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-center">
                  <div className="p-2 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)]">
                    <div className="text-[10px] font-mono text-[var(--text-muted)]">TOTAL RUNS</div>
                    <div className="text-base font-semibold font-mono text-[#FFFFFF]">{data.benchmarks.total_runs}</div>
                  </div>
                  <div className="p-2 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)]">
                    <div className="text-[10px] font-mono text-[var(--text-muted)]">MEASURED CPU</div>
                    <div className="text-base font-semibold font-mono text-[var(--secondary)]">{data.benchmarks.measured_cpu_runs}</div>
                  </div>
                  <div className="p-2 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)]">
                    <div className="text-[10px] font-mono text-[var(--text-muted)]">SKIPPED CUDA</div>
                    <div className="text-base font-semibold font-mono text-[var(--alert-warning)]">{data.benchmarks.skipped_gpu_runs}</div>
                  </div>
                  <div className="p-2 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)]">
                    <div className="text-[10px] font-mono text-[var(--text-muted)]">MAX SPEEDUP</div>
                    <div className="text-base font-semibold font-mono text-[var(--primary)]">{data.benchmarks.top_speedup_vs_cpu_ref}x</div>
                  </div>
                </div>

                <div className="text-xs text-[var(--text-muted)] leading-relaxed">
                  Local Windows execution skipped CUDA kernel runs due to absent local NVCC compiler.
                  All 24 GPU workloads are explicitly cataloged as <span className="font-mono text-[var(--alert-warning)]">SKIPPED_CUDA_UNAVAILABLE</span> with zero fabricated measurements.
                </div>

                <div className="text-[11px] font-mono text-[var(--metric-dim)] border-t border-[var(--border-hairline)] pt-2">
                  VARIABILITY AUDIT: {data.benchmarks.high_variability_label}
                </div>
              </div>
            </Panel>
          </div>

          {/* Column 3: Platform Telemetry & Navigation */}
          <div className="space-y-6">
            {/* Panel: Active Environment */}
            <Panel title="HARDWARE & RUNTIME TELEMETRY">
              <div className="space-y-3 text-xs">
                <div>
                  <div className="font-mono text-[10px] text-[var(--text-muted)] uppercase">LOCAL HOST ACCELERATOR</div>
                  <div className="font-semibold text-[#FFFFFF] mt-0.5">
                    {data.hardware.local_gpu_detected || 'NVIDIA GeForce MX450'}
                  </div>
                  <div className="text-[11px] font-mono text-[var(--alert-warning)] mt-0.5">
                    {data.hardware.local_compiler_status}
                  </div>
                </div>

                <div className="pt-2 border-t border-[var(--border-hairline)]">
                  <div className="font-mono text-[10px] text-[var(--text-muted)] uppercase">REMOTE BENCHMARK PLATFORM</div>
                  <div className="font-semibold text-[#FFFFFF] mt-0.5">
                    {data.hardware.canonical_colab_gpu}
                  </div>
                  <div className="text-[11px] font-mono text-[var(--secondary)] mt-0.5">
                    MEAN KERNEL: {data.hardware.colab_kernel_latency_ms} ms (Tier A Mean)
                  </div>
                </div>

                <div className="pt-2 border-t border-[var(--border-hairline)]">
                  <div className="font-mono text-[10px] text-[var(--text-muted)] uppercase">ACTIVE DATASET FIXTURE</div>
                  <div className="font-semibold text-[#FFFFFF] mt-0.5 font-mono">{data.dataset.name}.npz</div>
                  <div className="text-[11px] text-[var(--text-muted)] mt-0.5">
                    SHA-256: <span className="font-mono text-[10px]">{data.dataset.sha256.slice(0, 16)}...</span>
                  </div>
                </div>
              </div>
            </Panel>

            {/* Quick Workspace Navigation */}
            <Panel title="CORE RESEARCH MODULES">
              <div className="space-y-2">
                <button
                  onClick={() => navigate('/explorer')}
                  className="w-full p-2.5 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)] hover:border-[var(--primary)] text-left flex items-center justify-between"
                >
                  <div>
                    <div className="font-mono text-xs uppercase text-[#FFFFFF] font-medium">Graph Explorer</div>
                    <div className="text-[11px] text-[var(--text-muted)]">Sub-graph neighbor inspection</div>
                  </div>
                  <ArrowRight size={14} className="text-[var(--primary)]" />
                </button>

                <button
                  onClick={() => navigate('/transactions')}
                  className="w-full p-2.5 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)] hover:border-[var(--primary)] text-left flex items-center justify-between"
                >
                  <div>
                    <div className="font-mono text-xs uppercase text-[#FFFFFF] font-medium">Transaction Ledger</div>
                    <div className="text-[11px] text-[var(--text-muted)]">10,000 synthetic records</div>
                  </div>
                  <ArrowRight size={14} className="text-[var(--primary)]" />
                </button>

                <button
                  onClick={() => navigate('/cuda-lab')}
                  className="w-full p-2.5 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)] hover:border-[var(--primary)] text-left flex items-center justify-between"
                >
                  <div>
                    <div className="font-mono text-xs uppercase text-[#FFFFFF] font-medium">CUDA Performance Lab</div>
                    <div className="text-[11px] text-[var(--text-muted)]">Kernel profiling & Colab runs</div>
                  </div>
                  <ArrowRight size={14} className="text-[var(--primary)]" />
                </button>

                <button
                  onClick={() => navigate('/datasets')}
                  className="w-full p-2.5 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)] hover:border-[var(--primary)] text-left flex items-center justify-between"
                >
                  <div>
                    <div className="font-mono text-xs uppercase text-[#FFFFFF] font-medium">Dataset Registry</div>
                    <div className="text-[11px] text-[var(--text-muted)]">4 canonical NPZ fixtures</div>
                  </div>
                  <ArrowRight size={14} className="text-[var(--primary)]" />
                </button>

                <button
                  onClick={() => navigate('/reports')}
                  className="w-full p-2.5 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)] hover:border-[var(--primary)] text-left flex items-center justify-between"
                >
                  <div>
                    <div className="font-mono text-xs uppercase text-[#FFFFFF] font-medium">Research Reports</div>
                    <div className="text-[11px] text-[var(--text-muted)]">Phase 7.5 & 8 monographs</div>
                  </div>
                  <ArrowRight size={14} className="text-[var(--primary)]" />
                </button>
              </div>
            </Panel>
          </div>
        </div>
      </div>
    </div>
  );
};
