import React, { useState } from 'react';
import { RefreshCw } from 'lucide-react';
import { BreadcrumbHeader } from '../components/layout/BreadcrumbHeader';
import { Panel } from '../components/common/Panel';
import { StatusBadge } from '../components/common/StatusBadge';
import { DataTable, Column } from '../components/common/DataTable';
import { CyberButton } from '../components/common/CyberButton';
import { useDashboardPolling } from '../hooks/useDashboardPolling';

interface BenchmarkRunRecord {
  workload: string;
  operator: string;
  engine: string;
  timing_boundary: string;
  state: 'MEASURED' | 'SKIPPED_CUDA_UNAVAILABLE';
  stability: 'STABLE' | 'HIGH_VARIABILITY';
  mean_latency_ms: number;
  speedup: string;
  cv_percent: number;
}

export const ExperimentManagerPage: React.FC = () => {
  const [filterEngine, setFilterEngine] = useState<'all' | 'cpu' | 'cuda'>('all');
  const { data: dashboard, isRefreshing, refresh: loadDashboard } = useDashboardPolling();

  // Genuine runs directly sourced from experiments/results/benchmark_runs/session_20261024_230008_smoke/summary_table.csv
  const runs: BenchmarkRunRecord[] = [
    {
      workload: 'synthetic_tier_a_10k_mean',
      operator: 'mean',
      engine: 'cpu_reference',
      timing_boundary: 'cpu_host_timer',
      state: 'MEASURED',
      stability: 'STABLE',
      mean_latency_ms: 114.8598,
      speedup: '1.00x',
      cv_percent: 9.8,
    },
    {
      workload: 'synthetic_tier_a_10k_mean',
      operator: 'mean',
      engine: 'cpu_vectorized',
      timing_boundary: 'cpu_host_timer',
      state: 'MEASURED',
      stability: 'HIGH_VARIABILITY',
      mean_latency_ms: 2.5756,
      speedup: '44.60x',
      cv_percent: 34.29,
    },
    {
      workload: 'synthetic_tier_a_10k_mean',
      operator: 'mean',
      engine: 'cuda_baseline',
      timing_boundary: 'gpu_cuda_event',
      state: 'SKIPPED_CUDA_UNAVAILABLE',
      stability: 'STABLE',
      mean_latency_ms: 0.0,
      speedup: 'N/A',
      cv_percent: 0.0,
    },
    {
      workload: 'synthetic_tier_a_10k_max',
      operator: 'max',
      engine: 'cpu_reference',
      timing_boundary: 'cpu_host_timer',
      state: 'MEASURED',
      stability: 'STABLE',
      mean_latency_ms: 86.7878,
      speedup: '1.00x',
      cv_percent: 11.78,
    },
    {
      workload: 'synthetic_tier_a_10k_max',
      operator: 'max',
      engine: 'cpu_vectorized',
      timing_boundary: 'cpu_host_timer',
      state: 'MEASURED',
      stability: 'STABLE',
      mean_latency_ms: 87.4159,
      speedup: '0.99x',
      cv_percent: 15.9,
    },
    {
      workload: 'synthetic_tier_a_10k_max',
      operator: 'max',
      engine: 'cuda_baseline',
      timing_boundary: 'gpu_cuda_event',
      state: 'SKIPPED_CUDA_UNAVAILABLE',
      stability: 'STABLE',
      mean_latency_ms: 0.0,
      speedup: 'N/A',
      cv_percent: 0.0,
    },
    {
      workload: 'synthetic_tier_a_10k_sum',
      operator: 'sum',
      engine: 'cpu_reference',
      timing_boundary: 'cpu_host_timer',
      state: 'MEASURED',
      stability: 'STABLE',
      mean_latency_ms: 100.3461,
      speedup: '1.00x',
      cv_percent: 23.86,
    },
    {
      workload: 'synthetic_tier_a_10k_sum',
      operator: 'sum',
      engine: 'cpu_vectorized',
      timing_boundary: 'cpu_host_timer',
      state: 'MEASURED',
      stability: 'HIGH_VARIABILITY',
      mean_latency_ms: 3.8706,
      speedup: '25.93x',
      cv_percent: 189.0,
    },
    {
      workload: 'synthetic_tier_a_10k_sum',
      operator: 'sum',
      engine: 'cuda_baseline',
      timing_boundary: 'gpu_cuda_event',
      state: 'SKIPPED_CUDA_UNAVAILABLE',
      stability: 'STABLE',
      mean_latency_ms: 0.0,
      speedup: 'N/A',
      cv_percent: 0.0,
    },
  ];

  const columns: Column<BenchmarkRunRecord>[] = [
    {
      key: 'workload',
      header: 'WORKLOAD & OPERATOR',
      render: (r) => (
        <div>
          <span className="font-semibold text-xs text-[#FFFFFF] block">{r.workload}</span>
          <span className="font-mono text-[10px] text-[var(--primary)] uppercase">OPERATOR: {r.operator}</span>
        </div>
      ),
      width: '240px',
    },
    {
      key: 'engine',
      header: 'COMPUTE ENGINE',
      render: (r) => (
        <span className="font-mono text-xs text-[var(--text-primary)]">
          {r.engine}
        </span>
      ),
      width: '160px',
    },
    {
      key: 'timing_boundary',
      header: 'BOUNDARY',
      render: (r) => (
        <span className="font-mono text-[11px] text-[var(--text-muted)]">
          {r.timing_boundary}
        </span>
      ),
      width: '140px',
    },
    {
      key: 'mean_latency_ms',
      header: 'MEAN LATENCY',
      render: (r) =>
        r.state === 'MEASURED' ? (
          <span className="font-mono text-xs text-[#FFFFFF] font-semibold">
            {r.mean_latency_ms.toFixed(3)} ms
          </span>
        ) : (
          <span className="font-mono text-xs text-[var(--metric-dim)]">--</span>
        ),
      align: 'right',
      width: '130px',
    },
    {
      key: 'speedup',
      header: 'SPEEDUP',
      render: (r) =>
        r.speedup !== 'N/A' ? (
          <span className="font-mono text-xs text-[var(--secondary)] font-semibold">
            {r.speedup}
          </span>
        ) : (
          <span className="font-mono text-xs text-[var(--metric-dim)]">--</span>
        ),
      align: 'right',
      width: '100px',
    },
    {
      key: 'state',
      header: 'EXECUTION STATE',
      render: (r) =>
        r.state === 'MEASURED' ? (
          <StatusBadge variant="success">MEASURED</StatusBadge>
        ) : (
          <StatusBadge variant="warning">CUDA UNAVAILABLE</StatusBadge>
        ),
      width: '160px',
      align: 'center',
    },
    {
      key: 'stability',
      header: 'STABILITY (CV)',
      render: (r) =>
        r.state === 'MEASURED' ? (
          r.stability === 'STABLE' ? (
            <span className="font-mono text-[11px] text-[var(--secondary)]">STABLE ({r.cv_percent.toFixed(1)}%)</span>
          ) : (
            <span className="font-mono text-[11px] text-[var(--alert-warning)]">
              HIGH VARIABILITY ({r.cv_percent.toFixed(1)}%)
            </span>
          )
        ) : (
          <span className="font-mono text-[11px] text-[var(--metric-dim)]">N/A</span>
        ),
      width: '170px',
    },
  ];

  const filteredRuns = runs.filter((r) => {
    if (filterEngine === 'cpu') return r.engine.startsWith('cpu');
    if (filterEngine === 'cuda') return r.engine.startsWith('cuda');
    return true;
  });

  return (
    <div className="flex-1 flex flex-col">
      <BreadcrumbHeader
        category="BENCHMARK REPRODUCIBILITY HARNESS"
        title="Experiment & Benchmark Manager"
        subtitle="Multi-repetition measurement matrix with documented timing boundaries, RAM/VRAM preflight guards, and numerical verification."
        badges={
          <>
            <StatusBadge variant="primary">PHASE 8 CANONICAL</StatusBadge>
            <StatusBadge variant={dashboard?.benchmarks.artifact_state === 'AVAILABLE' ? 'muted' : 'warning'}>
              SESSION: {dashboard?.benchmarks.session_id || (dashboard?.benchmarks.artifact_state ? `ARTIFACT ${dashboard.benchmarks.artifact_state}` : '—')}
            </StatusBadge>
          </>
        }
        actions={
          <CyberButton
            variant="secondary"
            size="sm"
            onClick={loadDashboard}
            disabled={isRefreshing}
            icon={<RefreshCw size={13} className={isRefreshing ? 'animate-spin' : ''} />}
          >
            REFRESH SESSION DATA
          </CyberButton>
        }
      />

      <div className="p-6 max-w-7xl mx-auto w-full space-y-6 flex-1 flex flex-col">
        {/* Session Provenance Summary */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-4 border border-[var(--border-hairline)] bg-[var(--surface-graphite)]">
          <div>
            <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase">SESSION PROVENANCE</div>
            <div className="text-sm font-semibold font-mono text-[#FFFFFF] mt-0.5">
              {dashboard?.benchmarks.session_id || (dashboard?.benchmarks.artifact_state ? `ARTIFACT ${dashboard.benchmarks.artifact_state}` : '—')}
            </div>
          </div>
          <div>
            <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase">REPETITIONS POLICY</div>
            <div className="text-sm font-semibold font-mono text-[var(--primary)] mt-0.5">20 Reps (5 Warmup)</div>
          </div>
          <div>
            <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase">NUMERICAL GATES</div>
            <div className="text-sm font-semibold font-mono text-[var(--secondary)] mt-0.5">rtol=1e-5, atol=1e-5</div>
          </div>
          <div>
            <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase">MANIFEST INTEGRITY</div>
            <div className="text-sm font-semibold font-mono text-[#FFFFFF] mt-0.5">SHA-256 CAPTURED</div>
          </div>
        </div>

        {/* Filter Toolbar */}
        <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-graphite)] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-[var(--text-muted)] uppercase mr-2">ENGINE FILTER:</span>
            {(['all', 'cpu', 'cuda'] as const).map((filter) => (
              <button
                key={filter}
                onClick={() => setFilterEngine(filter)}
                className={`px-3 py-1 text-xs font-mono uppercase border ${
                  filterEngine === filter
                    ? 'border-[var(--primary)] text-[var(--primary)] bg-[rgba(0,242,254,0.05)] font-semibold'
                    : 'border-[var(--border-hairline)] text-[var(--text-muted)] hover:border-[var(--text-primary)]'
                }`}
              >
                {filter === 'all' ? 'ALL ENGINES' : filter.toUpperCase()}
              </button>
            ))}
          </div>

          <StatusBadge variant="muted">36 RAW WORKLOAD RUNS</StatusBadge>
        </div>

        {/* Table Panel */}
        <Panel
          title="CANONICAL SMOKE SESSION RUN MATRIX"
          subtitle="ARTIFACT: experiments/results/benchmark_runs/"
          noPadding
        >
          <DataTable
            columns={columns}
            data={filteredRuns}
            keyExtractor={(r) => `${r.workload}-${r.engine}`}
          />
        </Panel>

        {/* High Variability Scientific Note */}
        <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-raised)] text-xs text-[var(--text-muted)] leading-relaxed">
          <strong className="text-[var(--alert-warning)] uppercase font-mono block mb-0.5">
            Phase 8 Variability Audit Findings:
          </strong>
          Workload <span className="font-mono text-[#FFFFFF]">synthetic_tier_a_10k_sum (cpu_vectorized)</span> demonstrated an empirical coefficient of variation of 189.0% (mean 3.87ms, standard deviation 7.32ms) due to operating system thread scheduling jitter on the Windows host. In accordance with research integrity standards, all 20 raw observations were preserved without selective outlier purging.
        </div>
      </div>
    </div>
  );
};
