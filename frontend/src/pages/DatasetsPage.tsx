import React from 'react';
import { RefreshCw } from 'lucide-react';
import { BreadcrumbHeader } from '../components/layout/BreadcrumbHeader';
import { Panel } from '../components/common/Panel';
import { StatusBadge } from '../components/common/StatusBadge';
import { DataTable, Column } from '../components/common/DataTable';
import { StateView } from '../components/common/StateView';
import { CyberButton } from '../components/common/CyberButton';
import { useDatasetsPolling } from '../hooks/useDatasetsPolling';
import { DatasetFixtureItem } from '../types/api';

export const DatasetsPage: React.FC = () => {
  const {
    data: datasetResponse,
    state: telemetryState,
    isLoading: loading,
    isRefreshing,
    error: pollingError,
    lastSuccessTime,
    refresh: loadDatasets,
  } = useDatasetsPolling({ intervalMs: 30000, enabled: true });
  const datasets = datasetResponse?.fixtures || [];
  const error = pollingError ? pollingError.message : null;

  const availableCount = datasets.filter((d) => d.is_available).length;
  const totalCount = datasets.length;
  const allVerified = totalCount > 0 && availableCount === totalCount;

  const columns: Column<DatasetFixtureItem>[] = [
    {
      key: 'name',
      header: 'FIXTURE NAME',
      render: (r) => (
        <div>
          <span className="font-semibold text-xs text-[#FFFFFF] block">{r.name}</span>
          <span className="font-mono text-[10px] text-[var(--primary)]">{r.file_name}</span>
        </div>
      ),
      width: '260px',
    },
    {
      key: 'num_nodes',
      header: 'NODES',
      render: (r) => (
        <span className="font-mono text-xs text-[#FFFFFF] font-semibold">
          {r.is_available ? r.num_nodes.toLocaleString() : '—'}
        </span>
      ),
      align: 'right',
      width: '100px',
    },
    {
      key: 'num_edges',
      header: 'EDGES',
      render: (r) => (
        <span className="font-mono text-xs text-[#FFFFFF] font-semibold">
          {r.is_available ? r.num_edges.toLocaleString() : '—'}
        </span>
      ),
      align: 'right',
      width: '110px',
    },
    {
      key: 'feature_dim',
      header: 'FEATURE DIM',
      render: (r) => (
        <span className="font-mono text-xs text-[var(--text-muted)]">
          {r.is_available ? `${r.feature_dim}D` : '—'}
        </span>
      ),
      align: 'right',
      width: '110px',
    },
    {
      key: 'size_bytes',
      header: 'FILE SIZE',
      render: (r) => (
        <span className="font-mono text-xs text-[var(--text-muted)]">
          {r.size_bytes > 0
            ? `${(r.size_bytes / 1024 / 1024).toFixed(2)} MB`
            : r.is_available
            ? '0.00 MB'
            : '—'}
        </span>
      ),
      align: 'right',
      width: '110px',
    },
    {
      key: 'sha256',
      header: 'SHA-256 PROVENANCE CHECKSUM',
      render: (r) => (
        <span className="font-mono text-[10px] text-[var(--text-muted)] tracking-wider">
          {r.sha256 ? `${r.sha256.slice(0, 24)}...` : 'UNAVAILABLE'}
        </span>
      ),
    },
    {
      key: 'status',
      header: 'INTEGRITY',
      render: (r) => {
        if (r.is_available) {
          return <StatusBadge variant="success">VERIFIED</StatusBadge>;
        }
        if (r.sha256) {
          return <StatusBadge variant="critical">CORRUPT</StatusBadge>;
        }
        return <StatusBadge variant="critical">MISSING</StatusBadge>;
      },
      width: '110px',
      align: 'center',
    },
  ];

  if (!datasetResponse) {
    if (loading) {
      return (
        <div className="p-8 max-w-7xl mx-auto w-full">
          <StateView type="loading" title="VERIFYING REPOSITORY DATASET FIXTURES" />
        </div>
      );
    }

    return (
      <div className="p-8 max-w-7xl mx-auto w-full">
        <StateView
          type="error"
          title="FAILED TO RETRIEVE DATASET FIXTURES"
          message={error || 'Unable to connect to FastAPI backend.'}
          onRetry={loadDatasets}
        />
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col">
      <BreadcrumbHeader
        category="GRAPH TOPOLOGY REGISTRY"
        title="Dataset Management & Fixture Registry"
        subtitle="Canonical Compressed Sparse Row (CSR) graph fixtures verified under strict cryptographic SHA-256 provenance."
        badges={
          <>
            <StatusBadge variant={allVerified ? 'primary' : 'critical'}>
              {availableCount} / {totalCount} VERIFIED FIXTURES
            </StatusBadge>
            {telemetryState === 'DEGRADED' && (
              <StatusBadge variant="warning">DEGRADED TELEMETRY</StatusBadge>
            )}
            {isRefreshing && (
              <StatusBadge variant="primary">REFRESHING...</StatusBadge>
            )}
            <StatusBadge variant="success">NPZ COMPRESSION</StatusBadge>
          </>
        }
        actions={
          <CyberButton
            variant="secondary"
            size="sm"
            onClick={loadDatasets}
            disabled={isRefreshing}
            icon={<RefreshCw size={13} className={isRefreshing ? 'animate-spin' : ''} />}
          >
            RECHECK FIXTURES
          </CyberButton>
        }
      />

      <div className="p-6 max-w-7xl mx-auto w-full space-y-6 flex-1 flex flex-col">
        {/* Degraded Telemetry Notice Banner */}
        {telemetryState === 'DEGRADED' && (
          <div className="p-3 bg-[rgba(255,159,10,0.08)] border border-[var(--alert-warning)] text-xs text-[var(--text-muted)] flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-[var(--alert-warning)] shrink-0" />
              <span>
                <strong className="text-[var(--alert-warning)] font-mono uppercase">Dataset Telemetry Degraded:</strong>{' '}
                Background refresh failed. Retaining verified fixture inventory from{' '}
                {lastSuccessTime ? lastSuccessTime.toLocaleTimeString() : 'previous probe'}.
                {error && (
                  <span className="font-mono text-[11px] block text-[var(--metric-dim)] mt-0.5">
                    Reason: {error}
                  </span>
                )}
              </span>
            </div>
            <CyberButton
              variant="secondary"
              size="sm"
              onClick={loadDatasets}
              disabled={isRefreshing}
              icon={<RefreshCw size={12} className={isRefreshing ? 'animate-spin' : ''} />}
            >
              RETRY
            </CyberButton>
          </div>
        )}

        {/* Table Panel */}
        <Panel
          title="AUTHENTIC REPOSITORY GRAPH FIXTURES"
          subtitle="DIRECTORY: data/fixtures/"
          noPadding
        >
          <DataTable
            columns={columns}
            data={datasets}
            keyExtractor={(r) => r.id}
          />
        </Panel>

        {/* CSR Schema Specification Panel */}
        <Panel title="CSR IN-MEMORY ARRAY SCHEMA SPECIFICATION">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3 font-mono text-xs">
            <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)]">
              <span className="text-[var(--primary)] font-semibold block mb-1">row_ptr</span>
              <span className="text-[11px] text-[var(--text-muted)] block">1D np.int32 (N + 1,)</span>
              <span className="text-[10px] text-[var(--metric-dim)] block mt-1">CSR row cumulative neighbor offsets</span>
            </div>
            <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)]">
              <span className="text-[var(--primary)] font-semibold block mb-1">col_idx</span>
              <span className="text-[11px] text-[var(--text-muted)] block">1D np.int32 (M,)</span>
              <span className="text-[10px] text-[var(--metric-dim)] block mt-1">Contiguous neighbor target node indices</span>
            </div>
            <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)]">
              <span className="text-[var(--primary)] font-semibold block mb-1">features</span>
              <span className="text-[11px] text-[var(--text-muted)] block">2D np.float32 (N, D)</span>
              <span className="text-[10px] text-[var(--metric-dim)] block mt-1">Continuous transaction node feature vectors</span>
            </div>
            <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)]">
              <span className="text-[var(--primary)] font-semibold block mb-1">labels</span>
              <span className="text-[11px] text-[var(--text-muted)] block">1D np.int32 (N,)</span>
              <span className="text-[10px] text-[var(--metric-dim)] block mt-1">Ground truth targets (0=licit, 1=fraud)</span>
            </div>
            <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)]">
              <span className="text-[var(--primary)] font-semibold block mb-1">timestamps</span>
              <span className="text-[11px] text-[var(--text-muted)] block">1D np.float64 (N,)</span>
              <span className="text-[10px] text-[var(--metric-dim)] block mt-1">Chronological transaction offsets in seconds</span>
            </div>
          </div>
        </Panel>

        {/* Integrity Verification Protocol & Scope Notice (Phase 9D Requirement 9) */}
        <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-raised)] text-xs text-[var(--text-muted)] leading-relaxed">
          <strong className="text-[var(--primary)] uppercase font-mono block mb-0.5">
            Integrity Verification Protocol & Scope Notice:
          </strong>
          The <span className="font-mono text-[var(--secondary)] font-semibold">VERIFIED</span> integrity badge certifies filesystem existence in <span className="font-mono text-[#FFFFFF]">data/fixtures/</span>, parseable CSR array headers (row_ptr, col_idx, features), and cryptographic SHA-256 checksum matching. It validates physical file availability and structural metadata integrity only; it does not denote external scientific validation or live financial banking customer endorsement.
        </div>

        {/* Academic Grounding Notice */}
        <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-raised)] text-xs text-[var(--text-muted)] leading-relaxed">
          <strong className="text-[var(--tertiary)] uppercase font-mono block mb-0.5">
            Synthetic Benchmark Fixture Notice:
          </strong>
          All four graph fixtures are generated deterministically using seed 42 in <span className="font-mono text-[#FFFFFF]">backend/cpu_engine/generate_fixtures.py</span>. The graphs model power-law degree distributions and transaction relational structures without containing any empirical banking customer data.
        </div>
      </div>
    </div>
  );
};
