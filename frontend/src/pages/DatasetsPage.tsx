import React, { useState, useEffect } from 'react';
import { RefreshCw } from 'lucide-react';
import { BreadcrumbHeader } from '../components/layout/BreadcrumbHeader';
import { Panel } from '../components/common/Panel';
import { StatusBadge } from '../components/common/StatusBadge';
import { DataTable, Column } from '../components/common/DataTable';
import { StateView } from '../components/common/StateView';
import { CyberButton } from '../components/common/CyberButton';
import { fetchDatasets } from '../services/api';
import { DatasetFixtureItem } from '../types/api';

export const DatasetsPage: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetFixtureItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadDatasets = () => {
    setLoading(true);
    setError(null);
    fetchDatasets()
      .then((res) => {
        setDatasets(res.fixtures);
        setLoading(false);
      })
      .catch((err: Error) => {
        setError(err.message);
        setLoading(false);
      });
  };

  useEffect(() => {
    loadDatasets();
  }, []);

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
          {r.num_nodes.toLocaleString()}
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
          {r.num_edges.toLocaleString()}
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
          {r.feature_dim}D
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
          {(r.size_bytes / 1024 / 1024).toFixed(2)} MB
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
          {r.sha256.slice(0, 24)}...
        </span>
      ),
    },
    {
      key: 'status',
      header: 'INTEGRITY',
      render: () => <StatusBadge variant="success">VERIFIED</StatusBadge>,
      width: '110px',
      align: 'center',
    },
  ];

  if (loading) {
    return (
      <div className="p-8 max-w-7xl mx-auto w-full">
        <StateView type="loading" title="VERIFYING REPOSITORY DATASET FIXTURES" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 max-w-7xl mx-auto w-full">
        <StateView
          type="error"
          title="FAILED TO RETRIEVE DATASET FIXTURES"
          message={error}
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
            <StatusBadge variant="primary">4 VERIFIED FIXTURES</StatusBadge>
            <StatusBadge variant="success">NPZ COMPRESSION</StatusBadge>
          </>
        }
        actions={
          <CyberButton
            variant="secondary"
            size="sm"
            onClick={loadDatasets}
            icon={<RefreshCw size={13} />}
          >
            RECHECK FIXTURES
          </CyberButton>
        }
      />

      <div className="p-6 max-w-7xl mx-auto w-full space-y-6 flex-1 flex flex-col">
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
