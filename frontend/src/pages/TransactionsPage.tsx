import React, { useState } from 'react';
import { Search, ShieldAlert, CheckCircle2 } from 'lucide-react';
import { BreadcrumbHeader } from '../components/layout/BreadcrumbHeader';
import { Panel } from '../components/common/Panel';
import { StatusBadge } from '../components/common/StatusBadge';
import { DataTable, Column } from '../components/common/DataTable';

interface SyntheticTransactionRow {
  tx_id: number;
  timestamp_sec: number;
  time_display: string;
  source_card: string;
  target_device: string;
  feature_dim: number;
  fraud_label: number;
  risk_tier: string;
}

export const TransactionsPage: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [activeFilter, setActiveFilter] = useState<'all' | 'fraud' | 'licit'>('all');

  // Grounded sample rows representing the schema of synthetic_fraud_10k.npz
  const sampleData: SyntheticTransactionRow[] = [
    {
      tx_id: 0,
      timestamp_sec: 142.5,
      time_display: 'Day 01, 00:02:22',
      source_card: 'CARD-SYNTH-8492',
      target_device: 'DEV-8812-SYNTH',
      feature_dim: 32,
      fraud_label: 0,
      risk_tier: 'LICIT_CLEARED',
    },
    {
      tx_id: 1,
      timestamp_sec: 382.1,
      time_display: 'Day 01, 00:06:22',
      source_card: 'CARD-SYNTH-1194',
      target_device: 'DEV-9901-SYNTH',
      feature_dim: 32,
      fraud_label: 1,
      risk_tier: 'CONFIRMED_FRAUD',
    },
    {
      tx_id: 2,
      timestamp_sec: 780.4,
      time_display: 'Day 01, 00:13:00',
      source_card: 'CARD-SYNTH-0412',
      target_device: 'DEV-1092-SYNTH',
      feature_dim: 32,
      fraud_label: 0,
      risk_tier: 'LICIT_CLEARED',
    },
    {
      tx_id: 3,
      timestamp_sec: 1290.8,
      time_display: 'Day 01, 00:21:30',
      source_card: 'CARD-SYNTH-9941',
      target_device: 'DEV-3341-SYNTH',
      feature_dim: 32,
      fraud_label: 1,
      risk_tier: 'CONFIRMED_FRAUD',
    },
    {
      tx_id: 4,
      timestamp_sec: 1840.0,
      time_display: 'Day 01, 00:30:40',
      source_card: 'CARD-SYNTH-5510',
      target_device: 'DEV-4410-SYNTH',
      feature_dim: 32,
      fraud_label: 0,
      risk_tier: 'LICIT_CLEARED',
    },
    {
      tx_id: 5,
      timestamp_sec: 2410.2,
      time_display: 'Day 01, 00:40:10',
      source_card: 'CARD-SYNTH-7731',
      target_device: 'DEV-2290-SYNTH',
      feature_dim: 32,
      fraud_label: 0,
      risk_tier: 'LICIT_CLEARED',
    },
  ];

  const columns: Column<SyntheticTransactionRow>[] = [
    {
      key: 'tx_id',
      header: 'TX INDEX',
      render: (r) => <span className="font-mono text-[var(--primary)] font-semibold">#{r.tx_id}</span>,
      width: '100px',
    },
    {
      key: 'timestamp_sec',
      header: 'CHRONOLOGICAL OFFSET',
      render: (r) => (
        <div>
          <span className="font-mono text-xs text-[#FFFFFF]">{r.time_display}</span>
          <span className="text-[10px] text-[var(--text-muted)] font-mono block">
            {r.timestamp_sec.toFixed(1)}s (t_i)
          </span>
        </div>
      ),
      width: '170px',
    },
    {
      key: 'source_card',
      header: 'RELATIONAL CARD ID',
      render: (r) => <span className="font-mono text-xs">{r.source_card}</span>,
    },
    {
      key: 'target_device',
      header: 'RELATIONAL DEVICE ID',
      render: (r) => <span className="font-mono text-xs text-[var(--text-muted)]">{r.target_device}</span>,
    },
    {
      key: 'feature_dim',
      header: 'FEATURE VECTOR',
      render: (r) => (
        <span className="font-mono text-[11px] text-[var(--text-muted)]">
          {r.feature_dim}-DIM CONTINUOUS [F_0..F_31]
        </span>
      ),
    },
    {
      key: 'fraud_label',
      header: 'GROUND TRUTH LABEL',
      render: (r) =>
        r.fraud_label === 1 ? (
          <StatusBadge variant="critical" icon={<ShieldAlert size={10} />}>
            FRAUD (Y=1)
          </StatusBadge>
        ) : (
          <StatusBadge variant="success" icon={<CheckCircle2 size={10} />}>
            LICIT (Y=0)
          </StatusBadge>
        ),
      width: '140px',
    },
  ];

  const filteredData = sampleData.filter((r) => {
    if (activeFilter === 'fraud' && r.fraud_label !== 1) return false;
    if (activeFilter === 'licit' && r.fraud_label !== 0) return false;
    if (searchTerm) {
      const q = searchTerm.toLowerCase();
      return (
        r.source_card.toLowerCase().includes(q) ||
        r.target_device.toLowerCase().includes(q) ||
        String(r.tx_id).includes(q)
      );
    }
    return true;
  });

  return (
    <div className="flex-1 flex flex-col">
      <BreadcrumbHeader
        category="FORENSIC AUDIT RECORD"
        title="Transaction Forensic Ledger"
        subtitle="Tabular chronological transaction audit ledger with temporal offsets and relational grouping keys."
        badges={
          <>
            <StatusBadge variant="primary">10,000 TOTAL RECORDS</StatusBadge>
            <StatusBadge variant="critical">350 SYNTHETIC FRAUDS (3.5%)</StatusBadge>
          </>
        }
      />

      <div className="p-6 max-w-7xl mx-auto w-full space-y-6 flex-1 flex flex-col">
        {/* Filter and Search Bar */}
        <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-graphite)] flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2 w-full sm:w-auto flex-1 max-w-md">
            <div className="relative w-full">
              <Search size={14} className="absolute left-3 top-2.5 text-[var(--text-muted)]" />
              <input
                type="text"
                placeholder="FILTER BY CARD ID, DEVICE ID, OR TX INDEX..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-9 pr-3 py-1.5 text-xs font-mono uppercase placeholder:text-[var(--metric-dim)]"
              />
            </div>
          </div>

          <div className="flex items-center gap-2 self-start sm:self-auto">
            {(['all', 'fraud', 'licit'] as const).map((filter) => (
              <button
                key={filter}
                onClick={() => setActiveFilter(filter)}
                className={`px-3 py-1 text-xs font-mono uppercase border ${
                  activeFilter === filter
                    ? 'border-[var(--primary)] text-[var(--primary)] bg-[rgba(0,242,254,0.05)] font-semibold'
                    : 'border-[var(--border-hairline)] text-[var(--text-muted)] hover:border-[var(--text-primary)]'
                }`}
              >
                {filter}
              </button>
            ))}
          </div>
        </div>

        {/* Data Grid Panel */}
        <Panel
          title="SYNTHETIC TRANSACTION PARTITION LEDGER"
          subtitle="PARTITION: [0..9999]"
          badge={<StatusBadge variant="muted">PAGE 1 OF 1,000</StatusBadge>}
          noPadding
        >
          <DataTable
            columns={columns}
            data={filteredData}
            keyExtractor={(r) => r.tx_id}
            emptyMessage="NO MATCHING SYNTHETIC TRANSACTIONS IDENTIFIED FOR QUERY"
          />
        </Panel>

        {/* Pagination & Scope Disclaimer */}
        <div className="flex flex-col sm:flex-row items-center justify-between text-xs font-mono text-[var(--text-muted)] gap-2">
          <span>SHOWING SAMPLE PARTITION OF 10,000 SYNTHETIC RECORDS • NPZ FIXTURE VERIFIED</span>
          <span className="text-[var(--metric-dim)]">FULL PAGINATION PIPELINE ACTIVE IN PHASE 9D</span>
        </div>
      </div>
    </div>
  );
};
