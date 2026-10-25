import React, { useState, useEffect, useCallback } from 'react';
import { Search, ShieldAlert, CheckCircle2, RotateCcw, ChevronLeft, ChevronRight, Loader2, Info } from 'lucide-react';
import { BreadcrumbHeader } from '../components/layout/BreadcrumbHeader';
import { Panel } from '../components/common/Panel';
import { StatusBadge } from '../components/common/StatusBadge';
import { DataTable, Column } from '../components/common/DataTable';
import { CyberButton } from '../components/common/CyberButton';
import { fetchTransactions } from '../services/api';
import type { TransactionItem, TransactionFilter, TransactionListResponse } from '../types/api';

export const TransactionsPage: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [activeFilter, setActiveFilter] = useState<TransactionFilter>('all');
  const [offset, setOffset] = useState<number>(0);
  const [limit, setLimit] = useState<number>(50);

  const [data, setData] = useState<TransactionListResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadTransactions = useCallback(async (off: number, lim: number, flt: TransactionFilter) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchTransactions({
        offset: off,
        limit: lim,
        filter: flt,
      });
      setData(response);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to load transaction ledger';
      setError(msg);
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadTransactions(offset, limit, activeFilter);
  }, [loadTransactions, offset, limit, activeFilter]);

  const handleFilterChange = (newFilter: TransactionFilter) => {
    setActiveFilter(newFilter);
    setOffset(0); // Reset to page 1 on filter switch
  };

  const handleLimitChange = (newLimit: number) => {
    setLimit(newLimit);
    setOffset(0);
  };

  const handlePrevPage = () => {
    setOffset((prev) => Math.max(0, prev - limit));
  };

  const handleNextPage = () => {
    if (data && data.has_next) {
      setOffset((prev) => prev + limit);
    }
  };

  const columns: Column<TransactionItem>[] = [
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
          <span className="font-mono text-xs text-[#FFFFFF]">{r.timestamp_sec.toFixed(2)}s</span>
          <span className="text-[10px] text-[var(--text-muted)] font-mono block">
            Day {(r.timestamp_sec / 86400).toFixed(1)}
          </span>
        </div>
      ),
      width: '150px',
    },
    {
      key: 'in_degree',
      header: 'HISTORICAL DEGREE',
      render: (r) => (
        <span className="font-mono text-xs text-[var(--text-primary)]">
          {r.in_degree} {r.in_degree === 1 ? 'edge' : 'edges'}
        </span>
      ),
      width: '130px',
    },
    {
      key: 'source_card',
      header: 'RELATIONAL KEYS',
      render: () => (
        <div className="font-mono text-[11px] text-[var(--text-muted)]" title="Card and device strings omitted during CSR graph compression">
          <span>N/A </span>
          <span className="text-[10px] text-[var(--metric-dim)]">(Omitted in CSR)</span>
        </div>
      ),
      width: '160px',
    },
    {
      key: 'feature_sample',
      header: 'FEATURE SAMPLE (F0..F3)',
      render: (r) => (
        <span className="font-mono text-[11px] text-[var(--text-muted)]">
          [{r.feature_sample.map((v) => v.toFixed(3)).join(', ')}]
        </span>
      ),
    },
    {
      key: 'fraud_label',
      header: 'GROUND TRUTH LABEL',
      render: (r) =>
        r.is_fraud ? (
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

  const displayedRows = (data?.items || []).filter((r) => {
    if (!searchTerm) return true;
    const q = searchTerm.toLowerCase();
    return String(r.tx_id).includes(q) || r.risk_tier.toLowerCase().includes(q);
  });

  const totalFiltered = data?.filtered_count || 0;
  const currentPage = Math.floor(offset / limit) + 1;
  const totalPages = Math.max(1, Math.ceil(totalFiltered / limit));

  return (
    <div className="flex-1 flex flex-col">
      <BreadcrumbHeader
        category="FORENSIC AUDIT RECORD"
        title="Transaction Forensic Ledger"
        subtitle="Tabular chronological transaction audit ledger with temporal offsets and relational CSR neighbor connectivity."
        badges={
          <>
            <StatusBadge variant="primary">REST PAGINATION ACTIVE</StatusBadge>
            <StatusBadge variant="muted">
              {data ? `${data.total_dataset_records.toLocaleString()} TOTAL RECORDS` : '10,000 TOTAL RECORDS'}
            </StatusBadge>
            <StatusBadge variant="critical">
              {data ? `${data.total_fraud_records} SYNTHETIC FRAUDS (3.5%)` : '350 SYNTHETIC FRAUDS (3.5%)'}
            </StatusBadge>
          </>
        }
      />

      <div className="p-6 max-w-7xl mx-auto w-full space-y-6 flex-1 flex flex-col">
        {/* Schema Disclosure Callout */}
        <div className="p-3.5 border border-[var(--border-hairline)] bg-[var(--surface-graphite)] flex items-start gap-3 text-xs">
          <Info size={16} className="text-[var(--secondary)] flex-shrink-0 mt-0.5" />
          <div className="text-[var(--text-muted)] font-mono text-[11px] leading-relaxed">
            <span className="text-[var(--text-primary)] font-semibold uppercase mr-1">CSR SCHEMA SPECIFICATION:</span>
            Relational entity identifiers (Card ID, Device ID) were intentionally omitted during CSR graph compression. The underlying transaction connectivity is authentically preserved via directed CSR graph edges.
          </div>
        </div>

        {/* Filter and Search Bar */}
        <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-graphite)] flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2 w-full sm:w-auto flex-1 max-w-md">
            <div className="relative w-full">
              <Search size={14} className="absolute left-3 top-2.5 text-[var(--text-muted)]" />
              <input
                type="text"
                placeholder="FILTER CURRENT PAGE BY TX INDEX..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-9 pr-3 py-1.5 text-xs font-mono uppercase bg-[var(--surface-obsidian)] border border-[var(--border-hairline)] text-[var(--text-primary)] placeholder:text-[var(--metric-dim)] focus:outline-none focus:border-[var(--primary)]"
              />
            </div>
          </div>

          <div className="flex items-center gap-2 self-start sm:self-auto flex-wrap">
            <button
              onClick={() => handleFilterChange('all')}
              className={`px-3 py-1 text-xs font-mono uppercase border ${
                activeFilter === 'all'
                  ? 'border-[var(--primary)] text-[var(--primary)] bg-[rgba(0,242,254,0.05)] font-semibold'
                  : 'border-[var(--border-hairline)] text-[var(--text-muted)] hover:border-[var(--text-primary)]'
              }`}
            >
              ALL ({data ? data.total_dataset_records.toLocaleString() : '10,000'})
            </button>
            <button
              onClick={() => handleFilterChange('fraud')}
              className={`px-3 py-1 text-xs font-mono uppercase border ${
                activeFilter === 'fraud'
                  ? 'border-[var(--alert-critical)] text-[var(--alert-critical)] bg-[rgba(255,59,48,0.05)] font-semibold'
                  : 'border-[var(--border-hairline)] text-[var(--text-muted)] hover:border-[var(--text-primary)]'
              }`}
            >
              FRAUD ({data ? data.total_fraud_records : '350'})
            </button>
            <button
              onClick={() => handleFilterChange('licit')}
              className={`px-3 py-1 text-xs font-mono uppercase border ${
                activeFilter === 'licit'
                  ? 'border-[var(--alert-success)] text-[var(--alert-success)] bg-[rgba(48,209,88,0.05)] font-semibold'
                  : 'border-[var(--border-hairline)] text-[var(--text-muted)] hover:border-[var(--text-primary)]'
              }`}
            >
              LICIT ({data ? data.total_licit_records.toLocaleString() : '9,650'})
            </button>

            {searchTerm && (
              <CyberButton
                variant="secondary"
                size="sm"
                onClick={() => setSearchTerm('')}
                icon={<RotateCcw size={11} />}
              >
                CLEAR
              </CyberButton>
            )}
          </div>
        </div>

        {/* Error State Banner */}
        {error && (
          <div className="p-4 border border-[var(--alert-critical)] bg-[rgba(255,59,48,0.06)] flex items-center justify-between gap-3 text-xs">
            <div className="font-mono text-[var(--alert-critical)]">
              <span className="font-semibold uppercase mr-2">LEDGER ERROR:</span>
              {error}
            </div>
            <CyberButton
              variant="secondary"
              size="sm"
              onClick={() => loadTransactions(offset, limit, activeFilter)}
              icon={<RotateCcw size={11} />}
            >
              RETRY
            </CyberButton>
          </div>
        )}

        {/* Data Grid Panel */}
        <Panel
          title="SYNTHETIC TRANSACTION FORENSIC LEDGER"
          subtitle={`FILTER: ${activeFilter.toUpperCase()} • SHOWING ${totalFiltered > 0 ? offset + 1 : 0} - ${Math.min(offset + limit, totalFiltered)} OF ${totalFiltered.toLocaleString()} RECORDS`}
          badge={
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono text-[var(--text-muted)]">ROWS/PAGE:</span>
              {[25, 50, 100].map((pageSize) => (
                <button
                  key={pageSize}
                  onClick={() => handleLimitChange(pageSize)}
                  className={`px-2 py-0.5 text-[10px] font-mono border ${
                    limit === pageSize
                      ? 'border-[var(--primary)] text-[var(--primary)] bg-[rgba(0,242,254,0.08)]'
                      : 'border-[var(--border-hairline)] text-[var(--text-muted)]'
                  }`}
                >
                  {pageSize}
                </button>
              ))}
            </div>
          }
          noPadding
        >
          {loading ? (
            <div className="p-12 flex flex-col items-center justify-center space-y-3">
              <Loader2 size={28} className="animate-spin text-[var(--primary)]" />
              <span className="font-mono text-xs text-[var(--text-muted)] uppercase tracking-wider">
                LOADING TRANSACTION LEDGER PAGE...
              </span>
            </div>
          ) : (
            <DataTable
              columns={columns}
              data={displayedRows}
              keyExtractor={(r) => r.tx_id}
              emptyMessage="NO MATCHING SYNTHETIC TRANSACTIONS IDENTIFIED FOR QUERY"
            />
          )}
        </Panel>

        {/* Real Pagination Bar */}
        <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-graphite)] flex flex-col sm:flex-row items-center justify-between text-xs font-mono text-[var(--text-muted)] gap-3">
          <div className="flex items-center gap-2">
            <span>PAGE {currentPage} OF {totalPages}</span>
            <span className="text-[var(--border-hairline)]">•</span>
            <span>
              RECORDS {totalFiltered > 0 ? (offset + 1).toLocaleString() : 0} – {Math.min(offset + limit, totalFiltered).toLocaleString()} OF {totalFiltered.toLocaleString()}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <CyberButton
              variant="secondary"
              size="sm"
              icon={<ChevronLeft size={13} />}
              onClick={handlePrevPage}
              disabled={offset === 0 || loading}
            >
              PREVIOUS
            </CyberButton>
            <CyberButton
              variant="secondary"
              size="sm"
              icon={<ChevronRight size={13} />}
              onClick={handleNextPage}
              disabled={!data?.has_next || loading}
            >
              NEXT
            </CyberButton>
          </div>
        </div>
      </div>
    </div>
  );
};
