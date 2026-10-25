import React, { useState } from 'react';
import { Copy, Check, Image } from 'lucide-react';
import { BreadcrumbHeader } from '../components/layout/BreadcrumbHeader';
import { Panel } from '../components/common/Panel';
import { StatusBadge } from '../components/common/StatusBadge';
import { CyberButton } from '../components/common/CyberButton';
import { DataTable, Column } from '../components/common/DataTable';

interface ReportDossier {
  id: string;
  title: string;
  phase: string;
  path: string;
  description: string;
  status: string;
}

interface PublicationFigure {
  id: string;
  name: string;
  description: string;
  formats: string[];
}

export const ReportsPage: React.FC = () => {
  const [copiedBibtex, setCopiedBibtex] = useState<boolean>(false);

  const reports: ReportDossier[] = [
    {
      id: 'REP-7.5',
      title: 'Phase 7.5 — Research-Grade Evaluation Reporting & Visualizations',
      phase: 'Phase 7.5',
      path: 'docs/PHASE_7_5_EVALUATION_REPORT.md',
      description: 'Comprehensive downstream fraud classification evaluation report under causal streaming and sensitivity policies.',
      status: 'VERIFIED_CANONICAL',
    },
    {
      id: 'REP-8.0',
      title: 'Phase 8 — Benchmark Automation & Experiment Management Protocol',
      phase: 'Phase 8',
      path: 'docs/BENCHMARK_AUTOMATION.md',
      description: 'Specification and validation report for multi-repetition benchmark matrix with RAM/VRAM telemetry and variability audit.',
      status: 'VERIFIED_CANONICAL',
    },
    {
      id: 'REP-9A',
      title: 'Phase 9A — Stitch UI and Existing Application Audit Report',
      phase: 'Phase 9A',
      path: 'docs/PHASE_9A_STITCH_FRONTEND_AUDIT.md',
      description: 'Strictly read-only pre-implementation audit comparing 13 Stitch screens to repository capabilities, specifying an 8-view architecture.',
      status: 'AUDIT_COMPLETE',
    },
    {
      id: 'REP-SYS',
      title: 'System Architecture & Parallel Algorithm Specification',
      phase: 'Foundation',
      path: 'docs/SYSTEM_ARCHITECTURE.md',
      description: 'Mathematical formulation of Compressed Sparse Row (CSR) neighbor aggregation and GPU warp-level parallelization.',
      status: 'FOUNDATION',
    },
  ];

  const figures: PublicationFigure[] = [
    { id: 'fig1', name: 'Fig 1: Test PR-AUC Comparison', description: 'Bar chart comparing downstream test PR-AUC across 5 neighbor representations.', formats: ['PNG', 'SVG', 'PDF'] },
    { id: 'fig2', name: 'Fig 2: Test ROC-AUC Comparison', description: 'ROC-AUC performance comparing Logistic Regression and Random Forest.', formats: ['PNG', 'SVG', 'PDF'] },
    { id: 'fig3', name: 'Fig 3: PR Curve Omission Notice', description: 'Formal methodology notice explaining the ADR-007 discrete thresholding protocol.', formats: ['PNG', 'SVG', 'PDF'] },
    { id: 'fig4', name: 'Fig 4: Confusion Matrices', description: 'Normalized test confusion matrices across decision boundaries.', formats: ['PNG', 'SVG', 'PDF'] },
    { id: 'fig5', name: 'Fig 5: Operating Tradeoffs', description: 'Precision vs recall curves at target operating thresholds.', formats: ['PNG', 'SVG', 'PDF'] },
    { id: 'fig6', name: 'Fig 6: Runtime Breakdown', description: 'End-to-end latency decomposition: feature enrichment vs model inference.', formats: ['PNG', 'SVG', 'PDF'] },
    { id: 'fig7', name: 'Fig 7: Context Policy Sensitivity', description: 'Comparative ablation: causal_history vs train_only_context.', formats: ['PNG', 'SVG', 'PDF'] },
  ];

  const bibtex = `@techreport{fgi2026memory,
  title       = {Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation},
  author      = {FraudGraph Intelligence Research Group},
  institution = {Vellore Institute of Technology, CSI4007 GPU Programming},
  year        = {2026},
  address     = {Vellore, India},
  note        = {Software-only Academic Research Prototype}
}`;

  const copyBibtex = () => {
    navigator.clipboard.writeText(bibtex);
    setCopiedBibtex(true);
    setTimeout(() => setCopiedBibtex(false), 2000);
  };

  const reportColumns: Column<ReportDossier>[] = [
    {
      key: 'title',
      header: 'RESEARCH DOSSIER',
      render: (r) => (
        <div>
          <span className="font-semibold text-xs text-[#FFFFFF] block">{r.title}</span>
          <span className="font-mono text-[10px] text-[var(--primary)]">{r.path}</span>
        </div>
      ),
      width: '320px',
    },
    {
      key: 'description',
      header: 'RESEARCH SCOPE & SUMMARY',
      render: (r) => <span className="text-xs text-[var(--text-muted)]">{r.description}</span>,
    },
    {
      key: 'status',
      header: 'PROVENANCE STATUS',
      render: (r) => (
        <StatusBadge variant={r.status.startsWith('VERIFIED') ? 'success' : 'primary'}>
          {r.status}
        </StatusBadge>
      ),
      width: '180px',
      align: 'center',
    },
  ];

  return (
    <div className="flex-1 flex flex-col">
      <BreadcrumbHeader
        category="TECHNICAL DOCUMENTATION & MONOGRAPHS"
        title="Research Reports & Publication Artifacts"
        subtitle="Peer-reviewed technical evaluation dossiers, academic monographs, and multi-format vector figure assets."
        badges={
          <>
            <StatusBadge variant="primary">4 FORMAL DOSSIERS</StatusBadge>
            <StatusBadge variant="success">7 PUBLICATION FIGURES</StatusBadge>
          </>
        }
      />

      <div className="p-6 max-w-7xl mx-auto w-full space-y-6 flex-1 flex flex-col">
        {/* Table Panel: Research Dossiers */}
        <Panel
          title="AUTHENTIC RESEARCH DOSSIERS"
          subtitle="DIRECTORY: docs/"
          noPadding
        >
          <DataTable
            columns={reportColumns}
            data={reports}
            keyExtractor={(r) => r.id}
          />
        </Panel>

        {/* Publication Figures Panel */}
        <Panel
          title="PHASE 7.5 PUBLICATION FIGURES & MULTI-FORMAT ARTIFACTS"
          subtitle="DIRECTORY: experiments/results/evaluation_reporting_7_5/"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {figures.map((fig) => (
              <div
                key={fig.id}
                className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)] flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <Image size={14} className="text-[var(--primary)]" />
                    <span className="font-mono text-xs font-semibold text-[#FFFFFF]">{fig.name}</span>
                  </div>
                  <p className="text-[11px] text-[var(--text-muted)] leading-relaxed mb-3">
                    {fig.description}
                  </p>
                </div>
                <div className="flex items-center justify-between pt-2 border-t border-[var(--border-hairline)] text-[10px] font-mono">
                  <span className="text-[var(--text-muted)]">FORMATS:</span>
                  <div className="flex gap-1.5">
                    {fig.formats.map((fmt) => (
                      <span key={fmt} className="px-1.5 py-0.2 border border-[var(--border-hairline)] bg-[var(--surface-raised)] text-[var(--text-primary)]">
                        {fmt}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Panel>

        {/* BibTeX Citation Panel */}
        <Panel
          title="ACADEMIC CITATION SPECIFICATION (BIBTEX)"
          actions={
            <CyberButton
              variant="secondary"
              size="sm"
              onClick={copyBibtex}
              icon={copiedBibtex ? <Check size={12} className="text-[var(--secondary)]" /> : <Copy size={12} />}
            >
              {copiedBibtex ? 'COPIED TO CLIPBOARD' : 'COPY BIBTEX'}
            </CyberButton>
          }
        >
          <pre className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)] text-xs font-mono text-[var(--text-primary)] overflow-x-auto leading-relaxed">
            {bibtex}
          </pre>
        </Panel>
      </div>
    </div>
  );
};
