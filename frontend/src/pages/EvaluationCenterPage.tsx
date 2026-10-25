import React, { useState } from 'react';
import { Activity, ShieldAlert, Sliders, Info } from 'lucide-react';
import { BreadcrumbHeader } from '../components/layout/BreadcrumbHeader';
import { Panel } from '../components/common/Panel';
import { StatusBadge } from '../components/common/StatusBadge';
import { TelemetryCard } from '../components/common/TelemetryCard';
import { DataTable, Column } from '../components/common/DataTable';
import { Tabs } from '../components/common/Tabs';

interface ModelRow {
  model: string;
  representation: string;
  dim: number;
  val_pr_auc: number;
  val_roc_auc: number;
  test_pr_auc: number;
  test_roc_auc: number;
  val_threshold: number;
  test_precision: number;
  test_recall: number;
  test_f1: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export const EvaluationCenterPage: React.FC = () => {
  const [activePolicy, setActivePolicy] = useState<string>('causal');
  const [threshold, setThreshold] = useState<number>(0.1090);

  // Exact data from experiments/results/evaluation_reporting_7_5/summary_evaluation_metrics.csv
  const causalData: ModelRow[] = [
    {
      model: 'Logistic Regression',
      representation: 'max',
      dim: 64,
      val_pr_auc: 0.2405,
      val_roc_auc: 0.8209,
      test_pr_auc: 0.3315,
      test_roc_auc: 0.7865,
      val_threshold: 0.109,
      test_precision: 0.2553,
      test_recall: 0.4528,
      test_f1: 0.3265,
      tp: 24,
      fp: 70,
      tn: 1377,
      fn: 29,
    },
    {
      model: 'Logistic Regression',
      representation: 'raw',
      dim: 32,
      val_pr_auc: 0.259,
      val_roc_auc: 0.8213,
      test_pr_auc: 0.3088,
      test_roc_auc: 0.7855,
      val_threshold: 0.1684,
      test_precision: 0.3175,
      test_recall: 0.3774,
      test_f1: 0.3448,
      tp: 20,
      fp: 43,
      tn: 1404,
      fn: 33,
    },
    {
      model: 'Logistic Regression',
      representation: 'mean',
      dim: 64,
      val_pr_auc: 0.2016,
      val_roc_auc: 0.8239,
      test_pr_auc: 0.301,
      test_roc_auc: 0.7763,
      val_threshold: 0.109,
      test_precision: 0.2474,
      test_recall: 0.4528,
      test_f1: 0.32,
      tp: 24,
      fp: 73,
      tn: 1374,
      fn: 29,
    },
    {
      model: 'Logistic Regression',
      representation: 'multi_operator',
      dim: 96,
      val_pr_auc: 0.207,
      val_roc_auc: 0.8084,
      test_pr_auc: 0.2471,
      test_roc_auc: 0.78,
      val_threshold: 0.1387,
      test_precision: 0.2958,
      test_recall: 0.3962,
      test_f1: 0.3387,
      tp: 21,
      fp: 50,
      tn: 1397,
      fn: 32,
    },
    {
      model: 'Logistic Regression',
      representation: 'sum',
      dim: 64,
      val_pr_auc: 0.3054,
      val_roc_auc: 0.81,
      test_pr_auc: 0.1488,
      test_roc_auc: 0.7499,
      val_threshold: 0.2278,
      test_precision: 0.2174,
      test_recall: 0.3774,
      test_f1: 0.2759,
      tp: 20,
      fp: 72,
      tn: 1375,
      fn: 33,
    },
    {
      model: 'Random Forest',
      representation: 'mean',
      dim: 64,
      val_pr_auc: 0.2259,
      val_roc_auc: 0.7484,
      test_pr_auc: 0.2587,
      test_roc_auc: 0.7792,
      val_threshold: 0.1882,
      test_precision: 0.3333,
      test_recall: 0.283,
      test_f1: 0.3061,
      tp: 15,
      fp: 30,
      tn: 1417,
      fn: 38,
    },
    {
      model: 'Random Forest',
      representation: 'raw',
      dim: 32,
      val_pr_auc: 0.1967,
      val_roc_auc: 0.7495,
      test_pr_auc: 0.2445,
      test_roc_auc: 0.7786,
      val_threshold: 0.0991,
      test_precision: 0.1687,
      test_recall: 0.5283,
      test_f1: 0.2557,
      tp: 28,
      fp: 138,
      tn: 1309,
      fn: 25,
    },
  ];

  const columns: Column<ModelRow>[] = [
    {
      key: 'model',
      header: 'MODEL & OPERATOR',
      render: (r) => (
        <div>
          <span className="font-semibold text-xs text-[#FFFFFF] block">{r.model}</span>
          <span className="font-mono text-[10px] text-[var(--primary)] uppercase">
            OPERATOR: {r.representation} ({r.dim}D)
          </span>
        </div>
      ),
      width: '210px',
    },
    {
      key: 'test_pr_auc',
      header: 'TEST PR-AUC',
      render: (r) => (
        <span className="font-mono text-xs font-semibold text-[var(--primary)]">
          {r.test_pr_auc.toFixed(4)}
        </span>
      ),
      align: 'right',
      width: '120px',
    },
    {
      key: 'test_roc_auc',
      header: 'TEST ROC-AUC',
      render: (r) => (
        <span className="font-mono text-xs text-[var(--secondary)]">
          {r.test_roc_auc.toFixed(4)}
        </span>
      ),
      align: 'right',
      width: '120px',
    },
    {
      key: 'val_threshold',
      header: 'FROZEN τ*',
      render: (r) => (
        <span className="font-mono text-xs text-[var(--text-muted)]">
          {r.val_threshold.toFixed(4)}
        </span>
      ),
      align: 'right',
      width: '110px',
    },
    {
      key: 'test_f1',
      header: 'TEST F1',
      render: (r) => (
        <span className="font-mono text-xs text-[#FFFFFF] font-semibold">
          {r.test_f1.toFixed(4)}
        </span>
      ),
      align: 'right',
      width: '110px',
    },
    {
      key: 'test_precision',
      header: 'PRECISION',
      render: (r) => (
        <span className="font-mono text-xs text-[var(--text-muted)]">
          {r.test_precision.toFixed(4)}
        </span>
      ),
      align: 'right',
      width: '110px',
    },
    {
      key: 'test_recall',
      header: 'RECALL',
      render: (r) => (
        <span className="font-mono text-xs text-[var(--text-muted)]">
          {r.test_recall.toFixed(4)}
        </span>
      ),
      align: 'right',
      width: '110px',
    },
    {
      key: 'confusion',
      header: 'CONFUSION (TP / FP / TN / FN)',
      render: (r) => (
        <span className="font-mono text-[11px] text-[var(--text-muted)]">
          {r.tp} / {r.fp} / {r.tn} / {r.fn}
        </span>
      ),
      width: '180px',
    },
  ];

  return (
    <div className="flex-1 flex flex-col">
      <BreadcrumbHeader
        category="DOWNSTREAM CLASSIFICATION EVALUATION"
        title="Dedicated Model Evaluation Center"
        subtitle="Downstream fraud classification benchmarks evaluated on frozen validation thresholds under causal streaming graph policies."
        badges={
          <>
            <StatusBadge variant="primary">PHASE 7.4 / 7.5 CANONICAL</StatusBadge>
            <StatusBadge variant="success">FROZEN THRESHOLD EVALUATION</StatusBadge>
          </>
        }
      />

      <div className="p-6 max-w-7xl mx-auto w-full space-y-6 flex-1 flex flex-col">
        {/* KPI Strip */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <TelemetryCard
            label="TOP DOWNSTREAM PR-AUC"
            value="0.3315"
            unit="TEST"
            subtext="Logistic Regression (max neighbor aggregation, 64D)"
            icon={<Activity size={16} />}
            statusColor="cyan"
          />
          <TelemetryCard
            label="TOP DOWNSTREAM ROC-AUC"
            value="0.7865"
            unit="TEST"
            subtext="Logistic Regression (max neighbor aggregation, 64D)"
            icon={<Activity size={16} />}
            statusColor="mint"
          />
          <TelemetryCard
            label="TEST EVALUATION PARTITION"
            value="1,500"
            unit="TXS"
            subtext="53 synthetic fraud cases (prevalence: 3.53%)"
            icon={<ShieldAlert size={16} />}
            statusColor="cyan"
          />
          <TelemetryCard
            label="DECISION THRESHOLD PROTOCOL"
            value="FROZEN τ*"
            unit="VAL"
            subtext="Optimal threshold chosen on validation; evaluated blindly on test"
            icon={<Sliders size={16} />}
            statusColor="mint"
          />
        </div>

        {/* Policy Selector Tabs */}
        <Tabs
          tabs={[
            { id: 'causal', label: 'Primary Policy: causal_history', badge: 'BENCHMARK' },
            { id: 'sensitivity', label: 'Sensitivity Policy: train_only_context', badge: 'ABLATION' },
          ]}
          activeTab={activePolicy}
          onChange={setActivePolicy}
        />

        {/* Table Panel */}
        <Panel
          title="CANONICAL EVALUATION METRICS MATRIX"
          subtitle="ARTIFACT: summary_evaluation_metrics.csv"
          noPadding
        >
          <DataTable
            columns={columns}
            data={causalData}
            keyExtractor={(r) => `${r.model}-${r.representation}`}
          />
        </Panel>

        {/* Decision Threshold Inspector Panel */}
        <Panel
          title="FROZEN VALIDATION THRESHOLD SIMULATOR"
          subtitle="PARAMETER: tau in [0.01, 0.99]"
        >
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-[var(--text-muted)]">ACTIVE THRESHOLD (tau):</span>
              <span className="text-[var(--primary)] font-semibold text-sm">{threshold.toFixed(4)}</span>
            </div>
            <input
              type="range"
              min="0.01"
              max="0.99"
              step="0.005"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-[var(--surface-obsidian)] border border-[var(--border-hairline)]"
            />
            <div className="flex justify-between text-[11px] font-mono text-[var(--metric-dim)]">
              <span>0.01 (HIGH SENSITIVITY)</span>
              <span>OPTIMAL tau* = 0.1090 (MAX OPERATOR)</span>
              <span>0.99 (HIGH SPECIFICITY)</span>
            </div>
          </div>
        </Panel>

        {/* ADR-007 Methodological Notice Callout */}
        <div className="p-4 border border-[var(--alert-warning)] bg-[rgba(255,159,10,0.04)] space-y-2">
          <div className="flex items-center gap-2">
            <Info size={16} className="text-[var(--alert-warning)] shrink-0" />
            <span className="font-mono text-xs uppercase text-[var(--alert-warning)] font-semibold">
              ADR-007: Discrete Validation Thresholding Policy & PR Curve Omission Notice
            </span>
          </div>
          <p className="text-xs text-[var(--text-muted)] leading-relaxed pl-6">
            In financial fraud surveillance under class imbalance (3.5%), continuous Precision-Recall curves plotted directly on the test set introduce evaluation leakage by optimizing decision thresholds on unobserved test data. Per Architectural Decision Record ADR-007, decision thresholds are frozen strictly on the independent validation partition (<span className="font-mono text-[var(--text-primary)]">N=1,500, 58 frauds</span>) and applied blindly to the test partition. Continuous PR curves were deliberately omitted and replaced with the formal omission notice (<span className="font-mono text-[var(--text-primary)]">fig3_pr_curve_omission_notice.png</span>).
          </p>
        </div>
      </div>
    </div>
  );
};
