import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  apiClient,
  getHealth,
  getDashboardSummary,
  getDatasets,
  ApiError,
  isApiError,
  isTimeoutError,
  isAbortError,
} from '../src/services/apiClient.ts';

const BACKEND_URL = 'http://127.0.0.1:8000';
const VITE_DEV_URL = 'http://127.0.0.1:5173';

describe('Phase 10: End-to-End API Integration & Smoke Verification', () => {

  describe('1. Direct Backend Connectivity (FastAPI @ :8000)', () => {
    it('exercises GET /api/v1/health with full schema verification', async () => {
      const health = await getHealth({ baseUrl: BACKEND_URL, timeoutMs: 5000 });
      assert.strictEqual(health.status, 'healthy');
      assert.strictEqual(health.project, 'FraudGraph Intelligence');
      assert.strictEqual(health.project_id, 'FGI');
      assert.strictEqual(health.version, '1.0.0');

      // Compute engine telemetry
      assert.ok(health.compute_engine, 'compute_engine must be present');
      assert.strictEqual(typeof health.compute_engine.backend, 'string');
      assert.strictEqual(typeof health.compute_engine.cuda_available_locally, 'boolean');
      assert.ok(
        health.compute_engine.local_gpu_detected === null ||
        typeof health.compute_engine.local_gpu_detected === 'string'
      );
      assert.strictEqual(health.compute_engine.colab_workflow_enabled, true);

      // System timestamp
      assert.ok(health.system, 'system info must be present');
      assert.ok(typeof health.system.python_version === 'string');
      assert.ok(typeof health.system.timestamp === 'string');
    });

    it('exercises GET /api/v1/dashboard/summary reading genuine repository artifacts', async () => {
      const summary = await getDashboardSummary({ baseUrl: BACKEND_URL, timeoutMs: 5000 });
      assert.strictEqual(summary.status, 'operational');
      assert.strictEqual(summary.project, 'FraudGraph Intelligence');

      // Dataset summary
      assert.strictEqual(summary.dataset.name, 'synthetic_fraud_10k');
      assert.strictEqual(summary.dataset.num_nodes, 10000);
      assert.strictEqual(summary.dataset.num_edges, 231984);
      assert.strictEqual(summary.dataset.feature_dim, 32);
      assert.strictEqual(summary.dataset.fraud_count, 350);
      assert.strictEqual(summary.dataset.licit_count, 9650);
      assert.strictEqual(summary.dataset.fraud_rate_pct, 3.5);
      assert.strictEqual(summary.dataset.is_synthetic, true);
      assert.strictEqual(summary.dataset.provenance, 'SYNTHETIC');

      // Genuine evaluation summary parsed from Phase 7.5 CSV
      assert.strictEqual(summary.evaluation.artifact_state, 'AVAILABLE');
      assert.strictEqual(summary.evaluation.experiment_id, 'EXP-EVAL-PHASE7-4');
      assert.strictEqual(summary.evaluation.top_model, 'Logistic Regression (max)');
      assert.strictEqual(summary.evaluation.top_test_pr_auc, 0.3315);
      assert.strictEqual(summary.evaluation.top_test_roc_auc, 0.7865);
      assert.strictEqual(summary.evaluation.top_test_f1, 0.3265);
      assert.strictEqual(summary.evaluation.primary_policy, 'causal_history');
      assert.ok(summary.evaluation.adr_007_notice.includes('ADR-007'));

      // Genuine benchmark summary parsed from Phase 8 smoke CSV
      assert.strictEqual(summary.benchmarks.artifact_state, 'AVAILABLE');
      assert.strictEqual(summary.benchmarks.session_id, 'session_20261024_230008_smoke');
      assert.strictEqual(summary.benchmarks.total_runs, 36);
      assert.strictEqual(summary.benchmarks.measured_cpu_runs, 12);
      assert.strictEqual(summary.benchmarks.skipped_gpu_runs, 24);
      assert.strictEqual(summary.benchmarks.top_speedup_vs_cpu_ref, 44.60);
      assert.strictEqual(summary.benchmarks.local_gpu_status, 'SKIPPED_CUDA_UNAVAILABLE');

      // Hardware summary
      assert.strictEqual(summary.hardware.canonical_colab_gpu, 'NVIDIA Tesla T4 (Phase 6 Archive)');
      assert.strictEqual(summary.hardware.colab_kernel_latency_ms, 0.0679);
    });

    it('exercises GET /api/v1/datasets verifying physical CSR fixtures on disk', async () => {
      const res = await getDatasets({ baseUrl: BACKEND_URL, timeoutMs: 5000 });
      assert.strictEqual(res.status, 'success');
      assert.strictEqual(res.total_count, 4);
      assert.strictEqual(res.fixtures.length, 4);

      const byId = Object.fromEntries(res.fixtures.map((f) => [f.id, f]));

      // 10k temporal fraud fixture
      assert.ok(byId['synthetic_fraud_10k']);
      assert.strictEqual(byId['synthetic_fraud_10k'].is_available, true);
      assert.strictEqual(byId['synthetic_fraud_10k'].num_nodes, 10000);
      assert.strictEqual(byId['synthetic_fraud_10k'].num_edges, 231984);
      assert.strictEqual(byId['synthetic_fraud_10k'].feature_dim, 32);
      assert.strictEqual(byId['synthetic_fraud_10k'].sha256, 'aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430');

      // Toy graph fixture
      assert.ok(byId['toy_graph_5nodes']);
      assert.strictEqual(byId['toy_graph_5nodes'].is_available, true);
      assert.strictEqual(byId['toy_graph_5nodes'].num_nodes, 5);
      assert.strictEqual(byId['toy_graph_5nodes'].num_edges, 8);
      assert.strictEqual(byId['toy_graph_5nodes'].feature_dim, 4);
      assert.strictEqual(byId['toy_graph_5nodes'].sha256, '3ea8f702bc1f944d9f164222bad637666307c7be8e4a9767f813e40edc137909');
    });
  });

  describe('2. Vite Dev Server Proxy Routing (@ :5173/api/v1/*)', () => {
    it('proxies GET /api/v1/health transparently through Vite dev proxy', async () => {
      const health = await getHealth({ baseUrl: VITE_DEV_URL, timeoutMs: 5000 });
      assert.strictEqual(health.status, 'healthy');
      assert.strictEqual(health.project, 'FraudGraph Intelligence');
    });

    it('proxies GET /api/v1/dashboard/summary transparently through Vite dev proxy', async () => {
      const summary = await getDashboardSummary({ baseUrl: VITE_DEV_URL, timeoutMs: 5000 });
      assert.strictEqual(summary.status, 'operational');
      assert.strictEqual(summary.dataset.num_nodes, 10000);
      assert.strictEqual(summary.evaluation.top_test_pr_auc, 0.3315);
    });

    it('proxies GET /api/v1/datasets transparently through Vite dev proxy', async () => {
      const res = await getDatasets({ baseUrl: VITE_DEV_URL, timeoutMs: 5000 });
      assert.strictEqual(res.status, 'success');
      assert.strictEqual(res.total_count, 4);
    });
  });

  describe('3. Error Handling, Timeouts, Aborts & Network Recovery', () => {
    it('properly classifies 404 HTTP errors with structured ApiError', async () => {
      await assert.rejects(
        async () => {
          await apiClient.get(`${BACKEND_URL}/api/v1/non_existent_route`);
        },
        (err) => {
          assert.ok(isApiError(err));
          assert.strictEqual(err.kind, 'HTTP_ERROR');
          assert.strictEqual(err.status, 404);
          return true;
        }
      );
    });

    it('properly classifies network connection failure when backend is unreachable', async () => {
      await assert.rejects(
        async () => {
          // Port 65534 is closed / unused
          await apiClient.get('http://127.0.0.1:65534/api/v1/health', { timeoutMs: 1000 });
        },
        (err) => {
          assert.ok(isApiError(err));
          assert.ok(err.kind === 'NETWORK_ERROR' || err.kind === 'TIMEOUT_ERROR');
          return true;
        }
      );
    });

    it('handles caller request abortion cleanly via AbortSignal', async () => {
      const controller = new AbortController();
      controller.abort(); // pre-aborted
      await assert.rejects(
        async () => {
          await getHealth({ baseUrl: BACKEND_URL, signal: controller.signal });
        },
        (err) => {
          assert.ok(isAbortError(err));
          return true;
        }
      );
    });
  });
});
