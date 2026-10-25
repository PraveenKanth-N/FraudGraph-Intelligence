import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  apiClient,
  getSubgraph,
  getTransactions,
  formatErrorDetail,
} from '../src/services/apiClient.ts';

describe('Phase 11.2B: Subgraph and Transactions Client Tests', () => {

  it('getSubgraph constructs the expected query string and parses response', async () => {
    let capturedUrl = '';
    const mockResponse = {
      status: 'success',
      query: {
        root_node_id: 5407,
        requested_hops: 2,
        actual_hops: 2,
        direction: 'directed_csr',
        max_nodes: 150,
        max_edges: 500,
      },
      summary: {
        total_nodes: 150,
        total_edges: 312,
        fraud_node_count: 8,
        fraud_percentage: 5.33,
        is_truncated: true,
        node_cap_hit: true,
        edge_cap_hit: false,
        truncation_reason: 'MAX_NODES_LIMIT_REACHED',
      },
      nodes: [
        {
          id: 5407,
          depth: 0,
          is_root: true,
          timestamp_sec: 1420550.2,
          fraud_label: 0,
          is_fraud: false,
          degree: 252,
        },
      ],
      edges: [
        {
          source: 5407,
          target: 3812,
          temporal_delta_sec: 3410.5,
        },
      ],
      provenance: {
        fixture_name: 'synthetic_fraud_10k.npz',
        sha256: 'aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430',
        causality_policy: 'causal_history (directed CSR)',
      },
    };

    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url) => {
      capturedUrl = String(url);
      return {
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => mockResponse,
      };
    };

    try {
      const result = await getSubgraph({
        nodeId: 5407,
        hops: 2,
        maxNodes: 150,
        maxEdges: 500,
      });

      assert.ok(capturedUrl.includes('/graph/subgraph?'));
      assert.ok(capturedUrl.includes('node_id=5407'));
      assert.ok(capturedUrl.includes('hops=2'));
      assert.ok(capturedUrl.includes('max_nodes=150'));
      assert.ok(capturedUrl.includes('max_edges=500'));

      assert.strictEqual(result.status, 'success');
      assert.strictEqual(result.query.root_node_id, 5407);
      assert.strictEqual(result.summary.is_truncated, true);
      assert.strictEqual(result.summary.node_cap_hit, true);
      assert.strictEqual(result.nodes.length, 1);
      assert.strictEqual(result.nodes[0].is_root, true);
      assert.strictEqual(result.edges.length, 1);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('getTransactions constructs query string with offset, limit, filter and parses response', async () => {
    let capturedUrl = '';
    const mockResponse = {
      status: 'success',
      total_dataset_records: 10000,
      total_fraud_records: 350,
      total_licit_records: 9650,
      filtered_count: 350,
      offset: 50,
      limit: 25,
      returned_count: 25,
      has_next: true,
      has_prev: true,
      current_filter: 'fraud',
      ordering: 'original_transaction_index_chronological',
      items: [
        {
          tx_id: 105,
          timestamp_sec: 4200.5,
          fraud_label: 1,
          is_fraud: true,
          risk_tier: 'CONFIRMED_FRAUD',
          in_degree: 12,
          feature_dim: 32,
          feature_sample: [0.12, -0.45, 0.88, 0.02],
          source_card: null,
          target_device: null,
          notes: 'Relational entity strings were discarded during CSR compression; graph topology preserves entity relations via directed edges.',
        },
      ],
      schema_notes: 'source_card and target_device are omitted in CSR graph representation; in_degree indicates connected transactions in historical graph.',
      provenance_sha256: 'aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430',
    };

    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url) => {
      capturedUrl = String(url);
      return {
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => mockResponse,
      };
    };

    try {
      const result = await getTransactions({
        offset: 50,
        limit: 25,
        filter: 'fraud',
      });

      assert.ok(capturedUrl.includes('/transactions?'));
      assert.ok(capturedUrl.includes('offset=50'));
      assert.ok(capturedUrl.includes('limit=25'));
      assert.ok(capturedUrl.includes('filter=fraud'));

      assert.strictEqual(result.status, 'success');
      assert.strictEqual(result.total_dataset_records, 10000);
      assert.strictEqual(result.total_fraud_records, 350);
      assert.strictEqual(result.filtered_count, 350);
      assert.strictEqual(result.has_next, true);
      assert.strictEqual(result.has_prev, true);
      assert.strictEqual(result.items[0].source_card, null);
      assert.strictEqual(result.items[0].target_device, null);
      assert.ok(result.schema_notes.includes('source_card and target_device are omitted'));
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('handles empty query parameters on getTransactions cleanly', async () => {
    let capturedUrl = '';
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url) => {
      capturedUrl = String(url);
      return {
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => ({ status: 'success', items: [] }),
      };
    };

    try {
      await getTransactions();
      assert.ok(capturedUrl.endsWith('/transactions'));
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
