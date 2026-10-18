import React, { useEffect, useState } from 'react';
import { Activity, Cpu, ShieldAlert, ArrowUpRight } from 'lucide-react';
import './styles/design-tokens.css';

interface HealthData {
  status: string;
  project: string;
  project_id: string;
  version: string;
  compute_engine: {
    backend: string;
    cuda_available_locally: boolean;
    local_gpu_detected: string | null;
    vram_free_mb: number | null;
    colab_workflow_enabled: boolean;
  };
  system: {
    python_version: string;
    timestamp: string;
  };
}

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/v1/health')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: HealthData) => {
        setHealth(data);
        setLoading(false);
      })
      .catch((err: Error) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Telemetry Header */}
      <header
        style={{
          height: '64px',
          borderBottom: '1px solid var(--border-hairline)',
          backgroundColor: 'rgba(13, 17, 23, 0.95)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 2rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div
              style={{
                width: '8px',
                height: '8px',
                backgroundColor: 'var(--primary)',
                boxShadow: '0 0 8px var(--primary)',
              }}
            />
            <span
              style={{
                fontFamily: 'var(--font-sans)',
                fontWeight: 700,
                fontSize: '14px',
                letterSpacing: '0.1em',
                color: '#FFFFFF',
              }}
            >
              FRAUDGRAPH INTELLIGENCE
            </span>
          </div>

          <span
            style={{
              fontSize: '11px',
              fontFamily: 'var(--font-mono)',
              color: 'var(--metric-dim)',
              borderLeft: '1px solid var(--border-hairline)',
              paddingLeft: '1rem',
            }}
          >
            FGI-CORE // v1.0.0
          </span>
        </div>

        {/* Center Topic Pill */}
        <div
          style={{
            backgroundColor: 'var(--surface-obsidian)',
            border: '1px solid var(--border-hairline)',
            padding: '4px 16px',
            fontSize: '12px',
            color: 'var(--text-muted)',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <span style={{ color: 'var(--secondary)' }}>●</span>
          <span>Phase 1 — Project Foundation & Dependency Architecture</span>
        </div>

        {/* Right Status Cluster */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div
            style={{
              border: '1px solid var(--border-hairline)',
              padding: '4px 10px',
              fontSize: '11px',
              fontFamily: 'var(--font-mono)',
              color: loading ? 'var(--metric-dim)' : error ? 'var(--alert-critical)' : 'var(--secondary)',
            }}
          >
            {loading ? 'SYNCING API...' : error ? 'API OFFLINE (STANDALONE)' : 'BACKEND SYNCHRONIZED'}
          </div>

          <button
            style={{
              backgroundColor: 'var(--primary)',
              color: 'var(--on-primary)',
              border: 'none',
              padding: '6px 14px',
              fontSize: '12px',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}
          >
            DOCUMENTATION
            <ArrowUpRight size={14} />
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main style={{ flex: 1, padding: '2.5rem', maxWidth: '1440px', margin: '0 auto', width: '100%' }}>
        {/* Hero Section */}
        <div style={{ marginBottom: '2.5rem' }}>
          <div style={{ fontSize: '11px', color: 'var(--primary)', fontFamily: 'var(--font-mono)', marginBottom: '0.5rem' }}>
            VIT VELLORE // CSI4007 GPU PROGRAMMING
          </div>
          <h1
            style={{
              fontFamily: 'var(--font-serif)',
              fontSize: '44px',
              fontWeight: 400,
              lineHeight: 1.15,
              color: 'var(--tertiary)',
              maxWidth: '900px',
              marginBottom: '1rem',
            }}
          >
            Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation
          </h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '15px', maxWidth: '720px', lineHeight: 1.6 }}>
            High-performance parallel graph compute architecture engineered to eliminate memory-bandwidth bottlenecks in 
            transaction graph feature propagation across large-scale financial fraud datasets.
          </p>
        </div>

        {/* Grid of Diagnostic Panels */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem', marginBottom: '2.5rem' }}>
          {/* Card 1: Compute Engine */}
          <div
            style={{
              backgroundColor: 'var(--surface-graphite)',
              border: '1px solid var(--border-hairline)',
              padding: '1.5rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <span className="telemetry-label" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                COMPUTE ENGINE
              </span>
              <Cpu size={16} color="var(--primary)" />
            </div>
            <div style={{ fontSize: '24px', fontWeight: 600, marginBottom: '0.5rem', color: '#FFFFFF' }}>
              {health?.compute_engine?.backend ? health.compute_engine.backend.toUpperCase() : 'CPU REFERENCE'}
            </div>
            <div style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              CSR-based deterministic parallel neighbor aggregator. Verified CPU baseline for numerical tolerance testing.
            </div>
          </div>

          {/* Card 2: GPU Telemetry */}
          <div
            style={{
              backgroundColor: 'var(--surface-graphite)',
              border: '1px solid var(--border-hairline)',
              padding: '1.5rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <span className="telemetry-label" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                HARDWARE ACCELERATOR
              </span>
              <Activity size={16} color="var(--secondary)" />
            </div>
            <div style={{ fontSize: '24px', fontWeight: 600, marginBottom: '0.5rem', color: '#FFFFFF' }}>
              {health?.compute_engine?.local_gpu_detected || 'NVIDIA GeForce MX450'}
            </div>
            <div style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              Compute Capability 7.5 (Turing) • 2048 MiB VRAM. Colab GPU adapter configured for compilation and verification.
            </div>
          </div>

          {/* Card 3: Integrity Guarantee */}
          <div
            style={{
              backgroundColor: 'var(--surface-graphite)',
              border: '1px solid var(--border-hairline)',
              padding: '1.5rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <span className="telemetry-label" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                RESEARCH INTEGRITY
              </span>
              <ShieldAlert size={16} color="var(--primary)" />
            </div>
            <div style={{ fontSize: '24px', fontWeight: 600, marginBottom: '0.5rem', color: '#FFFFFF' }}>
              ZERO SYNTHETIC BIAS
            </div>
            <div style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              Zero fabricated GPU benchmarks. Unexecuted kernels explicitly marked as pending. Exact timing via CUDA events.
            </div>
          </div>
        </div>

        {/* Phase Status Banner */}
        <div
          style={{
            backgroundColor: 'var(--surface-raised)',
            border: '1px solid var(--border-hairline)',
            padding: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div>
            <div style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--secondary)', marginBottom: '0.25rem' }}>
              FOUNDATION ACCEPTANCE GATE
            </div>
            <div style={{ fontSize: '16px', fontWeight: 600, color: '#FFFFFF' }}>
              Phase 1 Deliverables: Virtual Environment, Scaffolding, Documentation Suite & Test Suite
            </div>
          </div>
          <div style={{ display: 'flex', gap: '1rem' }}>
            <span
              style={{
                backgroundColor: 'rgba(0, 242, 254, 0.1)',
                color: 'var(--primary)',
                border: '1px solid var(--primary)',
                padding: '4px 12px',
                fontSize: '11px',
                fontFamily: 'var(--font-mono)',
              }}
            >
              PHASE 1 IN PROGRESS
            </span>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer
        style={{
          borderTop: '1px solid var(--border-hairline)',
          padding: '1rem 2rem',
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: '12px',
          color: 'var(--metric-dim)',
          fontFamily: 'var(--font-mono)',
        }}
      >
        <div>WORKSPACE: D:\FraudGraph Intelligence</div>
        <div>VIT VELLORE — CSI4007 GPU PROGRAMMING CAPSTONE</div>
      </footer>
    </div>
  );
};

export default App;
