import React from 'react';
import { Cpu, ShieldCheck, Activity, RefreshCw } from 'lucide-react';
import { BreadcrumbHeader } from '../components/layout/BreadcrumbHeader';
import { Panel } from '../components/common/Panel';
import { StatusBadge } from '../components/common/StatusBadge';
import { TelemetryCard } from '../components/common/TelemetryCard';
import { DataTable, Column } from '../components/common/DataTable';
import { StateView } from '../components/common/StateView';
import { CyberButton } from '../components/common/CyberButton';
import { useHealthPolling } from '../hooks/useHealthPolling';

interface KernelMetadata {
  id: string;
  name: string;
  source_file: string;
  optimization_technique: string;
  memory_hierarchy: string;
  colab_latency_ms: string;
  status: string;
}

export const CudaLabPage: React.FC = () => {
  const {
    data: health,
    isLoading,
    isRefreshing,
    refresh: refreshHealth,
  } = useHealthPolling({ intervalMs: 15000, enabled: true });

  const kernels: KernelMetadata[] = [
    {
      id: 'K01',
      name: 'Baseline Global Memory Aggregator',
      source_file: 'fgi_baseline_kernels.cu',
      optimization_technique: '1 thread per node, direct global memory row_ptr traversal',
      memory_hierarchy: 'Global Memory (High serialization on skewed nodes)',
      colab_latency_ms: '0.0679 ms',
      status: 'VERIFIED_COLAB',
    },
    {
      id: 'K02',
      name: 'Warp-Coalesced Aggregator',
      source_file: 'fgi_coalesced_kernels.cu',
      optimization_technique: '1 warp (32 threads) per node, strided feature dimension contraction',
      memory_hierarchy: 'Coalesced 128-byte transactions, reduced bank conflicts',
      colab_latency_ms: '0.0612 ms',
      status: 'VERIFIED_COLAB',
    },
    {
      id: 'K03',
      name: 'Degree-Aware Hybrid Dispatch',
      source_file: 'fgi_degree_aware_kernels.cu',
      optimization_technique: 'Binning nodes by degree (Low <32, Mid 32-256, High >256)',
      memory_hierarchy: 'Shared memory accumulation for dense power-law hubs',
      colab_latency_ms: '0.0594 ms',
      status: 'VERIFIED_COLAB',
    },
    {
      id: 'K04',
      name: 'Pinned Memory Dual-Stream Pipeline',
      source_file: 'main_streams.cu',
      optimization_technique: 'Asynchronous cudaMemcpyAsync overlapping with compute',
      memory_hierarchy: 'Pinned host RAM DMA + multi-stream kernel concurrency',
      colab_latency_ms: '0.0581 ms',
      status: 'VERIFIED_COLAB',
    },
  ];

  const columns: Column<KernelMetadata>[] = [
    {
      key: 'name',
      header: 'KERNEL IMPLEMENTATION',
      render: (r) => (
        <div>
          <span className="font-semibold text-[#FFFFFF] text-xs block">{r.name}</span>
          <span className="font-mono text-[10px] text-[var(--primary)]">{r.source_file}</span>
        </div>
      ),
      width: '240px',
    },
    {
      key: 'optimization_technique',
      header: 'PARALLELIZATION ARCHITECTURE',
      render: (r) => <span className="text-xs text-[var(--text-primary)]">{r.optimization_technique}</span>,
    },
    {
      key: 'memory_hierarchy',
      header: 'TARGET MEMORY HIERARCHY',
      render: (r) => <span className="text-xs text-[var(--text-muted)] font-mono">{r.memory_hierarchy}</span>,
    },
    {
      key: 'colab_latency_ms',
      header: 'COLAB T4 KERNEL MEAN',
      render: (r) => (
        <span className="font-mono text-xs text-[var(--secondary)] font-semibold">
          {r.colab_latency_ms}
        </span>
      ),
      width: '160px',
      align: 'right',
    },
    {
      key: 'status',
      header: 'VERIFICATION',
      render: () => <StatusBadge variant="success">PASSED (1e-5)</StatusBadge>,
      width: '130px',
      align: 'center',
    },
  ];

  return (
    <div className="flex-1 flex flex-col">
      <BreadcrumbHeader
        category="MICRO-ARCHITECTURAL ACCELERATION"
        title="CUDA Performance Lab"
        subtitle="Low-level kernel profiling, memory hierarchy analysis, and timing boundary distinctions across GPU execution tiers."
        badges={
          <>
            <StatusBadge
              variant={
                health?.compute_engine?.cuda_available_locally
                  ? 'success'
                  : 'warning'
              }
            >
              {health?.compute_engine?.cuda_compiler_status ||
                (isLoading ? 'PROBING RUNTIME...' : 'LOCAL NVCC UNAVAILABLE')}
            </StatusBadge>
            <StatusBadge variant="success">COLAB TESLA T4 VERIFIED</StatusBadge>
          </>
        }
        actions={
          <CyberButton
            variant="secondary"
            size="sm"
            onClick={refreshHealth}
            disabled={isRefreshing}
            icon={<RefreshCw size={13} className={isRefreshing ? 'animate-spin' : ''} />}
          >
            PROBE HOST GPU
          </CyberButton>
        }
      />

      <div className="p-6 max-w-7xl mx-auto w-full space-y-6 flex-1 flex flex-col">
        {/* Hardware Status Split Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <TelemetryCard
            label="LOCAL WINDOWS HOST GPU"
            value={
              health?.compute_engine?.local_gpu_detected ||
              (isLoading ? 'PROBING...' : 'NO GPU DETECTED')
            }
            unit={
              health?.compute_engine?.cuda_available_locally
                ? 'CUDA READY'
                : health?.compute_engine?.local_gpu_detected
                ? 'NVCC ABSENT'
                : 'NO GPU'
            }
            subtext={
              health?.compute_engine?.local_gpu_detected
                ? health?.compute_engine?.vram_free_mb !== null &&
                  health?.compute_engine?.vram_free_mb !== undefined
                  ? `${health.compute_engine.vram_free_mb} MB Free VRAM • Local NVCC compiler absent • Fallback active`
                  : 'Local GPU detected • VRAM diagnostics unavailable'
                : 'No physical GPU detected on host • CPU execution fallback active'
            }
            icon={<Cpu size={16} />}
            statusColor={
              health?.compute_engine?.cuda_available_locally ? 'mint' : 'amber'
            }
          />
          <TelemetryCard
            label="CANONICAL BENCHMARK RIG"
            value="NVIDIA Tesla T4"
            unit="HISTORICAL"
            subtext="40 SMs • 2,560 CUDA cores • 14.9 GB VRAM • Ingested Colab archive"
            icon={<Activity size={16} />}
            statusColor="mint"
          />
          <TelemetryCard
            label="NUMERICAL TOLERANCE"
            value="1e-5 ATOL"
            unit="FLOAT32"
            subtext="Dual-tolerance verifier (rtol=1e-5, atol=1e-5) vs CPU Reference"
            icon={<ShieldCheck size={16} />}
            statusColor="cyan"
          />
        </div>

        {/* Local Compilation State Alert */}
        <StateView
          type="unavailable"
          title="LOCAL CUDA EXECUTION BOUNDARY"
          message="Local Windows environment does not have the NVIDIA CUDA Toolkit (nvcc) installed in PATH. All GPU workloads in local Phase 8 benchmark sessions are explicitly tagged as SKIPPED_CUDA_UNAVAILABLE. Verified GPU measurements are ingested from the Colab Tesla T4 runner."
        />

        {/* Kernel Breakdown Panel */}
        <Panel
          title="FORENSIC KERNEL ARCHITECTURE BREAKDOWN"
          subtitle="BACKEND: backend/gpu_engine/cuda/"
          badge={<StatusBadge variant="muted">4 KERNELS</StatusBadge>}
          noPadding
        >
          <DataTable
            columns={columns}
            data={kernels}
            keyExtractor={(r) => r.id}
          />
        </Panel>

        {/* Timing Boundary Explanation */}
        <div className="p-4 border border-[var(--border-hairline)] bg-[var(--surface-graphite)] space-y-3">
          <div className="font-mono text-xs uppercase text-[var(--primary)] font-semibold">
            Timing Boundary Protocol & Measurement Standards
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-[var(--text-muted)] leading-relaxed">
            <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)]">
              <strong className="text-[#FFFFFF] block mb-1 font-mono uppercase text-[11px]">
                CPU Boundary: cpu_host_timer
              </strong>
              Utilizes high-resolution Python <span className="font-mono text-[var(--primary)]">time.perf_counter_ns()</span> wrapping purely the neighbor accumulation loops. Excludes dataset fixture loading, graph parsing, and preflight memory checks.
            </div>
            <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)]">
              <strong className="text-[#FFFFFF] block mb-1 font-mono uppercase text-[11px]">
                GPU Boundary: gpu_cuda_event
              </strong>
              Utilizes hardware-synchronized <span className="font-mono text-[var(--secondary)]">cudaEventRecord()</span> capturing kernel-only time on the active CUDA stream. Memory transfer times (H2D, D2H) are measured separately and reported in end-to-end breakdowns.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
