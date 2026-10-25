import React, { useState } from 'react';
import { Network, ZoomIn, Maximize2 } from 'lucide-react';
import { BreadcrumbHeader } from '../components/layout/BreadcrumbHeader';
import { Panel } from '../components/common/Panel';
import { StatusBadge } from '../components/common/StatusBadge';
import { CyberButton } from '../components/common/CyberButton';

export const GraphExplorerPage: React.FC = () => {
  const [selectedHop, setSelectedHop] = useState<number>(2);
  const [selectedEntityFilter, setSelectedEntityFilter] = useState<string>('all');

  return (
    <div className="flex-1 flex flex-col">
      <BreadcrumbHeader
        category="TOPOLOGICAL GRAPH EXPLORATION"
        title="Topological Graph Explorer"
        subtitle="Sub-graph extraction and parallel neighbor inspection across Compressed Sparse Row (CSR) transaction graph topology."
        badges={
          <>
            <StatusBadge variant="primary">PHASE 9B SCAFFOLD</StatusBadge>
            <StatusBadge variant="muted">CYTOSCAPE.JS INTEGRATION (PHASE 9D)</StatusBadge>
          </>
        }
        actions={
          <div className="flex items-center gap-2">
            <CyberButton variant="secondary" size="sm" icon={<ZoomIn size={13} />}>
              ZOOM
            </CyberButton>
            <CyberButton variant="secondary" size="sm" icon={<Maximize2 size={13} />}>
              FIT VIEW
            </CyberButton>
          </div>
        }
      />

      <div className="p-6 max-w-7xl mx-auto w-full space-y-6 flex-1 flex flex-col">
        {/* Topology Boundary Milestone Banner */}
        <div className="p-4 border border-[var(--primary)] bg-[rgba(0,242,254,0.03)] flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div>
            <div className="font-mono text-xs text-[var(--primary)] font-semibold uppercase tracking-wider mb-0.5">
              Phase 9B Scaffolding Active // Phase 9D Integration Target
            </div>
            <div className="text-xs text-[var(--text-muted)]">
              Interactive Cytoscape.js WebGL canvas and CSR subgraph extraction endpoints will be integrated in Phase 9D.
              Below is the verified structural topology boundary for <span className="font-mono text-[var(--text-primary)]">synthetic_fraud_10k.npz</span>.
            </div>
          </div>
          <StatusBadge variant="primary">CSR TOPOLOGY READY</StatusBadge>
        </div>

        {/* 2-Column Explorer Scaffold */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 flex-1">
          {/* Main Canvas Area Placeholder */}
          <div className="lg:col-span-3">
            <Panel
              title="TOPOLOGICAL SUBGRAPH CANVAS"
              subtitle="DIRECTED NEIGHBOR EMBEDDINGS"
              bodyClassName="h-[520px] flex flex-col items-center justify-center relative bg-[var(--surface-obsidian)] border border-dashed border-[var(--border-hairline)]"
            >
              <div className="flex flex-col items-center text-center p-8 max-w-md">
                <Network size={44} className="text-[var(--primary)] mb-4 animate-pulse" />
                <h3 className="font-serif text-lg text-[var(--tertiary)] mb-2">
                  Topological Network Visualizer
                </h3>
                <p className="text-xs text-[var(--text-muted)] leading-relaxed mb-4">
                  In Phase 9D, this viewport renders high-density subgraphs using Cytoscape.js, extracting up to 250 directed neighbors around flagged transaction hubs with zero border radius node cards.
                </p>

                <div className="grid grid-cols-3 gap-3 w-full p-3 bg-[var(--surface-graphite)] border border-[var(--border-hairline)] font-mono text-[11px] text-center">
                  <div>
                    <span className="text-[var(--text-muted)] block">NODES</span>
                    <span className="text-[var(--text-primary)] font-semibold">10,000</span>
                  </div>
                  <div>
                    <span className="text-[var(--text-muted)] block">DIRECTED EDGES</span>
                    <span className="text-[var(--text-primary)] font-semibold">231,984</span>
                  </div>
                  <div>
                    <span className="text-[var(--text-muted)] block">FRAUD PREVALENCE</span>
                    <span className="text-[var(--alert-critical)] font-semibold">3.5% (350)</span>
                  </div>
                </div>
              </div>
            </Panel>
          </div>

          {/* Sidebar Inspector & Filters Scaffold */}
          <div className="space-y-6">
            <Panel title="EXPLORER FILTERS">
              <div className="space-y-4">
                <div>
                  <label className="font-mono text-[10px] uppercase text-[var(--text-muted)] block mb-1.5">
                    NEIGHBORHOOD DEPTH (HOPS)
                  </label>
                  <div className="flex items-center gap-2">
                    {[1, 2, 3].map((hop) => (
                      <button
                        key={hop}
                        onClick={() => setSelectedHop(hop)}
                        className={`flex-1 py-1.5 font-mono text-xs border ${
                          selectedHop === hop
                            ? 'border-[var(--primary)] bg-[rgba(0,242,254,0.1)] text-[var(--primary)] font-semibold'
                            : 'border-[var(--border-hairline)] text-[var(--text-muted)] hover:border-[var(--text-primary)]'
                        }`}
                      >
                        {hop}-HOP
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="font-mono text-[10px] uppercase text-[var(--text-muted)] block mb-1.5">
                    ENTITY CLASSIFICATION
                  </label>
                  <div className="space-y-1.5">
                    {['all', 'fraud_nodes', 'high_degree', 'isolated'].map((filter) => (
                      <button
                        key={filter}
                        onClick={() => setSelectedEntityFilter(filter)}
                        className={`w-full px-3 py-1.5 text-xs text-left font-mono uppercase border ${
                          selectedEntityFilter === filter
                            ? 'border-[var(--primary)] text-[var(--primary)] bg-[rgba(0,242,254,0.05)]'
                            : 'border-[var(--border-hairline)] text-[var(--text-muted)] hover:border-[var(--text-primary)]'
                        }`}
                      >
                        {filter.replace('_', ' ')}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)] text-xs text-[var(--text-muted)] space-y-1.5 font-mono text-[11px]">
                  <div className="text-[var(--primary)] font-semibold">DEGREE STATISTICS:</div>
                  <div className="flex justify-between">
                    <span>MEAN DEGREE:</span>
                    <span className="text-[#FFFFFF]">23.20</span>
                  </div>
                  <div className="flex justify-between">
                    <span>MAX DEGREE:</span>
                    <span className="text-[#FFFFFF]">252</span>
                  </div>
                  <div className="flex justify-between">
                    <span>ISOLATED NODES:</span>
                    <span className="text-[#FFFFFF]">1,155 (11.55%)</span>
                  </div>
                </div>
              </div>
            </Panel>
          </div>
        </div>
      </div>
    </div>
  );
};
