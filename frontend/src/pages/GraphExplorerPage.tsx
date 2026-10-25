import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Network, ZoomIn, ZoomOut, Maximize2, RotateCcw, AlertTriangle, Loader2, Info } from 'lucide-react';
import { BreadcrumbHeader } from '../components/layout/BreadcrumbHeader';
import { Panel } from '../components/common/Panel';
import { StatusBadge } from '../components/common/StatusBadge';
import { CyberButton } from '../components/common/CyberButton';
import { fetchSubgraph } from '../services/api';
import type { SubgraphResponse, SubgraphNode } from '../types/api';

export const GraphExplorerPage: React.FC = () => {
  const [nodeIdInput, setNodeIdInput] = useState<string>('5407');
  const [activeNodeId, setActiveNodeId] = useState<number>(5407);
  const [selectedHop, setSelectedHop] = useState<number>(1);
  const [maxNodes, setMaxNodes] = useState<number>(150);

  const [data, setData] = useState<SubgraphResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // SVG Pan and Zoom state
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [hoveredNode, setHoveredNode] = useState<SubgraphNode | null>(null);

  const loadGraph = useCallback(async (nId: number, hops: number, cap: number) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchSubgraph({
        nodeId: nId,
        hops,
        maxNodes: cap,
      });
      setData(response);
      setActiveNodeId(nId);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to extract subgraph';
      setError(msg);
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadGraph(5407, 1, 150);
  }, [loadGraph]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const parsed = parseInt(nodeIdInput.trim(), 10);
    if (isNaN(parsed) || parsed < 0 || parsed >= 10000) {
      setError(`Invalid node index: ${nodeIdInput}. Must be an integer between 0 and 9,999.`);
      return;
    }
    loadGraph(parsed, selectedHop, maxNodes);
  };

  const handleHopChange = (h: number) => {
    setSelectedHop(h);
    loadGraph(activeNodeId, h, maxNodes);
  };

  const handleQuickPick = (id: number, defaultHops: number = 1) => {
    setNodeIdInput(String(id));
    setSelectedHop(defaultHops);
    loadGraph(id, defaultHops, maxNodes);
  };

  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 0.25, 3.0));
  const handleZoomOut = () => setZoom((prev) => Math.max(prev - 0.25, 0.4));
  const handleResetView = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  };

  // Concentric radial layout coordinates
  const canvasWidth = 720;
  const canvasHeight = 480;
  const cx = canvasWidth / 2;
  const cy = canvasHeight / 2;

  const nodePositions = useMemo(() => {
    if (!data || data.nodes.length === 0) return new Map<number, { x: number; y: number }>();
    const posMap = new Map<number, { x: number; y: number }>();

    // Group nodes by depth
    const depthGroups: Record<number, SubgraphNode[]> = { 0: [], 1: [], 2: [], 3: [] };
    for (const n of data.nodes) {
      if (depthGroups[n.depth]) {
        depthGroups[n.depth].push(n);
      }
    }

    // Root at center
    if (depthGroups[0].length > 0) {
      posMap.set(depthGroups[0][0].id, { x: cx, y: cy });
    }

    const radiusForDepth: Record<number, number> = {
      1: 110,
      2: 185,
      3: 235,
    };

    [1, 2, 3].forEach((d) => {
      const nodesAtD = depthGroups[d];
      const count = nodesAtD.length;
      if (count === 0) return;
      const r = radiusForDepth[d];
      nodesAtD.forEach((node, idx) => {
        const angle = (2 * Math.PI * idx) / count - Math.PI / 2;
        const x = cx + r * Math.cos(angle);
        const y = cy + r * Math.sin(angle);
        posMap.set(node.id, { x, y });
      });
    });

    return posMap;
  }, [data, cx, cy]);

  return (
    <div className="flex-1 flex flex-col">
      <BreadcrumbHeader
        category="TOPOLOGICAL GRAPH EXPLORATION"
        title="Topological Graph Explorer"
        subtitle="Sub-graph extraction and parallel neighbor inspection across Compressed Sparse Row (CSR) transaction graph topology."
        badges={
          <>
            <StatusBadge variant="primary">LIVE SUBGRAPH REST</StatusBadge>
            <StatusBadge variant="success">CSR TOPOLOGY VALIDATED</StatusBadge>
            {data?.summary.is_truncated && (
              <StatusBadge variant="warning">RESULT TRUNCATED</StatusBadge>
            )}
          </>
        }
        actions={
          <div className="flex items-center gap-2">
            <CyberButton
              variant="secondary"
              size="sm"
              icon={<ZoomIn size={13} />}
              onClick={handleZoomIn}
              title="Zoom In"
            >
              ZOOM +
            </CyberButton>
            <CyberButton
              variant="secondary"
              size="sm"
              icon={<ZoomOut size={13} />}
              onClick={handleZoomOut}
              title="Zoom Out"
            >
              ZOOM -
            </CyberButton>
            <CyberButton
              variant="secondary"
              size="sm"
              icon={<Maximize2 size={13} />}
              onClick={handleResetView}
              title="Fit and Center View"
            >
              FIT VIEW
            </CyberButton>
          </div>
        }
      />

      <div className="p-6 max-w-7xl mx-auto w-full space-y-6 flex-1 flex flex-col">
        {/* Controls and Search Bar */}
        <div className="p-4 border border-[var(--border-hairline)] bg-[var(--surface-graphite)] flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-4">
          <form onSubmit={handleSearchSubmit} className="flex items-center gap-2 flex-1 max-w-md">
            <div className="relative flex-1">
              <input
                type="text"
                placeholder="TARGET NODE ID (0 - 9999)..."
                value={nodeIdInput}
                onChange={(e) => setNodeIdInput(e.target.value)}
                className="w-full px-3 py-1.5 text-xs font-mono uppercase bg-[var(--surface-obsidian)] border border-[var(--border-hairline)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--primary)]"
              />
            </div>
            <CyberButton
              variant="primary"
              size="sm"
              type="submit"
              disabled={loading}
              icon={loading ? <Loader2 size={12} className="animate-spin" /> : <Network size={12} />}
            >
              EXTRACT
            </CyberButton>
          </form>

          {/* Quick Picks for Canonical Representative Nodes */}
          <div className="flex items-center gap-2 flex-wrap text-xs font-mono">
            <span className="text-[var(--text-muted)] text-[10px] uppercase">PRESETS:</span>
            <button
              onClick={() => handleQuickPick(5407, 1)}
              className={`px-2.5 py-1 border text-[11px] ${
                activeNodeId === 5407
                  ? 'border-[var(--primary)] text-[var(--primary)] bg-[rgba(0,242,254,0.1)]'
                  : 'border-[var(--border-hairline)] text-[var(--text-muted)] hover:border-[var(--text-primary)]'
              }`}
            >
              HUB #5407 (DEG 252)
            </button>
            <button
              onClick={() => handleQuickPick(1, 1)}
              className={`px-2.5 py-1 border text-[11px] ${
                activeNodeId === 1
                  ? 'border-[var(--primary)] text-[var(--primary)] bg-[rgba(0,242,254,0.1)]'
                  : 'border-[var(--border-hairline)] text-[var(--text-muted)] hover:border-[var(--text-primary)]'
              }`}
            >
              ISOLATED #1 (DEG 0)
            </button>
            <button
              onClick={() => handleQuickPick(4834, 1)}
              className={`px-2.5 py-1 border text-[11px] ${
                activeNodeId === 4834
                  ? 'border-[var(--primary)] text-[var(--primary)] bg-[rgba(0,242,254,0.1)]'
                  : 'border-[var(--border-hairline)] text-[var(--text-muted)] hover:border-[var(--text-primary)]'
              }`}
            >
              FRAUD #4834 (DEG 245)
            </button>
          </div>
        </div>

        {/* Truncation or Warning Alert Banner */}
        {data?.summary.is_truncated && (
          <div className="p-3 border border-[var(--alert-warning)] bg-[rgba(255,159,10,0.06)] flex items-center gap-3 text-xs">
            <AlertTriangle size={16} className="text-[var(--alert-warning)] flex-shrink-0" />
            <div className="flex-1 font-mono">
              <span className="text-[var(--alert-warning)] font-semibold uppercase mr-2">SUBGRAPH TRUNCATED:</span>
              <span className="text-[var(--text-muted)]">
                Rendering bounded to {data.summary.total_nodes} nodes and {data.summary.total_edges} edges ({data.summary.truncation_reason}). Server-side caps prevent browser rendering degradation.
              </span>
            </div>
            <StatusBadge variant="warning">CAP ENFORCED</StatusBadge>
          </div>
        )}

        {/* Error State Banner */}
        {error && (
          <div className="p-4 border border-[var(--alert-critical)] bg-[rgba(255,59,48,0.06)] flex items-center justify-between gap-3 text-xs">
            <div className="font-mono text-[var(--alert-critical)]">
              <span className="font-semibold uppercase mr-2">EXTRACTION ERROR:</span>
              {error}
            </div>
            <CyberButton
              variant="secondary"
              size="sm"
              onClick={() => loadGraph(activeNodeId, selectedHop, maxNodes)}
              icon={<RotateCcw size={11} />}
            >
              RETRY
            </CyberButton>
          </div>
        )}

        {/* 2-Column Explorer Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 flex-1">
          {/* Main Visualizer Area */}
          <div className="lg:col-span-3">
            <Panel
              title="TOPOLOGICAL SUBGRAPH CANVAS"
              subtitle={`CENTERED ON NODE #${activeNodeId} • ${selectedHop}-HOP DIRECTED CSR`}
              badge={
                data ? (
                  <StatusBadge variant={data.summary.fraud_node_count > 0 ? 'critical' : 'success'}>
                    {data.summary.total_nodes} NODES ({data.summary.fraud_node_count} FRAUD)
                  </StatusBadge>
                ) : undefined
              }
              bodyClassName="h-[520px] flex flex-col items-center justify-center relative bg-[var(--surface-obsidian)] border border-[var(--border-hairline)] overflow-hidden"
            >
              {loading ? (
                <div className="flex flex-col items-center justify-center space-y-3">
                  <Loader2 size={32} className="animate-spin text-[var(--primary)]" />
                  <span className="font-mono text-xs text-[var(--text-muted)] tracking-wider uppercase">
                    EXTRACTING AUTHENTIC CSR SUBGRAPH...
                  </span>
                </div>
              ) : data && data.nodes.length > 0 ? (
                <div className="w-full h-full relative">
                  <svg
                    viewBox={`0 0 ${canvasWidth} ${canvasHeight}`}
                    className="w-full h-full cursor-crosshair select-none"
                  >
                    <defs>
                      <marker
                        id="directed-arrow"
                        viewBox="0 0 10 10"
                        refX="18"
                        refY="5"
                        markerWidth="6"
                        markerHeight="6"
                        orient="auto-start-reverse"
                      >
                        <path d="M 0 1 L 8 5 L 0 9 z" fill="rgba(0, 242, 254, 0.4)" />
                      </marker>
                    </defs>

                    <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
                      {/* Concentric Guide Rings */}
                      <circle cx={cx} cy={cy} r={110} fill="none" stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3" />
                      <circle cx={cx} cy={cy} r={185} fill="none" stroke="rgba(255,255,255,0.03)" strokeDasharray="3 3" />
                      <circle cx={cx} cy={cy} r={235} fill="none" stroke="rgba(255,255,255,0.02)" strokeDasharray="3 3" />

                      {/* Directed Edges */}
                      {data.edges.map((edge, idx) => {
                        const pSource = nodePositions.get(edge.source);
                        const pTarget = nodePositions.get(edge.target);
                        if (!pSource || !pTarget) return null;

                        return (
                          <line
                            key={`edge-${idx}-${edge.source}-${edge.target}`}
                            x1={pSource.x}
                            y1={pSource.y}
                            x2={pTarget.x}
                            y2={pTarget.y}
                            stroke="rgba(0, 242, 254, 0.22)"
                            strokeWidth={1}
                            markerEnd="url(#directed-arrow)"
                          />
                        );
                      })}

                      {/* Nodes */}
                      {data.nodes.map((node) => {
                        const pos = nodePositions.get(node.id);
                        if (!pos) return null;

                        const isRoot = node.is_root;
                        const isFraud = node.is_fraud;
                        const radius = isRoot ? 11 : isFraud ? 6.5 : 5;
                        const fillColor = isRoot
                          ? '#00F2FE'
                          : isFraud
                          ? '#FF3B30'
                          : '#00A8B5';

                        return (
                          <g
                            key={`node-${node.id}`}
                            transform={`translate(${pos.x}, ${pos.y})`}
                            onMouseEnter={() => setHoveredNode(node)}
                            onMouseLeave={() => setHoveredNode(null)}
                            className="cursor-pointer transition-all duration-150"
                          >
                            {isRoot && (
                              <circle
                                r={radius + 4}
                                fill="none"
                                stroke="#00F2FE"
                                strokeWidth={1.5}
                                opacity={0.4}
                                className="animate-pulse"
                              />
                            )}
                            <circle
                              r={radius}
                              fill={fillColor}
                              stroke="#000000"
                              strokeWidth={1.5}
                            />
                            {isRoot && (
                              <text
                                y={-16}
                                textAnchor="middle"
                                fill="#00F2FE"
                                fontSize={10}
                                fontFamily="monospace"
                                fontWeight="bold"
                              >
                                #{node.id} (ROOT)
                              </text>
                            )}
                          </g>
                        );
                      })}
                    </g>
                  </svg>

                  {/* Tooltip Overlay */}
                  {hoveredNode && (
                    <div className="absolute bottom-3 left-3 p-3 bg-[var(--surface-graphite)] border border-[var(--primary)] text-xs font-mono shadow-xl z-20 space-y-1">
                      <div className="text-[var(--primary)] font-bold flex items-center gap-1.5">
                        <span>TRANSACTION #{hoveredNode.id}</span>
                        {hoveredNode.is_root && <span className="text-[10px] text-[var(--secondary)]">(ROOT)</span>}
                      </div>
                      <div className="text-[var(--text-muted)] text-[11px]">
                        HOP DEPTH: <span className="text-white">{hoveredNode.depth}</span>
                      </div>
                      <div className="text-[var(--text-muted)] text-[11px]">
                        DEGREE (NEIGHBORS): <span className="text-white">{hoveredNode.degree}</span>
                      </div>
                      <div className="text-[var(--text-muted)] text-[11px]">
                        TIMESTAMP: <span className="text-white">{hoveredNode.timestamp_sec.toFixed(2)}s</span>
                      </div>
                      <div className="text-[11px]">
                        LABEL:{' '}
                        {hoveredNode.is_fraud ? (
                          <span className="text-[var(--alert-critical)] font-bold">FRAUD (Y=1)</span>
                        ) : (
                          <span className="text-[var(--alert-success)]">LICIT (Y=0)</span>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Canvas Legend */}
                  <div className="absolute top-3 right-3 p-2 bg-[rgba(10,14,20,0.85)] border border-[var(--border-hairline)] text-[10px] font-mono space-y-1">
                    <div className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-[#00F2FE]" />
                      <span className="text-white">QUERY ROOT</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-[#FF3B30]" />
                      <span className="text-white">FRAUD (Y=1)</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-[#00A8B5]" />
                      <span className="text-white">LICIT (Y=0)</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center p-8 text-center max-w-sm">
                  <Network size={40} className="text-[var(--text-muted)] mb-3" />
                  <h4 className="font-serif text-sm text-[var(--text-primary)] mb-1">
                    No Subgraph Vertices Found
                  </h4>
                  <p className="text-xs text-[var(--text-muted)]">
                    Query returned 0 nodes. Please choose a valid transaction node index between 0 and 9,999.
                  </p>
                </div>
              )}
            </Panel>
          </div>

          {/* Sidebar Inspector & Filters */}
          <div className="space-y-6">
            <Panel title="EXPLORER SETTINGS">
              <div className="space-y-4">
                <div>
                  <label className="font-mono text-[10px] uppercase text-[var(--text-muted)] block mb-1.5">
                    NEIGHBORHOOD DEPTH (HOPS)
                  </label>
                  <div className="flex items-center gap-2">
                    {[1, 2, 3].map((hop) => (
                      <button
                        key={hop}
                        onClick={() => handleHopChange(hop)}
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
                    MAX EXTRACTED NODES
                  </label>
                  <div className="grid grid-cols-3 gap-1.5">
                    {[50, 150, 300].map((cap) => (
                      <button
                        key={cap}
                        onClick={() => {
                          setMaxNodes(cap);
                          loadGraph(activeNodeId, selectedHop, cap);
                        }}
                        className={`py-1 text-[11px] font-mono border ${
                          maxNodes === cap
                            ? 'border-[var(--primary)] text-[var(--primary)] bg-[rgba(0,242,254,0.05)] font-semibold'
                            : 'border-[var(--border-hairline)] text-[var(--text-muted)] hover:border-[var(--text-primary)]'
                        }`}
                      >
                        {cap}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Subgraph Metrics Summary */}
                {data && (
                  <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)] text-xs text-[var(--text-muted)] space-y-1.5 font-mono text-[11px]">
                    <div className="text-[var(--primary)] font-semibold uppercase">EXTRACTED TOPOLOGY:</div>
                    <div className="flex justify-between">
                      <span>ROOT NODE:</span>
                      <span className="text-[#FFFFFF]">#{data.query.root_node_id}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>ACTUAL HOPS:</span>
                      <span className="text-[#FFFFFF]">{data.query.actual_hops}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>SUBGRAPH NODES:</span>
                      <span className="text-[#FFFFFF]">{data.summary.total_nodes}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>SUBGRAPH EDGES:</span>
                      <span className="text-[#FFFFFF]">{data.summary.total_edges}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>FRAUD VERTICES:</span>
                      <span className="text-[var(--alert-critical)] font-semibold">
                        {data.summary.fraud_node_count} ({data.summary.fraud_percentage}%)
                      </span>
                    </div>
                  </div>
                )}

                {/* Fixture Provenance Card */}
                {data && (
                  <div className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)] text-[10px] font-mono space-y-1 text-[var(--text-muted)]">
                    <div className="flex items-center gap-1 text-[var(--secondary)] font-semibold uppercase">
                      <Info size={11} />
                      PROVENANCE & INTEGRITY
                    </div>
                    <div>FIXTURE: {data.provenance.fixture_name}</div>
                    <div className="truncate" title={data.provenance.sha256}>
                      SHA256: {data.provenance.sha256.slice(0, 16)}...
                    </div>
                    <div>POLICY: {data.provenance.causality_policy}</div>
                  </div>
                )}
              </div>
            </Panel>
          </div>
        </div>
      </div>
    </div>
  );
};
