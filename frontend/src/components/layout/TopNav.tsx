import React, { useState, useEffect, useRef } from 'react';
import { NavLink } from 'react-router-dom';
import { Settings, Menu, X } from 'lucide-react';
import clsx from 'clsx';
import { NavItem } from '../../types/navigation';
import { HealthData } from '../../types/api';

export interface TopNavProps {
  health: HealthData | null;
  healthLoading: boolean;
  onOpenHealth: () => void;
  onOpenSettings: () => void;
}

export const NAV_ITEMS: NavItem[] = [
  { id: 'overview', label: 'Overview', path: '/', description: 'Operational forensic cockpit & telemetry highlights' },
  { id: 'explorer', label: 'Graph Explorer', path: '/explorer', description: 'Topological sub-graph neighbor analysis' },
  { id: 'transactions', label: 'Transactions', path: '/transactions', badge: '10K', description: 'Tabular synthetic forensic ledger' },
  { id: 'cuda-lab', label: 'CUDA Lab', path: '/cuda-lab', badge: 'GPU', description: 'Kernel profiling & memory hierarchy telemetry' },
  { id: 'experiments', label: 'Experiments', path: '/experiments', description: 'Benchmark matrix & reproducibility manager' },
  { id: 'evaluation', label: 'Evaluation', path: '/evaluation', badge: 'ML', description: 'Classifier metrics & ROC/PR decision thresholds' },
  { id: 'datasets', label: 'Datasets', path: '/datasets', description: 'Canonical graph fixtures & schema registry' },
  { id: 'reports', label: 'Reports', path: '/reports', description: 'Publication monographs & research dossier viewer' },
];

export const TopNav: React.FC<TopNavProps> = ({
  health,
  healthLoading,
  onOpenHealth,
  onOpenSettings,
}) => {
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  const menuToggleRef = useRef<HTMLButtonElement>(null);
  const headerRef = useRef<HTMLElement>(null);
  const isHealthy = health?.status === 'healthy';

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isMobileOpen) {
        setIsMobileOpen(false);
        menuToggleRef.current?.focus();
      }
    };

    const handleResize = () => {
      if (window.innerWidth >= 1024 && isMobileOpen) {
        setIsMobileOpen(false);
      }
    };

    const handleClickOutside = (e: MouseEvent) => {
      if (
        isMobileOpen &&
        headerRef.current &&
        !headerRef.current.contains(e.target as Node)
      ) {
        setIsMobileOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('resize', handleResize);
    document.addEventListener('mousedown', handleClickOutside);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('resize', handleResize);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isMobileOpen]);

  return (
    <header
      ref={headerRef}
      className="sticky top-0 z-40 border-b border-[var(--border-hairline)] bg-[rgba(13,17,23,0.95)] backdrop-blur-md"
    >
      <div className="h-14 px-4 flex items-center justify-between">
        {/* Brand Cluster */}
        <div className="flex items-center gap-4 shrink-0">
          <NavLink
            to="/"
            onClick={() => setIsMobileOpen(false)}
            className="flex items-center gap-2.5 text-inherit no-underline focus-visible:ring-1 focus-visible:ring-[var(--primary)] focus-visible:outline-none"
          >
            <div className="w-2.5 h-2.5 bg-[var(--primary)] shadow-[0_0_8px_var(--primary)] shrink-0" />
            <div className="flex flex-col">
              <span className="font-sans font-bold text-xs tracking-widest text-[#FFFFFF] leading-tight">
                FRAUDGRAPH
              </span>
              <span className="font-mono text-[9px] tracking-widest text-[var(--primary)]">
                INTELLIGENCE
              </span>
            </div>
          </NavLink>

          <span className="hidden xl:inline-block font-mono text-[10px] text-[var(--metric-dim)] border-l border-[var(--border-hairline)] pl-3">
            CSI4007 // CAPSTONE
          </span>
        </div>

        {/* Primary Navigation Ribbon (Desktop) */}
        <nav className="hidden lg:flex items-center gap-1 overflow-x-auto h-full px-2" aria-label="Primary navigation">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.id}
              to={item.path}
              end={item.path === '/'}
              className={({ isActive }) =>
                clsx(
                  'h-full px-3 font-mono text-xs tracking-wider uppercase flex items-center gap-1.5 border-b-2 transition-none select-none text-[var(--text-muted)] hover:text-[#FFFFFF] focus-visible:ring-1 focus-visible:ring-[var(--primary)] focus-visible:outline-none',
                  isActive
                    ? 'border-[var(--primary)] text-[var(--primary)] font-semibold bg-[rgba(0,242,254,0.03)]'
                    : 'border-transparent hover:border-[var(--border-hairline)]'
                )
              }
            >
              <span>{item.label}</span>
              {item.badge && (
                <span className="text-[9px] font-mono px-1 py-0.2 border border-[var(--border-hairline)] bg-[var(--surface-raised)] text-[var(--metric-dim)]">
                  {item.badge}
                </span>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Utility Controls Cluster */}
        <div className="flex items-center gap-2 shrink-0">
          {/* System Health Diagnostic Pill */}
          <button
            onClick={onOpenHealth}
            className="flex items-center gap-2 px-2.5 py-1 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)] text-xs font-mono text-[var(--text-muted)] hover:border-[var(--primary)] hover:text-[var(--primary)] focus-visible:ring-1 focus-visible:ring-[var(--primary)] focus-visible:outline-none transition-none"
            title="Open System Health & Hardware Diagnostics"
            aria-label="System Health"
          >
            <div
              className={clsx(
                'w-2 h-2',
                healthLoading
                  ? 'bg-[var(--alert-warning)] animate-pulse'
                  : isHealthy
                  ? 'bg-[var(--secondary)]'
                  : 'bg-[var(--alert-critical)]'
              )}
            />
            <span className="hidden sm:inline text-[11px]">
              {healthLoading
                ? 'PROBING...'
                : isHealthy
                ? 'SYS HEALTHY'
                : 'OFFLINE'}
            </span>
          </button>

          {/* Workstation Settings Trigger */}
          <button
            onClick={onOpenSettings}
            className="w-8 h-8 flex items-center justify-center border border-[var(--border-hairline)] bg-[var(--surface-obsidian)] text-[var(--text-muted)] hover:text-[var(--primary)] hover:border-[var(--primary)] focus-visible:ring-1 focus-visible:ring-[var(--primary)] focus-visible:outline-none transition-none"
            title="Workstation Settings & Preferences"
            aria-label="Settings"
          >
            <Settings size={14} />
          </button>

          {/* Mobile Menu Toggle Button (lg:hidden) */}
          <button
            ref={menuToggleRef}
            type="button"
            id="mobile-nav-toggle"
            aria-expanded={isMobileOpen}
            aria-controls="mobile-navigation-menu"
            aria-label={isMobileOpen ? 'Close navigation menu' : 'Open navigation menu'}
            onClick={() => setIsMobileOpen((prev) => !prev)}
            className="flex lg:hidden w-8 h-8 items-center justify-center border border-[var(--border-hairline)] bg-[var(--surface-obsidian)] text-[var(--text-muted)] hover:text-[#FFFFFF] hover:border-[var(--primary)] focus-visible:ring-1 focus-visible:ring-[var(--primary)] focus-visible:outline-none transition-none"
          >
            {isMobileOpen ? <X size={15} /> : <Menu size={15} />}
          </button>
        </div>
      </div>

      {/* Mobile Navigation Dropdown Menu (lg:hidden) */}
      {isMobileOpen && (
        <nav
          id="mobile-navigation-menu"
          aria-label="Mobile navigation"
          className="lg:hidden border-t border-[var(--border-hairline)] bg-[var(--surface-ground)] max-h-[calc(100vh-3.5rem)] overflow-y-auto px-4 py-3 space-y-2 shadow-2xl"
        >
          <div className="flex items-center justify-between pb-2 border-b border-[var(--border-hairline)]">
            <span className="font-mono text-[10px] tracking-widest text-[var(--metric-dim)] uppercase">
              NAVIGATION MATRIX // 8 PRIMARY ROUTES
            </span>
            <span className="font-mono text-[10px] text-[var(--primary)]">
              ESC TO CLOSE
            </span>
          </div>

          <div className="space-y-1.5 pt-1">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.id}
                to={item.path}
                end={item.path === '/'}
                onClick={() => setIsMobileOpen(false)}
                className={({ isActive }) =>
                  clsx(
                    'block p-3 border font-mono transition-none select-none focus-visible:ring-1 focus-visible:ring-[var(--primary)] focus-visible:outline-none',
                    isActive
                      ? 'border-[var(--primary)] bg-[rgba(0,242,254,0.06)] text-[var(--primary)] font-semibold'
                      : 'border-[var(--border-hairline)] bg-[var(--surface-obsidian)] text-[var(--text-muted)] hover:text-[#FFFFFF] hover:border-[var(--border-strong)]'
                  )
                }
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs uppercase tracking-wider">{item.label}</span>
                  {item.badge && (
                    <span className="text-[9px] font-mono px-1 py-0.5 border border-[var(--border-hairline)] bg-[var(--surface-raised)] text-[var(--metric-dim)]">
                      {item.badge}
                    </span>
                  )}
                </div>
                <div className="text-[11px] text-[var(--metric-dim)] font-mono leading-tight">
                  {item.description}
                </div>
              </NavLink>
            ))}
          </div>
        </nav>
      )}
    </header>
  );
};
