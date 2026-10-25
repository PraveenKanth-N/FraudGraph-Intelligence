import React, { useState, useMemo } from 'react';
import { Outlet } from 'react-router-dom';
import { TopNav } from './TopNav';
import { SystemHealthDrawer } from './SystemHealthDrawer';
import { SettingsModal } from './SettingsModal';
import { useHealthPolling } from '../../hooks/useHealthPolling';
import { WorkstationSettings } from '../../types/navigation';

const DEFAULT_SETTINGS: WorkstationSettings = {
  tableDensity: 'standard',
  telemetryStreamSimulated: false,
  highContrastGrid: true,
  activeDatasetId: 'synthetic_fraud_10k',
};

export const AppLayout: React.FC = () => {
  const healthResource = useHealthPolling({ intervalMs: 15000, enabled: true });
  const [isHealthOpen, setIsHealthOpen] = useState<boolean>(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [settings, setSettings] = useState<WorkstationSettings>(() => {
    try {
      const stored = localStorage.getItem('fgi_workstation_settings');
      return stored ? JSON.parse(stored) : DEFAULT_SETTINGS;
    } catch {
      return DEFAULT_SETTINGS;
    }
  });

  const handleSaveSettings = (newSettings: WorkstationSettings) => {
    setSettings(newSettings);
    try {
      localStorage.setItem('fgi_workstation_settings', JSON.stringify(newSettings));
    } catch {
      // safe fallback
    }
  };

  // F-06 Fix: Stabilize context object reference to prevent unneeded child route re-renders
  const outletContext = useMemo(
    () => ({
      settings,
      health: healthResource.data,
      healthState: healthResource.state,
      onRefreshHealth: healthResource.refresh,
    }),
    [settings, healthResource.data, healthResource.state, healthResource.refresh]
  );

  return (
    <div className="min-h-screen flex flex-col bg-[var(--surface-obsidian)] text-[var(--text-primary)]">
      {/* Universal Top Navigation Header */}
      <TopNav
        health={healthResource.data}
        healthLoading={healthResource.isLoading}
        telemetryState={healthResource.state}
        onOpenHealth={() => setIsHealthOpen(true)}
        onOpenSettings={() => setIsSettingsOpen(true)}
      />

      {/* Main Page Content */}
      <main className="flex-1 flex flex-col" id="main-content">
        <Outlet context={outletContext} />
      </main>

      {/* Institutional Academic Footer */}
      <footer className="border-t border-[var(--border-hairline)] bg-[var(--surface-obsidian)] px-6 py-4 flex flex-col sm:flex-row items-center justify-between text-xs font-mono text-[var(--metric-dim)] gap-2">
        <div className="flex items-center gap-3">
          <span>WORKSPACE: D:\FraudGraph Intelligence</span>
          <span className="hidden sm:inline border-l border-[var(--border-hairline)] pl-3">
            HEAD: ae5b1aa (Phase 9B)
          </span>
        </div>
        <div className="flex items-center gap-3 text-right">
          <span>VIT VELLORE // CSI4007 GPU PROGRAMMING CAPSTONE</span>
          <span className="hidden md:inline border-l border-[var(--border-hairline)] pl-3 text-[var(--text-muted)]">
            SYNTHETIC BENCHMARK DATA ONLY
          </span>
        </div>
      </footer>

      {/* Diagnostics Drawer Modal */}
      <SystemHealthDrawer
        isOpen={isHealthOpen}
        onClose={() => setIsHealthOpen(false)}
        health={healthResource.data}
        isLoading={healthResource.isLoading || healthResource.isRefreshing}
        telemetryState={healthResource.state}
        lastSuccessTime={healthResource.lastSuccessTime}
        error={healthResource.error}
        onRefresh={healthResource.refresh}
      />

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onSaveSettings={handleSaveSettings}
      />
    </div>
  );
};
