import React, { useState, useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import { TopNav } from './TopNav';
import { SystemHealthDrawer } from './SystemHealthDrawer';
import { SettingsModal } from './SettingsModal';
import { fetchHealth } from '../../services/api';
import { HealthData } from '../../types/api';
import { WorkstationSettings } from '../../types/navigation';

const DEFAULT_SETTINGS: WorkstationSettings = {
  tableDensity: 'standard',
  telemetryStreamSimulated: false,
  highContrastGrid: true,
  activeDatasetId: 'synthetic_fraud_10k',
};

export const AppLayout: React.FC = () => {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);
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

  const loadHealth = () => {
    setHealthLoading(true);
    fetchHealth()
      .then((data) => {
        setHealth(data);
        setHealthLoading(false);
      })
      .catch(() => {
        setHealth(null);
        setHealthLoading(false);
      });
  };

  useEffect(() => {
    loadHealth();
  }, []);

  const handleSaveSettings = (newSettings: WorkstationSettings) => {
    setSettings(newSettings);
    try {
      localStorage.setItem('fgi_workstation_settings', JSON.stringify(newSettings));
    } catch {
      // safe fallback
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[var(--surface-obsidian)] text-[var(--text-primary)]">
      {/* Universal Top Navigation Header */}
      <TopNav
        health={health}
        healthLoading={healthLoading}
        onOpenHealth={() => setIsHealthOpen(true)}
        onOpenSettings={() => setIsSettingsOpen(true)}
      />

      {/* Main Page Content */}
      <main className="flex-1 flex flex-col" id="main-content">
        <Outlet context={{ settings, health, onRefreshHealth: loadHealth }} />
      </main>

      {/* Institutional Academic Footer */}
      <footer className="border-t border-[var(--border-hairline)] bg-[var(--surface-obsidian)] px-6 py-4 flex flex-col sm:flex-row items-center justify-between text-xs font-mono text-[var(--metric-dim)] gap-2">
        <div className="flex items-center gap-3">
          <span>WORKSPACE: D:\FraudGraph Intelligence</span>
          <span className="hidden sm:inline border-l border-[var(--border-hairline)] pl-3">
            HEAD: d0ea85a (Phase 9B)
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
        health={health}
        isLoading={healthLoading}
        onRefresh={loadHealth}
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
