import React, { useState, useEffect } from 'react';
import { Modal } from '../common/Modal';
import { CyberButton } from '../common/CyberButton';
import { TableDensity, WorkstationSettings } from '../../types/navigation';

export interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: WorkstationSettings;
  onSaveSettings: (settings: WorkstationSettings) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onSaveSettings,
}) => {
  const [localSettings, setLocalSettings] = useState<WorkstationSettings>(settings);

  useEffect(() => {
    setLocalSettings(settings);
  }, [settings, isOpen]);

  const handleSave = () => {
    onSaveSettings(localSettings);
    onClose();
  };

  const densities: { id: TableDensity; label: string; desc: string }[] = [
    { id: 'compact', label: 'Compact (28px)', desc: 'Maximum information density for forensic table scanning' },
    { id: 'standard', label: 'Standard (36px)', desc: 'Balanced visual cadence and comfortable touch targets' },
    { id: 'spacious', label: 'Spacious (44px)', desc: 'Expanded spacing for high-resolution audit inspection' },
  ];

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Workstation Preferences"
      subtitle="Local Client State"
      maxWidth="md"
      actions={
        <div className="flex justify-end gap-2 w-full">
          <CyberButton variant="secondary" size="sm" onClick={onClose}>
            CANCEL
          </CyberButton>
          <CyberButton variant="primary" size="sm" onClick={handleSave}>
            APPLY PREFERENCES
          </CyberButton>
        </div>
      }
    >
      <div className="space-y-6">
        {/* Table Density Section */}
        <div>
          <label className="font-mono text-xs uppercase tracking-wider text-[var(--primary)] font-semibold block mb-2">
            Table Density Cadence
          </label>
          <div className="space-y-2">
            {densities.map((d) => (
              <label
                key={d.id}
                className={`p-3 border flex items-start gap-3 cursor-pointer ${
                  localSettings.tableDensity === d.id
                    ? 'border-[var(--primary)] bg-[rgba(0,242,254,0.04)]'
                    : 'border-[var(--border-hairline)] bg-[var(--surface-obsidian)] hover:border-[var(--text-muted)]'
                }`}
              >
                <input
                  type="radio"
                  name="tableDensity"
                  checked={localSettings.tableDensity === d.id}
                  onChange={() =>
                    setLocalSettings({ ...localSettings, tableDensity: d.id })
                  }
                  className="mt-0.5"
                />
                <div>
                  <div className="text-xs font-semibold text-[#FFFFFF]">{d.label}</div>
                  <div className="text-[11px] text-[var(--text-muted)] mt-0.5">{d.desc}</div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Display Toggles */}
        <div>
          <label className="font-mono text-xs uppercase tracking-wider text-[var(--primary)] font-semibold block mb-2">
            Visual Grid Options
          </label>
          <div className="space-y-2">
            <label className="p-3 border border-[var(--border-hairline)] bg-[var(--surface-obsidian)] flex items-center justify-between cursor-pointer">
              <div>
                <div className="text-xs font-semibold text-[#FFFFFF]">High-Contrast Grid Lines</div>
                <div className="text-[11px] text-[var(--text-muted)]">
                  Renders 1px hairline structural grid boundaries across data grids
                </div>
              </div>
              <input
                type="checkbox"
                checked={localSettings.highContrastGrid}
                onChange={(e) =>
                  setLocalSettings({ ...localSettings, highContrastGrid: e.target.checked })
                }
              />
            </label>
          </div>
        </div>

        {/* Client Storage Notice */}
        <div className="text-[11px] text-[var(--metric-dim)] font-mono border-t border-[var(--border-hairline)] pt-3">
          PREFERENCES PERSISTED LOCALLY IN CLIENT STORAGE // ZERO CLOUD TRACKING
        </div>
      </div>
    </Modal>
  );
};
