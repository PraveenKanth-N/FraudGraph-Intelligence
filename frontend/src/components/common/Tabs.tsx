import React from 'react';
import clsx from 'clsx';

export interface TabItem {
  id: string;
  label: string;
  badge?: string | number;
}

export interface TabsProps {
  tabs: TabItem[];
  activeTab: string;
  onChange: (id: string) => void;
  className?: string;
}

export const Tabs: React.FC<TabsProps> = ({ tabs, activeTab, onChange, className }) => {
  return (
    <div
      className={clsx(
        'flex items-center border-b border-[var(--border-hairline)] bg-[var(--surface-graphite)] gap-1 px-2',
        className
      )}
    >
      {tabs.map((tab) => {
        const isActive = tab.id === activeTab;
        return (
          <button
            key={tab.id}
            onClick={() => onChange(tab.id)}
            className={clsx(
              'h-10 px-4 font-mono text-xs tracking-wider uppercase flex items-center gap-2 border-b-2 transition-none',
              isActive
                ? 'border-[var(--primary)] text-[var(--primary)] font-semibold bg-[rgba(0,242,254,0.03)]'
                : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:border-[var(--border-hairline)]'
            )}
          >
            <span>{tab.label}</span>
            {tab.badge !== undefined && (
              <span
                className={clsx(
                  'px-1.5 py-0.2 text-[10px] font-mono border',
                  isActive
                    ? 'border-[var(--primary)] text-[var(--primary)] bg-[rgba(0,242,254,0.1)]'
                    : 'border-[var(--border-hairline)] text-[var(--text-muted)] bg-[var(--surface-raised)]'
                )}
              >
                {tab.badge}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
};
