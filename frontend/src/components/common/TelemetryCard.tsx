import React from 'react';
import clsx from 'clsx';

export interface TelemetryCardProps {
  label: string;
  value: string | number;
  unit?: string;
  subtext?: string;
  icon?: React.ReactNode;
  statusColor?: 'cyan' | 'mint' | 'red' | 'amber' | 'dim';
  className?: string;
}

export const TelemetryCard: React.FC<TelemetryCardProps> = ({
  label,
  value,
  unit,
  subtext,
  icon,
  statusColor = 'cyan',
  className,
}) => {
  const accentBorder = {
    cyan: 'border-l-2 border-l-[var(--primary)]',
    mint: 'border-l-2 border-l-[var(--secondary)]',
    red: 'border-l-2 border-l-[var(--alert-critical)]',
    amber: 'border-l-2 border-l-[var(--alert-warning)]',
    dim: 'border-l-2 border-l-[var(--metric-dim)]',
  };

  const accentColor = {
    cyan: 'text-[var(--primary)]',
    mint: 'text-[var(--secondary)]',
    red: 'text-[var(--alert-critical)]',
    amber: 'text-[var(--alert-warning)]',
    dim: 'text-[var(--text-muted)]',
  };

  return (
    <div
      className={clsx(
        'bg-[var(--surface-graphite)] border border-[var(--border-hairline)] p-4 flex flex-col justify-between',
        accentBorder[statusColor],
        className
      )}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="font-mono text-[11px] tracking-wider uppercase text-[var(--text-muted)]">
          {label}
        </span>
        {icon && <span className={clsx('shrink-0', accentColor[statusColor])}>{icon}</span>}
      </div>

      <div className="flex items-baseline gap-1.5 my-1">
        <span className="text-2xl font-semibold tracking-tight text-[#FFFFFF] font-sans">
          {value}
        </span>
        {unit && (
          <span className="text-xs font-mono text-[var(--text-muted)] tracking-wider">
            {unit}
          </span>
        )}
      </div>

      {subtext && (
        <div className="text-xs text-[var(--text-muted)] font-sans line-clamp-1 mt-1">
          {subtext}
        </div>
      )}
    </div>
  );
};
