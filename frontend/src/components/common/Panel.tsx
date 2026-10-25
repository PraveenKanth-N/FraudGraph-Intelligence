import React from 'react';
import clsx from 'clsx';

export interface PanelProps {
  title?: string;
  subtitle?: string;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
  noPadding?: boolean;
}

export const Panel: React.FC<PanelProps> = ({
  title,
  subtitle,
  badge,
  actions,
  children,
  className,
  bodyClassName,
  noPadding = false,
}) => {
  return (
    <div
      className={clsx(
        'bg-[var(--surface-graphite)] border border-[var(--border-hairline)] flex flex-col',
        className
      )}
    >
      {(title || actions || badge) && (
        <div className="h-11 px-4 border-b border-[var(--border-hairline)] flex items-center justify-between bg-[var(--surface-graphite)]">
          <div className="flex items-center gap-3">
            {title && (
              <span className="font-mono text-xs tracking-wider uppercase text-[var(--text-primary)] font-medium">
                {title}
              </span>
            )}
            {subtitle && (
              <span className="text-xs text-[var(--text-muted)] font-mono border-l border-[var(--border-hairline)] pl-3">
                {subtitle}
              </span>
            )}
            {badge && <div>{badge}</div>}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </div>
      )}
      <div className={clsx(!noPadding && 'p-4', 'flex-1', bodyClassName)}>
        {children}
      </div>
    </div>
  );
};
