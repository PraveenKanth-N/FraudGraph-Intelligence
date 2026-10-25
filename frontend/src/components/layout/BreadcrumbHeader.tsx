import React from 'react';
import clsx from 'clsx';

export interface BreadcrumbHeaderProps {
  title: string;
  subtitle?: string;
  category?: string;
  badges?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}

export const BreadcrumbHeader: React.FC<BreadcrumbHeaderProps> = ({
  title,
  subtitle,
  category,
  badges,
  actions,
  className,
}) => {
  return (
    <div
      className={clsx(
        'border-b border-[var(--border-hairline)] bg-[var(--surface-obsidian)] px-6 py-4 flex flex-col md:flex-row md:items-center justify-between gap-4',
        className
      )}
    >
      <div className="flex flex-col gap-1 max-w-4xl">
        {category && (
          <div className="font-mono text-[10px] text-[var(--primary)] uppercase tracking-wider font-semibold">
            {category}
          </div>
        )}
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="font-serif text-2xl md:text-3xl font-normal text-[var(--tertiary)] leading-tight tracking-tight">
            {title}
          </h1>
          {badges && <div className="flex items-center gap-2">{badges}</div>}
        </div>
        {subtitle && (
          <p className="text-xs text-[var(--text-muted)] font-sans max-w-2xl leading-relaxed mt-0.5">
            {subtitle}
          </p>
        )}
      </div>

      {actions && (
        <div className="flex items-center gap-2 shrink-0 self-start md:self-auto">
          {actions}
        </div>
      )}
    </div>
  );
};
