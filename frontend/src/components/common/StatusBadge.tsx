import React from 'react';
import clsx from 'clsx';

export type BadgeVariant = 'primary' | 'success' | 'critical' | 'warning' | 'muted';

export interface StatusBadgeProps {
  variant?: BadgeVariant;
  children: React.ReactNode;
  icon?: React.ReactNode;
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  variant = 'primary',
  children,
  icon,
  className,
}) => {
  const variantStyles = {
    primary:
      'bg-[rgba(0,242,254,0.08)] border-[var(--primary)] text-[var(--primary)]',
    success:
      'bg-[rgba(10,232,183,0.08)] border-[var(--secondary)] text-[var(--secondary)]',
    critical:
      'bg-[rgba(255,59,48,0.1)] border-[var(--alert-critical)] text-[var(--alert-critical)]',
    warning:
      'bg-[rgba(255,159,10,0.1)] border-[var(--alert-warning)] text-[var(--alert-warning)]',
    muted:
      'bg-[var(--surface-raised)] border-[var(--border-hairline)] text-[var(--text-muted)]',
  };

  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1.5 px-2 py-0.5 text-[10px] font-mono tracking-wider uppercase border',
        variantStyles[variant],
        className
      )}
    >
      {icon && <span className="inline-flex shrink-0">{icon}</span>}
      <span>{children}</span>
    </span>
  );
};
