import React from 'react';
import clsx from 'clsx';

export type ButtonVariant = 'primary' | 'secondary' | 'terminal' | 'danger' | 'ghost';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface CyberButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  icon?: React.ReactNode;
  iconRight?: React.ReactNode;
  children: React.ReactNode;
}

export const CyberButton: React.FC<CyberButtonProps> = ({
  variant = 'primary',
  size = 'md',
  icon,
  iconRight,
  children,
  className,
  disabled,
  ...props
}) => {
  const baseStyles =
    'inline-flex items-center justify-center font-medium transition-none select-none text-left';

  const sizeStyles = {
    sm: 'text-xs px-2.5 py-1 gap-1.5',
    md: 'text-xs px-3.5 py-1.5 gap-2',
    lg: 'text-sm px-4 py-2.5 gap-2.5',
  };

  const variantStyles = {
    primary:
      'bg-[var(--primary)] text-[var(--on-primary)] hover:bg-[var(--primary-hover)] active:scale-[0.99] font-semibold border-none',
    secondary:
      'bg-transparent text-[var(--tertiary)] border border-[var(--border-hairline)] hover:border-[var(--tertiary)] hover:bg-[rgba(255,255,255,0.03)]',
    terminal:
      'bg-transparent text-[var(--text-muted)] border border-transparent font-mono tracking-wider hover:text-[var(--primary)] hover:border-[var(--border-hairline)]',
    danger:
      'bg-[rgba(255,59,48,0.1)] text-[var(--alert-critical)] border border-[var(--alert-critical)] hover:bg-[rgba(255,59,48,0.2)]',
    ghost:
      'bg-transparent text-[var(--text-primary)] hover:bg-[var(--surface-raised)] border border-transparent',
  };

  return (
    <button
      className={clsx(
        baseStyles,
        sizeStyles[size],
        variantStyles[variant],
        disabled && 'opacity-40 cursor-not-allowed pointer-events-none',
        className
      )}
      disabled={disabled}
      {...props}
    >
      {icon && <span className="inline-flex shrink-0">{icon}</span>}
      <span>{children}</span>
      {iconRight && <span className="inline-flex shrink-0">{iconRight}</span>}
    </button>
  );
};
