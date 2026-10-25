import React, { useEffect } from 'react';
import { X } from 'lucide-react';
import clsx from 'clsx';

export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  actions?: React.ReactNode;
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl';
}

export const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  subtitle,
  children,
  actions,
  maxWidth = 'lg',
}) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'auto';
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const maxWidthStyles = {
    sm: 'max-w-md',
    md: 'max-w-lg',
    lg: 'max-w-2xl',
    xl: 'max-w-4xl',
    '2xl': 'max-w-5xl',
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[rgba(5,7,10,0.85)] backdrop-blur-md"
      onClick={onClose}
    >
      <div
        className={clsx(
          'w-full bg-[var(--surface-graphite)] border border-[var(--border-hairline)] shadow-2xl flex flex-col max-h-[90vh]',
          maxWidthStyles[maxWidth]
        )}
        style={{
          boxShadow: '0 0 0 1px var(--border-hairline), inset 0 1px 0 0 rgba(255, 255, 255, 0.05)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="h-12 px-5 border-b border-[var(--border-hairline)] flex items-center justify-between bg-[var(--surface-graphite)] shrink-0">
          <div className="flex items-center gap-3">
            <span className="font-mono text-xs font-semibold uppercase tracking-wider text-[var(--text-primary)]">
              {title}
            </span>
            {subtitle && (
              <span className="text-xs text-[var(--text-muted)] font-mono border-l border-[var(--border-hairline)] pl-3">
                {subtitle}
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            className="w-7 h-7 flex items-center justify-center text-[var(--text-muted)] hover:text-[var(--primary)] border border-transparent hover:border-[var(--border-hairline)]"
            aria-label="Close dialog"
          >
            <X size={16} />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto flex-1">{children}</div>

        {/* Modal Actions */}
        {actions && (
          <div className="px-5 py-3 border-t border-[var(--border-hairline)] bg-[var(--surface-raised)] flex items-center justify-end gap-3 shrink-0">
            {actions}
          </div>
        )}
      </div>
    </div>
  );
};
