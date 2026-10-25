import React from 'react';
import clsx from 'clsx';
import { TableDensity } from '../../types/navigation';

export interface Column<T> {
  key: string;
  header: string;
  render?: (row: T) => React.ReactNode;
  align?: 'left' | 'center' | 'right';
  width?: string;
  className?: string;
}

export interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  density?: TableDensity;
  keyExtractor: (row: T) => string | number;
  onRowClick?: (row: T) => void;
  isLoading?: boolean;
  emptyMessage?: string;
  className?: string;
}

export function DataTable<T>({
  columns,
  data,
  density = 'standard',
  keyExtractor,
  onRowClick,
  isLoading = false,
  emptyMessage = 'NO RECORDS REGISTERED IN ACTIVE REPOSITORY PARTITION',
  className,
}: DataTableProps<T>) {
  const densityStyles = {
    compact: 'py-1 px-3 text-xs h-7',
    standard: 'py-2 px-3 text-xs h-9',
    spacious: 'py-3 px-3 text-sm h-11',
  };

  const alignStyles = {
    left: 'text-left',
    center: 'text-center',
    right: 'text-right',
  };

  return (
    <div className={clsx('w-full overflow-x-auto border border-[var(--border-hairline)] bg-[var(--surface-graphite)]', className)}>
      <table className="w-full border-collapse">
        <thead>
          <tr className="border-b border-[var(--border-hairline)] bg-[var(--surface-raised)]">
            {columns.map((col) => (
              <th
                key={col.key}
                className={clsx(
                  'font-mono text-[11px] uppercase tracking-wider text-[var(--text-muted)] font-medium px-3 py-2 border-r border-[var(--border-hairline)] last:border-r-0',
                  alignStyles[col.align || 'left'],
                  col.className
                )}
                style={{ width: col.width }}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {isLoading ? (
            <tr>
              <td
                colSpan={columns.length}
                className="py-10 text-center font-mono text-xs text-[var(--text-muted)]"
              >
                <div className="flex items-center justify-center gap-2">
                  <div className="w-2 h-2 bg-[var(--primary)] animate-ping" />
                  <span>SYNCHRONIZING TABULAR ENCLAVE...</span>
                </div>
              </td>
            </tr>
          ) : data.length === 0 ? (
            <tr>
              <td
                colSpan={columns.length}
                className="py-10 text-center font-mono text-xs text-[var(--text-muted)]"
              >
                {emptyMessage}
              </td>
            </tr>
          ) : (
            data.map((row) => (
              <tr
                key={keyExtractor(row)}
                onClick={() => onRowClick?.(row)}
                className={clsx(
                  'border-b border-[var(--border-hairline)] hover:bg-[rgba(0,242,254,0.02)] transition-none',
                  onRowClick && 'cursor-pointer hover:border-l-2 hover:border-l-[var(--primary)]'
                )}
              >
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={clsx(
                      'border-r border-[var(--border-hairline)] last:border-r-0 text-[var(--text-primary)]',
                      densityStyles[density],
                      alignStyles[col.align || 'left'],
                      col.className
                    )}
                  >
                    {col.render ? col.render(row) : (row as Record<string, unknown>)[col.key] as React.ReactNode}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
