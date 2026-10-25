export type ViewId =
  | 'overview'
  | 'explorer'
  | 'transactions'
  | 'cuda-lab'
  | 'experiments'
  | 'evaluation'
  | 'datasets'
  | 'reports';

export interface NavItem {
  id: ViewId;
  label: string;
  path: string;
  badge?: string;
  description: string;
}

export type TableDensity = 'compact' | 'standard' | 'spacious';

export interface WorkstationSettings {
  tableDensity: TableDensity;
  telemetryStreamSimulated: boolean;
  highContrastGrid: boolean;
  activeDatasetId: string;
}
