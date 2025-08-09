export interface Triple {
  subject: string;
  predicate: string;
  object: string;
  type?: 'resource' | 'literal';
}

export interface Node {
  id: string;
  label: string;
  type: 'resource' | 'literal' | 'class';
  position?: { x: number; y: number; z?: number };
  color?: string;
}

export interface Edge {
  id: string;
  source: string;
  target: string;
  label: string;
}

export type ViewMode = '2D' | '3D';
export type LayoutType = 'force-directed' | 'hierarchical' | 'circular' | 'grid';

export interface GraphViewConfig {
  viewMode: ViewMode;
  layout: LayoutType;
  showLabels: boolean;
  nodeSize: number;
  linkDistance: number;
}

export interface FilterOptions {
  showClasses: boolean;
  showProperties: boolean;
  showInstances: boolean;
  showLiterals: boolean;
}