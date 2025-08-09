'use client';

import React from 'react';
import dynamic from 'next/dynamic';
import { Triple, ViewMode, LayoutType, FilterOptions, GraphViewConfig } from './types';

// Dynamically import the 3D component to avoid SSR issues
const Graph3D = dynamic(
  () => import('@/components/3d-graph/Graph3D').then(mod => ({ default: mod.Graph3D })),
  { ssr: false }
);

// Import 2D graph component (we'll create this)
const Graph2D = dynamic(
  () => import('./Graph2D').then(mod => ({ default: mod.Graph2D })),
  { ssr: false }
);

interface GraphViewerProps {
  triples: Triple[];
  viewConfig: GraphViewConfig;
  filterOptions: FilterOptions;
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  onNodeSelect: (nodeId: string | null) => void;
  onEdgeSelect: (edgeId: string | null) => void;
  currentTool?: 'select' | 'move' | 'add-node' | 'add-edge' | 'delete';
  pendingNode?: { label: string; type: 'resource' | 'literal' | 'class' } | null;
  pendingEdge?: { source: string; predicate: string } | null;
  onPlaceNode?: (position: { x: number; y: number }) => void;
  onAddEdgeSelect?: (nodeId: string) => void;
}

export const GraphViewer: React.FC<GraphViewerProps> = ({
  triples,
  viewConfig,
  filterOptions,
  selectedNodeId,
  selectedEdgeId,
  onNodeSelect,
  onEdgeSelect,
  currentTool,
  pendingNode,
  pendingEdge,
  onPlaceNode,
  onAddEdgeSelect
}) => {
  // Filter triples based on filter options
  const filteredTriples = triples.filter(triple => {
    // In a real implementation, you would check the actual type of each node
    // For now, we'll include all triples
    return true;
  });

  return (
    <div className="w-full h-full relative bg-gray-900">
      {viewConfig.viewMode === '3D' ? (
        <Graph3D
          triples={filteredTriples}
          selectedNode={selectedNodeId}
          onNodeSelect={onNodeSelect}
        />
      ) : (
        <Graph2D
          triples={filteredTriples}
          layout={viewConfig.layout}
          selectedNodeId={selectedNodeId}
          selectedEdgeId={selectedEdgeId}
          onNodeSelect={onNodeSelect}
          onEdgeSelect={onEdgeSelect}
          nodeSize={viewConfig.nodeSize}
          linkDistance={viewConfig.linkDistance}
          showLabels={viewConfig.showLabels}
          currentTool={currentTool}
          pendingNode={pendingNode}
          pendingEdge={pendingEdge}
          onPlaceNode={onPlaceNode}
          onAddEdgeSelect={onAddEdgeSelect}
        />
      )}

      {/* Info Overlay */}
      <div className="absolute top-4 right-4 bg-gray-800/90 p-3 rounded-lg text-sm text-gray-300">
        <div className="flex items-center gap-2 mb-2">
          <div className={`w-2 h-2 rounded-full ${viewConfig.viewMode === '3D' ? 'bg-green-400' : 'bg-blue-400'} animate-pulse`}></div>
          <span>{viewConfig.viewMode} Mode Active</span>
        </div>
        <div className="text-xs text-gray-400">
          <p>Nodes: {new Set([...filteredTriples.map(t => t.subject), ...filteredTriples.map(t => t.object)]).size}</p>
          <p>Edges: {filteredTriples.length}</p>
          <p>Layout: {viewConfig.layout}</p>
        </div>
      </div>
    </div>
  );
};