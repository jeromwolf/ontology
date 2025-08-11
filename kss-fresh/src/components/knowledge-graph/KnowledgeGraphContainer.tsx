'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { Triple, ViewMode, LayoutType, FilterOptions, GraphViewConfig } from './types';
import { ToolPanel } from './ToolPanel';
import { GraphViewer } from './GraphViewer';
import { SparqlPanel } from './SparqlPanel';

interface KnowledgeGraphContainerProps {
  initialTriples?: Triple[];
  onTriplesChange?: (triples: Triple[]) => void;
  labelType?: 'html' | 'sprite' | 'text' | 'billboard';
}

export const KnowledgeGraphContainer: React.FC<KnowledgeGraphContainerProps> = ({
  initialTriples = [],
  onTriplesChange,
  labelType = 'html'
}) => {
  const [triples, setTriples] = useState<Triple[]>(initialTriples);
  const [history, setHistory] = useState<Triple[][]>([initialTriples]);
  const [historyIndex, setHistoryIndex] = useState<number>(0);
  
  // Debug log
  useEffect(() => {
    console.log('KnowledgeGraphContainer - initialTriples:', initialTriples);
    console.log('KnowledgeGraphContainer - triples state:', triples);
  }, [initialTriples, triples]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [currentTool, setCurrentTool] = useState<'select' | 'move' | 'add-node' | 'add-edge' | 'delete'>('select');
  const [viewConfig, setViewConfig] = useState<GraphViewConfig>({
    viewMode: '3D',
    layout: 'force-directed',
    showLabels: true,
    nodeSize: 1,
    linkDistance: 50
  });
  const [filterOptions, setFilterOptions] = useState<FilterOptions>({
    showClasses: true,
    showProperties: true,
    showInstances: true,
    showLiterals: true
  });
  const [sparqlQuery, setSparqlQuery] = useState<string>(`SELECT ?subject ?predicate ?object
WHERE {
  ?subject ?predicate ?object
}`);

  const [pendingNode, setPendingNode] = useState<{ label: string; type: 'resource' | 'literal' | 'class' } | null>(null);
  const [pendingEdge, setPendingEdge] = useState<{ source: string; predicate: string } | null>(null);

  // Helper function to update triples with history
  const updateTriplesWithHistory = useCallback((newTriples: Triple[]) => {
    // Remove any future history if we're not at the end
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(newTriples);
    
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
    setTriples(newTriples);
    onTriplesChange?.(newTriples);
  }, [history, historyIndex, onTriplesChange]);

  // Undo function
  const handleUndo = useCallback(() => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      const previousTriples = history[newIndex];
      setHistoryIndex(newIndex);
      setTriples(previousTriples);
      onTriplesChange?.(previousTriples);
    }
  }, [history, historyIndex, onTriplesChange]);

  // Redo function
  const handleRedo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1;
      const nextTriples = history[newIndex];
      setHistoryIndex(newIndex);
      setTriples(nextTriples);
      onTriplesChange?.(nextTriples);
    }
  }, [history, historyIndex, onTriplesChange]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        handleUndo();
      } else if ((e.metaKey || e.ctrlKey) && (e.key === 'Z' || (e.key === 'z' && e.shiftKey))) {
        e.preventDefault();
        handleRedo();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleUndo, handleRedo]);

  const handleAddNode = useCallback((node: { label: string; type: 'resource' | 'literal' | 'class' }) => {
    // Set pending node to be placed by clicking on the graph
    setPendingNode(node);
    setCurrentTool('add-node');
  }, []);

  const handleAddEdge = useCallback((edge: { source: string; target: string; predicate: string }) => {
    // For edge mode, we'll handle this differently - user needs to select source node first
    if (!edge.source || !edge.target) {
      const predicate = prompt('엣지 이름 (predicate):');
      if (predicate) {
        setPendingEdge({ source: '', predicate });
        setCurrentTool('add-edge');
      }
    } else {
      const newTriple: Triple = {
        subject: edge.source,
        predicate: edge.predicate,
        object: edge.target,
        type: 'resource'
      };
      const updatedTriples = [...triples, newTriple];
      updateTriplesWithHistory(updatedTriples);
    }
  }, [triples, onTriplesChange]);

  const handleDeleteSelected = useCallback(() => {
    if (selectedNodeId) {
      const updatedTriples = triples.filter(
        t => t.subject !== selectedNodeId && t.object !== selectedNodeId
      );
      updateTriplesWithHistory(updatedTriples);
      setSelectedNodeId(null);
    }
  }, [selectedNodeId, triples, onTriplesChange]);

  const handleExecuteSparql = useCallback((query: string) => {
    // TODO: Implement SPARQL execution
    console.log('Executing SPARQL:', query);
    // For now, just filter triples based on simple pattern matching
    // In a real implementation, this would use a proper SPARQL engine
  }, []);

  return (
    <div className="flex h-full bg-gray-900">
      {/* Left Sidebar - Tool Panel */}
      <div className="w-64 bg-gray-800 border-r border-gray-700 flex-shrink-0">
        <ToolPanel
          viewConfig={viewConfig}
          filterOptions={filterOptions}
          currentTool={currentTool}
          onViewConfigChange={setViewConfig}
          onFilterChange={setFilterOptions}
          onToolChange={setCurrentTool}
          onAddNode={handleAddNode}
          onAddEdge={handleAddEdge}
          onDeleteSelected={handleDeleteSelected}
          selectedNodeId={selectedNodeId}
          selectedEdgeId={selectedEdgeId}
          onUndo={handleUndo}
          onRedo={handleRedo}
          canUndo={historyIndex > 0}
          canRedo={historyIndex < history.length - 1}
          triples={triples}
        />
      </div>

      {/* Center - Graph Viewer */}
      <div className="flex-1 relative">
        <GraphViewer
          triples={triples}
          viewConfig={viewConfig}
          filterOptions={filterOptions}
          selectedNodeId={selectedNodeId}
          selectedEdgeId={selectedEdgeId}
          onNodeSelect={setSelectedNodeId}
          onEdgeSelect={setSelectedEdgeId}
          currentTool={currentTool}
          pendingNode={pendingNode}
          pendingEdge={pendingEdge}
          labelType={labelType}
          onPlaceNode={(position) => {
            if (pendingNode) {
              const newTriple: Triple = {
                subject: pendingNode.label,
                predicate: 'rdf:type',
                object: pendingNode.type === 'class' ? 'rdfs:Class' : 'rdfs:Resource',
                type: 'resource'
                // Store position as metadata (would be used in a real RDF store)
                // position would be stored separately in a graph visualization layer
              };
              const updatedTriples = [...triples, newTriple];
              updateTriplesWithHistory(updatedTriples);
              setPendingNode(null);
              setCurrentTool('select');
            }
          }}
          onAddEdgeSelect={(nodeId) => {
            if (pendingEdge && !pendingEdge.source) {
              setPendingEdge({ ...pendingEdge, source: nodeId });
            } else if (pendingEdge && pendingEdge.source && pendingEdge.source !== nodeId) {
              const newTriple: Triple = {
                subject: pendingEdge.source,
                predicate: pendingEdge.predicate,
                object: nodeId,
                type: 'resource'
              };
              const updatedTriples = [...triples, newTriple];
              updateTriplesWithHistory(updatedTriples);
              setPendingEdge(null);
              setCurrentTool('select');
            }
          }}
        />
      </div>

      {/* Right Sidebar - SPARQL Panel */}
      <div className="w-96 bg-gray-800 border-l border-gray-700 flex-shrink-0">
        <SparqlPanel
          query={sparqlQuery}
          onQueryChange={setSparqlQuery}
          onExecute={handleExecuteSparql}
          triples={triples}
        />
      </div>
    </div>
  );
};

export default KnowledgeGraphContainer;