'use client';

import React, { useState } from 'react';
import { 
  Plus, 
  Link, 
  Trash2, 
  Move, 
  Eye,
  EyeOff,
  Box,
  Circle,
  Grid3X3,
  GitBranch,
  Undo,
  Redo
} from 'lucide-react';
import { ViewMode, LayoutType, FilterOptions, GraphViewConfig, Triple } from './types';

interface ToolPanelProps {
  viewConfig: GraphViewConfig;
  filterOptions: FilterOptions;
  currentTool: 'select' | 'move' | 'add-node' | 'add-edge' | 'delete';
  onViewConfigChange: (config: GraphViewConfig) => void;
  onFilterChange: (options: FilterOptions) => void;
  onToolChange: (tool: 'select' | 'move' | 'add-node' | 'add-edge' | 'delete') => void;
  onAddNode: (node: { label: string; type: 'resource' | 'literal' | 'class' }) => void;
  onAddEdge: (edge: { source: string; target: string; predicate: string }) => void;
  onDeleteSelected: () => void;
  selectedNodeId: string | null;
  triples?: Triple[];
  selectedEdgeId: string | null;
  onUndo?: () => void;
  onRedo?: () => void;
  canUndo?: boolean;
  canRedo?: boolean;
}

export const ToolPanel: React.FC<ToolPanelProps> = ({
  viewConfig,
  filterOptions,
  currentTool,
  onViewConfigChange,
  onFilterChange,
  onToolChange,
  onAddNode,
  onAddEdge,
  onDeleteSelected,
  selectedNodeId,
  selectedEdgeId,
  onUndo,
  onRedo,
  canUndo = false,
  canRedo = false,
  triples = []
}) => {
  const [activeSection, setActiveSection] = useState<string>('도구');

  return (
    <div className="h-full flex flex-col bg-gray-800 text-gray-200">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-semibold">지식그래프 시뮬레이터</h2>
        <p className="text-xs text-gray-400 mt-1">전문적인 지식그래프 편집, 시각화, 분석 도구</p>
      </div>

      {/* View Mode Toggle */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-medium">노드</span>
          <span className="text-sm text-gray-400">
            {new Set([
              ...triples.map(t => t.subject), 
              ...triples.filter(t => t.type !== 'literal').map(t => t.object)
            ]).size}
          </span>
        </div>
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-medium">엣지</span>
          <span className="text-sm text-gray-400">{triples.length}</span>
        </div>
        
        {/* Undo/Redo buttons */}
        <div className="flex gap-2 mt-3">
          <button
            onClick={onUndo}
            disabled={!canUndo}
            className="flex-1 flex items-center justify-center gap-1 py-1.5 px-2 text-xs bg-gray-700 hover:bg-gray-600 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="실행 취소 (Cmd/Ctrl+Z)"
          >
            <Undo className="w-3.5 h-3.5" />
            <span>실행 취소</span>
          </button>
          <button
            onClick={onRedo}
            disabled={!canRedo}
            className="flex-1 flex items-center justify-center gap-1 py-1.5 px-2 text-xs bg-gray-700 hover:bg-gray-600 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="다시 실행 (Cmd/Ctrl+Shift+Z)"
          >
            <Redo className="w-3.5 h-3.5" />
            <span>다시 실행</span>
          </button>
        </div>
      </div>

      {/* Tool Buttons */}
      <div className="p-4 border-b border-gray-700">
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => {
              const label = prompt('노드 이름:');
              if (label) {
                onAddNode({ label, type: 'resource' });
              }
            }}
            className="flex flex-col items-center justify-center p-3 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
          >
            <Plus className="w-5 h-5 mb-1" />
            <span className="text-xs">노드 추가</span>
          </button>
          <button
            onClick={() => {
              const source = prompt('시작 노드:');
              const target = prompt('도착 노드:');
              const predicate = prompt('관계:');
              if (source && target && predicate) {
                onAddEdge({ source, target, predicate });
              }
            }}
            className="flex flex-col items-center justify-center p-3 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
          >
            <Link className="w-5 h-5 mb-1" />
            <span className="text-xs">엣지 추가</span>
          </button>
          <button
            onClick={onDeleteSelected}
            disabled={!selectedNodeId && !selectedEdgeId}
            className="flex flex-col items-center justify-center p-3 bg-gray-700 hover:bg-gray-600 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Trash2 className="w-5 h-5 mb-1" />
            <span className="text-xs">삭제</span>
          </button>
          <button
            onClick={() => onToolChange('move')}
            className={`flex flex-col items-center justify-center p-3 rounded transition-colors ${
              currentTool === 'move'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 hover:bg-gray-600'
            }`}
          >
            <Move className="w-5 h-5 mb-1" />
            <span className="text-xs">이동</span>
          </button>
        </div>
      </div>

      {/* View Mode */}
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-sm font-medium mb-3">뷰 모드</h3>
        <div className="flex gap-2">
          <button
            onClick={() => onViewConfigChange({ ...viewConfig, viewMode: '2D' })}
            className={`flex-1 py-2 px-3 rounded text-sm transition-colors ${
              viewConfig.viewMode === '2D'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 hover:bg-gray-600'
            }`}
          >
            2D 뷰
          </button>
          <button
            onClick={() => onViewConfigChange({ ...viewConfig, viewMode: '3D' })}
            className={`flex-1 py-2 px-3 rounded text-sm transition-colors ${
              viewConfig.viewMode === '3D'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 hover:bg-gray-600'
            }`}
          >
            3D 뷰
          </button>
        </div>
      </div>

      {/* Layout Options */}
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-sm font-medium mb-3">레이아웃</h3>
        <div className="space-y-2">
          {[
            { id: 'force-directed', label: 'Force-directed', icon: GitBranch },
            { id: 'hierarchical', label: 'Hierarchical', icon: GitBranch },
            { id: 'circular', label: 'Circular', icon: Circle },
            { id: 'grid', label: 'Grid', icon: Grid3X3 }
          ].map(layout => {
            const Icon = layout.icon;
            return (
              <button
                key={layout.id}
                onClick={() => onViewConfigChange({ ...viewConfig, layout: layout.id as LayoutType })}
                className={`w-full flex items-center gap-2 px-3 py-2 rounded text-sm transition-colors ${
                  viewConfig.layout === layout.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 hover:bg-gray-600'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{layout.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Filters */}
      <div className="flex-1 p-4 overflow-y-auto">
        <h3 className="text-sm font-medium mb-3">필터</h3>
        <div className="space-y-2">
          {[
            { id: 'showClasses', label: '클래스 표시' },
            { id: 'showProperties', label: '속성 표시' },
            { id: 'showInstances', label: '개체 표시' },
            { id: 'showLiterals', label: '리터럴 표시' }
          ].map(filter => (
            <label key={filter.id} className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={filterOptions[filter.id as keyof FilterOptions]}
                onChange={(e) => onFilterChange({
                  ...filterOptions,
                  [filter.id]: e.target.checked
                })}
                className="rounded border-gray-600 bg-gray-700 text-blue-600"
              />
              <span className="text-sm">{filter.label}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-700">
        <p className="text-xs text-gray-500">수정됨 작업이 있습니다</p>
      </div>
    </div>
  );
};