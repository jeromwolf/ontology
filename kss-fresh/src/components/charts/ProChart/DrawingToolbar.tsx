'use client';

import { useState } from 'react';
import { 
  TrendingUp, 
  Minus, 
  BarChart3, 
  Square, 
  Circle, 
  Type, 
  Trash2,
  Move,
  Lock,
  Unlock,
  Eye,
  EyeOff,
  Palette,
  MoreVertical
} from 'lucide-react';
import { DrawingTool } from './types';

interface DrawingToolbarProps {
  selectedTool: string | null;
  onToolSelect: (tool: string | null) => void;
  drawings: DrawingTool[];
  onDrawingUpdate: (drawings: DrawingTool[]) => void;
  magnetMode: boolean;
  onMagnetToggle: (enabled: boolean) => void;
  showDrawings: boolean;
  onShowDrawingsToggle: (show: boolean) => void;
}

const drawingTools = [
  { id: 'trendline', icon: TrendingUp, name: '추세선', hotkey: 'T' },
  { id: 'horizontal', icon: Minus, name: '수평선', hotkey: 'H' },
  { id: 'vertical', icon: Minus, name: '수직선', hotkey: 'V', rotate: true },
  { id: 'fibonacci', icon: BarChart3, name: '피보나치', hotkey: 'F' },
  { id: 'channel', icon: TrendingUp, name: '채널', hotkey: 'C' },
  { id: 'rectangle', icon: Square, name: '사각형', hotkey: 'R' },
  { id: 'ellipse', icon: Circle, name: '타원', hotkey: 'E' },
  { id: 'text', icon: Type, name: '텍스트', hotkey: 'X' }
];

const colors = [
  '#ef4444', // red
  '#f59e0b', // amber
  '#10b981', // emerald
  '#3b82f6', // blue
  '#8b5cf6', // violet
  '#ec4899', // pink
  '#6b7280', // gray
  '#ffffff', // white
];

export default function DrawingToolbar({
  selectedTool,
  onToolSelect,
  drawings,
  onDrawingUpdate,
  magnetMode,
  onMagnetToggle,
  showDrawings,
  onShowDrawingsToggle
}: DrawingToolbarProps) {
  const [showColorPicker, setShowColorPicker] = useState(false);
  const [selectedColor, setSelectedColor] = useState('#3b82f6');
  const [lineWidth, setLineWidth] = useState(2);

  // 모든 그리기 삭제
  const clearAllDrawings = () => {
    if (confirm('모든 그리기를 삭제하시겠습니까?')) {
      onDrawingUpdate([]);
    }
  };

  // 선택된 그리기 삭제
  const deleteSelectedDrawings = () => {
    const selectedDrawings = drawings.filter(d => d.locked);
    if (selectedDrawings.length > 0) {
      onDrawingUpdate(drawings.filter(d => !d.locked));
    }
  };

  return (
    <div className="flex items-center justify-between p-2 bg-gray-900/50 border-b border-gray-700">
      {/* 그리기 도구 */}
      <div className="flex items-center gap-1">
        {drawingTools.map(tool => (
          <div key={tool.id} className="relative group">
            <button
              onClick={() => onToolSelect(selectedTool === tool.id ? null : tool.id)}
              className={`p-2 rounded-lg transition-colors ${
                selectedTool === tool.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 hover:bg-gray-700'
              }`}
              title={`${tool.name} (${tool.hotkey})`}
            >
              <tool.icon className={`w-4 h-4 ${tool.rotate ? 'rotate-90' : ''}`} />
            </button>
            <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-800 text-xs rounded opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap">
              {tool.name} ({tool.hotkey})
            </div>
          </div>
        ))}
        
        <div className="w-px h-6 bg-gray-700 mx-1" />
        
        {/* 색상 선택 */}
        <div className="relative">
          <button
            onClick={() => setShowColorPicker(!showColorPicker)}
            className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors"
            title="색상"
          >
            <div className="w-4 h-4 rounded" style={{ backgroundColor: selectedColor }} />
          </button>
          
          {showColorPicker && (
            <div className="absolute top-full left-0 mt-2 p-2 bg-gray-800 rounded-lg shadow-xl z-10">
              <div className="grid grid-cols-4 gap-1">
                {colors.map(color => (
                  <button
                    key={color}
                    onClick={() => {
                      setSelectedColor(color);
                      setShowColorPicker(false);
                    }}
                    className={`w-6 h-6 rounded ${
                      selectedColor === color ? 'ring-2 ring-blue-500' : ''
                    }`}
                    style={{ backgroundColor: color }}
                  />
                ))}
              </div>
              
              <div className="mt-2 pt-2 border-t border-gray-700">
                <label className="text-xs text-gray-400">선 굵기</label>
                <input
                  type="range"
                  min="1"
                  max="5"
                  value={lineWidth}
                  onChange={(e) => setLineWidth(Number(e.target.value))}
                  className="w-full mt-1"
                />
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* 도구 옵션 */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => onMagnetToggle(!magnetMode)}
          className={`p-2 rounded-lg transition-colors ${
            magnetMode ? 'bg-blue-600 text-white' : 'bg-gray-800 hover:bg-gray-700'
          }`}
          title="자석 모드"
        >
          <Move className="w-4 h-4" />
        </button>
        
        <button
          onClick={() => onShowDrawingsToggle(!showDrawings)}
          className={`p-2 rounded-lg transition-colors ${
            showDrawings ? 'bg-gray-800 hover:bg-gray-700' : 'bg-gray-700'
          }`}
          title={showDrawings ? '그리기 숨기기' : '그리기 보이기'}
        >
          {showDrawings ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
        </button>
        
        <div className="w-px h-6 bg-gray-700 mx-1" />
        
        <button
          onClick={deleteSelectedDrawings}
          className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors"
          title="선택 삭제"
        >
          <Trash2 className="w-4 h-4" />
        </button>
        
        <button
          onClick={clearAllDrawings}
          className="p-2 rounded-lg bg-gray-800 hover:bg-red-600 transition-colors"
          title="모두 삭제"
        >
          <Trash2 className="w-4 h-4" />
        </button>
        
        <div className="relative group">
          <button className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors">
            <MoreVertical className="w-4 h-4" />
          </button>
          
          <div className="absolute top-full right-0 mt-2 w-48 bg-gray-800 rounded-lg shadow-xl opacity-0 group-hover:opacity-100 pointer-events-none group-hover:pointer-events-auto transition-opacity">
            <div className="p-2 space-y-1">
              <button className="w-full px-3 py-2 text-sm text-left hover:bg-gray-700 rounded">
                그리기 저장
              </button>
              <button className="w-full px-3 py-2 text-sm text-left hover:bg-gray-700 rounded">
                그리기 불러오기
              </button>
              <div className="border-t border-gray-700 my-1"></div>
              <button className="w-full px-3 py-2 text-sm text-left hover:bg-gray-700 rounded">
                템플릿 저장
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}