'use client';

import { useState } from 'react';
import { Settings, TrendingUp, Download, Maximize2, RefreshCw } from 'lucide-react';

interface ChartControlsProps {
  onRefresh?: () => void;
  onDownload?: () => void;
  onFullscreen?: () => void;
  onSettingsChange?: (settings: any) => void;
}

export default function ChartControls({
  onRefresh,
  onDownload,
  onFullscreen,
  onSettingsChange
}: ChartControlsProps) {
  const [showSettings, setShowSettings] = useState(false);
  const [chartSettings, setChartSettings] = useState({
    showVolume: true,
    showMA: true,
    showGrid: true,
    theme: 'dark'
  });

  const handleSettingChange = (key: string, value: any) => {
    const newSettings = { ...chartSettings, [key]: value };
    setChartSettings(newSettings);
    onSettingsChange?.(newSettings);
  };

  return (
    <div className="h-full flex flex-col bg-gray-900/50">
      {/* 헤더 */}
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-sm font-semibold mb-2">차트 도구</h3>
        <div className="text-xs text-gray-400">차트 설정 및 도구</div>
      </div>
      
      {/* 도구 버튼들 */}
      <div className="p-4 space-y-3">
        <button
          onClick={onRefresh}
          className="w-full flex items-center gap-3 px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          <span className="text-sm">새로고침</span>
        </button>
        
        <button
          onClick={onDownload}
          className="w-full flex items-center gap-3 px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <Download className="w-4 h-4" />
          <span className="text-sm">차트 다운로드</span>
        </button>
        
        <button
          onClick={onFullscreen}
          className="w-full flex items-center gap-3 px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <Maximize2 className="w-4 h-4" />
          <span className="text-sm">전체화면</span>
        </button>
        
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="w-full flex items-center gap-3 px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <Settings className="w-4 h-4" />
          <span className="text-sm">차트 설정</span>
        </button>
      </div>
      
      {/* 설정 패널 */}
      {showSettings && (
        <div className="px-4 pb-4 space-y-3 border-t border-gray-700 pt-4">
          <label className="flex items-center justify-between">
            <span className="text-sm">거래량 표시</span>
            <input
              type="checkbox"
              checked={chartSettings.showVolume}
              onChange={(e) => handleSettingChange('showVolume', e.target.checked)}
              className="rounded"
            />
          </label>
          
          <label className="flex items-center justify-between">
            <span className="text-sm">이동평균선</span>
            <input
              type="checkbox"
              checked={chartSettings.showMA}
              onChange={(e) => handleSettingChange('showMA', e.target.checked)}
              className="rounded"
            />
          </label>
          
          <label className="flex items-center justify-between">
            <span className="text-sm">격자 표시</span>
            <input
              type="checkbox"
              checked={chartSettings.showGrid}
              onChange={(e) => handleSettingChange('showGrid', e.target.checked)}
              className="rounded"
            />
          </label>
        </div>
      )}
      
      {/* 범례 */}
      <div className="mt-auto p-4 border-t border-gray-700">
        <div className="space-y-2 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded"></div>
            <span className="text-gray-400">상승</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded"></div>
            <span className="text-gray-400">하락</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-500 rounded"></div>
            <span className="text-gray-400">MA5</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-orange-500 rounded"></div>
            <span className="text-gray-400">MA20</span>
          </div>
        </div>
      </div>
    </div>
  );
}