'use client';

import { useState } from 'react';
import { Settings, Plus, X, ChevronDown, ChevronUp } from 'lucide-react';
import { Indicator } from './types';

interface IndicatorPanelProps {
  indicators: Indicator[];
  onIndicatorUpdate: (indicators: Indicator[]) => void;
  onClose?: () => void;
}

// 사용 가능한 지표 목록
const availableIndicators = [
  // 오버레이 지표
  { id: 'ma', name: '이동평균선 (MA)', type: 'overlay' as const, params: { period: 20 } },
  { id: 'ema', name: '지수이동평균 (EMA)', type: 'overlay' as const, params: { period: 20 } },
  { id: 'bollinger', name: '볼린저 밴드', type: 'overlay' as const, params: { period: 20, stdDev: 2 } },
  { id: 'sma', name: '단순이동평균 (SMA)', type: 'overlay' as const, params: { period: 20 } },
  { id: 'vwap', name: 'VWAP', type: 'overlay' as const, params: {} },
  { id: 'sar', name: 'Parabolic SAR', type: 'overlay' as const, params: { step: 0.02, max: 0.2 } },
  { id: 'ichimoku', name: '일목균형표', type: 'overlay' as const, params: { conversionPeriod: 9, basePeriod: 26, spanPeriod: 52 } },
  
  // 오실레이터 지표
  { id: 'rsi', name: 'RSI', type: 'oscillator' as const, params: { period: 14 } },
  { id: 'macd', name: 'MACD', type: 'oscillator' as const, params: { fast: 12, slow: 26, signal: 9 } },
  { id: 'stochastic', name: 'Stochastic', type: 'oscillator' as const, params: { k: 14, d: 3, smooth: 3 } },
  { id: 'cci', name: 'CCI', type: 'oscillator' as const, params: { period: 20 } },
  { id: 'williams', name: 'Williams %R', type: 'oscillator' as const, params: { period: 14 } },
  { id: 'momentum', name: 'Momentum', type: 'oscillator' as const, params: { period: 10 } },
  { id: 'roc', name: 'ROC', type: 'oscillator' as const, params: { period: 12 } },
  { id: 'atr', name: 'ATR', type: 'oscillator' as const, params: { period: 14 } },
  { id: 'adx', name: 'ADX', type: 'oscillator' as const, params: { period: 14 } },
  
  // 거래량 지표
  { id: 'volume', name: '거래량', type: 'volume' as const, params: {} },
  { id: 'obv', name: 'OBV', type: 'volume' as const, params: {} },
  { id: 'mfi', name: 'MFI', type: 'volume' as const, params: { period: 14 } },
  { id: 'ad', name: 'A/D Line', type: 'volume' as const, params: {} },
  { id: 'cmf', name: 'CMF', type: 'volume' as const, params: { period: 20 } },
];

const defaultColors = {
  overlay: ['#3b82f6', '#f59e0b', '#10b981', '#8b5cf6', '#ef4444'],
  oscillator: ['#8b5cf6', '#ec4899', '#06b6d4', '#f97316', '#84cc16'],
  volume: ['#6366f1', '#a855f7', '#14b8a6', '#f43f5e', '#eab308']
};

export default function IndicatorPanel({
  indicators,
  onIndicatorUpdate,
  onClose
}: IndicatorPanelProps) {
  const [expandedCategory, setExpandedCategory] = useState<string | null>('overlay');
  const [editingIndicator, setEditingIndicator] = useState<string | null>(null);

  // 지표 추가
  const addIndicator = (template: typeof availableIndicators[0]) => {
    const existingCount = indicators.filter(ind => ind.id.startsWith(template.id)).length;
    const newId = existingCount > 0 ? `${template.id}${existingCount + 1}` : template.id;
    const colorIndex = indicators.filter(ind => ind.type === template.type).length;
    const colors = defaultColors[template.type];
    
    const newIndicator: Indicator = {
      id: newId,
      name: template.name,
      type: template.type,
      enabled: true,
      params: { ...template.params },
      color: colors[colorIndex % colors.length]
    };
    
    onIndicatorUpdate([...indicators, newIndicator]);
  };

  // 지표 제거
  const removeIndicator = (id: string) => {
    onIndicatorUpdate(indicators.filter(ind => ind.id !== id));
  };

  // 지표 토글
  const toggleIndicator = (id: string) => {
    onIndicatorUpdate(
      indicators.map(ind => 
        ind.id === id ? { ...ind, enabled: !ind.enabled } : ind
      )
    );
  };

  // 지표 파라미터 업데이트
  const updateIndicatorParams = (id: string, params: any) => {
    onIndicatorUpdate(
      indicators.map(ind => 
        ind.id === id ? { ...ind, params } : ind
      )
    );
  };

  const categories = [
    { id: 'overlay', name: '오버레이 지표', indicators: availableIndicators.filter(i => i.type === 'overlay') },
    { id: 'oscillator', name: '오실레이터', indicators: availableIndicators.filter(i => i.type === 'oscillator') },
    { id: 'volume', name: '거래량 지표', indicators: availableIndicators.filter(i => i.type === 'volume') }
  ];

  const activeIndicators = indicators.filter(ind => ind.enabled);

  return (
    <div className="w-80 h-full bg-gray-900/50 border-r border-gray-700 flex flex-col">
      {/* 헤더 */}
      <div className="p-4 border-b border-gray-700 flex items-center justify-between">
        <h3 className="text-sm font-semibold">기술적 지표</h3>
        {onClose && (
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-800 rounded transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* 활성 지표 */}
      <div className="p-4 border-b border-gray-700">
        <div className="text-xs text-gray-400 mb-2">활성 지표 ({activeIndicators.length})</div>
        <div className="space-y-2">
          {indicators.map(indicator => (
            <div
              key={indicator.id}
              className="flex items-center justify-between p-2 bg-gray-800 rounded-lg"
            >
              <label className="flex items-center gap-2 cursor-pointer flex-1">
                <input
                  type="checkbox"
                  checked={indicator.enabled}
                  onChange={() => toggleIndicator(indicator.id)}
                  className="rounded border-gray-600"
                />
                <span className="text-sm">{indicator.name}</span>
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: indicator.color }}
                />
              </label>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setEditingIndicator(
                    editingIndicator === indicator.id ? null : indicator.id
                  )}
                  className="p-1 hover:bg-gray-700 rounded"
                >
                  <Settings className="w-3 h-3" />
                </button>
                <button
                  onClick={() => removeIndicator(indicator.id)}
                  className="p-1 hover:bg-gray-700 rounded text-red-400"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 지표 추가 */}
      <div className="flex-1 overflow-y-auto">
        {categories.map(category => (
          <div key={category.id} className="border-b border-gray-700">
            <button
              onClick={() => setExpandedCategory(
                expandedCategory === category.id ? null : category.id
              )}
              className="w-full p-4 flex items-center justify-between hover:bg-gray-800/50 transition-colors"
            >
              <span className="text-sm font-medium">{category.name}</span>
              {expandedCategory === category.id ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </button>
            
            {expandedCategory === category.id && (
              <div className="px-4 pb-4 space-y-1">
                {category.indicators.map(indicator => (
                  <button
                    key={indicator.id}
                    onClick={() => addIndicator(indicator)}
                    className="w-full px-3 py-2 text-sm text-left hover:bg-gray-800 rounded-lg transition-colors flex items-center gap-2"
                  >
                    <Plus className="w-3 h-3" />
                    {indicator.name}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* 하단 액션 */}
      <div className="p-4 border-t border-gray-700 space-y-2">
        <button className="w-full py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm transition-colors">
          지표 설정 저장
        </button>
        <button className="w-full py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm transition-colors">
          기본값으로 재설정
        </button>
      </div>
    </div>
  );
}