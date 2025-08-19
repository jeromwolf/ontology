'use client';

import { useState, useRef, ReactNode } from 'react';
import { Maximize2, Settings, Download, Camera, RefreshCcw } from 'lucide-react';
import { ChartData, Indicator, DrawingTool, OrderLevel } from './types';

export interface ProChartConfig {
  symbol: string;
  symbolName: string;
  timeframe: string;
  chartType: 'candle' | 'line' | 'bar' | 'heikinashi';
  indicators: Indicator[];
  theme?: 'dark' | 'light';
  height?: number;
  realtime?: boolean;
  showVolume?: boolean;
  showOrderBook?: boolean;
  showDrawingTools?: boolean;
  showIndicatorPanel?: boolean;
  onSymbolChange?: (symbol: string) => void;
  onTimeframeChange?: (timeframe: string) => void;
  onIndicatorChange?: (indicators: Indicator[]) => void;
  onDrawingSave?: (drawings: DrawingTool[]) => void;
  onScreenshot?: (canvas: HTMLCanvasElement) => void;
}

interface ProChartContainerProps {
  config: ProChartConfig;
  children: ReactNode;
  headerContent?: ReactNode;
  sidebarContent?: ReactNode;
  bottomContent?: ReactNode;
}

export default function ProChartContainer({
  config,
  children,
  headerContent,
  sidebarContent,
  bottomContent
}: ProChartContainerProps) {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // 전체화면 토글
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  // 스크린샷 캡처
  const captureScreenshot = async () => {
    if (!containerRef.current) return;
    
    try {
      const { default: html2canvas } = await import('html2canvas');
      const canvas = await html2canvas(containerRef.current, {
        backgroundColor: '#0a0a0a',
        scale: 2
      });
      
      // 다운로드
      const link = document.createElement('a');
      link.download = `${config.symbol}_${config.timeframe}_${Date.now()}.png`;
      link.href = canvas.toDataURL();
      link.click();
      
      // 콜백 실행
      config.onScreenshot?.(canvas);
    } catch (error) {
      console.error('Screenshot failed:', error);
    }
  };

  return (
    <div 
      ref={containerRef}
      className={`flex flex-col bg-gray-950 text-white ${
        isFullscreen ? 'fixed inset-0 z-50' : 'h-full'
      }`}
    >
      {/* 헤더 */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-900/50 border-b border-gray-800">
        <div className="flex items-center gap-4">
          {headerContent}
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={() => window.location.reload()}
            className="p-2 rounded-lg hover:bg-gray-800 transition-colors"
            title="새로고침"
          >
            <RefreshCcw className="w-4 h-4" />
          </button>
          
          <button
            onClick={captureScreenshot}
            className="p-2 rounded-lg hover:bg-gray-800 transition-colors"
            title="스크린샷"
          >
            <Camera className="w-4 h-4" />
          </button>
          
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 rounded-lg hover:bg-gray-800 transition-colors"
            title="설정"
          >
            <Settings className="w-4 h-4" />
          </button>
          
          <button
            onClick={toggleFullscreen}
            className="p-2 rounded-lg hover:bg-gray-800 transition-colors"
            title={isFullscreen ? "전체화면 종료" : "전체화면"}
          >
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* 메인 컨텐츠 영역 */}
      <div className="flex-1 flex overflow-hidden">
        {/* 사이드바 (옵션) */}
        {sidebarContent && (
          <div className="w-64 bg-gray-900/30 border-r border-gray-800 overflow-y-auto">
            {sidebarContent}
          </div>
        )}

        {/* 차트 영역 */}
        <div className="flex-1 flex flex-col">
          {/* 메인 차트 */}
          <div className="flex-1 relative">
            {children}
          </div>

          {/* 하단 컨텐츠 (보조 지표 등) */}
          {bottomContent && (
            <div className="h-48 border-t border-gray-800">
              {bottomContent}
            </div>
          )}
        </div>
      </div>

      {/* 설정 패널 */}
      {showSettings && (
        <div className="absolute top-16 right-4 w-80 bg-gray-900 border border-gray-700 rounded-lg shadow-xl p-4 z-40">
          <h3 className="text-sm font-semibold mb-4">차트 설정</h3>
          
          <div className="space-y-4">
            <div>
              <label className="text-xs text-gray-400">테마</label>
              <select className="w-full mt-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm">
                <option value="dark">다크</option>
                <option value="light">라이트</option>
              </select>
            </div>
            
            <div>
              <label className="text-xs text-gray-400">차트 타입</label>
              <select className="w-full mt-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm">
                <option value="candle">캔들스틱</option>
                <option value="line">라인</option>
                <option value="bar">바</option>
                <option value="heikinashi">평균봉</option>
              </select>
            </div>
            
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" defaultChecked className="rounded border-gray-600" />
                <span className="text-sm">거래량 표시</span>
              </label>
              
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" defaultChecked className="rounded border-gray-600" />
                <span className="text-sm">그리드 표시</span>
              </label>
            </div>
          </div>
          
          <div className="mt-4 pt-4 border-t border-gray-700">
            <button 
              onClick={() => setShowSettings(false)}
              className="w-full py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition-colors"
            >
              적용
            </button>
          </div>
        </div>
      )}
    </div>
  );
}