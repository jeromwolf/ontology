'use client';

import { useEffect, useRef } from 'react';
import { ChartData } from './types';

interface TradingViewChartProps {
  data: ChartData[];
  height?: number;
  showVolume?: boolean;
  indicators?: {
    ma5?: boolean;
    ma20?: boolean;
    ma60?: boolean;
    bollinger?: boolean;
  };
  onCrosshairMove?: (price: number | null, time: string | null) => void;
}

export default function TradingViewChart({
  data,
  height = 400,
  showVolume = true,
  indicators = {},
  onCrosshairMove
}: TradingViewChartProps) {
  const chartRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!chartRef.current || !data || data.length === 0) return;

    const canvas = chartRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Canvas 크기 설정
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * devicePixelRatio;
    canvas.height = rect.height * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);

    // 차트 영역 계산
    const padding = 40;
    const chartWidth = rect.width - padding * 2;
    const chartHeight = rect.height - padding * 2;

    // 데이터 범위 계산
    if (data.length === 0) return;

    const prices = data.flatMap(d => [d.high, d.low, d.open, d.close]);
    const minPrice = Math.min(...prices) * 0.99;
    const maxPrice = Math.max(...prices) * 1.01;
    const priceRange = maxPrice - minPrice;

    // 캔들 너비 계산
    const candleWidth = Math.max(2, chartWidth / data.length * 0.8);
    const candleSpacing = chartWidth / data.length;

    // 배경 그리기
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, rect.width, rect.height);

    // 격자 그리기
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    
    // 수평선 (가격 레벨)
    for (let i = 0; i <= 5; i++) {
      const y = padding + (chartHeight * i / 5);
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(rect.width - padding, y);
      ctx.stroke();
      
      // 가격 레이블
      const price = maxPrice - (priceRange * i / 5);
      ctx.fillStyle = '#64748b';
      ctx.font = '12px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(price.toLocaleString(), padding - 5, y + 4);
    }

    // 수직선 (시간)
    const timeStep = Math.max(1, Math.floor(data.length / 6));
    for (let i = 0; i < data.length; i += timeStep) {
      const x = padding + (i * candleSpacing);
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, rect.height - padding);
      ctx.stroke();
    }

    // 캔들스틱 그리기
    data.forEach((candle, index) => {
      const x = padding + (index * candleSpacing);
      
      // 가격을 Y 좌표로 변환
      const openY = padding + ((maxPrice - candle.open) / priceRange) * chartHeight;
      const closeY = padding + ((maxPrice - candle.close) / priceRange) * chartHeight;
      const highY = padding + ((maxPrice - candle.high) / priceRange) * chartHeight;
      const lowY = padding + ((maxPrice - candle.low) / priceRange) * chartHeight;

      const isGreen = candle.close > candle.open;
      const bodyTop = Math.min(openY, closeY);
      const bodyBottom = Math.max(openY, closeY);
      const bodyHeight = Math.abs(closeY - openY);

      // 심지 그리기
      ctx.strokeStyle = isGreen ? '#10b981' : '#ef4444';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, highY);
      ctx.lineTo(x, lowY);
      ctx.stroke();

      // 캔들 몸체 그리기
      if (bodyHeight < 1) {
        // 도지(시가 = 종가)
        ctx.strokeStyle = '#64748b';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x - candleWidth/2, openY);
        ctx.lineTo(x + candleWidth/2, openY);
        ctx.stroke();
      } else {
        // 일반 캔들
        ctx.fillStyle = isGreen ? '#10b981' : '#ef4444';
        ctx.fillRect(x - candleWidth/2, bodyTop, candleWidth, bodyHeight);
        
        ctx.strokeStyle = isGreen ? '#059669' : '#dc2626';
        ctx.lineWidth = 1;
        ctx.strokeRect(x - candleWidth/2, bodyTop, candleWidth, bodyHeight);
      }
    });

    // 이동평균선 그리기
    if (indicators.ma5 && data.length > 5) {
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      for (let i = 4; i < data.length; i++) {
        const ma5 = data.slice(i-4, i+1).reduce((sum, d) => sum + d.close, 0) / 5;
        const x = padding + (i * candleSpacing);
        const y = padding + ((maxPrice - ma5) / priceRange) * chartHeight;
        
        if (i === 4) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }

    if (indicators.ma20 && data.length > 20) {
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      for (let i = 19; i < data.length; i++) {
        const ma20 = data.slice(i-19, i+1).reduce((sum, d) => sum + d.close, 0) / 20;
        const x = padding + (i * candleSpacing);
        const y = padding + ((maxPrice - ma20) / priceRange) * chartHeight;
        
        if (i === 19) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }

    // 범례
    ctx.fillStyle = '#f1f5f9';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'left';
    
    let legendY = 25;
    if (data.length > 0) {
      const lastCandle = data[data.length - 1];
      ctx.fillText(`현재가: ₩${lastCandle.close.toLocaleString()}`, 10, legendY);
      legendY += 20;
      
      const change = lastCandle.close - lastCandle.open;
      const changePercent = ((change / lastCandle.open) * 100).toFixed(2);
      const changeColor = change >= 0 ? '#10b981' : '#ef4444';
      
      ctx.fillStyle = changeColor;
      ctx.fillText(`${change >= 0 ? '+' : ''}${change.toLocaleString()} (${changePercent}%)`, 10, legendY);
    }

  }, [data, height, showVolume, indicators]);

  return (
    <div className="relative w-full bg-gray-950 rounded-lg overflow-hidden" style={{ height }}>
      <canvas
        ref={chartRef}
        className="w-full h-full cursor-crosshair"
        style={{ width: '100%', height: '100%' }}
        onMouseMove={(e) => {
          if (!onCrosshairMove || !data.length) return;
          
          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const y = e.clientY - rect.top;
          
          // 간단한 크로스헤어 계산
          const padding = 40;
          const chartWidth = rect.width - padding * 2;
          const chartHeight = rect.height - padding * 2;
          
          if (x >= padding && x <= rect.width - padding && y >= padding && y <= rect.height - padding) {
            const dataIndex = Math.floor((x - padding) / chartWidth * data.length);
            if (dataIndex >= 0 && dataIndex < data.length) {
              const prices = data.flatMap(d => [d.high, d.low, d.open, d.close]);
              const minPrice = Math.min(...prices) * 0.99;
              const maxPrice = Math.max(...prices) * 1.01;
              const priceRange = maxPrice - minPrice;
              
              const price = maxPrice - ((y - padding) / chartHeight) * priceRange;
              const time = data[dataIndex].time;
              
              onCrosshairMove(price, time);
            }
          } else {
            onCrosshairMove(null, null);
          }
        }}
        onMouseLeave={() => {
          if (onCrosshairMove) {
            onCrosshairMove(null, null);
          }
        }}
      />
      
      {/* 차트 제목 */}
      <div className="absolute top-2 left-2 text-white bg-gray-900/80 px-3 py-1 rounded text-sm">
        Professional Trading Chart
      </div>
      
      {/* 범례 */}
      <div className="absolute top-2 right-2 text-xs text-gray-300 bg-gray-900/80 px-3 py-2 rounded">
        <div className="flex items-center gap-4">
          {indicators.ma5 && (
            <div className="flex items-center gap-1">
              <div className="w-3 h-0.5 bg-blue-500"></div>
              <span>MA5</span>
            </div>
          )}
          {indicators.ma20 && (
            <div className="flex items-center gap-1">
              <div className="w-3 h-0.5 bg-yellow-500"></div>
              <span>MA20</span>
            </div>
          )}
        </div>
      </div>
      
      {/* 로딩 상태 */}
      {(!data || data.length === 0) && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center text-gray-400">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mx-auto mb-2"></div>
            <p className="text-sm">차트 데이터 로딩 중...</p>
          </div>
        </div>
      )}
    </div>
  );
}