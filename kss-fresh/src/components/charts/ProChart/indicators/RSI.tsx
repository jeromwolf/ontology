'use client';

import { useEffect, useRef } from 'react';
import { 
  createChart, 
  IChartApi, 
  ISeriesApi,
  LineData,
  Time,
  ColorType
} from 'lightweight-charts';

interface RSIProps {
  data: Array<{
    time: string;
    rsi: number;
  }>;
  height?: number;
  period?: number;
  overbought?: number;
  oversold?: number;
  onCrosshairMove?: (rsi: number | null) => void;
}

export default function RSI({
  data,
  height = 150,
  period = 14,
  overbought = 70,
  oversold = 30,
  onCrosshairMove
}: RSIProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const rsiSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: height,
      layout: {
        background: { type: ColorType.Solid, color: '#0a0a0a' },
        textColor: '#d1d5db',
      },
      grid: {
        vertLines: { color: '#1f2937' },
        horzLines: { color: '#1f2937' },
      },
      rightPriceScale: {
        borderColor: '#1f2937',
        autoScale: false,
        scaleMargins: {
          top: 0.1,
          bottom: 0.1,
        },
      },
      timeScale: {
        borderColor: '#1f2937',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    // RSI 라인
    const rsiSeries = chart.addLineSeries({
      color: '#8b5cf6',
      lineWidth: 2,
      title: `RSI(${period})`,
    });
    rsiSeriesRef.current = rsiSeries;

    // 과매수/과매도 영역
    const overboughtSeries = chart.addLineSeries({
      color: '#ef4444',
      lineWidth: 1,
      lineStyle: 2, // Dashed
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      priceLineVisible: false,
    });

    const oversoldSeries = chart.addLineSeries({
      color: '#10b981',
      lineWidth: 1,
      lineStyle: 2, // Dashed
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      priceLineVisible: false,
    });

    const middleSeries = chart.addLineSeries({
      color: '#6b7280',
      lineWidth: 1,
      lineStyle: 3, // Dotted
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      priceLineVisible: false,
    });

    // 가격 범위 설정
    chart.priceScale('right').applyOptions({
      scaleMargins: {
        top: 0.05,
        bottom: 0.05,
      },
    });

    // 크로스헤어 이동 이벤트
    chart.subscribeCrosshairMove((param) => {
      if (param.time && param.point && param.seriesData.has(rsiSeries)) {
        const data = param.seriesData.get(rsiSeries) as LineData;
        onCrosshairMove?.(data.value);
      } else {
        onCrosshairMove?.(null);
      }
    });

    // 반응형 리사이즈
    const handleResize = () => {
      if (chartContainerRef.current && chart) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [height, period, overbought, oversold, onCrosshairMove]);

  // 데이터 업데이트
  useEffect(() => {
    if (!rsiSeriesRef.current || !chartRef.current || !data.length) return;

    // RSI 데이터
    const rsiData: LineData[] = data.map(d => ({
      time: Math.floor(new Date(d.time).getTime() / 1000) as Time,
      value: d.rsi,
    }));
    rsiSeriesRef.current.setData(rsiData);

    // 고정 라인 데이터
    const times = rsiData.map(d => d.time);
    const overboughtData = times.map(time => ({ time, value: overbought }));
    const oversoldData = times.map(time => ({ time, value: oversold }));
    const middleData = times.map(time => ({ time, value: 50 }));

    // 시간 범위 맞추기
    chartRef.current.timeScale().fitContent();
  }, [data, overbought, oversold]);

  return (
    <div className="relative h-full">
      <div className="absolute top-2 left-2 z-10">
        <div className="text-xs text-gray-400">RSI({period})</div>
        <div className="flex items-center gap-4 mt-1">
          <div className="flex items-center gap-1">
            <div className="w-3 h-px bg-red-500"></div>
            <span className="text-xs text-gray-400">과매수 {overbought}</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-px bg-green-500"></div>
            <span className="text-xs text-gray-400">과매도 {oversold}</span>
          </div>
        </div>
      </div>
      <div ref={chartContainerRef} className="w-full h-full" />
    </div>
  );
}