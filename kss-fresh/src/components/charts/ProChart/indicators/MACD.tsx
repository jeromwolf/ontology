'use client';

import { useEffect, useRef } from 'react';
import { 
  createChart, 
  IChartApi, 
  ISeriesApi,
  LineData,
  HistogramData,
  Time,
  ColorType
} from 'lightweight-charts';

interface MACDProps {
  data: Array<{
    time: string;
    macd: number;
    signal: number;
    histogram: number;
  }>;
  height?: number;
  fastPeriod?: number;
  slowPeriod?: number;
  signalPeriod?: number;
  onCrosshairMove?: (values: { macd: number; signal: number; histogram: number } | null) => void;
}

export default function MACD({
  data,
  height = 150,
  fastPeriod = 12,
  slowPeriod = 26,
  signalPeriod = 9,
  onCrosshairMove
}: MACDProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const macdSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const signalSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const histogramSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

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

    // MACD 라인
    const macdSeries = chart.addLineSeries({
      color: '#3b82f6',
      lineWidth: 2,
      title: 'MACD',
    });
    macdSeriesRef.current = macdSeries;

    // Signal 라인
    const signalSeries = chart.addLineSeries({
      color: '#ef4444',
      lineWidth: 2,
      title: 'Signal',
    });
    signalSeriesRef.current = signalSeries;

    // Histogram
    const histogramSeries = chart.addHistogramSeries({
      color: '#10b981',
      priceFormat: {
        type: 'price',
        precision: 4,
        minMove: 0.0001,
      },
    });
    histogramSeriesRef.current = histogramSeries;

    // 0 라인
    const zeroLineSeries = chart.addLineSeries({
      color: '#6b7280',
      lineWidth: 1,
      lineStyle: 3, // Dotted
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      priceLineVisible: false,
    });

    // 크로스헤어 이동 이벤트
    chart.subscribeCrosshairMove((param) => {
      if (param.time && param.point) {
        const macdData = param.seriesData.get(macdSeries) as LineData;
        const signalData = param.seriesData.get(signalSeries) as LineData;
        const histogramData = param.seriesData.get(histogramSeries) as HistogramData;
        
        if (macdData && signalData && histogramData) {
          onCrosshairMove?.({
            macd: macdData.value,
            signal: signalData.value,
            histogram: histogramData.value,
          });
        } else {
          onCrosshairMove?.(null);
        }
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
  }, [height, onCrosshairMove]);

  // 데이터 업데이트
  useEffect(() => {
    if (!macdSeriesRef.current || !signalSeriesRef.current || !histogramSeriesRef.current || !data.length) return;

    // MACD 데이터
    const macdData: LineData[] = data.map(d => ({
      time: Math.floor(new Date(d.time).getTime() / 1000) as Time,
      value: d.macd,
    }));
    macdSeriesRef.current.setData(macdData);

    // Signal 데이터
    const signalData: LineData[] = data.map(d => ({
      time: Math.floor(new Date(d.time).getTime() / 1000) as Time,
      value: d.signal,
    }));
    signalSeriesRef.current.setData(signalData);

    // Histogram 데이터
    const histogramData: HistogramData[] = data.map(d => ({
      time: Math.floor(new Date(d.time).getTime() / 1000) as Time,
      value: d.histogram,
      color: d.histogram >= 0 ? '#10b981' : '#ef4444',
    }));
    histogramSeriesRef.current.setData(histogramData);

    // 시간 범위 맞추기
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  }, [data]);

  return (
    <div className="relative h-full">
      <div className="absolute top-2 left-2 z-10">
        <div className="text-xs text-gray-400">MACD({fastPeriod},{slowPeriod},{signalPeriod})</div>
        <div className="flex items-center gap-4 mt-1">
          <div className="flex items-center gap-1">
            <div className="w-3 h-px bg-blue-500"></div>
            <span className="text-xs text-gray-400">MACD</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-px bg-red-500"></div>
            <span className="text-xs text-gray-400">Signal</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 bg-green-500"></div>
            <span className="text-xs text-gray-400">Histogram</span>
          </div>
        </div>
      </div>
      <div ref={chartContainerRef} className="w-full h-full" />
    </div>
  );
}