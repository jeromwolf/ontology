'use client';

import { useEffect, useRef, useState } from 'react';
import { 
  createChart, 
  IChartApi, 
  ISeriesApi,
  CandlestickData,
  HistogramData,
  LineData,
  Time,
  ColorType,
  CrosshairMode
} from 'lightweight-charts';

interface ChartProps {
  data: CandlestickData[];
  indicators: {
    ma5?: LineData[];
    ma20?: LineData[];
    ma60?: LineData[];
    volume?: HistogramData[];
    rsi?: LineData[];
  };
  height?: number;
  realtime?: boolean;
}

export default function TradingViewChart({ data, indicators, height = 600, realtime = true }: ChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const ma5SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ma20SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ma60SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  // 차트 초기화
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
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          width: 1,
          color: '#4b5563',
          style: 0,
          labelBackgroundColor: '#1f2937',
        },
        horzLine: {
          width: 1,
          color: '#4b5563',
          style: 0,
          labelBackgroundColor: '#1f2937',
        },
      },
      rightPriceScale: {
        borderColor: '#1f2937',
        scaleMargins: {
          top: 0.1,
          bottom: 0.2,
        },
      },
      timeScale: {
        borderColor: '#1f2937',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    // 캔들스틱 시리즈
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#10b981',
      downColor: '#ef4444',
      borderUpColor: '#10b981',
      borderDownColor: '#ef4444',
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });
    candleSeriesRef.current = candleSeries;

    // 거래량 히스토그램
    const volumeSeries = chart.addHistogramSeries({
      color: '#3b82f6',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '',
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });
    volumeSeriesRef.current = volumeSeries;

    // 이동평균선들
    const ma5Series = chart.addLineSeries({
      color: '#3b82f6',
      lineWidth: 2,
      title: 'MA5',
    });
    ma5SeriesRef.current = ma5Series;

    const ma20Series = chart.addLineSeries({
      color: '#f59e0b',
      lineWidth: 2,
      title: 'MA20',
    });
    ma20SeriesRef.current = ma20Series;

    const ma60Series = chart.addLineSeries({
      color: '#10b981',
      lineWidth: 2,
      title: 'MA60',
    });
    ma60SeriesRef.current = ma60Series;

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
      if (chart) {
        chart.remove();
      }
    };
  }, [height]);

  // 데이터 업데이트
  useEffect(() => {
    if (!candleSeriesRef.current || !data.length) return;

    candleSeriesRef.current.setData(data);

    // 거래량 데이터
    if (volumeSeriesRef.current && indicators.volume) {
      volumeSeriesRef.current.setData(indicators.volume);
    }

    // 이동평균선 데이터
    if (ma5SeriesRef.current && indicators.ma5) {
      ma5SeriesRef.current.setData(indicators.ma5);
    }
    if (ma20SeriesRef.current && indicators.ma20) {
      ma20SeriesRef.current.setData(indicators.ma20);
    }
    if (ma60SeriesRef.current && indicators.ma60) {
      ma60SeriesRef.current.setData(indicators.ma60);
    }

    // 차트 시간 범위 맞추기
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  }, [data, indicators]);

  // 실시간 업데이트
  useEffect(() => {
    if (!realtime || !candleSeriesRef.current || !data.length) return;

    const interval = setInterval(() => {
      const lastCandle = data[data.length - 1];
      const time = Math.floor(Date.now() / 1000) as Time;
      
      // 새로운 가격 시뮬레이션
      const change = (Math.random() - 0.5) * 100;
      const newPrice = lastCandle.close + change;
      
      const update: CandlestickData = {
        time,
        open: lastCandle.close,
        high: Math.max(lastCandle.close, newPrice),
        low: Math.min(lastCandle.close, newPrice),
        close: newPrice,
      };

      candleSeriesRef.current?.update(update);

      // 거래량 업데이트
      if (volumeSeriesRef.current) {
        volumeSeriesRef.current.update({
          time,
          value: Math.random() * 1000000,
          color: newPrice >= lastCandle.close ? '#10b98180' : '#ef444480',
        });
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [realtime, data]);

  return (
    <div className="relative">
      <div ref={chartContainerRef} className="w-full" />
      
      {/* 차트 위 정보 오버레이 */}
      <div className="absolute top-4 left-4 bg-gray-900/80 backdrop-blur-sm rounded-lg p-3">
        <div className="text-xs text-gray-400 mb-1">삼성전자 (005930)</div>
        <div className="flex items-center gap-4">
          <div>
            <div className="text-lg font-bold text-white">₩69,800</div>
            <div className="text-xs text-green-400">+1,200 (+1.75%)</div>
          </div>
          <div className="text-xs text-gray-400">
            <div>고가: ₩70,500</div>
            <div>저가: ₩68,900</div>
          </div>
        </div>
      </div>

      {/* 우측 상단 시간프레임 */}
      <div className="absolute top-4 right-4 flex items-center gap-1 bg-gray-900/80 backdrop-blur-sm rounded-lg p-1">
        {['1m', '5m', '15m', '1h', 'D'].map(tf => (
          <button
            key={tf}
            className={`px-2 py-1 rounded text-xs transition-colors ${
              tf === '5m' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
            }`}
          >
            {tf}
          </button>
        ))}
      </div>
    </div>
  );
}