'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ArrowLeft, LineChart, BarChart3, Activity, TrendLine, Crosshair } from 'lucide-react';
import { ProChartContainer, OrderBook } from '@/components/charts/ProChart';
import TradingViewChart from '@/components/charts/ProChart/TradingViewChart';
import KISTokenStatus from '@/components/charts/ProChart/KISTokenStatus';
import type { ChartData, Indicator } from '@/components/charts/ProChart/types';

// 한국 주식 종목 목록
const koreanStocks = [
  { code: '005930', name: '삼성전자' },
  { code: '000660', name: 'SK하이닉스' },
  { code: '035720', name: '카카오' },
  { code: '035420', name: 'NAVER' },
  { code: '005380', name: '현대차' },
  { code: '051910', name: 'LG화학' },
  { code: '006400', name: '삼성SDI' },
  { code: '207940', name: '삼성바이오로직스' }
];

// 시간 프레임 옵션
const timeframes = [
  { value: '1m', label: '1분' },
  { value: '3m', label: '3분' },
  { value: '5m', label: '5분' },
  { value: '15m', label: '15분' },
  { value: '30m', label: '30분' },
  { value: '60m', label: '1시간' },
  { value: 'D', label: '일봉' },
  { value: 'W', label: '주봉' },
  { value: 'M', label: '월봉' }
];

export default function ProTradingChartClient() {
  const [selectedSymbol, setSelectedSymbol] = useState('005930');
  const [selectedTimeframe, setSelectedTimeframe] = useState('5m');
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [indicators, setIndicators] = useState<Indicator[]>([
    { id: 'ma5', name: 'MA 5', type: 'overlay', enabled: true, params: { period: 5 }, color: '#3b82f6' },
    { id: 'ma20', name: 'MA 20', type: 'overlay', enabled: true, params: { period: 20 }, color: '#f59e0b' },
    { id: 'ma60', name: 'MA 60', type: 'overlay', enabled: false, params: { period: 60 }, color: '#10b981' },
    { id: 'volume', name: 'Volume', type: 'volume', enabled: true, params: {}, color: '#6366f1' }
  ]);
  
  // 실시간 데이터
  const [lastPrice, setLastPrice] = useState(69800);
  const [priceChange, setPriceChange] = useState(1200);
  const [priceChangePercent, setPriceChangePercent] = useState(1.75);
  const [volume24h, setVolume24h] = useState(15234567);
  const [isRealtime, setIsRealtime] = useState(true);

  // 초기 데이터 생성
  useEffect(() => {
    const generateInitialData = () => {
      const data: ChartData[] = [];
      let basePrice = 69000;
      const now = new Date();
      
      for (let i = 100; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 5 * 60 * 1000);
        const open = basePrice;
        const volatility = 0.02; // 2% 변동성
        const change = (Math.random() - 0.5) * basePrice * volatility;
        const close = Math.max(1000, basePrice + change); // 최소 1000원
        const high = Math.max(open, close) * (1 + Math.random() * 0.01);
        const low = Math.min(open, close) * (1 - Math.random() * 0.01);
        
        data.push({
          time: time.toISOString(),
          open: Math.round(open),
          high: Math.round(high),
          low: Math.round(low),
          close: Math.round(close),
          volume: Math.floor(Math.random() * 1000000) + 100000,
        });
        
        basePrice = close;
      }
      
      setChartData(data);
      if (data.length > 0) {
        const lastCandle = data[data.length - 1];
        setLastPrice(lastCandle.close);
        
        const change = lastCandle.close - lastCandle.open;
        const changePercent = (change / lastCandle.open) * 100;
        setPriceChange(change);
        setPriceChangePercent(changePercent);
      }
    };

    generateInitialData();
  }, [selectedSymbol, selectedTimeframe]);

  // 실시간 업데이트
  useEffect(() => {
    if (!isRealtime || chartData.length === 0) return;

    const interval = setInterval(() => {
      setChartData(prev => {
        if (prev.length === 0) return prev;
        
        const lastCandle = prev[prev.length - 1];
        const volatility = 0.005; // 0.5% 변동성
        const change = (Math.random() - 0.5) * lastCandle.close * volatility;
        const newPrice = Math.max(1000, lastCandle.close + change);
        const high = Math.max(lastCandle.close, newPrice + Math.random() * newPrice * 0.002);
        const low = Math.min(lastCandle.close, newPrice - Math.random() * newPrice * 0.002);
        
        const now = new Date();
        const newCandle: ChartData = {
          time: now.toISOString(),
          open: lastCandle.close,
          high: Math.round(high),
          low: Math.round(low),
          close: Math.round(newPrice),
          volume: Math.floor(Math.random() * 100000) + 50000,
        };
        
        // 업데이트된 데이터로 상태 업데이트
        setLastPrice(newCandle.close);
        
        const priceChange = newCandle.close - newCandle.open;
        const changePercent = (priceChange / newCandle.open) * 100;
        setPriceChange(priceChange);
        setPriceChangePercent(changePercent);
        setVolume24h(prevVol => prevVol + newCandle.volume);
        
        return [...prev.slice(-99), newCandle]; // 최근 100개 캔들만 유지
      });
    }, 2000); // 2초마다 업데이트

    return () => clearInterval(interval);
  }, [isRealtime, chartData.length]);

  const selectedStock = koreanStocks.find(s => s.code === selectedSymbol) || koreanStocks[0];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <ProChartContainer
        config={{
          symbol: selectedSymbol,
          symbolName: selectedStock.name,
          timeframe: selectedTimeframe,
          chartType: 'candle',
          indicators: indicators,
          height: 600,
          realtime: isRealtime,
          showOrderBook: true,
          showDrawingTools: true,
          showIndicatorPanel: true,
          onSymbolChange: setSelectedSymbol,
          onTimeframeChange: setSelectedTimeframe,
          onIndicatorChange: setIndicators
        }}
        headerContent={
          <>
            <Link href="/modules/stock-analysis/tools" className="text-gray-400 hover:text-white transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              Professional Trading Chart
            </h1>
            
            {/* 종목 선택 */}
            <select 
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-blue-500"
            >
              {koreanStocks.map(stock => (
                <option key={stock.code} value={stock.code}>
                  {stock.name} ({stock.code})
                </option>
              ))}
            </select>
            
            {/* 시간 프레임 */}
            <div className="flex items-center gap-1 bg-gray-800 rounded-lg p-1">
              {timeframes.map(tf => (
                <button
                  key={tf.value}
                  onClick={() => setSelectedTimeframe(tf.value)}
                  className={`px-3 py-1 rounded text-sm transition-colors ${
                    selectedTimeframe === tf.value
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  {tf.label}
                </button>
              ))}
            </div>
          </>
        }
        sidebarContent={
          <div className="space-y-4">
            <OrderBook
              symbol={selectedSymbol}
              lastPrice={lastPrice}
              priceChange={priceChange}
              priceChangePercent={priceChangePercent}
              volume24h={volume24h}
              realtime={isRealtime}
            />
            <KISTokenStatus />
          </div>
        }
      >
        <TradingViewChart
          data={chartData}
          height={600}
          showVolume={true}
          indicators={{
            ma5: indicators.find(i => i.id === 'ma5')?.enabled,
            ma20: indicators.find(i => i.id === 'ma20')?.enabled,
            ma60: indicators.find(i => i.id === 'ma60')?.enabled,
            bollinger: indicators.find(i => i.id === 'bollinger')?.enabled
          }}
          onCrosshairMove={(price, time) => {
            // 크로스헤어 이동 시 처리
            console.log('Crosshair:', price, time);
          }}
        />
      </ProChartContainer>
    </div>
  );
}