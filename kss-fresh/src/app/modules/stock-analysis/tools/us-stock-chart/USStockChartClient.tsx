'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ArrowLeft, LineChart } from 'lucide-react';
import { ProChartContainer } from '@/components/charts/ProChart';
import TradingViewChart from '@/components/charts/ProChart/TradingViewChart';
import USMarketStatus from '@/components/charts/ProChart/USMarketStatus';
import ChartControls from '@/components/charts/ProChart/ChartControls';
import type { ChartData, Indicator } from '@/components/charts/ProChart/types';
import { usStockService } from '@/lib/services/us-stock-service';

// 미국 주식 종목 목록
const usStocks = [
  // Tech Giants
  { code: 'AAPL', name: 'Apple', sector: 'Technology' },
  { code: 'MSFT', name: 'Microsoft', sector: 'Technology' },
  { code: 'GOOGL', name: 'Alphabet', sector: 'Technology' },
  { code: 'AMZN', name: 'Amazon', sector: 'Consumer' },
  { code: 'NVDA', name: 'NVIDIA', sector: 'Technology' },
  { code: 'META', name: 'Meta', sector: 'Technology' },
  { code: 'TSLA', name: 'Tesla', sector: 'Auto' },
  { code: 'NFLX', name: 'Netflix', sector: 'Entertainment' },
  // Finance
  { code: 'JPM', name: 'JP Morgan', sector: 'Finance' },
  { code: 'BAC', name: 'Bank of America', sector: 'Finance' },
  { code: 'WFC', name: 'Wells Fargo', sector: 'Finance' },
  { code: 'GS', name: 'Goldman Sachs', sector: 'Finance' },
  // Others
  { code: 'DIS', name: 'Disney', sector: 'Entertainment' },
  { code: 'KO', name: 'Coca-Cola', sector: 'Consumer' },
  { code: 'PEP', name: 'PepsiCo', sector: 'Consumer' },
  { code: 'JNJ', name: 'Johnson & Johnson', sector: 'Healthcare' },
];

// 시간 프레임 옵션
const timeframes = [
  { value: '1min', label: '1분' },
  { value: '5min', label: '5분' },
  { value: '15min', label: '15분' },
  { value: '30min', label: '30분' },
  { value: '60min', label: '1시간' },
  { value: 'day', label: '일봉' },
  { value: 'week', label: '주봉' },
  { value: 'month', label: '월봉' }
];

export default function USStockChartClient() {
  const [selectedSymbol, setSelectedSymbol] = useState('TSLA');
  const [selectedTimeframe, setSelectedTimeframe] = useState('5min');
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [indicators, setIndicators] = useState<Indicator[]>([
    { id: 'ma5', name: 'MA 5', type: 'overlay', enabled: true, params: { period: 5 }, color: '#3b82f6' },
    { id: 'ma20', name: 'MA 20', type: 'overlay', enabled: true, params: { period: 20 }, color: '#f59e0b' },
    { id: 'ma60', name: 'MA 60', type: 'overlay', enabled: false, params: { period: 60 }, color: '#10b981' },
    { id: 'volume', name: 'Volume', type: 'volume', enabled: true, params: {}, color: '#6366f1' }
  ]);
  
  // 실시간 데이터
  const [lastPrice, setLastPrice] = useState(0);
  const [priceChange, setPriceChange] = useState(0);
  const [priceChangePercent, setPriceChangePercent] = useState(0);
  const [volume24h, setVolume24h] = useState(0);
  const [isRealtime, setIsRealtime] = useState(true);
  const [isLoading, setIsLoading] = useState(true);

  // 초기 데이터 가져오기
  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        console.log('Fetching US stock data for:', selectedSymbol);
        const data = await usStockService.getChartData(selectedSymbol, selectedTimeframe);
        console.log('Received data:', data.length, 'candles');
        
        if (data && data.length > 0) {
          setChartData(data);
          const lastCandle = data[data.length - 1];
          const firstCandle = data[0];
          
          setLastPrice(lastCandle.close);
          
          const change = lastCandle.close - firstCandle.close;
          const changePercent = (change / firstCandle.close) * 100;
          setPriceChange(change);
          setPriceChangePercent(changePercent);
          
          // 거래량 합계
          const totalVolume = data.reduce((sum, d) => sum + d.volume, 0);
          setVolume24h(totalVolume);
        }
      } catch (error) {
        console.error('Failed to fetch US stock data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [selectedSymbol, selectedTimeframe]);

  // 실시간 업데이트 (시뮬레이션)
  useEffect(() => {
    if (!isRealtime || chartData.length === 0) return;

    const interval = setInterval(() => {
      setChartData(prev => {
        if (prev.length === 0) return prev;
        
        const lastCandle = prev[prev.length - 1];
        const volatility = 0.005;
        const change = (Math.random() - 0.5) * lastCandle.close * volatility;
        const newPrice = Math.max(1, lastCandle.close + change);
        const high = Math.max(lastCandle.close, newPrice + Math.random() * newPrice * 0.002);
        const low = Math.min(lastCandle.close, newPrice - Math.random() * newPrice * 0.002);
        
        const now = new Date();
        const newCandle: ChartData = {
          time: now.toISOString(),
          open: lastCandle.close,
          high: Math.round(high * 100) / 100,
          low: Math.round(low * 100) / 100,
          close: Math.round(newPrice * 100) / 100,
          volume: Math.floor(Math.random() * 500000) + 100000,
        };
        
        setLastPrice(newCandle.close);
        
        const firstCandle = prev[0];
        const priceChange = newCandle.close - firstCandle.close;
        const changePercent = (priceChange / firstCandle.close) * 100;
        setPriceChange(priceChange);
        setPriceChangePercent(changePercent);
        setVolume24h(prevVol => prevVol + newCandle.volume);
        
        return [...prev.slice(-99), newCandle];
      });
    }, 3000); // 3초마다 업데이트

    return () => clearInterval(interval);
  }, [isRealtime, chartData.length]);

  const selectedStock = usStocks.find(s => s.code === selectedSymbol) || usStocks[0];

  const handleRefresh = () => {
    window.location.reload();
  };

  const handleDownload = async () => {
    const canvas = document.querySelector('canvas');
    if (canvas) {
      const link = document.createElement('a');
      link.download = `${selectedStock.name}_chart_${new Date().toISOString().split('T')[0]}.png`;
      link.href = canvas.toDataURL();
      link.click();
    }
  };

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
          showOrderBook: false,
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
            <h1 className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
              미국 주식 차트 🇺🇸
            </h1>
            
            {/* 종목 선택 */}
            <select 
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-purple-500"
            >
              {Object.entries(
                usStocks.reduce((acc, stock) => {
                  if (!acc[stock.sector]) acc[stock.sector] = [];
                  acc[stock.sector].push(stock);
                  return acc;
                }, {} as Record<string, typeof usStocks>)
              ).map(([sector, stocks]) => (
                <optgroup key={sector} label={sector}>
                  {stocks.map(stock => (
                    <option key={stock.code} value={stock.code}>
                      {stock.name} ({stock.code})
                    </option>
                  ))}
                </optgroup>
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
                      ? 'bg-purple-600 text-white'
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
            {/* 주식 정보 */}
            <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
              <h3 className="text-sm font-semibold mb-3">주식 정보</h3>
              
              {/* 현재가 */}
              <div className="mb-4">
                <div className="text-2xl font-bold">
                  ${lastPrice.toFixed(2)}
                </div>
                <div className={`flex items-center gap-1 text-sm ${
                  priceChange >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  <span>{priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}</span>
                  <span>({priceChangePercent.toFixed(2)}%)</span>
                </div>
              </div>
              
              {/* 거래량 */}
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">거래량</span>
                  <span>{volume24h.toLocaleString()}</span>
                </div>
              </div>
              
              {/* 데이터 정보 */}
              <div className="mt-4 p-3 bg-blue-900/20 border border-blue-500/30 rounded">
                <div className="text-xs text-blue-300">
                  <div className="font-medium mb-1">데이터 정보</div>
                  <div>• 실시간 가격 (15분 지연)</div>
                  <div>• API: Twelve Data / Alpha Vantage</div>
                </div>
              </div>
            </div>
            
            <USMarketStatus />
          </div>
        }
        rightSidebarContent={
          <ChartControls
            onRefresh={handleRefresh}
            onDownload={handleDownload}
            onFullscreen={() => {
              document.documentElement.requestFullscreen();
            }}
          />
        }
      >
        {isLoading ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500 mx-auto mb-4"></div>
              <p className="text-gray-400">차트 데이터 로딩 중...</p>
            </div>
          </div>
        ) : (
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
              console.log('Crosshair:', price, time);
            }}
          />
        )}
      </ProChartContainer>
    </div>
  );
}