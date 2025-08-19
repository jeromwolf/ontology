'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ArrowLeft, BarChart3 } from 'lucide-react';
import { ProChartContainer, OrderBook } from '@/components/charts/ProChart';
import TradingViewChart from '@/components/charts/ProChart/TradingViewChart';
import KISTokenStatus from '@/components/charts/ProChart/KISTokenStatus';
import ChartControls from '@/components/charts/ProChart/ChartControls';
import type { ChartData, Indicator } from '@/components/charts/ProChart/types';
import { kisApiService } from '@/lib/services/kis-api-service';

// 한국 주식 종목 목록
const koreanStocks = [
  { code: '005930', name: '삼성전자' },
  { code: '000660', name: 'SK하이닉스' },
  { code: '035720', name: '카카오' },
  { code: '035420', name: 'NAVER' },
  { code: '005380', name: '현대차' },
  { code: '051910', name: 'LG화학' },
  { code: '006400', name: '삼성SDI' },
  { code: '207940', name: '삼성바이오로직스' },
  { code: '000270', name: '기아' },
  { code: '005490', name: 'POSCO홀딩스' },
  { code: '068270', name: '셀트리온' },
  { code: '105560', name: 'KB금융' },
  { code: '055550', name: '신한지주' },
  { code: '086790', name: '하나금융지주' },
  { code: '003550', name: 'LG' },
  { code: '034730', name: 'SK' },
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

export default function KRStockChartClient() {
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

  // 초기 데이터 가져오기
  useEffect(() => {
    const fetchData = async () => {
      try {
        // KIS API를 통해 실제 데이터 가져오기
        const chartData = await kisApiService.getChartHistory(selectedSymbol, 100);
        if (chartData && chartData.length > 0) {
          setChartData(chartData);
          const lastCandle = chartData[chartData.length - 1];
          const firstCandle = chartData[0];
          
          setLastPrice(lastCandle.close);
          
          const change = lastCandle.close - firstCandle.close;
          const changePercent = (change / firstCandle.close) * 100;
          setPriceChange(change);
          setPriceChangePercent(changePercent);
          
          // 거래량 합계
          const totalVolume = chartData.reduce((sum, d) => sum + d.volume, 0);
          setVolume24h(totalVolume);
          return;
        }
      } catch (error) {
        console.error('Failed to fetch KR stock data:', error);
      }
      
      // 실패 시 데모 데이터 생성
      generateDemoData();
    };
    
    const generateDemoData = () => {
      const data: ChartData[] = [];
      let basePrice = 69000;
      const now = new Date();
      
      for (let i = 100; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 5 * 60 * 1000);
        const open = basePrice;
        const volatility = 0.02;
        const change = (Math.random() - 0.5) * basePrice * volatility;
        const close = Math.max(1000, basePrice + change);
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
        
        const change = lastCandle.close - data[0].close;
        const changePercent = (change / data[0].close) * 100;
        setPriceChange(change);
        setPriceChangePercent(changePercent);
      }
    };

    fetchData();
  }, [selectedSymbol, selectedTimeframe]);

  // 실시간 업데이트
  useEffect(() => {
    if (!isRealtime || chartData.length === 0) return;

    const interval = setInterval(() => {
      setChartData(prev => {
        if (prev.length === 0) return prev;
        
        const lastCandle = prev[prev.length - 1];
        const volatility = 0.005;
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
        
        setLastPrice(newCandle.close);
        
        const firstCandle = prev[0];
        const priceChange = newCandle.close - firstCandle.close;
        const changePercent = (priceChange / firstCandle.close) * 100;
        setPriceChange(priceChange);
        setPriceChangePercent(changePercent);
        setVolume24h(prevVol => prevVol + newCandle.volume);
        
        return [...prev.slice(-99), newCandle];
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [isRealtime, chartData.length]);

  const selectedStock = koreanStocks.find(s => s.code === selectedSymbol) || koreanStocks[0];

  const handleRefresh = () => {
    window.location.reload();
  };

  const handleDownload = async () => {
    // 차트 다운로드 기능
    const canvas = document.querySelector('canvas');
    if (canvas) {
      const link = document.createElement('a');
      link.download = `${selectedStock.name}_차트_${new Date().toISOString().split('T')[0]}.png`;
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
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-blue-600 bg-clip-text text-transparent">
              한국 주식 차트 🇰🇷
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
              market="KR"
            />
            <KISTokenStatus />
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
      </ProChartContainer>
    </div>
  );
}