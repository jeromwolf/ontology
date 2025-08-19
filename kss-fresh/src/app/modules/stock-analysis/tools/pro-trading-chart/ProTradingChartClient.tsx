'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ArrowLeft, LineChart, BarChart3, Activity, TrendLine, Crosshair } from 'lucide-react';
import { ProChartContainer, OrderBook } from '@/components/charts/ProChart';
import TradingViewChart from '@/components/charts/ProChart/TradingViewChart';
import KISTokenStatus from '@/components/charts/ProChart/KISTokenStatus';
import USMarketStatus from '@/components/charts/ProChart/USMarketStatus';
import ChartControls from '@/components/charts/ProChart/ChartControls';
import type { ChartData, Indicator } from '@/components/charts/ProChart/types';
import { kisApiService } from '@/lib/services/kis-api-service';

// 주식 종목 목록
const stockList = [
  // 한국 주식
  { code: '005930', name: '삼성전자', market: 'KR' },
  { code: '000660', name: 'SK하이닉스', market: 'KR' },
  { code: '035720', name: '카카오', market: 'KR' },
  { code: '035420', name: 'NAVER', market: 'KR' },
  { code: '005380', name: '현대차', market: 'KR' },
  { code: '051910', name: 'LG화학', market: 'KR' },
  { code: '006400', name: '삼성SDI', market: 'KR' },
  { code: '207940', name: '삼성바이오로직스', market: 'KR' },
  // 미국 주식
  { code: 'TSLA', name: '테슬라', market: 'US' },
  { code: 'AAPL', name: '애플', market: 'US' },
  { code: 'MSFT', name: '마이크로소프트', market: 'US' },
  { code: 'GOOGL', name: '구글', market: 'US' },
  { code: 'AMZN', name: '아마존', market: 'US' },
  { code: 'NVDA', name: '엔비디아', market: 'US' },
  { code: 'META', name: '메타', market: 'US' },
  { code: 'NFLX', name: '넷플릭스', market: 'US' }
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
  const [marketType, setMarketType] = useState<'KR' | 'US'>('KR');
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
  const [lastPrice, setLastPrice] = useState(0);
  const [priceChange, setPriceChange] = useState(0);
  const [priceChangePercent, setPriceChangePercent] = useState(0);
  const [volume24h, setVolume24h] = useState(0);
  const [isRealtime, setIsRealtime] = useState(true);
  const [dataSource, setDataSource] = useState<'REAL' | 'DEMO'>('DEMO');
  const [previousClose, setPreviousClose] = useState(0);

  // 마켓 타입 변경 시 심볼 자동 변경
  useEffect(() => {
    if (marketType === 'KR') {
      setSelectedSymbol('005930'); // 삼성전자
    } else {
      setSelectedSymbol('TSLA'); // 테슬라
    }
  }, [marketType]);

  // 초기 데이터 가져오기
  useEffect(() => {
    const fetchData = async () => {
      const stock = stockList.find(s => s.code === selectedSymbol);
      
      if (stock?.market === 'US') {
        // 미국 주식 데이터 가져오기
        try {
          console.log('Fetching US stock data for:', selectedSymbol);
          const response = await fetch(`/api/stock/us?symbol=${selectedSymbol}&interval=${selectedTimeframe}&includeDaily=true`);
          
          if (response.ok) {
            const { data, previousClose: prevClose } = await response.json();
            
            if (data && data.length > 0) {
              setChartData(data);
              const lastCandle = data[data.length - 1];
              
              setLastPrice(lastCandle.close);
              
              // 전일 종가가 있으면 사용, 없으면 첫 캔들 사용
              const basePrice = prevClose || data[0].close;
              setPreviousClose(basePrice);
              
              const change = lastCandle.close - basePrice;
              const changePercent = (change / basePrice) * 100;
              setPriceChange(change);
              setPriceChangePercent(changePercent);
              
              // 거래량 합계
              const totalVolume = data.reduce((sum: number, d: ChartData) => sum + d.volume, 0);
              setVolume24h(totalVolume);
              
              console.log('Updated price:', lastCandle.close, 'Change:', change);
              setDataSource('REAL');
              return;
            }
          }
        } catch (error) {
          console.error('Failed to fetch US stock data:', error);
        }
      } else if (stock?.market === 'KR') {
        // 한국 주식 데이터 가져오기 (KIS API)
        try {
          const chartData = await kisApiService.getChartHistory(selectedSymbol, 100);
          if (chartData && chartData.length > 0) {
            setChartData(chartData);
            const lastCandle = chartData[chartData.length - 1];
            setLastPrice(lastCandle.close);
            
            // KIS API에서는 전일 종가 정보가 없으므로 첫 캔들 사용
            const basePrice = chartData[0].close;
            setPreviousClose(basePrice);
            
            const change = lastCandle.close - basePrice;
            const changePercent = (change / basePrice) * 100;
            setPriceChange(change);
            setPriceChangePercent(changePercent);
            
            // 거래량 합계
            const totalVolume = chartData.reduce((sum, d) => sum + d.volume, 0);
            setVolume24h(totalVolume);
            setDataSource('REAL');
            return;
          }
        } catch (error) {
          console.error('Failed to fetch KR stock data:', error);
        }
      }
      
      // 실패 시 데모 데이터 생성
      setDataSource('DEMO');
      generateDemoData();
    };
    
    const generateDemoData = () => {
      const data: ChartData[] = [];
      const stock = stockList.find(s => s.code === selectedSymbol);
      // 실제 가격에 근접한 기본값 설정
      const usBasePrices: Record<string, number> = {
        'TSLA': 212,
        'AAPL': 226,
        'MSFT': 410,
        'GOOGL': 163,
        'AMZN': 178,
        'NVDA': 125,
        'META': 513,
        'NFLX': 612,
      };
      
      let basePrice = stock?.market === 'US' 
        ? (usBasePrices[selectedSymbol] || 100) 
        : 69000;
      const now = new Date();
      
      for (let i = 100; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 5 * 60 * 1000);
        const open = basePrice;
        const volatility = 0.02; // 2% 변동성
        const change = (Math.random() - 0.5) * basePrice * volatility;
        const minPrice = stock?.market === 'US' ? 1 : 1000;
        const close = Math.max(minPrice, basePrice + change);
        const high = Math.max(open, close) * (1 + Math.random() * 0.01);
        const low = Math.min(open, close) * (1 - Math.random() * 0.01);
        
        data.push({
          time: time.toISOString(),
          open: Math.round(open * 100) / 100,
          high: Math.round(high * 100) / 100,
          low: Math.round(low * 100) / 100,
          close: Math.round(close * 100) / 100,
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

    const stock = stockList.find(s => s.code === selectedSymbol);
    const isUS = stock?.market === 'US';

    const interval = setInterval(() => {
      setChartData(prev => {
        if (prev.length === 0) return prev;
        
        const lastCandle = prev[prev.length - 1];
        const volatility = 0.005; // 0.5% 변동성
        const change = (Math.random() - 0.5) * lastCandle.close * volatility;
        const minPrice = isUS ? 1 : 1000;
        const newPrice = Math.max(minPrice, lastCandle.close + change);
        const high = Math.max(lastCandle.close, newPrice + Math.random() * newPrice * 0.002);
        const low = Math.min(lastCandle.close, newPrice - Math.random() * newPrice * 0.002);
        
        const now = new Date();
        const newCandle: ChartData = {
          time: now.toISOString(),
          open: lastCandle.close,
          high: isUS ? Math.round(high * 100) / 100 : Math.round(high),
          low: isUS ? Math.round(low * 100) / 100 : Math.round(low),
          close: isUS ? Math.round(newPrice * 100) / 100 : Math.round(newPrice),
          volume: Math.floor(Math.random() * (isUS ? 500000 : 100000)) + (isUS ? 100000 : 50000),
        };
        
        // 업데이트된 데이터로 상태 업데이트
        setLastPrice(newCandle.close);
        
        // 전일 종가 기준으로 계산
        const priceChange = newCandle.close - previousClose;
        const changePercent = previousClose > 0 ? (priceChange / previousClose) * 100 : 0;
        setPriceChange(priceChange);
        setPriceChangePercent(changePercent);
        setVolume24h(prevVol => prevVol + newCandle.volume);
        
        return [...prev.slice(-99), newCandle]; // 최근 100개 캔들만 유지
      });
    }, 2000); // 2초마다 업데이트

    return () => clearInterval(interval);
  }, [isRealtime, chartData.length, selectedSymbol, previousClose]);

  const selectedStock = stockList.find(s => s.code === selectedSymbol) || stockList[0];

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
            
            {/* 데이터 소스 표시 */}
            <div className={`flex items-center gap-2 px-3 py-1 rounded-lg text-xs font-medium ${
              dataSource === 'REAL' 
                ? 'bg-green-900/30 border border-green-500/30 text-green-400' 
                : 'bg-yellow-900/30 border border-yellow-500/30 text-yellow-400'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                dataSource === 'REAL' ? 'bg-green-400' : 'bg-yellow-400'
              } animate-pulse`} />
              {dataSource === 'REAL' ? 'REAL DATA' : 'DEMO DATA'}
            </div>
            
            {/* 마켓 타입 토글 */}
            <div className="flex items-center bg-gray-800 rounded-lg p-1">
              <button
                onClick={() => setMarketType('KR')}
                className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
                  marketType === 'KR'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                🇰🇷 국내
              </button>
              <button
                onClick={() => setMarketType('US')}
                className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
                  marketType === 'US'
                    ? 'bg-purple-600 text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                🇺🇸 해외
              </button>
            </div>
            
            {/* 종목 선택 */}
            <select 
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-blue-500"
            >
              {stockList
                .filter(s => s.market === marketType)
                .map(stock => (
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
              market={selectedStock.market}
            />
            {selectedStock.market === 'US' ? (
              <USMarketStatus />
            ) : (
              <KISTokenStatus />
            )}
          </div>
        }
      >
        <TradingViewChart
          data={chartData}
          height={600}
          showVolume={true}
          previousClose={previousClose}
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