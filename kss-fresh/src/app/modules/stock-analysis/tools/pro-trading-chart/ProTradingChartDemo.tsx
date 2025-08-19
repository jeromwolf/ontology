'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ArrowLeft, BarChart3, LineChart, Activity, TrendingUp, Settings } from 'lucide-react';

// 임시 차트 데이터
interface ChartData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export default function ProTradingChartDemo() {
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState('005930');
  const [selectedTimeframe, setSelectedTimeframe] = useState('5m');
  const [lastPrice, setLastPrice] = useState(69800);
  const [priceChange, setPriceChange] = useState(1.2);
  const [mounted, setMounted] = useState(false);

  // 한국 주식 종목 목록
  const koreanStocks = [
    { code: '005930', name: '삼성전자' },
    { code: '000660', name: 'SK하이닉스' },
    { code: '035720', name: '카카오' },
    { code: '035420', name: 'NAVER' },
  ];

  // 시간 프레임 옵션
  const timeframes = [
    { value: '1m', label: '1분' },
    { value: '5m', label: '5분' },
    { value: '15m', label: '15분' },
    { value: '1h', label: '1시간' },
    { value: 'D', label: '일봉' }
  ];

  // 컴포넌트 마운트 확인
  useEffect(() => {
    setMounted(true);
  }, []);

  // 실시간 가격 업데이트 시뮬레이션
  useEffect(() => {
    if (!mounted) return;
    
    const interval = setInterval(() => {
      setLastPrice(prev => {
        const change = (Math.random() - 0.5) * 100;
        return prev + change;
      });
      setPriceChange(prev => prev + (Math.random() - 0.5) * 0.1);
    }, 1000);

    return () => clearInterval(interval);
  }, [mounted]);

  // 마운트되지 않았으면 로딩 표시
  if (!mounted) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Demo Chart 로딩 중...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      {/* Header */}
      <div className="border-b border-gray-700 bg-gray-900/50 backdrop-blur-xl">
        <div className="max-w-[1920px] mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/modules/stock-analysis/tools" className="text-gray-400 hover:text-white transition-colors">
                <ArrowLeft className="w-5 h-5" />
              </Link>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                Professional Trading Chart (Demo)
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
            </div>
            
            <div className="flex items-center gap-3">
              {/* 실시간 가격 정보 */}
              <div className="flex items-center gap-4 px-4 py-2 bg-gray-800 rounded-lg">
                <div>
                  <div className="text-2xl font-bold">
                    ₩{Math.floor(lastPrice).toLocaleString()}
                  </div>
                  <div className={`text-sm ${priceChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
                  </div>
                </div>
                <div className="text-sm text-gray-400">
                  <div>거래량</div>
                  <div className="font-medium text-white">15.2M</div>
                </div>
              </div>
              
              {/* 도구 버튼들 */}
              <div className="flex items-center gap-2">
                <button className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors" title="지표">
                  <Activity className="w-4 h-4" />
                </button>
                <button className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors" title="그리기 도구">
                  <TrendingUp className="w-4 h-4" />
                </button>
                <button className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors" title="설정">
                  <Settings className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex h-[calc(100vh-80px)]">
        {/* Main Chart Area */}
        <div className="flex-1 flex flex-col bg-gray-950">
          {/* Chart Container */}
          <div className="flex-1 relative bg-gray-950 flex items-center justify-center">
            <div className="text-center">
              <BarChart3 className="w-24 h-24 mx-auto mb-4 text-blue-500" />
              <h2 className="text-2xl font-bold mb-2">Professional Trading Chart</h2>
              <p className="text-gray-400 mb-4">TradingView 수준의 실시간 차트 시스템</p>
              
              {/* 차트 기능 미리보기 */}
              <div className="grid grid-cols-2 gap-4 max-w-md mx-auto">
                <div className="p-4 bg-gray-800 rounded-lg">
                  <LineChart className="w-8 h-8 mx-auto mb-2 text-green-400" />
                  <div className="text-sm">실시간 캔들차트</div>
                </div>
                <div className="p-4 bg-gray-800 rounded-lg">
                  <Activity className="w-8 h-8 mx-auto mb-2 text-blue-400" />
                  <div className="text-sm">30+ 기술지표</div>
                </div>
                <div className="p-4 bg-gray-800 rounded-lg">
                  <TrendingUp className="w-8 h-8 mx-auto mb-2 text-purple-400" />
                  <div className="text-sm">그리기 도구</div>
                </div>
                <div className="p-4 bg-gray-800 rounded-lg">
                  <BarChart3 className="w-8 h-8 mx-auto mb-2 text-orange-400" />
                  <div className="text-sm">실시간 호가창</div>
                </div>
              </div>
              
              <div className="mt-6 p-4 bg-green-900/20 border border-green-500/30 rounded-lg max-w-md mx-auto">
                <p className="text-sm text-green-400">✅ Demo 페이지가 정상적으로 로드되었습니다!</p>
                <p className="text-xs text-gray-400 mt-1">실제 차트는 ProTradingChartClient에서 확인하세요</p>
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel - 호가창 미리보기 */}
        <div className="w-80 bg-gray-900/50 border-l border-gray-700">
          <div className="p-4 border-b border-gray-700">
            <h3 className="text-sm font-semibold mb-2">호가창</h3>
            <div className="flex items-center gap-2 text-xs">
              <button className="flex-1 py-1 bg-gray-800 rounded hover:bg-gray-700">호가</button>
              <button className="flex-1 py-1 bg-gray-800 rounded hover:bg-gray-700">체결</button>
              <button className="flex-1 py-1 bg-gray-800 rounded hover:bg-gray-700">일별</button>
            </div>
          </div>
          
          <div className="p-4">
            {/* 매도 호가 미리보기 */}
            <div className="space-y-1 mb-4">
              {Array.from({ length: 5 }, (_, i) => {
                const price = Math.floor(lastPrice) + (5 - i) * 100;
                const volume = Math.floor(Math.random() * 900) + 100;
                return (
                  <div key={`sell-${i}`} className="flex items-center justify-between bg-red-900/20 px-2 py-1 rounded">
                    <span className="text-xs text-gray-400">1,{volume}</span>
                    <span className="text-red-400 font-medium text-sm">₩{price.toLocaleString()}</span>
                  </div>
                );
              })}
            </div>
            
            {/* 현재가 */}
            <div className="bg-gray-800 p-3 rounded mb-4 text-center">
              <div className="text-lg font-bold">₩{Math.floor(lastPrice).toLocaleString()}</div>
              <div className={`text-sm ${priceChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {priceChange >= 0 ? '▲' : '▼'} {Math.abs(priceChange).toFixed(2)}%
              </div>
            </div>
            
            {/* 매수 호가 미리보기 */}
            <div className="space-y-1">
              {Array.from({ length: 5 }, (_, i) => {
                const price = Math.floor(lastPrice) - (i + 1) * 100;
                const volume = Math.floor(Math.random() * 900) + 100;
                return (
                  <div key={`buy-${i}`} className="flex items-center justify-between bg-blue-900/20 px-2 py-1 rounded">
                    <span className="text-blue-400 font-medium text-sm">₩{price.toLocaleString()}</span>
                    <span className="text-xs text-gray-400">2,{volume}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Footer Status Bar */}
      <div className="h-8 bg-gray-900/50 border-t border-gray-700 flex items-center px-4 text-xs text-gray-400">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500"></div>
            <span>실시간 연결됨</span>
          </div>
          <div>서버 지연: 12ms</div>
          <div>데이터 제공: Demo</div>
        </div>
        <div className="ml-auto">
          마지막 업데이트: {new Date().toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
}