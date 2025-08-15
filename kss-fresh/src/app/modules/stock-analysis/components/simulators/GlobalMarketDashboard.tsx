'use client';

import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Clock, DollarSign, Globe, BarChart3, Activity } from 'lucide-react';

// 모의 데이터 (실제로는 API 연동 필요)
const mockExchangeRate = {
  USD_KRW: 1320.50,
  change: 5.20,
  changePercent: 0.40
};

const mockIndices = {
  SPX: { name: 'S&P 500', value: 4783.35, change: 29.09, changePercent: 0.61 },
  IXIC: { name: 'NASDAQ', value: 14972.76, change: 164.46, changePercent: 1.11 },
  DJI: { name: 'DOW JONES', value: 37466.11, change: 211.02, changePercent: 0.57 },
  VIX: { name: 'VIX', value: 13.14, change: -0.52, changePercent: -3.81 }
};

const mockTrendingStocks = {
  mostActive: [
    { symbol: 'AAPL', name: 'Apple', price: 195.89, change: 2.56, changePercent: 1.32, volume: '53.2M' },
    { symbol: 'NVDA', name: 'NVIDIA', price: 878.36, change: 42.12, changePercent: 5.03, volume: '41.7M' },
    { symbol: 'TSLA', name: 'Tesla', price: 246.38, change: -3.21, changePercent: -1.29, volume: '112.4M' }
  ],
  gainers: [
    { symbol: 'SMCI', name: 'Super Micro', price: 589.21, change: 89.32, changePercent: 17.87, volume: '12.3M' },
    { symbol: 'AMD', name: 'AMD', price: 178.45, change: 14.23, changePercent: 8.67, volume: '28.9M' },
    { symbol: 'META', name: 'Meta', price: 512.18, change: 24.67, changePercent: 5.06, volume: '18.2M' }
  ],
  losers: [
    { symbol: 'BA', name: 'Boeing', price: 218.45, change: -12.34, changePercent: -5.35, volume: '8.7M' },
    { symbol: 'DIS', name: 'Disney', price: 92.18, change: -3.45, changePercent: -3.61, volume: '14.5M' },
    { symbol: 'PYPL', name: 'PayPal', price: 58.92, change: -1.78, changePercent: -2.93, volume: '11.2M' }
  ]
};

const sectorData = [
  { name: 'Technology', change: 2.34, value: 100 },
  { name: 'Healthcare', change: 0.87, value: 85 },
  { name: 'Financial', change: 1.23, value: 92 },
  { name: 'Consumer', change: -0.45, value: 78 },
  { name: 'Energy', change: -1.89, value: 65 },
  { name: 'Industrial', change: 0.56, value: 88 },
  { name: 'Materials', change: -0.23, value: 80 },
  { name: 'Real Estate', change: -2.34, value: 60 },
  { name: 'Utilities', change: 0.12, value: 82 },
  { name: 'Communication', change: 1.78, value: 95 },
  { name: 'Staples', change: 0.34, value: 84 }
];

export default function GlobalMarketDashboard() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [marketStatus, setMarketStatus] = useState('');
  const [activeTab, setActiveTab] = useState('mostActive');

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    // 미국 동부 시간 기준 시장 상태 확인
    const now = new Date();
    const et = new Date(now.toLocaleString("en-US", {timeZone: "America/New_York"}));
    const hours = et.getHours();
    const minutes = et.getMinutes();
    const day = et.getDay();
    
    if (day === 0 || day === 6) {
      setMarketStatus('휴장');
    } else if (hours === 9 && minutes >= 30 || (hours > 9 && hours < 16)) {
      setMarketStatus('개장중');
    } else if (hours >= 4 && hours < 9 || (hours === 9 && minutes < 30)) {
      setMarketStatus('프리마켓');
    } else if (hours >= 16 && hours < 20) {
      setMarketStatus('애프터마켓');
    } else {
      setMarketStatus('휴장');
    }
  }, [currentTime]);

  const getStatusColor = (status: string) => {
    switch(status) {
      case '개장중': return 'text-green-500';
      case '프리마켓':
      case '애프터마켓': return 'text-yellow-500';
      default: return 'text-gray-500';
    }
  };

  const getHeatmapColor = (change: number) => {
    if (change > 2) return 'bg-green-600';
    if (change > 1) return 'bg-green-500';
    if (change > 0) return 'bg-green-400';
    if (change > -1) return 'bg-red-400';
    if (change > -2) return 'bg-red-500';
    return 'bg-red-600';
  };

  return (
    <div className="space-y-6">
      {/* 상단: 환율 & 시간 정보 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* 환율 카드 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">USD/KRW</h3>
            <DollarSign className="w-5 h-5 text-gray-400" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold">{mockExchangeRate.USD_KRW.toFixed(2)}</span>
            <span className={`text-sm font-medium ${mockExchangeRate.change > 0 ? 'text-red-500' : 'text-blue-500'}`}>
              {mockExchangeRate.change > 0 ? '+' : ''}{mockExchangeRate.change.toFixed(2)}
              ({mockExchangeRate.changePercent > 0 ? '+' : ''}{mockExchangeRate.changePercent.toFixed(2)}%)
            </span>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
            1달러 = {mockExchangeRate.USD_KRW.toFixed(2)}원
          </p>
        </div>

        {/* 시장 시간 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">미국 시장</h3>
            <Clock className="w-5 h-5 text-gray-400" />
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm">현재 시간 (ET)</span>
              <span className="text-sm font-medium">
                {currentTime.toLocaleTimeString('en-US', { 
                  timeZone: 'America/New_York',
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit',
                  hour12: false 
                })}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">한국 시간</span>
              <span className="text-sm font-medium">
                {currentTime.toLocaleTimeString('ko-KR', { 
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit',
                  hour12: false 
                })}
              </span>
            </div>
          </div>
        </div>

        {/* 시장 상태 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">시장 상태</h3>
            <Activity className="w-5 h-5 text-gray-400" />
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${marketStatus === '개장중' ? 'bg-green-500 animate-pulse' : marketStatus.includes('마켓') ? 'bg-yellow-500' : 'bg-gray-400'}`} />
            <span className={`text-lg font-medium ${getStatusColor(marketStatus)}`}>
              {marketStatus}
            </span>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
            정규장: 23:30 - 06:00 (한국시간)
          </p>
        </div>
      </div>

      {/* 주요 지수 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5" />
          미국 주요 지수
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(mockIndices).map(([key, index]) => (
            <div key={key} className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
              <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                {index.name}
              </h4>
              <p className="text-xl font-bold mb-1">{index.value.toLocaleString()}</p>
              <div className={`flex items-center gap-1 text-sm ${index.change > 0 ? 'text-green-500' : 'text-red-500'}`}>
                {index.change > 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                <span>
                  {index.change > 0 ? '+' : ''}{index.change.toFixed(2)} ({index.changePercent > 0 ? '+' : ''}{index.changePercent.toFixed(2)}%)
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 인기 종목 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            실시간 인기 종목
          </h3>
          <div className="flex gap-2">
            <button
              onClick={() => setActiveTab('mostActive')}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'mostActive' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
              }`}
            >
              거래량 상위
            </button>
            <button
              onClick={() => setActiveTab('gainers')}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'gainers' 
                  ? 'bg-green-500 text-white' 
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
              }`}
            >
              상승률 상위
            </button>
            <button
              onClick={() => setActiveTab('losers')}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'losers' 
                  ? 'bg-red-500 text-white' 
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
              }`}
            >
              하락률 상위
            </button>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-sm text-gray-600 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                <th className="pb-3 font-medium">종목</th>
                <th className="pb-3 font-medium text-right">현재가</th>
                <th className="pb-3 font-medium text-right">변동</th>
                <th className="pb-3 font-medium text-right">거래량</th>
              </tr>
            </thead>
            <tbody>
              {mockTrendingStocks[activeTab].map((stock) => (
                <tr key={stock.symbol} className="border-b border-gray-100 dark:border-gray-800">
                  <td className="py-3">
                    <div>
                      <p className="font-medium">{stock.symbol}</p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">{stock.name}</p>
                    </div>
                  </td>
                  <td className="py-3 text-right font-medium">${stock.price.toFixed(2)}</td>
                  <td className={`py-3 text-right ${stock.change > 0 ? 'text-green-500' : 'text-red-500'}`}>
                    <div className="flex items-center justify-end gap-1">
                      {stock.change > 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                      <span>
                        {stock.change > 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%
                      </span>
                    </div>
                  </td>
                  <td className="py-3 text-right text-sm text-gray-600 dark:text-gray-400">{stock.volume}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 섹터 히트맵 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Globe className="w-5 h-5" />
          S&P 500 섹터별 현황
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          {sectorData.map((sector) => (
            <div
              key={sector.name}
              className={`${getHeatmapColor(sector.change)} p-4 rounded-lg text-white text-center transition-transform hover:scale-105 cursor-pointer`}
            >
              <p className="text-sm font-medium mb-1">{sector.name}</p>
              <p className="text-lg font-bold">
                {sector.change > 0 ? '+' : ''}{sector.change.toFixed(2)}%
              </p>
            </div>
          ))}
        </div>
        <div className="flex items-center justify-center gap-4 mt-4 text-sm text-gray-600 dark:text-gray-400">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-500 rounded"></div>
            <span>상승</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-gray-400 rounded"></div>
            <span>보합</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-500 rounded"></div>
            <span>하락</span>
          </div>
        </div>
      </div>

      {/* 알림 메시지 */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 flex items-start gap-3">
        <div className="text-blue-600 dark:text-blue-400">ℹ️</div>
        <div className="text-sm text-blue-800 dark:text-blue-200">
          <p className="font-medium mb-1">실시간 데이터 안내</p>
          <p>현재 표시되는 데이터는 데모용 샘플 데이터입니다. 실제 서비스에서는 실시간 시세가 제공됩니다.</p>
        </div>
      </div>
    </div>
  );
}