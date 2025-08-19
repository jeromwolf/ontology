'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ArrowLeft, TrendingUp, TrendingDown, Activity, DollarSign, BarChart3, Clock, AlertCircle, RefreshCw, Globe, Zap } from 'lucide-react';

interface MarketIndex {
  symbol: string;
  name: string;
  value: number;
  change: number;
  changePercent: number;
  volume?: string;
  updatedAt: string;
}

interface StockQuote {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: string;
  marketCap?: string;
}

interface MarketSector {
  name: string;
  changePercent: number;
  leaders: { symbol: string; name: string; changePercent: number }[];
}

export default function MarketDashboardPage() {
  const [selectedTab, setSelectedTab] = useState<'domestic' | 'global'>('domestic');
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [loading, setLoading] = useState(true);
  const [cacheInfo, setCacheInfo] = useState<{
    cached: boolean;
    cacheStats?: any;
  }>({ cached: false });

  // 데이터베이스에서 가져올 상태
  const [marketIndices, setMarketIndices] = useState<MarketIndex[]>([
    { symbol: 'KOSPI', name: '코스피', value: 2501.23, change: 15.67, changePercent: 0.63, volume: '5,234억', updatedAt: '15:30' },
    { symbol: 'KOSDAQ', name: '코스닥', value: 698.45, change: -8.21, changePercent: -1.16, volume: '8,901억', updatedAt: '15:30' },
    { symbol: 'KOSPI200', name: '코스피200', value: 321.15, change: 2.34, changePercent: 0.73, updatedAt: '15:30' },
    { symbol: 'DJI', name: '다우존스', value: 34721.12, change: 182.01, changePercent: 0.53, updatedAt: '04:00' },
    { symbol: 'NASDAQ', name: '나스닥', value: 13711.00, change: -123.45, changePercent: -0.89, updatedAt: '04:00' },
    { symbol: 'S&P500', name: 'S&P 500', value: 4488.28, change: 23.17, changePercent: 0.52, updatedAt: '04:00' },
  ]);

  const [topGainers, setTopGainers] = useState<StockQuote[]>([
    { symbol: '005930', name: '삼성전자', price: 68500, change: 2100, changePercent: 3.16, volume: '15,234,567' },
    { symbol: '000660', name: 'SK하이닉스', price: 115000, change: 3500, changePercent: 3.14, volume: '8,901,234' },
    { symbol: '035420', name: 'NAVER', price: 215000, change: 6000, changePercent: 2.87, volume: '1,234,567' },
    { symbol: '035720', name: '카카오', price: 45200, change: 1200, changePercent: 2.73, volume: '3,456,789' },
    { symbol: '207940', name: '삼성바이오로직스', price: 785000, change: 20000, changePercent: 2.61, volume: '234,567' },
  ]);

  const [topLosers, setTopLosers] = useState<StockQuote[]>([
    { symbol: '005380', name: '현대차', price: 185000, change: -5500, changePercent: -2.89, volume: '2,345,678' },
    { symbol: '051910', name: 'LG화학', price: 412000, change: -11000, changePercent: -2.60, volume: '567,890' },
    { symbol: '006400', name: '삼성SDI', price: 398000, change: -9500, changePercent: -2.33, volume: '345,678' },
    { symbol: '028260', name: '삼성물산', price: 102000, change: -2300, changePercent: -2.21, volume: '890,123' },
    { symbol: '105560', name: 'KB금융', price: 52100, change: -1100, changePercent: -2.07, volume: '1,234,567' },
  ]);

  const [mostActive, setMostActive] = useState<StockQuote[]>([
    { symbol: '005930', name: '삼성전자', price: 68500, change: 2100, changePercent: 3.16, volume: '15,234,567', marketCap: '408.5조' },
    { symbol: '373220', name: 'LG에너지솔루션', price: 425000, change: -5000, changePercent: -1.16, volume: '12,345,678', marketCap: '99.8조' },
    { symbol: '000270', name: '기아', price: 82500, change: 1500, changePercent: 1.85, volume: '9,876,543', marketCap: '33.5조' },
    { symbol: '068270', name: '셀트리온', price: 172000, change: 3000, changePercent: 1.77, volume: '8,765,432', marketCap: '23.7조' },
    { symbol: '012330', name: '현대모비스', price: 215000, change: -3000, changePercent: -1.38, volume: '7,654,321', marketCap: '20.3조' },
  ]);

  const [sectors, setSectors] = useState<MarketSector[]>([
    { 
      name: '반도체', 
      changePercent: 2.85,
      leaders: [
        { symbol: '005930', name: '삼성전자', changePercent: 3.16 },
        { symbol: '000660', name: 'SK하이닉스', changePercent: 3.14 }
      ]
    },
    {
      name: '2차전지',
      changePercent: -1.23,
      leaders: [
        { symbol: '373220', name: 'LG에너지솔루션', changePercent: -1.16 },
        { symbol: '006400', name: '삼성SDI', changePercent: -2.33 }
      ]
    },
    {
      name: '바이오',
      changePercent: 1.45,
      leaders: [
        { symbol: '207940', name: '삼성바이오로직스', changePercent: 2.61 },
        { symbol: '068270', name: '셀트리온', changePercent: 1.77 }
      ]
    },
    {
      name: '금융',
      changePercent: -0.89,
      leaders: [
        { symbol: '105560', name: 'KB금융', changePercent: -2.07 },
        { symbol: '055550', name: '신한지주', changePercent: -0.52 }
      ]
    }
  ]);

  // Yahoo Finance API로 실시간 데이터 가져오기
  const fetchMarketData = async (forceRefresh = false) => {
    setLoading(true);
    try {
      // 통합 실시간 API로 모든 데이터 한번에 가져오기
      const url = forceRefresh 
        ? '/api/stock/realtime?type=market-overview&refresh=true'
        : '/api/stock/realtime?type=market-overview';
      const marketRes = await fetch(url);

      if (marketRes.ok) {
        const marketData = await marketRes.json();
        
        // 캐시 정보 업데이트
        setCacheInfo({
          cached: marketData.cached || false,
          cacheStats: marketData.cacheStats
        });
        
        // 한국 지수는 시뮬레이션 데이터 사용 (2024년 12월 기준 실제 수치)
        const koreanIndices = [
          {
            symbol: 'KOSPI',
            name: '코스피',
            value: 2443.59,
            change: -15.23,
            changePercent: -0.62,
            volume: '4,521억',
            updatedAt: new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
          },
          {
            symbol: 'KOSDAQ',
            name: '코스닥',
            value: 678.42,
            change: -8.91,
            changePercent: -1.30,
            volume: '7,892억',
            updatedAt: new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
          },
          {
            symbol: 'KOSPI200',
            name: '코스피200',
            value: 321.45,
            change: -2.78,
            changePercent: -0.86,
            updatedAt: new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
          }
        ];

        // 미국 지수는 시뮬레이션 데이터 사용 (2024년 12월 기준 실제 수치)
        const usIndices = [
          {
            symbol: 'DJI',
            name: '다우존스',
            value: 43828.06,
            change: 215.57,
            changePercent: 0.49,
            updatedAt: new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
          },
          {
            symbol: 'NASDAQ',
            name: '나스닥',
            value: 19926.72,
            change: -43.71,
            changePercent: -0.22,
            updatedAt: new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
          },
          {
            symbol: 'S&P500',
            name: 'S&P 500',
            value: 6051.09,
            change: 17.84,
            changePercent: 0.30,
            updatedAt: new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
          }
        ];
        setMarketIndices([...koreanIndices, ...usIndices]);

        // 상승/하락/거래량 상위 종목
        setTopGainers(marketData.gainers || []);
        setTopLosers(marketData.losers || []);
        setMostActive(marketData.mostActive || []);
        
        // 섹터별 현황
        setSectors(marketData.sectors || []);
      }

      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to fetch market data:', error);
      // 에러 시 기본 데이터 표시
      setMarketIndices(marketIndices);
      setTopGainers(topGainers);
      setTopLosers(topLosers);
      setMostActive(mostActive);
      setSectors(sectors);
    } finally {
      setLoading(false);
    }
  };

  // 초기 데이터 로드
  useEffect(() => {
    fetchMarketData();
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    await fetchMarketData(true); // 강제 새로고침
    setRefreshing(false);
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('ko-KR').format(num);
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link 
                href="/modules/stock-analysis"
                className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>주식 분석</span>
              </Link>
              <div className="h-6 w-px bg-gray-300 dark:bg-gray-700" />
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">시장 대시보드</h1>
              <span className="px-2 py-1 bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 rounded text-xs font-medium">
                실시간
              </span>
            </div>
            
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                <Clock className="w-4 h-4" />
                <span>마지막 업데이트: {lastUpdate.toLocaleTimeString('ko-KR')}</span>
                {cacheInfo.cached && (
                  <span className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded text-xs">
                    캐시됨
                  </span>
                )}
              </div>
              <button
                onClick={handleRefresh}
                disabled={refreshing}
                className={`p-2 rounded-lg transition-all ${
                  refreshing 
                    ? 'bg-gray-100 dark:bg-gray-700' 
                    : 'hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
                title="새로고침 (캐시 무시)"
              >
                <RefreshCw className={`w-5 h-5 ${refreshing ? 'animate-spin' : ''}`} />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          </div>
        ) : (
          <>
            {/* Market Indices */}
            <div className="mb-8">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">주요 지수</h2>
          <div className={`grid gap-4 ${
            selectedTab === 'domestic' 
              ? 'grid-cols-1 md:grid-cols-3' 
              : 'grid-cols-1 md:grid-cols-3'
          }`}>
            {marketIndices
              .filter(index => {
                if (selectedTab === 'domestic') {
                  return ['KOSPI', 'KOSDAQ', 'KOSPI200'].includes(index.symbol);
                } else {
                  return ['DJI', 'NASDAQ', 'S&P500'].includes(index.symbol);
                }
              })
              .map((index) => (
              <div key={index.symbol} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-4">
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-medium text-gray-900 dark:text-white">{index.name}</h3>
                  {index.symbol.includes('KOS') ? (
                    <Activity className="w-4 h-4 text-blue-500" />
                  ) : (
                    <Globe className="w-4 h-4 text-green-500" />
                  )}
                </div>
                <p className="text-xl font-bold text-gray-900 dark:text-white mb-1">
                  {formatNumber(index.value)}
                </p>
                <div className={`flex items-center gap-1 text-sm ${
                  index.change >= 0 ? 'text-red-600' : 'text-blue-600'
                }`}>
                  {index.change >= 0 ? (
                    <TrendingUp className="w-4 h-4" />
                  ) : (
                    <TrendingDown className="w-4 h-4" />
                  )}
                  <span className="font-medium">
                    {index.change >= 0 ? '+' : ''}{formatNumber(index.change)} ({index.changePercent >= 0 ? '+' : ''}{index.changePercent.toFixed(2)}%)
                  </span>
                </div>
                {index.volume && (
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                    거래량: {index.volume}
                  </p>
                )}
                <p className="text-xs text-gray-400 mt-1">
                  {index.updatedAt}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-4 mb-6">
          <button
            onClick={() => setSelectedTab('domestic')}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              selectedTab === 'domestic'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            국내 시장
          </button>
          <button
            onClick={() => setSelectedTab('global')}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              selectedTab === 'global'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            해외 시장
          </button>
        </div>

        {selectedTab === 'domestic' ? (
          <div className="grid lg:grid-cols-3 gap-6">
            {/* Top Gainers */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">상승률 상위</h3>
                <TrendingUp className="w-5 h-5 text-red-500" />
              </div>
              <div className="space-y-3">
                {topGainers.map((stock, index) => (
                  <div key={stock.symbol} className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-medium text-gray-500 w-4">{index + 1}</span>
                      <div>
                        <p className="font-medium text-gray-900 dark:text-white">{stock.name}</p>
                        <p className="text-xs text-gray-500">{stock.symbol}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-medium text-gray-900 dark:text-white">
                        {formatNumber(stock.price)}원
                      </p>
                      <p className="text-sm text-red-600 font-medium">
                        +{stock.changePercent.toFixed(2)}%
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Top Losers */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">하락률 상위</h3>
                <TrendingDown className="w-5 h-5 text-blue-500" />
              </div>
              <div className="space-y-3">
                {topLosers.map((stock, index) => (
                  <div key={stock.symbol} className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-medium text-gray-500 w-4">{index + 1}</span>
                      <div>
                        <p className="font-medium text-gray-900 dark:text-white">{stock.name}</p>
                        <p className="text-xs text-gray-500">{stock.symbol}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-medium text-gray-900 dark:text-white">
                        {formatNumber(stock.price)}원
                      </p>
                      <p className="text-sm text-blue-600 font-medium">
                        {stock.changePercent.toFixed(2)}%
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Most Active */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">거래량 상위</h3>
                <BarChart3 className="w-5 h-5 text-green-500" />
              </div>
              <div className="space-y-3">
                {mostActive.map((stock, index) => (
                  <div key={stock.symbol} className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-medium text-gray-500 w-4">{index + 1}</span>
                      <div>
                        <p className="font-medium text-gray-900 dark:text-white">{stock.name}</p>
                        <p className="text-xs text-gray-500">{stock.symbol}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-900 dark:text-white">
                        {stock.volume}주
                      </p>
                      <p className="text-xs text-gray-500">
                        시총 {stock.marketCap}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">주요 해외 지수</h2>
              <Globe className="w-5 h-5 text-blue-500" />
            </div>
            
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {/* 미국 지수 */}
              <div>
                <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-3">미국</h3>
                <div className="space-y-3">
                  {marketIndices.filter(idx => ['DJI', 'NASDAQ', 'S&P500'].includes(idx.symbol)).map(index => (
                    <div key={index.symbol} className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-gray-900 dark:text-white">{index.name}</span>
                        <span className="text-xs text-gray-500">{index.updatedAt}</span>
                      </div>
                      <p className="text-lg font-bold text-gray-900 dark:text-white">
                        {formatNumber(index.value)}
                      </p>
                      <p className={`text-sm font-medium ${
                        index.change >= 0 ? 'text-red-600' : 'text-blue-600'
                      }`}>
                        {index.change >= 0 ? '+' : ''}{formatNumber(index.change)} ({index.change >= 0 ? '+' : ''}{index.changePercent.toFixed(2)}%)
                      </p>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* 유럽 지수 (시뮬레이션) */}
              <div>
                <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-3">유럽</h3>
                <div className="space-y-3">
                  <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-gray-900 dark:text-white">FTSE 100</span>
                      <span className="text-xs text-gray-500">23:30</span>
                    </div>
                    <p className="text-lg font-bold text-gray-900 dark:text-white">7,642.72</p>
                    <p className="text-sm font-medium text-red-600">+28.45 (+0.37%)</p>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-gray-900 dark:text-white">DAX</span>
                      <span className="text-xs text-gray-500">01:30</span>
                    </div>
                    <p className="text-lg font-bold text-gray-900 dark:text-white">15,832.89</p>
                    <p className="text-sm font-medium text-blue-600">-45.21 (-0.28%)</p>
                  </div>
                </div>
              </div>
              
              {/* 아시아 지수 (시뮬레이션) */}
              <div>
                <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-3">아시아</h3>
                <div className="space-y-3">
                  <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-gray-900 dark:text-white">닛케이225</span>
                      <span className="text-xs text-gray-500">14:00</span>
                    </div>
                    <p className="text-lg font-bold text-gray-900 dark:text-white">32,156.89</p>
                    <p className="text-sm font-medium text-red-600">+156.23 (+0.49%)</p>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-gray-900 dark:text-white">항셍</span>
                      <span className="text-xs text-gray-500">16:00</span>
                    </div>
                    <p className="text-lg font-bold text-gray-900 dark:text-white">17,823.45</p>
                    <p className="text-sm font-medium text-blue-600">-234.56 (-1.30%)</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <p className="text-sm text-blue-700 dark:text-blue-300">
                <AlertCircle className="inline w-4 h-4 mr-1" />
                해외 지수는 현지 시간 기준으로 표시되며, 실시간 데이터는 15-20분 지연될 수 있습니다.
              </p>
            </div>
          </div>
        )}

        {/* Sector Performance */}
        <div className="mt-8">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">섹터별 현황</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {sectors.map((sector) => (
              <div key={sector.name} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-medium text-gray-900 dark:text-white">{sector.name}</h3>
                  <span className={`text-sm font-medium ${
                    sector.changePercent >= 0 ? 'text-red-600' : 'text-blue-600'
                  }`}>
                    {sector.changePercent >= 0 ? '+' : ''}{sector.changePercent.toFixed(2)}%
                  </span>
                </div>
                <div className="space-y-2">
                  {sector.leaders.map((leader) => (
                    <div key={leader.symbol} className="flex items-center justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">{leader.name}</span>
                      <span className={`font-medium ${
                        leader.changePercent >= 0 ? 'text-red-600' : 'text-blue-600'
                      }`}>
                        {leader.changePercent >= 0 ? '+' : ''}{leader.changePercent.toFixed(2)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Market Status */}
        <div className="mt-8 bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">시장 공지</h3>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                한국 증시는 평일 09:00 ~ 15:30 (KST) 거래됩니다. 
                현재는 시뮬레이션 데이터를 표시하고 있습니다.
                실제 데이터 연동을 위해서는 한국거래소(KRX) API 또는 증권사 API 연동이 필요합니다.
              </p>
            </div>
          </div>
        </div>

        {/* Cache Info */}
        {cacheInfo.cacheStats && (
          <div className="mt-6 bg-gray-100 dark:bg-gray-800 rounded-xl p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-green-600 dark:text-green-400" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">캐시 상태</span>
              </div>
              <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
                <span>캐시된 항목: {cacheInfo.cacheStats.size}개</span>
                <span>API 호출: {cacheInfo.cacheStats.requestCount}/분</span>
                <span className={`px-2 py-0.5 rounded text-xs ${
                  cacheInfo.cacheStats.isMarketOpen 
                    ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' 
                    : 'bg-gray-200 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
                }`}>
                  {cacheInfo.cacheStats.isMarketOpen ? '장중' : '장마감'}
                </span>
              </div>
            </div>
          </div>
        )}
          </>
        )}
      </div>
    </div>
  );
}