'use client';

import React, { useState, useEffect } from 'react';
import { Search, Filter, Download, Star, TrendingUp, DollarSign, BarChart3, Activity, AlertCircle, ChevronDown } from 'lucide-react';

interface ScreeningCriteria {
  // 기본 필터
  exchange: string[];
  marketCap: { min: number; max: number };
  price: { min: number; max: number };
  volume: { min: number; max: number };
  
  // 기본적 분석
  peRatio: { min: number; max: number };
  pegRatio: { min: number; max: number };
  pbRatio: { min: number; max: number };
  psRatio: { min: number; max: number };
  dividendYield: { min: number; max: number };
  
  // 수익성 지표
  roe: { min: number; max: number };
  roa: { min: number; max: number };
  profitMargin: { min: number; max: number };
  revenueGrowth: { min: number; max: number };
  epsGrowth: { min: number; max: number };
  
  // 기술적 지표
  rsi: { min: number; max: number };
  sma20Above: boolean;
  sma50Above: boolean;
  sma200Above: boolean;
  priceChange1M: { min: number; max: number };
  priceChange3M: { min: number; max: number };
}

interface StockResult {
  symbol: string;
  name: string;
  exchange: string;
  sector: string;
  price: number;
  change: number;
  changePercent: number;
  marketCap: number;
  volume: number;
  avgVolume: number;
  peRatio: number;
  dividendYield: number;
  beta: number;
  eps: number;
  revenue: number;
  revenueGrowth: number;
  roe: number;
  rsi: number;
  sma20: number;
  sma50: number;
  sma200: number;
}

// 모의 데이터
const mockStocks: StockResult[] = [
  {
    symbol: 'AAPL',
    name: 'Apple Inc.',
    exchange: 'NASDAQ',
    sector: 'Technology',
    price: 195.89,
    change: 2.45,
    changePercent: 1.27,
    marketCap: 3.04e12,
    volume: 52345678,
    avgVolume: 48567890,
    peRatio: 32.45,
    dividendYield: 0.44,
    beta: 1.25,
    eps: 6.04,
    revenue: 383.29e9,
    revenueGrowth: 2.4,
    roe: 145.2,
    rsi: 58.5,
    sma20: 193.45,
    sma50: 189.23,
    sma200: 178.56
  },
  {
    symbol: 'MSFT',
    name: 'Microsoft Corporation',
    exchange: 'NASDAQ',
    sector: 'Technology',
    price: 429.87,
    change: 5.23,
    changePercent: 1.23,
    marketCap: 3.19e12,
    volume: 23456789,
    avgVolume: 25678901,
    peRatio: 37.23,
    dividendYield: 0.68,
    beta: 0.92,
    eps: 11.54,
    revenue: 211.92e9,
    revenueGrowth: 12.5,
    roe: 43.7,
    rsi: 62.3,
    sma20: 425.67,
    sma50: 418.90,
    sma200: 385.45
  },
  {
    symbol: 'JPM',
    name: 'JPMorgan Chase & Co.',
    exchange: 'NYSE',
    sector: 'Financial',
    price: 208.56,
    change: -1.23,
    changePercent: -0.59,
    marketCap: 595.67e9,
    volume: 8765432,
    avgVolume: 9876543,
    peRatio: 11.85,
    dividendYield: 2.23,
    beta: 1.12,
    eps: 17.59,
    revenue: 158.10e9,
    revenueGrowth: 22.8,
    roe: 15.8,
    rsi: 45.2,
    sma20: 210.34,
    sma50: 205.78,
    sma200: 195.23
  },
  {
    symbol: 'XOM',
    name: 'Exxon Mobil Corporation',
    exchange: 'NYSE',
    sector: 'Energy',
    price: 104.32,
    change: -2.45,
    changePercent: -2.29,
    marketCap: 415.23e9,
    volume: 15678901,
    avgVolume: 18765432,
    peRatio: 8.92,
    dividendYield: 3.31,
    beta: 1.35,
    eps: 11.69,
    revenue: 413.68e9,
    revenueGrowth: -12.4,
    roe: 29.7,
    rsi: 38.7,
    sma20: 107.89,
    sma50: 109.45,
    sma200: 105.67
  }
];

// 섹터 목록
const sectors = [
  'Technology', 'Healthcare', 'Financial', 'Consumer Discretionary', 
  'Consumer Staples', 'Energy', 'Materials', 'Industrials', 
  'Real Estate', 'Utilities', 'Communication Services'
];

// 프리셋 전략
const presetStrategies = [
  {
    name: '가치주 발굴',
    description: 'PE < 15, PB < 2, 배당수익률 > 2%',
    criteria: {
      peRatio: { min: 0, max: 15 },
      pbRatio: { min: 0, max: 2 },
      dividendYield: { min: 2, max: 100 }
    }
  },
  {
    name: '성장주 탐색',
    description: '매출 성장률 > 15%, EPS 성장률 > 20%',
    criteria: {
      revenueGrowth: { min: 15, max: 100 },
      epsGrowth: { min: 20, max: 100 }
    }
  },
  {
    name: '우량 배당주',
    description: '배당수익률 > 3%, ROE > 15%, 시가총액 > $10B',
    criteria: {
      dividendYield: { min: 3, max: 100 },
      roe: { min: 15, max: 100 },
      marketCap: { min: 10e9, max: 10e12 }
    }
  },
  {
    name: '모멘텀 추종',
    description: 'RSI > 50, 20일선 위, 1개월 수익률 > 5%',
    criteria: {
      rsi: { min: 50, max: 70 },
      sma20Above: true,
      priceChange1M: { min: 5, max: 100 }
    }
  }
];

export default function USStockScreener() {
  const [criteria, setCriteria] = useState<ScreeningCriteria>({
    exchange: ['NYSE', 'NASDAQ'],
    marketCap: { min: 0, max: 10e12 },
    price: { min: 0, max: 10000 },
    volume: { min: 0, max: 1e9 },
    peRatio: { min: 0, max: 100 },
    pegRatio: { min: 0, max: 5 },
    pbRatio: { min: 0, max: 10 },
    psRatio: { min: 0, max: 20 },
    dividendYield: { min: 0, max: 10 },
    roe: { min: -100, max: 100 },
    roa: { min: -100, max: 100 },
    profitMargin: { min: -100, max: 100 },
    revenueGrowth: { min: -50, max: 100 },
    epsGrowth: { min: -50, max: 100 },
    rsi: { min: 0, max: 100 },
    sma20Above: false,
    sma50Above: false,
    sma200Above: false,
    priceChange1M: { min: -100, max: 100 },
    priceChange3M: { min: -100, max: 100 }
  });

  const [results, setResults] = useState<StockResult[]>(mockStocks);
  const [sortBy, setSortBy] = useState('marketCap');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [selectedSectors, setSelectedSectors] = useState<string[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [watchlist, setWatchlist] = useState<string[]>([]);

  // 스크리닝 실행
  const runScreening = () => {
    // 실제로는 API 호출, 여기서는 모의 필터링
    const filtered = mockStocks.filter(stock => {
      if (selectedSectors.length > 0 && !selectedSectors.includes(stock.sector)) return false;
      if (stock.price < criteria.price.min || stock.price > criteria.price.max) return false;
      if (stock.peRatio < criteria.peRatio.min || stock.peRatio > criteria.peRatio.max) return false;
      if (stock.dividendYield < criteria.dividendYield.min || stock.dividendYield > criteria.dividendYield.max) return false;
      if (stock.roe < criteria.roe.min || stock.roe > criteria.roe.max) return false;
      if (stock.rsi < criteria.rsi.min || stock.rsi > criteria.rsi.max) return false;
      if (criteria.sma20Above && stock.price < stock.sma20) return false;
      return true;
    });

    setResults(filtered);
  };

  // 프리셋 적용
  const applyPreset = (preset: any) => {
    setCriteria({
      ...criteria,
      ...preset.criteria
    });
  };

  // 정렬
  const handleSort = (field: string) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  // 정렬된 결과
  const sortedResults = [...results].sort((a, b) => {
    const aVal = a[sortBy as keyof StockResult];
    const bVal = b[sortBy as keyof StockResult];
    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return sortOrder === 'asc' ? aVal - bVal : bVal - aVal;
    }
    return 0;
  });

  // 관심종목 토글
  const toggleWatchlist = (symbol: string) => {
    setWatchlist(prev => 
      prev.includes(symbol) 
        ? prev.filter(s => s !== symbol)
        : [...prev, symbol]
    );
  };

  // 시가총액 포맷
  const formatMarketCap = (value: number) => {
    if (value >= 1e12) return `$${(value / 1e12).toFixed(1)}T`;
    if (value >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
    return `$${value.toFixed(0)}`;
  };

  // 거래량 포맷
  const formatVolume = (value: number) => {
    if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
    return value.toString();
  };

  return (
    <div className="space-y-6">
      {/* 프리셋 전략 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4">빠른 스크리닝 전략</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {presetStrategies.map((preset) => (
            <button
              key={preset.name}
              onClick={() => applyPreset(preset)}
              className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors text-left"
            >
              <h4 className="font-medium mb-1">{preset.name}</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">{preset.description}</p>
            </button>
          ))}
        </div>
      </div>

      {/* 필터 섹션 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Filter className="w-5 h-5" />
            스크리닝 조건
          </h3>
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm text-blue-600 dark:text-blue-400 hover:underline"
          >
            {showAdvanced ? '간단히 보기' : '고급 필터'}
            <ChevronDown className={`w-4 h-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
          </button>
        </div>

        {/* 기본 필터 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* 거래소 선택 */}
          <div>
            <label className="block text-sm font-medium mb-2">거래소</label>
            <div className="flex gap-2">
              {['NYSE', 'NASDAQ', 'AMEX'].map((exchange) => (
                <label key={exchange} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={criteria.exchange.includes(exchange)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setCriteria({ ...criteria, exchange: [...criteria.exchange, exchange] });
                      } else {
                        setCriteria({ ...criteria, exchange: criteria.exchange.filter(ex => ex !== exchange) });
                      }
                    }}
                    className="rounded border-gray-300 dark:border-gray-600"
                  />
                  <span className="text-sm">{exchange}</span>
                </label>
              ))}
            </div>
          </div>

          {/* 시가총액 */}
          <div>
            <label className="block text-sm font-medium mb-2">시가총액</label>
            <div className="flex gap-2">
              <select
                value={criteria.marketCap.min}
                onChange={(e) => setCriteria({ ...criteria, marketCap: { ...criteria.marketCap, min: Number(e.target.value) } })}
                className="flex-1 px-3 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
              >
                <option value="0">최소</option>
                <option value="1e6">$1M</option>
                <option value="10e6">$10M</option>
                <option value="100e6">$100M</option>
                <option value="1e9">$1B</option>
                <option value="10e9">$10B</option>
                <option value="100e9">$100B</option>
              </select>
              <span className="self-center">~</span>
              <select
                value={criteria.marketCap.max}
                onChange={(e) => setCriteria({ ...criteria, marketCap: { ...criteria.marketCap, max: Number(e.target.value) } })}
                className="flex-1 px-3 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
              >
                <option value="10e12">최대</option>
                <option value="1e12">$1T</option>
                <option value="500e9">$500B</option>
                <option value="100e9">$100B</option>
                <option value="10e9">$10B</option>
                <option value="1e9">$1B</option>
              </select>
            </div>
          </div>

          {/* 주가 범위 */}
          <div>
            <label className="block text-sm font-medium mb-2">주가</label>
            <div className="flex gap-2 items-center">
              <input
                type="number"
                value={criteria.price.min}
                onChange={(e) => setCriteria({ ...criteria, price: { ...criteria.price, min: Number(e.target.value) } })}
                placeholder="$0"
                className="flex-1 px-3 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
              />
              <span>~</span>
              <input
                type="number"
                value={criteria.price.max}
                onChange={(e) => setCriteria({ ...criteria, price: { ...criteria.price, max: Number(e.target.value) } })}
                placeholder="$10000"
                className="flex-1 px-3 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
              />
            </div>
          </div>

          {/* PE Ratio */}
          <div>
            <label className="block text-sm font-medium mb-2">PE Ratio</label>
            <div className="flex gap-2 items-center">
              <input
                type="number"
                value={criteria.peRatio.min}
                onChange={(e) => setCriteria({ ...criteria, peRatio: { ...criteria.peRatio, min: Number(e.target.value) } })}
                placeholder="0"
                className="flex-1 px-3 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
              />
              <span>~</span>
              <input
                type="number"
                value={criteria.peRatio.max}
                onChange={(e) => setCriteria({ ...criteria, peRatio: { ...criteria.peRatio, max: Number(e.target.value) } })}
                placeholder="100"
                className="flex-1 px-3 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
              />
            </div>
          </div>

          {/* 배당수익률 */}
          <div>
            <label className="block text-sm font-medium mb-2">배당수익률 (%)</label>
            <div className="flex gap-2 items-center">
              <input
                type="number"
                value={criteria.dividendYield.min}
                onChange={(e) => setCriteria({ ...criteria, dividendYield: { ...criteria.dividendYield, min: Number(e.target.value) } })}
                placeholder="0"
                step="0.1"
                className="flex-1 px-3 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
              />
              <span>~</span>
              <input
                type="number"
                value={criteria.dividendYield.max}
                onChange={(e) => setCriteria({ ...criteria, dividendYield: { ...criteria.dividendYield, max: Number(e.target.value) } })}
                placeholder="10"
                step="0.1"
                className="flex-1 px-3 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
              />
            </div>
          </div>

          {/* ROE */}
          <div>
            <label className="block text-sm font-medium mb-2">ROE (%)</label>
            <div className="flex gap-2 items-center">
              <input
                type="number"
                value={criteria.roe.min}
                onChange={(e) => setCriteria({ ...criteria, roe: { ...criteria.roe, min: Number(e.target.value) } })}
                placeholder="-100"
                className="flex-1 px-3 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
              />
              <span>~</span>
              <input
                type="number"
                value={criteria.roe.max}
                onChange={(e) => setCriteria({ ...criteria, roe: { ...criteria.roe, max: Number(e.target.value) } })}
                placeholder="100"
                className="flex-1 px-3 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
              />
            </div>
          </div>
        </div>

        {/* 고급 필터 */}
        {showAdvanced && (
          <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
            <h4 className="font-medium mb-4">고급 필터</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* RSI */}
              <div>
                <label className="block text-sm font-medium mb-2">RSI</label>
                <div className="flex gap-2 items-center">
                  <input
                    type="number"
                    value={criteria.rsi.min}
                    onChange={(e) => setCriteria({ ...criteria, rsi: { ...criteria.rsi, min: Number(e.target.value) } })}
                    placeholder="0"
                    className="flex-1 px-3 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                  />
                  <span>~</span>
                  <input
                    type="number"
                    value={criteria.rsi.max}
                    onChange={(e) => setCriteria({ ...criteria, rsi: { ...criteria.rsi, max: Number(e.target.value) } })}
                    placeholder="100"
                    className="flex-1 px-3 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                  />
                </div>
              </div>

              {/* 이동평균선 */}
              <div>
                <label className="block text-sm font-medium mb-2">이동평균선</label>
                <div className="space-y-2">
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={criteria.sma20Above}
                      onChange={(e) => setCriteria({ ...criteria, sma20Above: e.target.checked })}
                      className="rounded border-gray-300 dark:border-gray-600"
                    />
                    <span className="text-sm">20일선 위</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={criteria.sma50Above}
                      onChange={(e) => setCriteria({ ...criteria, sma50Above: e.target.checked })}
                      className="rounded border-gray-300 dark:border-gray-600"
                    />
                    <span className="text-sm">50일선 위</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={criteria.sma200Above}
                      onChange={(e) => setCriteria({ ...criteria, sma200Above: e.target.checked })}
                      className="rounded border-gray-300 dark:border-gray-600"
                    />
                    <span className="text-sm">200일선 위</span>
                  </label>
                </div>
              </div>

              {/* 수익률 */}
              <div>
                <label className="block text-sm font-medium mb-2">1개월 수익률 (%)</label>
                <div className="flex gap-2 items-center">
                  <input
                    type="number"
                    value={criteria.priceChange1M.min}
                    onChange={(e) => setCriteria({ ...criteria, priceChange1M: { ...criteria.priceChange1M, min: Number(e.target.value) } })}
                    placeholder="-100"
                    className="flex-1 px-3 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                  />
                  <span>~</span>
                  <input
                    type="number"
                    value={criteria.priceChange1M.max}
                    onChange={(e) => setCriteria({ ...criteria, priceChange1M: { ...criteria.priceChange1M, max: Number(e.target.value) } })}
                    placeholder="100"
                    className="flex-1 px-3 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 섹터 필터 */}
        <div className="mt-6">
          <label className="block text-sm font-medium mb-2">섹터 선택</label>
          <div className="flex flex-wrap gap-2">
            {sectors.map((sector) => (
              <button
                key={sector}
                onClick={() => {
                  if (selectedSectors.includes(sector)) {
                    setSelectedSectors(prev => prev.filter(s => s !== sector));
                  } else {
                    setSelectedSectors(prev => [...prev, sector]);
                  }
                }}
                className={`px-3 py-1 rounded-full text-sm transition-colors ${
                  selectedSectors.includes(sector)
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                {sector}
              </button>
            ))}
          </div>
        </div>

        {/* 실행 버튼 */}
        <div className="mt-6 flex gap-3">
          <button
            onClick={runScreening}
            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center gap-2"
          >
            <Search className="w-5 h-5" />
            스크리닝 실행
          </button>
          <button
            onClick={() => {
              setCriteria({
                exchange: ['NYSE', 'NASDAQ'],
                marketCap: { min: 0, max: 10e12 },
                price: { min: 0, max: 10000 },
                volume: { min: 0, max: 1e9 },
                peRatio: { min: 0, max: 100 },
                pegRatio: { min: 0, max: 5 },
                pbRatio: { min: 0, max: 10 },
                psRatio: { min: 0, max: 20 },
                dividendYield: { min: 0, max: 10 },
                roe: { min: -100, max: 100 },
                roa: { min: -100, max: 100 },
                profitMargin: { min: -100, max: 100 },
                revenueGrowth: { min: -50, max: 100 },
                epsGrowth: { min: -50, max: 100 },
                rsi: { min: 0, max: 100 },
                sma20Above: false,
                sma50Above: false,
                sma200Above: false,
                priceChange1M: { min: -100, max: 100 },
                priceChange3M: { min: -100, max: 100 }
              });
              setSelectedSectors([]);
            }}
            className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
          >
            초기화
          </button>
        </div>
      </div>

      {/* 결과 테이블 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              스크리닝 결과 ({sortedResults.length}개)
            </h3>
            <div className="flex gap-2">
              <button className="px-4 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors flex items-center gap-2">
                <Download className="w-4 h-4" />
                Excel 다운로드
              </button>
              {watchlist.length > 0 && (
                <div className="px-4 py-2 bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 rounded-lg flex items-center gap-2">
                  <Star className="w-4 h-4" />
                  관심종목 {watchlist.length}개
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-sm text-gray-600 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                <th className="p-4 font-medium">
                  <button className="hover:text-gray-900 dark:hover:text-gray-100">★</button>
                </th>
                <th className="p-4 font-medium">
                  <button 
                    onClick={() => handleSort('symbol')}
                    className="hover:text-gray-900 dark:hover:text-gray-100"
                  >
                    종목
                  </button>
                </th>
                <th className="p-4 font-medium text-right">
                  <button 
                    onClick={() => handleSort('price')}
                    className="hover:text-gray-900 dark:hover:text-gray-100"
                  >
                    주가
                  </button>
                </th>
                <th className="p-4 font-medium text-right">
                  <button 
                    onClick={() => handleSort('changePercent')}
                    className="hover:text-gray-900 dark:hover:text-gray-100"
                  >
                    변동률
                  </button>
                </th>
                <th className="p-4 font-medium text-right">
                  <button 
                    onClick={() => handleSort('marketCap')}
                    className="hover:text-gray-900 dark:hover:text-gray-100"
                  >
                    시가총액
                  </button>
                </th>
                <th className="p-4 font-medium text-right">
                  <button 
                    onClick={() => handleSort('volume')}
                    className="hover:text-gray-900 dark:hover:text-gray-100"
                  >
                    거래량
                  </button>
                </th>
                <th className="p-4 font-medium text-right">
                  <button 
                    onClick={() => handleSort('peRatio')}
                    className="hover:text-gray-900 dark:hover:text-gray-100"
                  >
                    PE
                  </button>
                </th>
                <th className="p-4 font-medium text-right">
                  <button 
                    onClick={() => handleSort('dividendYield')}
                    className="hover:text-gray-900 dark:hover:text-gray-100"
                  >
                    배당
                  </button>
                </th>
                <th className="p-4 font-medium text-right">
                  <button 
                    onClick={() => handleSort('roe')}
                    className="hover:text-gray-900 dark:hover:text-gray-100"
                  >
                    ROE
                  </button>
                </th>
                <th className="p-4 font-medium text-right">
                  <button 
                    onClick={() => handleSort('rsi')}
                    className="hover:text-gray-900 dark:hover:text-gray-100"
                  >
                    RSI
                  </button>
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedResults.map((stock) => (
                <tr key={stock.symbol} className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-900">
                  <td className="p-4">
                    <button
                      onClick={() => toggleWatchlist(stock.symbol)}
                      className={`text-lg ${watchlist.includes(stock.symbol) ? 'text-yellow-500' : 'text-gray-400 hover:text-yellow-500'}`}
                    >
                      {watchlist.includes(stock.symbol) ? '★' : '☆'}
                    </button>
                  </td>
                  <td className="p-4">
                    <div>
                      <p className="font-medium">{stock.symbol}</p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">{stock.name}</p>
                      <p className="text-xs text-gray-400 dark:text-gray-500">{stock.exchange} · {stock.sector}</p>
                    </div>
                  </td>
                  <td className="p-4 text-right font-medium">${stock.price.toFixed(2)}</td>
                  <td className={`p-4 text-right ${stock.changePercent > 0 ? 'text-green-500' : 'text-red-500'}`}>
                    <div className="flex items-center justify-end gap-1">
                      {stock.changePercent > 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                      <span>{stock.changePercent > 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%</span>
                    </div>
                  </td>
                  <td className="p-4 text-right">{formatMarketCap(stock.marketCap)}</td>
                  <td className="p-4 text-right">
                    <div>
                      <p>{formatVolume(stock.volume)}</p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">평균 {formatVolume(stock.avgVolume)}</p>
                    </div>
                  </td>
                  <td className="p-4 text-right">{stock.peRatio.toFixed(2)}</td>
                  <td className="p-4 text-right">{stock.dividendYield.toFixed(2)}%</td>
                  <td className="p-4 text-right">{stock.roe.toFixed(1)}%</td>
                  <td className="p-4 text-right">
                    <span className={`px-2 py-1 rounded text-xs ${
                      stock.rsi > 70 ? 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300' :
                      stock.rsi < 30 ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300' :
                      'bg-gray-100 dark:bg-gray-700'
                    }`}>
                      {stock.rsi.toFixed(0)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {sortedResults.length === 0 && (
          <div className="p-12 text-center">
            <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600 dark:text-gray-400">조건에 맞는 종목이 없습니다.</p>
            <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">스크리닝 조건을 조정해보세요.</p>
          </div>
        )}
      </div>

      {/* 사용 안내 */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
        <h4 className="font-medium mb-2 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          미국 주식 스크리너 사용법
        </h4>
        <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
          <li>• 빠른 스크리닝 전략을 선택하거나 직접 조건을 설정하세요</li>
          <li>• 고급 필터에서 기술적 지표와 재무 지표를 세밀하게 조정할 수 있습니다</li>
          <li>• 섹터를 선택하여 특정 산업군에서만 검색할 수 있습니다</li>
          <li>• 별표를 클릭하여 관심종목에 추가하고 관리하세요</li>
          <li>• 결과를 Excel로 다운로드하여 추가 분석에 활용하세요</li>
        </ul>
      </div>
    </div>
  );
}