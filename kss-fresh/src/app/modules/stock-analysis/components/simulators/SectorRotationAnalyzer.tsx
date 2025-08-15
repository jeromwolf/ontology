'use client';

import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, BarChart3, Activity, Calendar, Globe, ArrowUpDown, Info } from 'lucide-react';

interface SectorData {
  name: string;
  code: string;
  performance: {
    day: number;
    week: number;
    month: number;
    quarter: number;
    year: number;
  };
  momentum: number; // 모멘텀 점수
  relativeStrength: number; // 상대강도
  volume: number;
  marketCap: number;
  peRatio: number;
  topStocks: Array<{
    symbol: string;
    name: string;
    weight: number;
    performance: number;
  }>;
}

interface EconomicPhase {
  name: string;
  description: string;
  favoredSectors: string[];
  avoidSectors: string[];
}

// 모의 섹터 데이터
const sectorData: SectorData[] = [
  {
    name: 'Technology',
    code: 'XLK',
    performance: {
      day: 1.23,
      week: 3.45,
      month: 8.92,
      quarter: 15.67,
      year: 42.34
    },
    momentum: 85,
    relativeStrength: 92,
    volume: 123456789,
    marketCap: 12.5e12,
    peRatio: 28.5,
    topStocks: [
      { symbol: 'AAPL', name: 'Apple', weight: 23.2, performance: 45.6 },
      { symbol: 'MSFT', name: 'Microsoft', weight: 21.8, performance: 38.9 },
      { symbol: 'NVDA', name: 'NVIDIA', weight: 8.9, performance: 156.2 }
    ]
  },
  {
    name: 'Healthcare',
    code: 'XLV',
    performance: {
      day: 0.45,
      week: 1.23,
      month: 3.45,
      quarter: 7.89,
      year: 12.34
    },
    momentum: 65,
    relativeStrength: 70,
    volume: 87654321,
    marketCap: 8.2e12,
    peRatio: 18.2,
    topStocks: [
      { symbol: 'UNH', name: 'UnitedHealth', weight: 10.5, performance: 15.2 },
      { symbol: 'JNJ', name: 'Johnson & Johnson', weight: 8.7, performance: 8.9 },
      { symbol: 'PFE', name: 'Pfizer', weight: 5.4, performance: -12.3 }
    ]
  },
  {
    name: 'Financials',
    code: 'XLF',
    performance: {
      day: 0.89,
      week: 2.34,
      month: 5.67,
      quarter: 12.34,
      year: 28.9
    },
    momentum: 78,
    relativeStrength: 82,
    volume: 98765432,
    marketCap: 7.8e12,
    peRatio: 14.5,
    topStocks: [
      { symbol: 'BRK.B', name: 'Berkshire Hathaway', weight: 14.2, performance: 22.3 },
      { symbol: 'JPM', name: 'JPMorgan Chase', weight: 10.8, performance: 35.6 },
      { symbol: 'BAC', name: 'Bank of America', weight: 6.9, performance: 42.1 }
    ]
  },
  {
    name: 'Energy',
    code: 'XLE',
    performance: {
      day: -1.23,
      week: -3.45,
      month: -5.67,
      quarter: -8.92,
      year: 45.67
    },
    momentum: 45,
    relativeStrength: 55,
    volume: 76543210,
    marketCap: 3.2e12,
    peRatio: 12.3,
    topStocks: [
      { symbol: 'XOM', name: 'Exxon Mobil', weight: 22.5, performance: 52.3 },
      { symbol: 'CVX', name: 'Chevron', weight: 18.9, performance: 48.7 },
      { symbol: 'COP', name: 'ConocoPhillips', weight: 7.8, performance: 38.9 }
    ]
  },
  {
    name: 'Consumer Discretionary',
    code: 'XLY',
    performance: {
      day: 1.56,
      week: 4.23,
      month: 7.89,
      quarter: 18.92,
      year: 35.67
    },
    momentum: 82,
    relativeStrength: 88,
    volume: 65432109,
    marketCap: 5.6e12,
    peRatio: 25.8,
    topStocks: [
      { symbol: 'AMZN', name: 'Amazon', weight: 35.2, performance: 42.1 },
      { symbol: 'TSLA', name: 'Tesla', weight: 16.8, performance: 28.9 },
      { symbol: 'HD', name: 'Home Depot', weight: 8.2, performance: 18.7 }
    ]
  },
  {
    name: 'Consumer Staples',
    code: 'XLP',
    performance: {
      day: -0.23,
      week: 0.45,
      month: 1.23,
      quarter: 3.45,
      year: 8.92
    },
    momentum: 55,
    relativeStrength: 60,
    volume: 54321098,
    marketCap: 2.8e12,
    peRatio: 20.5,
    topStocks: [
      { symbol: 'PG', name: 'Procter & Gamble', weight: 13.5, performance: 12.3 },
      { symbol: 'KO', name: 'Coca-Cola', weight: 10.2, performance: 6.8 },
      { symbol: 'PEP', name: 'PepsiCo', weight: 9.8, performance: 8.2 }
    ]
  },
  {
    name: 'Industrials',
    code: 'XLI',
    performance: {
      day: 0.67,
      week: 1.89,
      month: 4.56,
      quarter: 10.23,
      year: 22.34
    },
    momentum: 72,
    relativeStrength: 75,
    volume: 43210987,
    marketCap: 4.2e12,
    peRatio: 19.8,
    topStocks: [
      { symbol: 'BA', name: 'Boeing', weight: 6.8, performance: -5.2 },
      { symbol: 'CAT', name: 'Caterpillar', weight: 5.9, performance: 32.1 },
      { symbol: 'UPS', name: 'UPS', weight: 4.2, performance: 15.6 }
    ]
  },
  {
    name: 'Real Estate',
    code: 'XLRE',
    performance: {
      day: -0.89,
      week: -2.34,
      month: -3.45,
      quarter: -5.67,
      year: -12.34
    },
    momentum: 35,
    relativeStrength: 40,
    volume: 32109876,
    marketCap: 1.5e12,
    peRatio: 32.5,
    topStocks: [
      { symbol: 'PLD', name: 'Prologis', weight: 9.8, performance: -8.9 },
      { symbol: 'AMT', name: 'American Tower', weight: 7.2, performance: -15.2 },
      { symbol: 'EQIX', name: 'Equinix', weight: 5.6, performance: -10.5 }
    ]
  },
  {
    name: 'Materials',
    code: 'XLB',
    performance: {
      day: 0.34,
      week: 1.23,
      month: 2.89,
      quarter: 6.78,
      year: 15.67
    },
    momentum: 62,
    relativeStrength: 65,
    volume: 21098765,
    marketCap: 1.2e12,
    peRatio: 16.2,
    topStocks: [
      { symbol: 'LIN', name: 'Linde', weight: 18.5, performance: 22.3 },
      { symbol: 'APD', name: 'Air Products', weight: 8.2, performance: 18.9 },
      { symbol: 'SHW', name: 'Sherwin-Williams', weight: 6.9, performance: 12.5 }
    ]
  },
  {
    name: 'Utilities',
    code: 'XLU',
    performance: {
      day: -0.12,
      week: -0.56,
      month: -1.23,
      quarter: -2.34,
      year: -5.67
    },
    momentum: 42,
    relativeStrength: 45,
    volume: 10987654,
    marketCap: 0.9e12,
    peRatio: 17.8,
    topStocks: [
      { symbol: 'NEE', name: 'NextEra Energy', weight: 14.2, performance: -3.2 },
      { symbol: 'SO', name: 'Southern Company', weight: 8.5, performance: -7.8 },
      { symbol: 'DUK', name: 'Duke Energy', weight: 7.8, performance: -6.5 }
    ]
  },
  {
    name: 'Communication Services',
    code: 'XLC',
    performance: {
      day: 2.34,
      week: 5.67,
      month: 12.34,
      quarter: 25.67,
      year: 56.78
    },
    momentum: 90,
    relativeStrength: 95,
    volume: 87654321,
    marketCap: 4.5e12,
    peRatio: 22.3,
    topStocks: [
      { symbol: 'META', name: 'Meta Platforms', weight: 22.8, performance: 156.2 },
      { symbol: 'GOOGL', name: 'Alphabet', weight: 20.5, performance: 45.8 },
      { symbol: 'NFLX', name: 'Netflix', weight: 5.2, performance: 62.3 }
    ]
  }
];

// 경제 사이클 단계
const economicPhases: EconomicPhase[] = [
  {
    name: '경기 회복기',
    description: '경제가 바닥을 찍고 회복하기 시작하는 시기',
    favoredSectors: ['Financials', 'Consumer Discretionary', 'Technology', 'Industrials'],
    avoidSectors: ['Consumer Staples', 'Utilities', 'Healthcare']
  },
  {
    name: '경기 확장기',
    description: '경제가 활발히 성장하는 시기',
    favoredSectors: ['Technology', 'Industrials', 'Materials', 'Energy'],
    avoidSectors: ['Utilities', 'Consumer Staples', 'Real Estate']
  },
  {
    name: '경기 둔화기',
    description: '성장이 정점을 찍고 둔화되기 시작하는 시기',
    favoredSectors: ['Consumer Staples', 'Healthcare', 'Utilities'],
    avoidSectors: ['Technology', 'Consumer Discretionary', 'Financials']
  },
  {
    name: '경기 침체기',
    description: '경제가 수축하는 시기',
    favoredSectors: ['Consumer Staples', 'Healthcare', 'Utilities'],
    avoidSectors: ['Financials', 'Materials', 'Energy', 'Industrials']
  }
];

export default function SectorRotationAnalyzer() {
  const [selectedPhase, setSelectedPhase] = useState(0);
  const [timeFrame, setTimeFrame] = useState<'month' | 'quarter' | 'year'>('quarter');
  const [viewMode, setViewMode] = useState<'heatmap' | 'table' | 'flow'>('heatmap');
  const [showDetails, setShowDetails] = useState(false);
  
  // 섹터 정렬
  const sortedSectors = [...sectorData].sort((a, b) => 
    b.performance[timeFrame] - a.performance[timeFrame]
  );
  
  // 색상 계산
  const getHeatmapColor = (value: number) => {
    if (value > 20) return 'bg-green-600';
    if (value > 10) return 'bg-green-500';
    if (value > 5) return 'bg-green-400';
    if (value > 0) return 'bg-green-300';
    if (value > -5) return 'bg-red-300';
    if (value > -10) return 'bg-red-400';
    if (value > -20) return 'bg-red-500';
    return 'bg-red-600';
  };
  
  const getMomentumColor = (value: number) => {
    if (value >= 80) return 'text-green-600';
    if (value >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };
  
  // 추천 섹터 확인
  const isRecommendedSector = (sectorName: string) => {
    return economicPhases[selectedPhase].favoredSectors.includes(sectorName);
  };
  
  const isAvoidSector = (sectorName: string) => {
    return economicPhases[selectedPhase].avoidSectors.includes(sectorName);
  };

  return (
    <div className="space-y-6">
      {/* 경제 사이클 선택 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Calendar className="w-5 h-5" />
          현재 경제 사이클 단계
        </h3>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {economicPhases.map((phase, index) => (
            <button
              key={phase.name}
              onClick={() => setSelectedPhase(index)}
              className={`p-4 rounded-lg transition-all ${
                selectedPhase === index
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              <h4 className="font-medium mb-1">{phase.name}</h4>
              <p className="text-xs opacity-80">{phase.description}</p>
            </button>
          ))}
        </div>
        
        <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <h4 className="font-medium mb-2">추천 섹터 전략</h4>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <p className="text-sm font-medium text-green-600 dark:text-green-400 mb-1">선호 섹터</p>
              <div className="flex flex-wrap gap-2">
                {economicPhases[selectedPhase].favoredSectors.map(sector => (
                  <span key={sector} className="px-2 py-1 bg-green-100 dark:bg-green-900 rounded text-xs">
                    {sector}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <p className="text-sm font-medium text-red-600 dark:text-red-400 mb-1">회피 섹터</p>
              <div className="flex flex-wrap gap-2">
                {economicPhases[selectedPhase].avoidSectors.map(sector => (
                  <span key={sector} className="px-2 py-1 bg-red-100 dark:bg-red-900 rounded text-xs">
                    {sector}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 뷰 모드 & 기간 선택 */}
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1 bg-white dark:bg-gray-800 rounded-lg p-2 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex gap-2">
            <button
              onClick={() => setViewMode('heatmap')}
              className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                viewMode === 'heatmap'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700'
              }`}
            >
              히트맵
            </button>
            <button
              onClick={() => setViewMode('table')}
              className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                viewMode === 'table'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700'
              }`}
            >
              테이블
            </button>
            <button
              onClick={() => setViewMode('flow')}
              className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                viewMode === 'flow'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700'
              }`}
            >
              자금 흐름
            </button>
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-2 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex gap-2">
            <button
              onClick={() => setTimeFrame('month')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                timeFrame === 'month'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700'
              }`}
            >
              1개월
            </button>
            <button
              onClick={() => setTimeFrame('quarter')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                timeFrame === 'quarter'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700'
              }`}
            >
              3개월
            </button>
            <button
              onClick={() => setTimeFrame('year')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                timeFrame === 'year'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700'
              }`}
            >
              1년
            </button>
          </div>
        </div>
      </div>

      {/* 히트맵 뷰 */}
      {viewMode === 'heatmap' && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-4">섹터별 성과 히트맵</h3>
          
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            {sortedSectors.map((sector) => (
              <div
                key={sector.code}
                className={`relative ${getHeatmapColor(sector.performance[timeFrame])} p-4 rounded-lg text-white cursor-pointer transform transition-transform hover:scale-105`}
                onClick={() => setShowDetails(!showDetails)}
              >
                {isRecommendedSector(sector.name) && (
                  <span className="absolute top-2 right-2 text-yellow-300">★</span>
                )}
                {isAvoidSector(sector.name) && (
                  <span className="absolute top-2 right-2 text-red-300">⚠</span>
                )}
                <p className="font-medium mb-1">{sector.name}</p>
                <p className="text-xs opacity-90 mb-2">{sector.code}</p>
                <p className="text-2xl font-bold">
                  {sector.performance[timeFrame] > 0 ? '+' : ''}{sector.performance[timeFrame].toFixed(1)}%
                </p>
                <div className="flex items-center gap-2 mt-2 text-xs">
                  <span>모멘텀: {sector.momentum}</span>
                  <span>RS: {sector.relativeStrength}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 테이블 뷰 */}
      {viewMode === 'table' && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="p-6">
            <h3 className="text-lg font-semibold mb-4">섹터별 상세 분석</h3>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-sm text-gray-600 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                  <th className="px-6 pb-3 font-medium">섹터</th>
                  <th className="px-6 pb-3 font-medium text-right">1일</th>
                  <th className="px-6 pb-3 font-medium text-right">1주</th>
                  <th className="px-6 pb-3 font-medium text-right">1개월</th>
                  <th className="px-6 pb-3 font-medium text-right">3개월</th>
                  <th className="px-6 pb-3 font-medium text-right">1년</th>
                  <th className="px-6 pb-3 font-medium text-right">모멘텀</th>
                  <th className="px-6 pb-3 font-medium text-right">RS</th>
                  <th className="px-6 pb-3 font-medium text-right">P/E</th>
                </tr>
              </thead>
              <tbody>
                {sortedSectors.map((sector) => (
                  <tr key={sector.code} className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-900">
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{sector.name}</span>
                        {isRecommendedSector(sector.name) && (
                          <span className="text-green-500">★</span>
                        )}
                        {isAvoidSector(sector.name) && (
                          <span className="text-red-500">⚠</span>
                        )}
                      </div>
                      <span className="text-sm text-gray-500">{sector.code}</span>
                    </td>
                    <td className={`px-6 py-4 text-right font-medium ${sector.performance.day > 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {sector.performance.day > 0 ? '+' : ''}{sector.performance.day.toFixed(2)}%
                    </td>
                    <td className={`px-6 py-4 text-right font-medium ${sector.performance.week > 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {sector.performance.week > 0 ? '+' : ''}{sector.performance.week.toFixed(2)}%
                    </td>
                    <td className={`px-6 py-4 text-right font-medium ${sector.performance.month > 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {sector.performance.month > 0 ? '+' : ''}{sector.performance.month.toFixed(2)}%
                    </td>
                    <td className={`px-6 py-4 text-right font-medium ${sector.performance.quarter > 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {sector.performance.quarter > 0 ? '+' : ''}{sector.performance.quarter.toFixed(2)}%
                    </td>
                    <td className={`px-6 py-4 text-right font-medium ${sector.performance.year > 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {sector.performance.year > 0 ? '+' : ''}{sector.performance.year.toFixed(2)}%
                    </td>
                    <td className={`px-6 py-4 text-right font-medium ${getMomentumColor(sector.momentum)}`}>
                      {sector.momentum}
                    </td>
                    <td className={`px-6 py-4 text-right font-medium ${getMomentumColor(sector.relativeStrength)}`}>
                      {sector.relativeStrength}
                    </td>
                    <td className="px-6 py-4 text-right">
                      {sector.peRatio.toFixed(1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 자금 흐름 뷰 */}
      {viewMode === 'flow' && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-4">섹터별 자금 흐름</h3>
          
          <div className="space-y-4">
            {/* 자금 유입 섹터 */}
            <div>
              <h4 className="font-medium mb-3 text-green-600 dark:text-green-400">
                자금 유입 섹터 (강세)
              </h4>
              <div className="space-y-2">
                {sortedSectors
                  .filter(s => s.performance[timeFrame] > 0)
                  .slice(0, 5)
                  .map((sector) => (
                    <div key={sector.code} className="flex items-center gap-4">
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-medium">{sector.name}</span>
                          <span className="text-green-600">+{sector.performance[timeFrame].toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4">
                          <div
                            className="bg-green-500 h-4 rounded-full flex items-center justify-end pr-2"
                            style={{ width: `${Math.min(100, sector.performance[timeFrame] * 2)}%` }}
                          >
                            <TrendingUp className="w-3 h-3 text-white" />
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium">${(sector.volume / 1e6).toFixed(0)}M</p>
                        <p className="text-xs text-gray-500">거래량</p>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
            
            {/* 자금 유출 섹터 */}
            <div>
              <h4 className="font-medium mb-3 text-red-600 dark:text-red-400">
                자금 유출 섹터 (약세)
              </h4>
              <div className="space-y-2">
                {sortedSectors
                  .filter(s => s.performance[timeFrame] < 0)
                  .slice(-5)
                  .reverse()
                  .map((sector) => (
                    <div key={sector.code} className="flex items-center gap-4">
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-medium">{sector.name}</span>
                          <span className="text-red-600">{sector.performance[timeFrame].toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4">
                          <div
                            className="bg-red-500 h-4 rounded-full flex items-center justify-end pr-2"
                            style={{ width: `${Math.min(100, Math.abs(sector.performance[timeFrame]) * 2)}%` }}
                          >
                            <TrendingDown className="w-3 h-3 text-white" />
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium">${(sector.volume / 1e6).toFixed(0)}M</p>
                        <p className="text-xs text-gray-500">거래량</p>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 상위 종목 분석 */}
      {showDetails && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-4">섹터별 상위 종목</h3>
          
          <div className="grid md:grid-cols-2 gap-4">
            {sortedSectors.slice(0, 4).map((sector) => (
              <div key={sector.code} className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <h4 className="font-medium mb-3 flex items-center justify-between">
                  <span>{sector.name}</span>
                  <span className="text-sm text-gray-500">{sector.code}</span>
                </h4>
                <div className="space-y-2">
                  {sector.topStocks.map((stock) => (
                    <div key={stock.symbol} className="flex items-center justify-between">
                      <div>
                        <span className="font-medium">{stock.symbol}</span>
                        <span className="text-sm text-gray-500 ml-2">{stock.name}</span>
                      </div>
                      <div className="text-right">
                        <span className={`font-medium ${stock.performance > 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {stock.performance > 0 ? '+' : ''}{stock.performance.toFixed(1)}%
                        </span>
                        <span className="text-xs text-gray-500 ml-2">({stock.weight.toFixed(1)}%)</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 섹터 로테이션 전략 가이드 */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Info className="w-5 h-5" />
          섹터 로테이션 전략 가이드
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-3">섹터 로테이션이란?</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              경제 사이클의 각 단계에서 특정 섹터가 다른 섹터보다 더 나은 성과를 보이는 경향을 활용하는 투자 전략입니다.
            </p>
            <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
              <li>• 경기 회복기: 금융, 부동산 섹터 선호</li>
              <li>• 경기 확장기: 기술, 산업재 섹터 강세</li>
              <li>• 경기 둔화기: 필수소비재, 헬스케어 방어</li>
              <li>• 경기 침체기: 유틸리티, 필수소비재 안전</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-3">투자 전략 팁</h4>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>모멘텀 점수 80 이상인 섹터에 집중 투자</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>상대강도(RS) 지표로 시장 대비 성과 확인</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>경제 지표와 함께 종합적으로 판단</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>섹터 ETF를 활용한 분산 투자</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}