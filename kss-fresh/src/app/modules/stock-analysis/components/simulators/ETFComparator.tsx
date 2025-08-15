'use client';

import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, DollarSign, Activity, AlertCircle, Info, Star, Download, ChevronDown } from 'lucide-react';

interface ETFData {
  symbol: string;
  name: string;
  category: string;
  expense: number; // 운용보수 %
  aum: number; // 운용자산 (AUM)
  nav: number; // 순자산가치
  price: number; // 현재가
  premium: number; // 프리미엄/디스카운트 %
  volume: number; // 거래량
  
  // 성과 데이터
  performance: {
    day1: number;
    week1: number;
    month1: number;
    month3: number;
    year1: number;
    year3: number;
    year5: number;
  };
  
  // 리스크 지표
  risk: {
    volatility: number; // 변동성
    sharpe: number; // 샤프 비율
    beta: number; // 베타
    maxDrawdown: number; // 최대 낙폭
  };
  
  // 배당 정보
  dividend: {
    yield: number; // 배당수익률
    frequency: string; // 배당 주기
    lastAmount: number; // 최근 배당금
  };
  
  // 구성 정보
  holdings: {
    top10Weight: number; // 상위 10종목 비중
    totalHoldings: number; // 총 보유 종목 수
    topHoldings: Array<{
      name: string;
      weight: number;
    }>;
  };
}

// 모의 ETF 데이터
const mockETFs: ETFData[] = [
  {
    symbol: 'SPY',
    name: 'SPDR S&P 500 ETF',
    category: '미국 대형주',
    expense: 0.09,
    aum: 450000000000,
    nav: 475.23,
    price: 475.45,
    premium: 0.05,
    volume: 75234567,
    performance: {
      day1: 0.45,
      week1: 1.23,
      month1: 3.45,
      month3: 8.92,
      year1: 24.56,
      year3: 42.34,
      year5: 89.23
    },
    risk: {
      volatility: 15.67,
      sharpe: 1.45,
      beta: 1.00,
      maxDrawdown: -33.72
    },
    dividend: {
      yield: 1.32,
      frequency: '분기',
      lastAmount: 1.58
    },
    holdings: {
      top10Weight: 31.5,
      totalHoldings: 503,
      topHoldings: [
        { name: 'Apple Inc.', weight: 7.2 },
        { name: 'Microsoft Corp.', weight: 6.8 },
        { name: 'Amazon.com Inc.', weight: 3.5 }
      ]
    }
  },
  {
    symbol: 'QQQ',
    name: 'Invesco QQQ Trust',
    category: '미국 기술주',
    expense: 0.20,
    aum: 220000000000,
    nav: 423.67,
    price: 423.89,
    premium: 0.05,
    volume: 45678901,
    performance: {
      day1: 0.78,
      week1: 2.45,
      month1: 5.67,
      month3: 12.34,
      year1: 35.67,
      year3: 78.90,
      year5: 156.78
    },
    risk: {
      volatility: 22.34,
      sharpe: 1.67,
      beta: 1.15,
      maxDrawdown: -32.58
    },
    dividend: {
      yield: 0.56,
      frequency: '분기',
      lastAmount: 0.59
    },
    holdings: {
      top10Weight: 42.3,
      totalHoldings: 101,
      topHoldings: [
        { name: 'Apple Inc.', weight: 11.5 },
        { name: 'Microsoft Corp.', weight: 10.2 },
        { name: 'NVIDIA Corp.', weight: 6.8 }
      ]
    }
  },
  {
    symbol: 'VTI',
    name: 'Vanguard Total Stock Market ETF',
    category: '미국 전체시장',
    expense: 0.03,
    aum: 320000000000,
    nav: 245.78,
    price: 245.82,
    premium: 0.02,
    volume: 23456789,
    performance: {
      day1: 0.42,
      week1: 1.34,
      month1: 3.78,
      month3: 9.23,
      year1: 25.89,
      year3: 45.67,
      year5: 95.34
    },
    risk: {
      volatility: 16.89,
      sharpe: 1.42,
      beta: 1.05,
      maxDrawdown: -35.41
    },
    dividend: {
      yield: 1.28,
      frequency: '분기',
      lastAmount: 0.78
    },
    holdings: {
      top10Weight: 25.6,
      totalHoldings: 4026,
      topHoldings: [
        { name: 'Apple Inc.', weight: 6.5 },
        { name: 'Microsoft Corp.', weight: 5.9 },
        { name: 'Amazon.com Inc.', weight: 2.8 }
      ]
    }
  },
  {
    symbol: 'SCHD',
    name: 'Schwab US Dividend Equity ETF',
    category: '미국 배당주',
    expense: 0.06,
    aum: 48000000000,
    nav: 82.34,
    price: 82.38,
    premium: 0.05,
    volume: 12345678,
    performance: {
      day1: 0.23,
      week1: 0.89,
      month1: 2.34,
      month3: 5.67,
      year1: 15.23,
      year3: 32.45,
      year5: 67.89
    },
    risk: {
      volatility: 12.34,
      sharpe: 1.23,
      beta: 0.85,
      maxDrawdown: -24.56
    },
    dividend: {
      yield: 3.45,
      frequency: '분기',
      lastAmount: 0.71
    },
    holdings: {
      top10Weight: 40.2,
      totalHoldings: 104,
      topHoldings: [
        { name: 'Broadcom Inc.', weight: 4.8 },
        { name: 'Johnson & Johnson', weight: 4.5 },
        { name: 'Exxon Mobil Corp.', weight: 4.2 }
      ]
    }
  }
];

// ETF 카테고리
const categories = [
  '전체',
  '미국 대형주',
  '미국 기술주',
  '미국 전체시장',
  '미국 배당주',
  '신흥시장',
  '선진국',
  '섹터',
  '채권',
  '원자재'
];

export default function ETFComparator() {
  const [selectedETFs, setSelectedETFs] = useState<string[]>(['SPY', 'QQQ']);
  const [compareMode, setCompareMode] = useState<'performance' | 'risk' | 'cost' | 'holdings'>('performance');
  const [selectedCategory, setSelectedCategory] = useState('전체');
  const [searchTerm, setSearchTerm] = useState('');
  const [showDetails, setShowDetails] = useState(false);
  
  // 선택된 ETF 데이터
  const selectedETFData = mockETFs.filter(etf => selectedETFs.includes(etf.symbol));
  
  // ETF 선택/해제
  const toggleETF = (symbol: string) => {
    if (selectedETFs.includes(symbol)) {
      setSelectedETFs(prev => prev.filter(s => s !== symbol));
    } else if (selectedETFs.length < 4) {
      setSelectedETFs(prev => [...prev, symbol]);
    }
  };
  
  // 카테고리별 필터링
  const filteredETFs = mockETFs.filter(etf => {
    const matchesCategory = selectedCategory === '전체' || etf.category === selectedCategory;
    const matchesSearch = etf.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         etf.name.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesCategory && matchesSearch;
  });
  
  // 숫자 포맷
  const formatNumber = (num: number, decimals: number = 2) => {
    return num.toFixed(decimals);
  };
  
  const formatCurrency = (num: number) => {
    if (num >= 1e12) return `$${(num / 1e12).toFixed(1)}T`;
    if (num >= 1e9) return `$${(num / 1e9).toFixed(1)}B`;
    if (num >= 1e6) return `$${(num / 1e6).toFixed(1)}M`;
    return `$${num.toFixed(0)}`;
  };
  
  const formatVolume = (num: number) => {
    if (num >= 1e9) return `${(num / 1e9).toFixed(1)}B`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(1)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(0)}K`;
    return num.toString();
  };
  
  // 성과 색상
  const getPerformanceColor = (value: number) => {
    if (value > 0) return 'text-green-600';
    if (value < 0) return 'text-red-600';
    return 'text-gray-600';
  };
  
  // 최고/최저값 표시
  const getBestWorst = (values: number[], isHigherBetter: boolean = true) => {
    const max = Math.max(...values);
    const min = Math.min(...values);
    return {
      best: isHigherBetter ? max : min,
      worst: isHigherBetter ? min : max
    };
  };

  return (
    <div className="space-y-6">
      {/* ETF 선택 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4">ETF 선택 (최대 4개)</h3>
        
        {/* 검색 및 필터 */}
        <div className="flex gap-4 mb-4">
          <input
            type="text"
            placeholder="ETF 검색..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="flex-1 px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
          />
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
          >
            {categories.map(cat => (
              <option key={cat} value={cat}>{cat}</option>
            ))}
          </select>
        </div>
        
        {/* ETF 목록 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {filteredETFs.map(etf => (
            <button
              key={etf.symbol}
              onClick={() => toggleETF(etf.symbol)}
              disabled={!selectedETFs.includes(etf.symbol) && selectedETFs.length >= 4}
              className={`p-4 rounded-lg border transition-all ${
                selectedETFs.includes(etf.symbol)
                  ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-500'
                  : 'bg-gray-50 dark:bg-gray-900 border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800'
              } ${!selectedETFs.includes(etf.symbol) && selectedETFs.length >= 4 ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <div className="flex items-start justify-between mb-2">
                <div>
                  <p className="font-bold text-left">{etf.symbol}</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 text-left">{etf.category}</p>
                </div>
                {selectedETFs.includes(etf.symbol) && (
                  <Star className="w-5 h-5 text-yellow-500 fill-current" />
                )}
              </div>
              <p className="text-sm text-left line-clamp-2">{etf.name}</p>
              <div className="flex items-center justify-between mt-2">
                <span className="text-lg font-medium">${etf.price.toFixed(2)}</span>
                <span className={`text-sm ${getPerformanceColor(etf.performance.day1)}`}>
                  {etf.performance.day1 > 0 ? '+' : ''}{etf.performance.day1.toFixed(2)}%
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* 비교 모드 선택 */}
      {selectedETFData.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-2 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex gap-2">
            <button
              onClick={() => setCompareMode('performance')}
              className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                compareMode === 'performance'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              성과 비교
            </button>
            <button
              onClick={() => setCompareMode('risk')}
              className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                compareMode === 'risk'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              리스크 분석
            </button>
            <button
              onClick={() => setCompareMode('cost')}
              className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                compareMode === 'cost'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              비용 & 배당
            </button>
            <button
              onClick={() => setCompareMode('holdings')}
              className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                compareMode === 'holdings'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              구성 종목
            </button>
          </div>
        </div>
      )}

      {/* 비교 결과 */}
      {selectedETFData.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">ETF 비교 분석</h3>
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="text-sm text-blue-600 dark:text-blue-400 hover:underline flex items-center gap-1"
            >
              {showDetails ? '간단히 보기' : '자세히 보기'}
              <ChevronDown className={`w-4 h-4 transition-transform ${showDetails ? 'rotate-180' : ''}`} />
            </button>
          </div>

          {/* 성과 비교 */}
          {compareMode === 'performance' && (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-sm text-gray-600 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                    <th className="pb-3 font-medium">기간</th>
                    {selectedETFData.map(etf => (
                      <th key={etf.symbol} className="pb-3 font-medium text-right">
                        {etf.symbol}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    { label: '1일', key: 'day1' },
                    { label: '1주', key: 'week1' },
                    { label: '1개월', key: 'month1' },
                    { label: '3개월', key: 'month3' },
                    { label: '1년', key: 'year1' },
                    { label: '3년', key: 'year3' },
                    { label: '5년', key: 'year5' }
                  ].map(({ label, key }) => {
                    const values = selectedETFData.map(etf => etf.performance[key as keyof typeof etf.performance]);
                    const { best, worst } = getBestWorst(values);
                    
                    return (
                      <tr key={key} className="border-b border-gray-100 dark:border-gray-800">
                        <td className="py-3 text-sm">{label}</td>
                        {selectedETFData.map(etf => {
                          const value = etf.performance[key as keyof typeof etf.performance];
                          return (
                            <td key={etf.symbol} className="py-3 text-right">
                              <span className={`font-medium ${getPerformanceColor(value)} ${
                                value === best ? 'font-bold' : ''
                              }`}>
                                {value > 0 ? '+' : ''}{formatNumber(value)}%
                                {value === best && ' 👑'}
                              </span>
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* 리스크 분석 */}
          {compareMode === 'risk' && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {selectedETFData.map(etf => (
                  <div key={etf.symbol} className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                    <h4 className="font-medium mb-3">{etf.symbol}</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">변동성</span>
                        <span className="font-medium">{formatNumber(etf.risk.volatility)}%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">샤프 비율</span>
                        <span className="font-medium">{formatNumber(etf.risk.sharpe)}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">베타</span>
                        <span className="font-medium">{formatNumber(etf.risk.beta)}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">최대 낙폭</span>
                        <span className="font-medium text-red-600">{formatNumber(etf.risk.maxDrawdown)}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              {showDetails && (
                <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <h4 className="font-medium mb-2">리스크 지표 해석</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• <strong>변동성</strong>: 낮을수록 안정적 (15% 이하 권장)</li>
                    <li>• <strong>샤프 비율</strong>: 높을수록 좋음 (1.0 이상 우수)</li>
                    <li>• <strong>베타</strong>: 1.0 = 시장과 동일, 1.0 초과 = 시장보다 변동성 큼</li>
                    <li>• <strong>최대 낙폭</strong>: 과거 최고점 대비 최대 하락폭</li>
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* 비용 & 배당 */}
          {compareMode === 'cost' && (
            <div className="space-y-4">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="text-left text-sm text-gray-600 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                      <th className="pb-3 font-medium">항목</th>
                      {selectedETFData.map(etf => (
                        <th key={etf.symbol} className="pb-3 font-medium text-right">
                          {etf.symbol}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-100 dark:border-gray-800">
                      <td className="py-3 text-sm">운용보수</td>
                      {selectedETFData.map(etf => {
                        const values = selectedETFData.map(e => e.expense);
                        const { best } = getBestWorst(values, false);
                        return (
                          <td key={etf.symbol} className="py-3 text-right">
                            <span className={`font-medium ${etf.expense === best ? 'text-green-600 font-bold' : ''}`}>
                              {formatNumber(etf.expense)}%
                              {etf.expense === best && ' 👑'}
                            </span>
                          </td>
                        );
                      })}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-800">
                      <td className="py-3 text-sm">운용자산 (AUM)</td>
                      {selectedETFData.map(etf => (
                        <td key={etf.symbol} className="py-3 text-right font-medium">
                          {formatCurrency(etf.aum)}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-800">
                      <td className="py-3 text-sm">배당수익률</td>
                      {selectedETFData.map(etf => {
                        const values = selectedETFData.map(e => e.dividend.yield);
                        const { best } = getBestWorst(values);
                        return (
                          <td key={etf.symbol} className="py-3 text-right">
                            <span className={`font-medium ${etf.dividend.yield === best ? 'text-green-600 font-bold' : ''}`}>
                              {formatNumber(etf.dividend.yield)}%
                              {etf.dividend.yield === best && ' 👑'}
                            </span>
                          </td>
                        );
                      })}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-800">
                      <td className="py-3 text-sm">배당 주기</td>
                      {selectedETFData.map(etf => (
                        <td key={etf.symbol} className="py-3 text-right text-sm">
                          {etf.dividend.frequency}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-800">
                      <td className="py-3 text-sm">일평균 거래량</td>
                      {selectedETFData.map(etf => (
                        <td key={etf.symbol} className="py-3 text-right font-medium">
                          {formatVolume(etf.volume)}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
              
              {/* 10년 투자 시뮬레이션 */}
              <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <h4 className="font-medium mb-3">$10,000 투자 시 10년 후 예상 비용</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                  {selectedETFData.map(etf => {
                    const totalCost = 10000 * etf.expense / 100 * 10;
                    return (
                      <div key={etf.symbol} className="text-center">
                        <p className="font-medium">{etf.symbol}</p>
                        <p className="text-2xl font-bold text-red-600">${totalCost.toFixed(0)}</p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">총 운용보수</p>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* 구성 종목 */}
          {compareMode === 'holdings' && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {selectedETFData.map(etf => (
                  <div key={etf.symbol} className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-medium">{etf.symbol}</h4>
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        총 {etf.holdings.totalHoldings}개 종목
                      </span>
                    </div>
                    
                    <div className="mb-3">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm text-gray-600 dark:text-gray-400">상위 10종목 비중</span>
                        <span className="font-medium">{formatNumber(etf.holdings.top10Weight)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full"
                          style={{ width: `${etf.holdings.top10Weight}%` }}
                        />
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <p className="text-sm font-medium mb-1">상위 보유 종목</p>
                      {etf.holdings.topHoldings.map((holding, idx) => (
                        <div key={idx} className="flex justify-between items-center text-sm">
                          <span className="text-gray-700 dark:text-gray-300">{holding.name}</span>
                          <span className="font-medium">{formatNumber(holding.weight)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* 추천 ETF 조합 */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Star className="w-5 h-5" />
          추천 ETF 포트폴리오
        </h3>
        
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-medium mb-2">보수적 포트폴리오</h4>
            <ul className="text-sm space-y-1 mb-3">
              <li>• SPY (40%) - S&P 500</li>
              <li>• AGG (30%) - 채권</li>
              <li>• VNQ (20%) - 부동산</li>
              <li>• GLD (10%) - 금</li>
            </ul>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              예상 연수익률: 6-8% | 변동성: 낮음
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-medium mb-2">균형 포트폴리오</h4>
            <ul className="text-sm space-y-1 mb-3">
              <li>• VTI (50%) - 미국 전체</li>
              <li>• VXUS (30%) - 해외 주식</li>
              <li>• BND (20%) - 채권</li>
            </ul>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              예상 연수익률: 8-10% | 변동성: 중간
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-medium mb-2">성장 포트폴리오</h4>
            <ul className="text-sm space-y-1 mb-3">
              <li>• QQQ (40%) - 기술주</li>
              <li>• VUG (30%) - 성장주</li>
              <li>• VWO (20%) - 신흥시장</li>
              <li>• ARKK (10%) - 혁신기업</li>
            </ul>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              예상 연수익률: 10-15% | 변동성: 높음
            </p>
          </div>
        </div>
      </div>

      {/* 사용 안내 */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
        <h4 className="font-medium mb-2 flex items-center gap-2">
          <Info className="w-5 h-5" />
          ETF 비교 분석기 사용법
        </h4>
        <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
          <li>• 최대 4개의 ETF를 선택하여 상세 비교 분석할 수 있습니다</li>
          <li>• 성과, 리스크, 비용, 구성 종목 등 다양한 관점에서 비교하세요</li>
          <li>• 👑 표시는 해당 지표에서 가장 우수한 ETF를 나타냅니다</li>
          <li>• 운용보수는 장기 투자 시 수익률에 큰 영향을 미치므로 주의깊게 확인하세요</li>
          <li>• 추천 포트폴리오는 일반적인 가이드라인이며, 개인 상황에 맞게 조정하세요</li>
        </ul>
      </div>
    </div>
  );
}