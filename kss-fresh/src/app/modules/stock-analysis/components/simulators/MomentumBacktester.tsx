'use client';

import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Activity, BarChart3, AlertCircle, Info, Target, Zap } from 'lucide-react';

interface Stock {
  symbol: string;
  name: string;
  sector: string;
  currentPrice: number;
  momentum1M: number;
  momentum3M: number;
  momentum6M: number;
  momentum12M: number;
  rsi: number;
  volume20D: number;
  volatility: number;
}

interface BacktestResult {
  totalReturn: number;
  annualizedReturn: number;
  maxDrawdown: number;
  sharpeRatio: number;
  winRate: number;
  totalTrades: number;
  avgWinReturn: number;
  avgLossReturn: number;
  profitFactor: number;
  monthlyReturns: { month: string; return: number }[];
  trades: Trade[];
}

interface Trade {
  entryDate: string;
  exitDate: string;
  symbol: string;
  entryPrice: number;
  exitPrice: number;
  return: number;
  holdingDays: number;
  signal: string;
}

interface MomentumSettings {
  lookbackPeriod: number;
  holdingPeriod: number;
  topN: number;
  rebalanceFrequency: 'weekly' | 'monthly' | 'quarterly';
  signalType: 'price' | 'rsi' | 'dual' | 'risk-adjusted';
  stopLoss: number;
  takeProfit: number;
}

// 모의 주식 데이터
const stockData: Stock[] = [
  { symbol: 'NVDA', name: 'NVIDIA', sector: 'Technology', currentPrice: 875.28, momentum1M: 15.2, momentum3M: 42.5, momentum6M: 85.3, momentum12M: 239.5, rsi: 72, volume20D: 45238000, volatility: 42.5 },
  { symbol: 'META', name: 'Meta Platforms', sector: 'Technology', currentPrice: 504.23, momentum1M: 8.7, momentum3M: 23.4, momentum6M: 41.2, momentum12M: 194.8, rsi: 65, volume20D: 23456000, volatility: 35.2 },
  { symbol: 'TSLA', name: 'Tesla', sector: 'Consumer Discretionary', currentPrice: 238.45, momentum1M: -5.2, momentum3M: 12.3, momentum6M: -8.5, momentum12M: 15.2, rsi: 45, volume20D: 118234000, volatility: 48.7 },
  { symbol: 'AAPL', name: 'Apple', sector: 'Technology', currentPrice: 189.95, momentum1M: 3.2, momentum3M: 8.5, momentum6M: 12.3, momentum12M: 48.5, rsi: 58, volume20D: 58234000, volatility: 24.3 },
  { symbol: 'GOOGL', name: 'Alphabet', sector: 'Technology', currentPrice: 155.34, momentum1M: 6.8, momentum3M: 15.2, momentum6M: 28.4, momentum12M: 55.2, rsi: 61, volume20D: 25678000, volatility: 28.5 },
  { symbol: 'MSFT', name: 'Microsoft', sector: 'Technology', currentPrice: 423.85, momentum1M: 4.5, momentum3M: 11.2, momentum6M: 18.7, momentum12M: 58.9, rsi: 59, volume20D: 22345000, volatility: 22.1 },
  { symbol: 'AMD', name: 'AMD', sector: 'Technology', currentPrice: 168.92, momentum1M: 12.3, momentum3M: 28.5, momentum6M: 45.2, momentum12M: 128.5, rsi: 68, volume20D: 65432000, volatility: 45.8 },
  { symbol: 'JPM', name: 'JPMorgan', sector: 'Financials', currentPrice: 195.42, momentum1M: 2.8, momentum3M: 8.9, momentum6M: 15.6, momentum12M: 38.2, rsi: 54, volume20D: 12345000, volatility: 19.8 },
  { symbol: 'V', name: 'Visa', sector: 'Financials', currentPrice: 276.48, momentum1M: 3.5, momentum3M: 7.2, momentum6M: 11.8, momentum12M: 24.5, rsi: 56, volume20D: 8765000, volatility: 18.5 },
  { symbol: 'UNH', name: 'UnitedHealth', sector: 'Healthcare', currentPrice: 524.65, momentum1M: -2.1, momentum3M: 5.4, momentum6M: 8.9, momentum12M: 12.3, rsi: 48, volume20D: 3456000, volatility: 21.2 }
];

export default function MomentumBacktester() {
  const [settings, setSettings] = useState<MomentumSettings>({
    lookbackPeriod: 3,
    holdingPeriod: 1,
    topN: 5,
    rebalanceFrequency: 'monthly',
    signalType: 'price',
    stopLoss: 10,
    takeProfit: 30
  });
  
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedStocks, setSelectedStocks] = useState<Stock[]>([]);
  const [viewMode, setViewMode] = useState<'setup' | 'results' | 'trades'>('setup');
  
  // 모멘텀 점수 계산
  const calculateMomentumScore = (stock: Stock): number => {
    const periodMap = {
      1: stock.momentum1M,
      3: stock.momentum3M,
      6: stock.momentum6M,
      12: stock.momentum12M
    };
    
    const momentum = periodMap[settings.lookbackPeriod as keyof typeof periodMap] || stock.momentum3M;
    
    switch (settings.signalType) {
      case 'price':
        return momentum;
      case 'rsi':
        return stock.rsi > 50 ? momentum : momentum * 0.5;
      case 'dual':
        return momentum * (stock.rsi / 50) * 0.5;
      case 'risk-adjusted':
        return momentum / stock.volatility;
      default:
        return momentum;
    }
  };
  
  // 백테스트 실행
  const runBacktest = () => {
    setIsRunning(true);
    
    // 모멘텀 스코어로 정렬
    const rankedStocks = [...stockData]
      .map(stock => ({ ...stock, score: calculateMomentumScore(stock) }))
      .sort((a, b) => b.score - a.score);
    
    // 상위 N개 선택
    const selected = rankedStocks.slice(0, settings.topN);
    setSelectedStocks(selected);
    
    // 모의 백테스트 결과 생성
    const trades: Trade[] = [];
    const monthlyReturns: { month: string; return: number }[] = [];
    
    // 12개월 백테스트
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    let cumulativeReturn = 100;
    
    for (let i = 0; i < 12; i++) {
      const monthReturn = (Math.random() - 0.5) * 20; // -10% ~ +10%
      cumulativeReturn *= (1 + monthReturn / 100);
      monthlyReturns.push({ month: months[i], return: monthReturn });
      
      // 월별 거래 생성
      if (i % (settings.rebalanceFrequency === 'monthly' ? 1 : 3) === 0) {
        selected.forEach(stock => {
          const tradeReturn = (Math.random() - 0.4) * 30; // -20% ~ +40% (모멘텀 편향)
          trades.push({
            entryDate: `2024-${String(i + 1).padStart(2, '0')}-01`,
            exitDate: `2024-${String(i + 2).padStart(2, '0')}-01`,
            symbol: stock.symbol,
            entryPrice: stock.currentPrice,
            exitPrice: stock.currentPrice * (1 + tradeReturn / 100),
            return: tradeReturn,
            holdingDays: 30,
            signal: `Momentum Score: ${stock.score.toFixed(1)}`
          });
        });
      }
    }
    
    const winningTrades = trades.filter(t => t.return > 0);
    const losingTrades = trades.filter(t => t.return <= 0);
    const totalReturn = cumulativeReturn - 100;
    
    const result: BacktestResult = {
      totalReturn,
      annualizedReturn: totalReturn,
      maxDrawdown: -15.3,
      sharpeRatio: 1.85,
      winRate: (winningTrades.length / trades.length) * 100,
      totalTrades: trades.length,
      avgWinReturn: winningTrades.length > 0 
        ? winningTrades.reduce((sum, t) => sum + t.return, 0) / winningTrades.length 
        : 0,
      avgLossReturn: losingTrades.length > 0
        ? losingTrades.reduce((sum, t) => sum + t.return, 0) / losingTrades.length
        : 0,
      profitFactor: Math.abs(
        winningTrades.reduce((sum, t) => sum + t.return, 0) / 
        (losingTrades.reduce((sum, t) => sum + t.return, 0) || 1)
      ),
      monthlyReturns,
      trades
    };
    
    setBacktestResult(result);
    setIsRunning(false);
    setViewMode('results');
  };
  
  const getSignalStrength = (score: number): { label: string; color: string } => {
    if (score > 30) return { label: '매우 강함', color: 'text-green-600' };
    if (score > 15) return { label: '강함', color: 'text-blue-600' };
    if (score > 0) return { label: '보통', color: 'text-yellow-600' };
    return { label: '약함', color: 'text-red-600' };
  };

  return (
    <div className="space-y-6">
      {/* 탭 네비게이션 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-2 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex gap-2">
          <button
            onClick={() => setViewMode('setup')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === 'setup'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            전략 설정
          </button>
          <button
            onClick={() => setViewMode('results')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === 'results'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            } ${!backtestResult && 'opacity-50 cursor-not-allowed'}`}
            disabled={!backtestResult}
          >
            백테스트 결과
          </button>
          <button
            onClick={() => setViewMode('trades')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === 'trades'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            } ${!backtestResult && 'opacity-50 cursor-not-allowed'}`}
            disabled={!backtestResult}
          >
            거래 내역
          </button>
        </div>
      </div>

      {viewMode === 'setup' && (
        <>
          {/* 전략 설정 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">모멘텀 전략 설정</h3>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    룩백 기간
                  </label>
                  <select
                    value={settings.lookbackPeriod}
                    onChange={(e) => setSettings({ ...settings, lookbackPeriod: Number(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                  >
                    <option value={1}>1개월</option>
                    <option value={3}>3개월</option>
                    <option value={6}>6개월</option>
                    <option value={12}>12개월</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">
                    보유 기간
                  </label>
                  <select
                    value={settings.holdingPeriod}
                    onChange={(e) => setSettings({ ...settings, holdingPeriod: Number(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                  >
                    <option value={1}>1개월</option>
                    <option value={3}>3개월</option>
                    <option value={6}>6개월</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">
                    종목 개수 (Top N)
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={settings.topN}
                    onChange={(e) => setSettings({ ...settings, topN: Number(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                  />
                </div>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    리밸런싱 주기
                  </label>
                  <select
                    value={settings.rebalanceFrequency}
                    onChange={(e) => setSettings({ ...settings, rebalanceFrequency: e.target.value as any })}
                    className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                  >
                    <option value="weekly">주간</option>
                    <option value="monthly">월간</option>
                    <option value="quarterly">분기</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">
                    신호 타입
                  </label>
                  <select
                    value={settings.signalType}
                    onChange={(e) => setSettings({ ...settings, signalType: e.target.value as any })}
                    className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                  >
                    <option value="price">순수 가격 모멘텀</option>
                    <option value="rsi">RSI 필터링</option>
                    <option value="dual">가격 + RSI 결합</option>
                    <option value="risk-adjusted">위험조정 모멘텀</option>
                  </select>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      손절매 (%)
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="50"
                      value={settings.stopLoss}
                      onChange={(e) => setSettings({ ...settings, stopLoss: Number(e.target.value) })}
                      className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      익절매 (%)
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={settings.takeProfit}
                      onChange={(e) => setSettings({ ...settings, takeProfit: Number(e.target.value) })}
                      className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                    />
                  </div>
                </div>
              </div>
            </div>
            
            <button
              onClick={runBacktest}
              disabled={isRunning}
              className="mt-6 w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isRunning ? (
                <>
                  <Activity className="w-4 h-4 animate-spin" />
                  백테스트 실행 중...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  백테스트 실행
                </>
              )}
            </button>
          </div>

          {/* 현재 모멘텀 랭킹 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">현재 모멘텀 랭킹</h3>
            
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <th className="text-left p-2">순위</th>
                    <th className="text-left p-2">종목</th>
                    <th className="text-right p-2">1개월</th>
                    <th className="text-right p-2">3개월</th>
                    <th className="text-right p-2">6개월</th>
                    <th className="text-right p-2">RSI</th>
                    <th className="text-right p-2">변동성</th>
                    <th className="text-right p-2">신호</th>
                  </tr>
                </thead>
                <tbody>
                  {[...stockData]
                    .map(stock => ({ ...stock, score: calculateMomentumScore(stock) }))
                    .sort((a, b) => b.score - a.score)
                    .slice(0, 10)
                    .map((stock, idx) => {
                      const signal = getSignalStrength(stock.score);
                      return (
                        <tr key={stock.symbol} className="border-b border-gray-100 dark:border-gray-900">
                          <td className="p-2">{idx + 1}</td>
                          <td className="p-2">
                            <div>
                              <span className="font-medium">{stock.symbol}</span>
                              <span className="text-gray-600 dark:text-gray-400 ml-2 text-xs">
                                {stock.name}
                              </span>
                            </div>
                          </td>
                          <td className="text-right p-2">
                            <span className={stock.momentum1M > 0 ? 'text-green-600' : 'text-red-600'}>
                              {stock.momentum1M > 0 ? '+' : ''}{stock.momentum1M.toFixed(1)}%
                            </span>
                          </td>
                          <td className="text-right p-2">
                            <span className={stock.momentum3M > 0 ? 'text-green-600' : 'text-red-600'}>
                              {stock.momentum3M > 0 ? '+' : ''}{stock.momentum3M.toFixed(1)}%
                            </span>
                          </td>
                          <td className="text-right p-2">
                            <span className={stock.momentum6M > 0 ? 'text-green-600' : 'text-red-600'}>
                              {stock.momentum6M > 0 ? '+' : ''}{stock.momentum6M.toFixed(1)}%
                            </span>
                          </td>
                          <td className="text-right p-2">{stock.rsi}</td>
                          <td className="text-right p-2">{stock.volatility.toFixed(1)}%</td>
                          <td className="text-right p-2">
                            <span className={`font-medium ${signal.color}`}>
                              {signal.label}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {viewMode === 'results' && backtestResult && (
        <>
          {/* 백테스트 결과 요약 */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">총 수익률</p>
              <p className={`text-2xl font-bold ${backtestResult.totalReturn > 0 ? 'text-green-600' : 'text-red-600'}`}>
                {backtestResult.totalReturn > 0 ? '+' : ''}{backtestResult.totalReturn.toFixed(2)}%
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">샤프 비율</p>
              <p className="text-2xl font-bold">{backtestResult.sharpeRatio.toFixed(2)}</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">최대 낙폭</p>
              <p className="text-2xl font-bold text-red-600">{backtestResult.maxDrawdown.toFixed(1)}%</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">승률</p>
              <p className="text-2xl font-bold text-blue-600">{backtestResult.winRate.toFixed(1)}%</p>
            </div>
          </div>

          {/* 추가 통계 */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">프로핏 팩터</p>
              <p className="text-xl font-bold">{backtestResult.profitFactor.toFixed(2)}</p>
              <p className="text-xs text-gray-500">평균 이익/평균 손실</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">평균 수익</p>
              <p className="text-xl font-bold text-green-600">+{backtestResult.avgWinReturn.toFixed(2)}%</p>
              <p className="text-xs text-gray-500">수익 거래 평균</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">평균 손실</p>
              <p className="text-xl font-bold text-red-600">{backtestResult.avgLossReturn.toFixed(2)}%</p>
              <p className="text-xs text-gray-500">손실 거래 평균</p>
            </div>
          </div>

          {/* 월별 수익률 차트 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">월별 수익률</h3>
            
            <div className="space-y-3">
              {backtestResult.monthlyReturns.map((month) => {
                const isPositive = month.return > 0;
                const barWidth = Math.abs(month.return) * 5;
                
                return (
                  <div key={month.month} className="flex items-center gap-3">
                    <span className="w-12 text-sm font-medium">{month.month}</span>
                    <div className="flex-1 flex items-center">
                      <div className="flex-1 bg-gray-100 dark:bg-gray-900 h-6 rounded relative">
                        <div
                          className={`absolute top-0 h-full rounded ${
                            isPositive ? 'bg-green-500' : 'bg-red-500'
                          }`}
                          style={{
                            width: `${barWidth}%`,
                            left: isPositive ? '50%' : `${50 - barWidth}%`
                          }}
                        />
                        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gray-400" />
                      </div>
                    </div>
                    <span className={`w-16 text-right text-sm font-medium ${
                      isPositive ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {isPositive ? '+' : ''}{month.return.toFixed(1)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 선택된 종목 */}
          {selectedStocks.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">백테스트 포트폴리오</h3>
              
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                {selectedStocks.map((stock) => (
                  <div key={stock.symbol} className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold">{stock.symbol}</span>
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        {stock.sector}
                      </span>
                    </div>
                    <div className="text-sm space-y-1">
                      <div className="flex justify-between">
                        <span>모멘텀 점수</span>
                        <span className="font-medium">{stock.score.toFixed(1)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>3개월 수익률</span>
                        <span className={`font-medium ${
                          stock.momentum3M > 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {stock.momentum3M > 0 ? '+' : ''}{stock.momentum3M.toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>변동성</span>
                        <span>{stock.volatility.toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {viewMode === 'trades' && backtestResult && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-4">거래 내역</h3>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-2">진입일</th>
                  <th className="text-left p-2">청산일</th>
                  <th className="text-left p-2">종목</th>
                  <th className="text-right p-2">진입가</th>
                  <th className="text-right p-2">청산가</th>
                  <th className="text-right p-2">수익률</th>
                  <th className="text-right p-2">보유일</th>
                  <th className="text-left p-2">신호</th>
                </tr>
              </thead>
              <tbody>
                {backtestResult.trades.map((trade, idx) => (
                  <tr key={idx} className="border-b border-gray-100 dark:border-gray-900">
                    <td className="p-2">{trade.entryDate}</td>
                    <td className="p-2">{trade.exitDate}</td>
                    <td className="p-2 font-medium">{trade.symbol}</td>
                    <td className="text-right p-2">${trade.entryPrice.toFixed(2)}</td>
                    <td className="text-right p-2">${trade.exitPrice.toFixed(2)}</td>
                    <td className="text-right p-2">
                      <span className={`font-medium ${
                        trade.return > 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {trade.return > 0 ? '+' : ''}{trade.return.toFixed(2)}%
                      </span>
                    </td>
                    <td className="text-right p-2">{trade.holdingDays}일</td>
                    <td className="p-2 text-xs text-gray-600 dark:text-gray-400">
                      {trade.signal}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 모멘텀 전략 가이드 */}
      <div className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Info className="w-5 h-5" />
          모멘텀 전략 가이드
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-3">모멘텀 투자의 핵심</h4>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-purple-500">✓</span>
                <span>상승 추세의 지속성을 활용 - "Winners keep winning"</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-500">✓</span>
                <span>상대적 강도가 높은 종목 선택 - 시장 대비 초과 성과</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-500">✓</span>
                <span>정기적 리밸런싱으로 추세 변화 대응</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-500">✓</span>
                <span>손절매 규칙으로 하방 리스크 제한</span>
              </li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-3">실전 적용 팁</h4>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>시장 전체가 하락 추세일 때는 현금 비중 확대</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>거래량 확인 - 모멘텀은 거래량이 뒷받침되어야 함</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>섹터 집중 리스크 관리 - 최대 30% 이하로 제한</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>급등 후 조정은 자연스러운 과정 - 장기 관점 유지</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}