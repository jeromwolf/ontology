'use client';

import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Activity, BarChart3, AlertCircle, Info, Target, Zap } from 'lucide-react';

interface StockPair {
  stock1: {
    symbol: string;
    name: string;
    price: number;
    sector: string;
  };
  stock2: {
    symbol: string;
    name: string;
    price: number;
    sector: string;
  };
  correlation: number;
  cointegration: number;
  spread: number;
  zScore: number;
  halfLife: number;
  sharpeRatio: number;
}

interface PairMetrics {
  currentSpread: number;
  meanSpread: number;
  stdSpread: number;
  zScore: number;
  signal: 'long' | 'short' | 'neutral';
  entryThreshold: number;
  exitThreshold: number;
  stopLoss: number;
}

interface BacktestResult {
  totalTrades: number;
  winRate: number;
  avgProfit: number;
  maxDrawdown: number;
  sharpeRatio: number;
  totalReturn: number;
  trades: Trade[];
}

interface Trade {
  entryDate: string;
  exitDate: string;
  type: 'long' | 'short';
  entrySpread: number;
  exitSpread: number;
  profit: number;
  duration: number;
}

// 모의 주식 쌍 데이터
const stockPairs: StockPair[] = [
  {
    stock1: { symbol: 'KO', name: 'Coca-Cola', price: 60.25, sector: 'Consumer Staples' },
    stock2: { symbol: 'PEP', name: 'PepsiCo', price: 172.45, sector: 'Consumer Staples' },
    correlation: 0.92,
    cointegration: 0.88,
    spread: 2.86,
    zScore: 1.45,
    halfLife: 15,
    sharpeRatio: 1.82
  },
  {
    stock1: { symbol: 'MA', name: 'Mastercard', price: 428.50, sector: 'Financial' },
    stock2: { symbol: 'V', name: 'Visa', price: 261.80, sector: 'Financial' },
    correlation: 0.95,
    cointegration: 0.91,
    spread: 1.64,
    zScore: -0.82,
    halfLife: 12,
    sharpeRatio: 2.15
  },
  {
    stock1: { symbol: 'HD', name: 'Home Depot', price: 342.15, sector: 'Consumer Discretionary' },
    stock2: { symbol: 'LOW', name: "Lowe's", price: 215.30, sector: 'Consumer Discretionary' },
    correlation: 0.89,
    cointegration: 0.85,
    spread: 1.59,
    zScore: 2.21,
    halfLife: 18,
    sharpeRatio: 1.65
  },
  {
    stock1: { symbol: 'GS', name: 'Goldman Sachs', price: 385.20, sector: 'Financial' },
    stock2: { symbol: 'MS', name: 'Morgan Stanley', price: 88.45, sector: 'Financial' },
    correlation: 0.87,
    cointegration: 0.82,
    spread: 4.35,
    zScore: -1.15,
    halfLife: 20,
    sharpeRatio: 1.43
  }
];

// 모의 가격 데이터 생성
const generatePriceHistory = (basePrice: number, days: number) => {
  const prices = [basePrice];
  for (let i = 1; i < days; i++) {
    const change = (Math.random() - 0.5) * 0.02 * basePrice;
    prices.push(prices[i - 1] + change);
  }
  return prices;
};

export default function PairTradingAnalyzer() {
  const [selectedPair, setSelectedPair] = useState<StockPair>(stockPairs[0]);
  const [timeframe, setTimeframe] = useState<30 | 60 | 90>(60);
  const [entryZScore, setEntryZScore] = useState(2.0);
  const [exitZScore, setExitZScore] = useState(0.5);
  const [stopLossZScore, setStopLossZScore] = useState(3.0);
  const [viewMode, setViewMode] = useState<'analysis' | 'backtest' | 'signals'>('analysis');
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  // 가격 히스토리 생성
  const [priceHistory1, setPriceHistory1] = useState<number[]>([]);
  const [priceHistory2, setPriceHistory2] = useState<number[]>([]);
  
  useEffect(() => {
    setPriceHistory1(generatePriceHistory(selectedPair.stock1.price, timeframe));
    setPriceHistory2(generatePriceHistory(selectedPair.stock2.price, timeframe));
  }, [selectedPair, timeframe]);
  
  // 스프레드 계산
  const calculateSpread = () => {
    const ratio = priceHistory1.map((p1, i) => p1 / priceHistory2[i]);
    const mean = ratio.reduce((a, b) => a + b, 0) / ratio.length;
    const std = Math.sqrt(ratio.map(r => Math.pow(r - mean, 2)).reduce((a, b) => a + b, 0) / ratio.length);
    const currentRatio = ratio[ratio.length - 1];
    const zScore = (currentRatio - mean) / std;
    
    return {
      spread: ratio,
      currentSpread: currentRatio,
      meanSpread: mean,
      stdSpread: std,
      zScore: zScore
    };
  };
  
  const spreadData = calculateSpread();
  
  // 트레이딩 신호 결정
  const getTradingSignal = (zScore: number): 'long' | 'short' | 'neutral' => {
    if (zScore > entryZScore) return 'short';
    if (zScore < -entryZScore) return 'long';
    return 'neutral';
  };
  
  const currentSignal = getTradingSignal(spreadData.zScore);
  
  // 백테스트 실행
  const runBacktest = (): BacktestResult => {
    const trades: Trade[] = [];
    let position: 'none' | 'long' | 'short' = 'none';
    let entryIndex = 0;
    let totalReturn = 0;
    
    for (let i = 20; i < spreadData.spread.length; i++) {
      const currentSpread = spreadData.spread[i];
      const mean = spreadData.spread.slice(i - 20, i).reduce((a, b) => a + b, 0) / 20;
      const std = Math.sqrt(spreadData.spread.slice(i - 20, i).map(s => Math.pow(s - mean, 2)).reduce((a, b) => a + b, 0) / 20);
      const zScore = (currentSpread - mean) / std;
      
      // Entry logic
      if (position === 'none') {
        if (zScore > entryZScore) {
          position = 'short';
          entryIndex = i;
        } else if (zScore < -entryZScore) {
          position = 'long';
          entryIndex = i;
        }
      }
      
      // Exit logic
      if (position !== 'none') {
        const shouldExit = (position === 'long' && zScore > -exitZScore) ||
                          (position === 'short' && zScore < exitZScore) ||
                          Math.abs(zScore) > stopLossZScore;
        
        if (shouldExit) {
          const entrySpread = spreadData.spread[entryIndex];
          const profit = position === 'long' 
            ? (currentSpread - entrySpread) / entrySpread * 100
            : (entrySpread - currentSpread) / entrySpread * 100;
          
          trades.push({
            entryDate: `Day ${entryIndex}`,
            exitDate: `Day ${i}`,
            type: position,
            entrySpread: entrySpread,
            exitSpread: currentSpread,
            profit: profit,
            duration: i - entryIndex
          });
          
          totalReturn += profit;
          position = 'none';
        }
      }
    }
    
    const winningTrades = trades.filter(t => t.profit > 0).length;
    const avgProfit = trades.length > 0 ? trades.reduce((sum, t) => sum + t.profit, 0) / trades.length : 0;
    const returns = trades.map(t => t.profit);
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const stdReturn = Math.sqrt(returns.map(r => Math.pow(r - avgReturn, 2)).reduce((a, b) => a + b, 0) / returns.length);
    
    return {
      totalTrades: trades.length,
      winRate: trades.length > 0 ? (winningTrades / trades.length) * 100 : 0,
      avgProfit: avgProfit,
      maxDrawdown: Math.min(...returns, 0),
      sharpeRatio: stdReturn > 0 ? (avgReturn / stdReturn) * Math.sqrt(252 / timeframe) : 0,
      totalReturn: totalReturn,
      trades: trades.slice(-10) // 최근 10개 거래만
    };
  };
  
  const backtestResult = runBacktest();
  
  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'long': return 'text-green-600';
      case 'short': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };
  
  const getSignalText = (signal: string) => {
    switch (signal) {
      case 'long': return '매수 신호 (스프레드 확대 예상)';
      case 'short': return '매도 신호 (스프레드 축소 예상)';
      default: return '중립 (진입 신호 없음)';
    }
  };

  return (
    <div className="space-y-6">
      {/* 주식 쌍 선택 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4">주식 쌍 선택</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {stockPairs.map((pair) => (
            <button
              key={`${pair.stock1.symbol}-${pair.stock2.symbol}`}
              onClick={() => setSelectedPair(pair)}
              className={`p-4 rounded-lg transition-all text-left ${
                selectedPair === pair
                  ? 'bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-500'
                  : 'bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="font-bold">{pair.stock1.symbol}</span>
                  <span className="text-gray-500">vs</span>
                  <span className="font-bold">{pair.stock2.symbol}</span>
                </div>
                <span className="text-xs text-gray-500">{pair.stock1.sector}</span>
              </div>
              
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-gray-600 dark:text-gray-400">상관계수:</span>
                  <span className="ml-1 font-medium">{pair.correlation.toFixed(2)}</span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">공적분:</span>
                  <span className="ml-1 font-medium">{pair.cointegration.toFixed(2)}</span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">Z-Score:</span>
                  <span className={`ml-1 font-medium ${
                    Math.abs(pair.zScore) > 2 ? 'text-red-600' : 'text-gray-900 dark:text-gray-100'
                  }`}>
                    {pair.zScore.toFixed(2)}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">반감기:</span>
                  <span className="ml-1 font-medium">{pair.halfLife}일</span>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* 전략 설정 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Target className="w-5 h-5" />
          전략 설정
        </h3>
        
        <div className="grid md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium mb-2">분석 기간</label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(Number(e.target.value) as any)}
              className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
            >
              <option value={30}>30일</option>
              <option value={60}>60일</option>
              <option value={90}>90일</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">진입 Z-Score</label>
            <input
              type="number"
              value={entryZScore}
              onChange={(e) => setEntryZScore(Number(e.target.value))}
              step="0.1"
              min="1"
              max="3"
              className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">청산 Z-Score</label>
            <input
              type="number"
              value={exitZScore}
              onChange={(e) => setExitZScore(Number(e.target.value))}
              step="0.1"
              min="0"
              max="1"
              className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
            />
          </div>
        </div>
        
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="mt-4 text-sm text-blue-600 dark:text-blue-400 hover:underline"
        >
          고급 설정 {showAdvanced ? '숨기기' : '보기'}
        </button>
        
        {showAdvanced && (
          <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">손절 Z-Score</label>
                <input
                  type="number"
                  value={stopLossZScore}
                  onChange={(e) => setStopLossZScore(Number(e.target.value))}
                  step="0.1"
                  min="2.5"
                  max="4"
                  className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 뷰 모드 선택 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-2 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex gap-2">
          <button
            onClick={() => setViewMode('analysis')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === 'analysis'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            스프레드 분석
          </button>
          <button
            onClick={() => setViewMode('signals')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === 'signals'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            실시간 신호
          </button>
          <button
            onClick={() => setViewMode('backtest')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === 'backtest'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            백테스트
          </button>
        </div>
      </div>

      {/* 스프레드 분석 */}
      {viewMode === 'analysis' && (
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">스프레드 차트</h3>
            
            {/* 가격 비율 차트 */}
            <div className="h-64 relative bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <div className="absolute top-2 left-2 text-sm">
                <span className="font-medium">{selectedPair.stock1.symbol}/{selectedPair.stock2.symbol}</span>
              </div>
              
              {/* Y축 레이블 */}
              <div className="absolute left-0 top-0 bottom-0 flex flex-col justify-between text-xs text-gray-500">
                <span>{(spreadData.meanSpread + 2 * spreadData.stdSpread).toFixed(3)}</span>
                <span>{spreadData.meanSpread.toFixed(3)}</span>
                <span>{(spreadData.meanSpread - 2 * spreadData.stdSpread).toFixed(3)}</span>
              </div>
              
              {/* 차트 영역 */}
              <div className="ml-12 h-full relative">
                {/* 평균선 */}
                <div className="absolute w-full border-t-2 border-gray-400 dark:border-gray-600" style={{ top: '50%' }} />
                
                {/* 2 표준편차 밴드 */}
                <div className="absolute w-full border-t border-dashed border-red-400" style={{ top: '16.7%' }} />
                <div className="absolute w-full border-t border-dashed border-red-400" style={{ bottom: '16.7%' }} />
                
                {/* 스프레드 라인 */}
                <svg className="w-full h-full">
                  <polyline
                    fill="none"
                    stroke="rgb(59, 130, 246)"
                    strokeWidth="2"
                    points={spreadData.spread.map((s, i) => {
                      const x = (i / (spreadData.spread.length - 1)) * 100;
                      const normalized = (s - spreadData.meanSpread) / (spreadData.stdSpread * 4) + 0.5;
                      const y = (1 - normalized) * 100;
                      return `${x}%,${y}%`;
                    }).join(' ')}
                  />
                </svg>
              </div>
              
              {/* X축 레이블 */}
              <div className="ml-12 mt-2 flex justify-between text-xs text-gray-500">
                <span>-{timeframe}일</span>
                <span>현재</span>
              </div>
            </div>
            
            {/* Z-Score 차트 */}
            <div className="mt-6 h-32 relative bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <div className="absolute top-2 left-2 text-sm">
                <span className="font-medium">Z-Score</span>
              </div>
              
              {/* 신호 영역 */}
              <div className="ml-12 h-full relative">
                <div className="absolute w-full h-full">
                  {/* 매도 영역 */}
                  <div className="absolute w-full bg-red-100 dark:bg-red-900/20" style={{ top: 0, height: '16.7%' }} />
                  {/* 매수 영역 */}
                  <div className="absolute w-full bg-green-100 dark:bg-green-900/20" style={{ bottom: 0, height: '16.7%' }} />
                </div>
                
                {/* 임계값 라인 */}
                <div className="absolute w-full border-t border-red-500" style={{ top: '16.7%' }} />
                <div className="absolute w-full border-t border-gray-400" style={{ top: '50%' }} />
                <div className="absolute w-full border-t border-green-500" style={{ bottom: '16.7%' }} />
              </div>
            </div>
          </div>
          
          {/* 통계 정보 */}
          <div className="grid md:grid-cols-4 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">현재 스프레드</p>
              <p className="text-xl font-bold">{spreadData.currentSpread.toFixed(4)}</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">평균 스프레드</p>
              <p className="text-xl font-bold">{spreadData.meanSpread.toFixed(4)}</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">표준편차</p>
              <p className="text-xl font-bold">{spreadData.stdSpread.toFixed(4)}</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">Z-Score</p>
              <p className={`text-xl font-bold ${
                Math.abs(spreadData.zScore) > 2 ? 'text-red-600' : 'text-gray-900 dark:text-gray-100'
              }`}>
                {spreadData.zScore.toFixed(2)}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* 실시간 신호 */}
      {viewMode === 'signals' && (
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5" />
              현재 트레이딩 신호
            </h3>
            
            <div className={`text-center py-8 px-4 rounded-lg ${
              currentSignal === 'long' ? 'bg-green-50 dark:bg-green-900/20' :
              currentSignal === 'short' ? 'bg-red-50 dark:bg-red-900/20' :
              'bg-gray-50 dark:bg-gray-900'
            }`}>
              <p className={`text-3xl font-bold mb-2 ${getSignalColor(currentSignal)}`}>
                {getSignalText(currentSignal)}
              </p>
              <p className="text-lg text-gray-600 dark:text-gray-400">
                현재 Z-Score: {spreadData.zScore.toFixed(2)}
              </p>
            </div>
            
            {currentSignal !== 'neutral' && (
              <div className="mt-4 grid md:grid-cols-2 gap-4">
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <h4 className="font-medium mb-2">추천 포지션</h4>
                  {currentSignal === 'long' ? (
                    <div className="text-sm space-y-1">
                      <p>• {selectedPair.stock1.symbol} 매수</p>
                      <p>• {selectedPair.stock2.symbol} 매도</p>
                      <p className="text-gray-600 dark:text-gray-400 mt-2">
                        스프레드가 평균 이하로 축소됨. 확대 예상
                      </p>
                    </div>
                  ) : (
                    <div className="text-sm space-y-1">
                      <p>• {selectedPair.stock1.symbol} 매도</p>
                      <p>• {selectedPair.stock2.symbol} 매수</p>
                      <p className="text-gray-600 dark:text-gray-400 mt-2">
                        스프레드가 평균 이상으로 확대됨. 축소 예상
                      </p>
                    </div>
                  )}
                </div>
                
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <h4 className="font-medium mb-2">리스크 관리</h4>
                  <div className="text-sm space-y-1">
                    <p>• 목표 청산: Z-Score {exitZScore.toFixed(1)}</p>
                    <p>• 손절 기준: Z-Score {stopLossZScore.toFixed(1)}</p>
                    <p>• 예상 보유기간: {selectedPair.halfLife}일</p>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* 실시간 가격 정보 */}
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-semibold">{selectedPair.stock1.symbol}</h4>
                <span className="text-xs text-gray-500">{selectedPair.stock1.name}</span>
              </div>
              <p className="text-2xl font-bold mb-2">${selectedPair.stock1.price.toFixed(2)}</p>
              <div className="flex items-center gap-2 text-sm">
                <TrendingUp className="w-4 h-4 text-green-500" />
                <span className="text-green-600">+1.23 (+0.62%)</span>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-semibold">{selectedPair.stock2.symbol}</h4>
                <span className="text-xs text-gray-500">{selectedPair.stock2.name}</span>
              </div>
              <p className="text-2xl font-bold mb-2">${selectedPair.stock2.price.toFixed(2)}</p>
              <div className="flex items-center gap-2 text-sm">
                <TrendingDown className="w-4 h-4 text-red-500" />
                <span className="text-red-600">-0.89 (-0.52%)</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 백테스트 결과 */}
      {viewMode === 'backtest' && (
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">백테스트 결과</h3>
            
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
              <div className="text-center">
                <p className="text-sm text-gray-600 dark:text-gray-400">총 거래수</p>
                <p className="text-2xl font-bold">{backtestResult.totalTrades}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600 dark:text-gray-400">승률</p>
                <p className="text-2xl font-bold text-green-600">{backtestResult.winRate.toFixed(1)}%</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600 dark:text-gray-400">평균 수익</p>
                <p className="text-2xl font-bold">{backtestResult.avgProfit.toFixed(2)}%</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600 dark:text-gray-400">최대 낙폭</p>
                <p className="text-2xl font-bold text-red-600">{backtestResult.maxDrawdown.toFixed(2)}%</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600 dark:text-gray-400">샤프 비율</p>
                <p className="text-2xl font-bold">{backtestResult.sharpeRatio.toFixed(2)}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600 dark:text-gray-400">총 수익률</p>
                <p className="text-2xl font-bold text-blue-600">{backtestResult.totalReturn.toFixed(2)}%</p>
              </div>
            </div>
            
            {/* 최근 거래 내역 */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-600 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                    <th className="pb-2">진입일</th>
                    <th className="pb-2">청산일</th>
                    <th className="pb-2">타입</th>
                    <th className="pb-2 text-right">진입 스프레드</th>
                    <th className="pb-2 text-right">청산 스프레드</th>
                    <th className="pb-2 text-right">수익률</th>
                    <th className="pb-2 text-right">보유일수</th>
                  </tr>
                </thead>
                <tbody>
                  {backtestResult.trades.map((trade, idx) => (
                    <tr key={idx} className="border-b border-gray-100 dark:border-gray-800">
                      <td className="py-2">{trade.entryDate}</td>
                      <td className="py-2">{trade.exitDate}</td>
                      <td className="py-2">
                        <span className={`font-medium ${trade.type === 'long' ? 'text-green-600' : 'text-red-600'}`}>
                          {trade.type === 'long' ? '매수' : '매도'}
                        </span>
                      </td>
                      <td className="py-2 text-right">{trade.entrySpread.toFixed(4)}</td>
                      <td className="py-2 text-right">{trade.exitSpread.toFixed(4)}</td>
                      <td className={`py-2 text-right font-medium ${trade.profit > 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {trade.profit > 0 ? '+' : ''}{trade.profit.toFixed(2)}%
                      </td>
                      <td className="py-2 text-right">{trade.duration}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* 페어 트레이딩 가이드 */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Info className="w-5 h-5" />
          페어 트레이딩 전략 가이드
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-3">페어 트레이딩이란?</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              상관관계가 높은 두 자산 간의 가격 차이(스프레드)가 평균으로 회귀하는 특성을 이용한 시장 중립적 전략입니다.
            </p>
            <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
              <li>• 시장 방향성과 무관한 수익 창출</li>
              <li>• 통계적 차익거래 기회 포착</li>
              <li>• 상대적으로 낮은 리스크</li>
              <li>• 높은 샤프 비율 달성 가능</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-3">성공적인 페어 선택 기준</h4>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>높은 상관계수 (0.8 이상)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>공적분 관계 존재 (p-value < 0.05)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>동일 섹터 또는 유사 비즈니스 모델</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>충분한 유동성과 낮은 거래 비용</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}