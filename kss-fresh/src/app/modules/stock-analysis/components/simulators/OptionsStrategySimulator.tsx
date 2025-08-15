'use client';

import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Shield, AlertCircle, Info, BarChart3, Calculator, Target } from 'lucide-react';

interface OptionLeg {
  type: 'call' | 'put';
  position: 'long' | 'short';
  strike: number;
  premium: number;
  quantity: number;
  expiry: string;
}

interface Strategy {
  name: string;
  description: string;
  legs: OptionLeg[];
  maxProfit: number | 'unlimited';
  maxLoss: number | 'unlimited';
  breakeven: number[];
  riskReward: number;
  sentiment: 'bullish' | 'bearish' | 'neutral' | 'volatile';
}

interface Greeks {
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
}

// 옵션 전략 프리셋
const strategyPresets: Strategy[] = [
  {
    name: 'Covered Call',
    description: '보유 주식에 대해 콜옵션 매도하여 추가 수익 창출',
    legs: [
      { type: 'call', position: 'short', strike: 105, premium: 2.5, quantity: 1, expiry: '30일' }
    ],
    maxProfit: 7.5,
    maxLoss: 97.5,
    breakeven: [97.5],
    riskReward: 0.077,
    sentiment: 'neutral'
  },
  {
    name: 'Bull Call Spread',
    description: '제한된 상승 이익을 노리는 저비용 불리시 전략',
    legs: [
      { type: 'call', position: 'long', strike: 100, premium: 3, quantity: 1, expiry: '30일' },
      { type: 'call', position: 'short', strike: 105, premium: 1, quantity: 1, expiry: '30일' }
    ],
    maxProfit: 3,
    maxLoss: 2,
    breakeven: [102],
    riskReward: 1.5,
    sentiment: 'bullish'
  },
  {
    name: 'Iron Condor',
    description: '변동성이 낮을 때 수익을 내는 중립 전략',
    legs: [
      { type: 'put', position: 'short', strike: 95, premium: 1, quantity: 1, expiry: '30일' },
      { type: 'put', position: 'long', strike: 90, premium: 0.3, quantity: 1, expiry: '30일' },
      { type: 'call', position: 'short', strike: 105, premium: 1, quantity: 1, expiry: '30일' },
      { type: 'call', position: 'long', strike: 110, premium: 0.3, quantity: 1, expiry: '30일' }
    ],
    maxProfit: 1.4,
    maxLoss: 3.6,
    breakeven: [93.6, 106.4],
    riskReward: 0.39,
    sentiment: 'neutral'
  },
  {
    name: 'Long Straddle',
    description: '큰 변동성을 예상할 때 사용하는 전략',
    legs: [
      { type: 'call', position: 'long', strike: 100, premium: 3, quantity: 1, expiry: '30일' },
      { type: 'put', position: 'long', strike: 100, premium: 3, quantity: 1, expiry: '30일' }
    ],
    maxProfit: 'unlimited',
    maxLoss: 6,
    breakeven: [94, 106],
    riskReward: Infinity,
    sentiment: 'volatile'
  },
  {
    name: 'Bear Put Spread',
    description: '제한된 하락 이익을 노리는 베어리시 전략',
    legs: [
      { type: 'put', position: 'long', strike: 100, premium: 3, quantity: 1, expiry: '30일' },
      { type: 'put', position: 'short', strike: 95, premium: 1, quantity: 1, expiry: '30일' }
    ],
    maxProfit: 3,
    maxLoss: 2,
    breakeven: [98],
    riskReward: 1.5,
    sentiment: 'bearish'
  },
  {
    name: 'Butterfly Spread',
    description: '특정 가격에서 최대 이익을 노리는 저위험 전략',
    legs: [
      { type: 'call', position: 'long', strike: 95, premium: 4, quantity: 1, expiry: '30일' },
      { type: 'call', position: 'short', strike: 100, premium: 2, quantity: 2, expiry: '30일' },
      { type: 'call', position: 'long', strike: 105, premium: 0.5, quantity: 1, expiry: '30일' }
    ],
    maxProfit: 4.5,
    maxLoss: 0.5,
    breakeven: [95.5, 104.5],
    riskReward: 9,
    sentiment: 'neutral'
  }
];

export default function OptionsStrategySimulator() {
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy>(strategyPresets[0]);
  const [currentPrice, setCurrentPrice] = useState(100);
  const [volatility, setVolatility] = useState(25);
  const [daysToExpiry, setDaysToExpiry] = useState(30);
  const [customLegs, setCustomLegs] = useState<OptionLeg[]>([]);
  const [viewMode, setViewMode] = useState<'payoff' | 'greeks' | 'analysis'>('payoff');
  const [priceRange, setPriceRange] = useState({ min: 80, max: 120 });
  
  // 손익 계산
  const calculatePayoff = (price: number, strategy: Strategy): number => {
    let totalPayoff = 0;
    
    strategy.legs.forEach(leg => {
      const intrinsicValue = leg.type === 'call' 
        ? Math.max(0, price - leg.strike)
        : Math.max(0, leg.strike - price);
      
      const legPayoff = leg.position === 'long'
        ? (intrinsicValue - leg.premium) * leg.quantity * 100
        : (leg.premium - intrinsicValue) * leg.quantity * 100;
      
      totalPayoff += legPayoff;
    });
    
    return totalPayoff;
  };
  
  // Greeks 계산 (간단한 근사치)
  const calculateGreeks = (strategy: Strategy): Greeks => {
    let totalDelta = 0;
    let totalGamma = 0;
    let totalTheta = 0;
    let totalVega = 0;
    let totalRho = 0;
    
    strategy.legs.forEach(leg => {
      const moneyness = currentPrice / leg.strike;
      const timeDecay = Math.exp(-daysToExpiry / 365);
      
      // 간단한 Black-Scholes 근사
      const d1 = (Math.log(moneyness) + (0.05 + (volatility/100)**2 / 2) * (daysToExpiry/365)) / 
                 ((volatility/100) * Math.sqrt(daysToExpiry/365));
      
      const delta = leg.type === 'call' ? 0.5 + 0.5 * Math.tanh(d1) : -0.5 + 0.5 * Math.tanh(d1);
      const gamma = Math.exp(-d1**2/2) / (Math.sqrt(2 * Math.PI) * currentPrice * (volatility/100) * Math.sqrt(daysToExpiry/365));
      const theta = -currentPrice * gamma * (volatility/100) / (2 * Math.sqrt(daysToExpiry/365)) / 365;
      const vega = currentPrice * gamma * Math.sqrt(daysToExpiry/365) / 100;
      const rho = leg.type === 'call' ? leg.strike * timeDecay * delta / 100 : -leg.strike * timeDecay * delta / 100;
      
      const multiplier = leg.position === 'long' ? 1 : -1;
      totalDelta += delta * multiplier * leg.quantity;
      totalGamma += gamma * multiplier * leg.quantity;
      totalTheta += theta * multiplier * leg.quantity;
      totalVega += vega * multiplier * leg.quantity;
      totalRho += rho * multiplier * leg.quantity;
    });
    
    return {
      delta: totalDelta,
      gamma: totalGamma,
      theta: totalTheta,
      vega: totalVega,
      rho: totalRho
    };
  };
  
  // 손익 차트 데이터 생성
  const generatePayoffData = () => {
    const data = [];
    const step = (priceRange.max - priceRange.min) / 100;
    
    for (let price = priceRange.min; price <= priceRange.max; price += step) {
      data.push({
        price: price,
        payoff: calculatePayoff(price, selectedStrategy)
      });
    }
    
    return data;
  };
  
  const payoffData = generatePayoffData();
  const maxPayoff = Math.max(...payoffData.map(d => d.payoff));
  const minPayoff = Math.min(...payoffData.map(d => d.payoff));
  const payoffRange = maxPayoff - minPayoff;
  
  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish': return 'text-green-600';
      case 'bearish': return 'text-red-600';
      case 'neutral': return 'text-gray-600';
      case 'volatile': return 'text-purple-600';
      default: return 'text-gray-600';
    }
  };
  
  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish': return '📈';
      case 'bearish': return '📉';
      case 'neutral': return '➡️';
      case 'volatile': return '⚡';
      default: return '📊';
    }
  };

  return (
    <div className="space-y-6">
      {/* 전략 선택 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4">옵션 전략 선택</h3>
        
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {strategyPresets.map((strategy) => (
            <button
              key={strategy.name}
              onClick={() => setSelectedStrategy(strategy)}
              className={`p-4 rounded-lg transition-all text-left ${
                selectedStrategy.name === strategy.name
                  ? 'bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-500'
                  : 'bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-medium">{strategy.name}</h4>
                <span className="text-2xl">{getSentimentIcon(strategy.sentiment)}</span>
              </div>
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                {strategy.description}
              </p>
              <div className="flex items-center justify-between text-xs">
                <span className={getSentimentColor(strategy.sentiment)}>
                  {strategy.sentiment.charAt(0).toUpperCase() + strategy.sentiment.slice(1)}
                </span>
                <span className="font-medium">
                  R/R: {strategy.riskReward === Infinity ? '∞' : strategy.riskReward.toFixed(1)}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* 파라미터 설정 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Calculator className="w-5 h-5" />
          파라미터 설정
        </h3>
        
        <div className="grid md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium mb-2">현재 주가</label>
            <input
              type="number"
              value={currentPrice}
              onChange={(e) => setCurrentPrice(Number(e.target.value))}
              className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
              step="1"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">변동성 (IV) %</label>
            <input
              type="number"
              value={volatility}
              onChange={(e) => setVolatility(Number(e.target.value))}
              className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
              step="1"
              min="0"
              max="100"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">만기까지 일수</label>
            <input
              type="number"
              value={daysToExpiry}
              onChange={(e) => setDaysToExpiry(Number(e.target.value))}
              className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
              step="1"
              min="1"
              max="365"
            />
          </div>
        </div>
      </div>

      {/* 뷰 모드 선택 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-2 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex gap-2">
          <button
            onClick={() => setViewMode('payoff')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === 'payoff'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            손익 차트
          </button>
          <button
            onClick={() => setViewMode('greeks')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === 'greeks'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            Greeks
          </button>
          <button
            onClick={() => setViewMode('analysis')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === 'analysis'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            전략 분석
          </button>
        </div>
      </div>

      {/* 손익 차트 */}
      {viewMode === 'payoff' && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-4">만기 손익 구조</h3>
          
          <div className="relative h-64 mb-4">
            {/* Y축 */}
            <div className="absolute left-0 top-0 bottom-0 flex flex-col justify-between text-xs text-gray-500">
              <span>${maxPayoff.toFixed(0)}</span>
              <span>$0</span>
              <span>${minPayoff.toFixed(0)}</span>
            </div>
            
            {/* 차트 영역 */}
            <div className="ml-12 h-full relative bg-gray-50 dark:bg-gray-900 rounded">
              {/* 0선 */}
              <div 
                className="absolute w-full border-t-2 border-gray-400 dark:border-gray-600"
                style={{ top: `${(maxPayoff / payoffRange) * 100}%` }}
              />
              
              {/* 손익 곡선 */}
              <svg className="w-full h-full">
                <polyline
                  fill="none"
                  stroke="rgb(59, 130, 246)"
                  strokeWidth="2"
                  points={payoffData.map((d, i) => 
                    `${(i / payoffData.length) * 100}%,${((maxPayoff - d.payoff) / payoffRange) * 100}%`
                  ).join(' ')}
                />
                
                {/* 현재가 표시 */}
                <line
                  x1={`${((currentPrice - priceRange.min) / (priceRange.max - priceRange.min)) * 100}%`}
                  y1="0"
                  x2={`${((currentPrice - priceRange.min) / (priceRange.max - priceRange.min)) * 100}%`}
                  y2="100%"
                  stroke="rgb(239, 68, 68)"
                  strokeWidth="2"
                  strokeDasharray="5,5"
                />
              </svg>
              
              {/* 손익분기점 표시 */}
              {selectedStrategy.breakeven.map((be, idx) => (
                <div
                  key={idx}
                  className="absolute top-0 bottom-0 w-0.5 bg-green-500"
                  style={{ left: `${((be - priceRange.min) / (priceRange.max - priceRange.min)) * 100}%` }}
                >
                  <span className="absolute -top-6 -left-4 text-xs text-green-600 font-medium">
                    ${be}
                  </span>
                </div>
              ))}
            </div>
            
            {/* X축 */}
            <div className="ml-12 mt-2 flex justify-between text-xs text-gray-500">
              <span>${priceRange.min}</span>
              <span>${currentPrice} (현재)</span>
              <span>${priceRange.max}</span>
            </div>
          </div>
          
          {/* 주요 지표 */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
              <p className="text-sm text-gray-600 dark:text-gray-400">최대 이익</p>
              <p className="text-lg font-bold text-green-600">
                {selectedStrategy.maxProfit === 'unlimited' 
                  ? '무제한' 
                  : `$${(selectedStrategy.maxProfit * 100).toFixed(0)}`}
              </p>
            </div>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
              <p className="text-sm text-gray-600 dark:text-gray-400">최대 손실</p>
              <p className="text-lg font-bold text-red-600">
                {selectedStrategy.maxLoss === 'unlimited' 
                  ? '무제한' 
                  : `$${(selectedStrategy.maxLoss * 100).toFixed(0)}`}
              </p>
            </div>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
              <p className="text-sm text-gray-600 dark:text-gray-400">손익분기점</p>
              <p className="text-lg font-bold">
                {selectedStrategy.breakeven.map(be => `$${be}`).join(', ')}
              </p>
            </div>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
              <p className="text-sm text-gray-600 dark:text-gray-400">위험/보상 비율</p>
              <p className="text-lg font-bold">
                {selectedStrategy.riskReward === Infinity ? '∞' : `1:${selectedStrategy.riskReward.toFixed(1)}`}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Greeks */}
      {viewMode === 'greeks' && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-4">Greeks 분석</h3>
          
          {(() => {
            const greeks = calculateGreeks(selectedStrategy);
            return (
              <div className="space-y-4">
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium">Delta (Δ)</h4>
                        <span className={`text-2xl font-bold ${greeks.delta > 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {greeks.delta.toFixed(3)}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        주가가 $1 변할 때 옵션 가격 변화
                      </p>
                    </div>
                    
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium">Gamma (Γ)</h4>
                        <span className="text-2xl font-bold">
                          {greeks.gamma.toFixed(4)}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        주가가 $1 변할 때 델타의 변화
                      </p>
                    </div>
                    
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium">Theta (Θ)</h4>
                        <span className="text-2xl font-bold text-red-600">
                          {greeks.theta.toFixed(2)}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        하루 지날 때 옵션 가격 변화
                      </p>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium">Vega (ν)</h4>
                        <span className="text-2xl font-bold">
                          {greeks.vega.toFixed(3)}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        변동성이 1% 변할 때 옵션 가격 변화
                      </p>
                    </div>
                    
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium">Rho (ρ)</h4>
                        <span className="text-2xl font-bold">
                          {greeks.rho.toFixed(3)}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        금리가 1% 변할 때 옵션 가격 변화
                      </p>
                    </div>
                    
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                      <h4 className="font-medium mb-2 flex items-center gap-2">
                        <Info className="w-4 h-4" />
                        포지션 민감도
                      </h4>
                      <p className="text-sm">
                        이 전략은 {Math.abs(greeks.delta) > 0.5 ? '방향성' : '중립적'} 포지션이며,
                        {greeks.theta < 0 ? ' 시간 가치 소멸에 취약' : ' 시간 가치를 수익으로 전환'}합니다.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* 전략 분석 */}
      {viewMode === 'analysis' && (
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">전략 상세 분석</h3>
            
            <div className="space-y-4">
              {/* 전략 구성 */}
              <div>
                <h4 className="font-medium mb-3">전략 구성</h4>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="text-left text-sm text-gray-600 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                        <th className="pb-2">포지션</th>
                        <th className="pb-2">타입</th>
                        <th className="pb-2">행사가</th>
                        <th className="pb-2">프리미엄</th>
                        <th className="pb-2">수량</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedStrategy.legs.map((leg, idx) => (
                        <tr key={idx} className="border-b border-gray-100 dark:border-gray-800">
                          <td className="py-2">
                            <span className={`font-medium ${leg.position === 'long' ? 'text-green-600' : 'text-red-600'}`}>
                              {leg.position === 'long' ? '매수' : '매도'}
                            </span>
                          </td>
                          <td className="py-2 capitalize">{leg.type}</td>
                          <td className="py-2">${leg.strike}</td>
                          <td className="py-2">${leg.premium}</td>
                          <td className="py-2">{leg.quantity}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              
              {/* 시장 전망 */}
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <h4 className="font-medium mb-3">적합한 시장 상황</h4>
                  <ul className="text-sm space-y-2">
                    {selectedStrategy.sentiment === 'bullish' && (
                      <>
                        <li className="flex items-start gap-2">
                          <span className="text-green-500">✓</span>
                          <span>상승 추세가 예상될 때</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <span className="text-green-500">✓</span>
                          <span>긍정적인 뉴스나 실적 발표 전</span>
                        </li>
                      </>
                    )}
                    {selectedStrategy.sentiment === 'bearish' && (
                      <>
                        <li className="flex items-start gap-2">
                          <span className="text-green-500">✓</span>
                          <span>하락 추세가 예상될 때</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <span className="text-green-500">✓</span>
                          <span>부정적인 시장 환경</span>
                        </li>
                      </>
                    )}
                    {selectedStrategy.sentiment === 'neutral' && (
                      <>
                        <li className="flex items-start gap-2">
                          <span className="text-green-500">✓</span>
                          <span>횡보장이 예상될 때</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <span className="text-green-500">✓</span>
                          <span>변동성이 낮을 것으로 예상</span>
                        </li>
                      </>
                    )}
                    {selectedStrategy.sentiment === 'volatile' && (
                      <>
                        <li className="flex items-start gap-2">
                          <span className="text-green-500">✓</span>
                          <span>큰 가격 변동이 예상될 때</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <span className="text-green-500">✓</span>
                          <span>중요한 이벤트 전</span>
                        </li>
                      </>
                    )}
                  </ul>
                </div>
                
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <h4 className="font-medium mb-3">리스크 관리</h4>
                  <ul className="text-sm space-y-2">
                    <li className="flex items-start gap-2">
                      <span className="text-yellow-500">⚠</span>
                      <span>최대 손실을 포트폴리오의 2-5% 이내로 제한</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-yellow-500">⚠</span>
                      <span>만기 전 조기 청산 계획 수립</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-yellow-500">⚠</span>
                      <span>변동성 급변 시 포지션 조정</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
          
          {/* 실행 가이드 */}
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">실행 가이드</h3>
            <div className="grid md:grid-cols-3 gap-4">
              <div>
                <h4 className="font-medium mb-2">진입 시점</h4>
                <ul className="text-sm space-y-1">
                  <li>• 기술적 지지/저항 확인</li>
                  <li>• IV 수준 체크</li>
                  <li>• 이벤트 일정 확인</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">포지션 관리</h4>
                <ul className="text-sm space-y-1">
                  <li>• 목표 수익률 도달 시 청산</li>
                  <li>• 손절선 엄격히 준수</li>
                  <li>• 만기 2주 전 평가</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">조정 전략</h4>
                <ul className="text-sm space-y-1">
                  <li>• 롤링 (만기 연장)</li>
                  <li>• 스프레드 폭 조정</li>
                  <li>• 방어적 헤지 추가</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}