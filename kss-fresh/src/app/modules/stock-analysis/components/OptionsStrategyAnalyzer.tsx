'use client';

import React, { useState, useEffect, useRef } from 'react';
import { TrendingUp, TrendingDown, AlertCircle, Activity, Calculator, BarChart3, PieChart, LineChart } from 'lucide-react';

interface OptionStrategy {
  name: string;
  type: string;
  legs: OptionLeg[];
  maxProfit: number | null;
  maxLoss: number | null;
  breakeven: number[];
  description: string;
}

interface OptionLeg {
  type: 'call' | 'put';
  position: 'long' | 'short';
  strike: number;
  premium: number;
  quantity: number;
}

interface Greeks {
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
}

export default function OptionsStrategyAnalyzer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [spotPrice, setSpotPrice] = useState(100);
  const [selectedStrategy, setSelectedStrategy] = useState('covered-call');
  const [currentStrategy, setCurrentStrategy] = useState<OptionStrategy | null>(null);
  const [greeks, setGreeks] = useState<Greeks>({
    delta: 0.45,
    gamma: 0.02,
    theta: -0.05,
    vega: 0.15,
    rho: 0.08
  });

  const strategies: Record<string, OptionStrategy> = {
    'covered-call': {
      name: '커버드 콜 (Covered Call)',
      type: 'income',
      legs: [
        { type: 'call', position: 'short', strike: 105, premium: 2.5, quantity: 1 }
      ],
      maxProfit: 750,
      maxLoss: -9750,
      breakeven: [97.5],
      description: '주식 보유 + 콜옵션 매도. 추가 수익 창출 전략'
    },
    'protective-put': {
      name: '보호적 풋 (Protective Put)',
      type: 'protection',
      legs: [
        { type: 'put', position: 'long', strike: 95, premium: 2, quantity: 1 }
      ],
      maxProfit: null,
      maxLoss: -700,
      breakeven: [102],
      description: '주식 보유 + 풋옵션 매수. 하락 리스크 헤지'
    },
    'bull-call-spread': {
      name: '불 콜 스프레드 (Bull Call Spread)',
      type: 'directional',
      legs: [
        { type: 'call', position: 'long', strike: 100, premium: 3, quantity: 1 },
        { type: 'call', position: 'short', strike: 105, premium: 1, quantity: 1 }
      ],
      maxProfit: 300,
      maxLoss: -200,
      breakeven: [102],
      description: '제한된 위험으로 상승 베팅'
    },
    'bear-put-spread': {
      name: '베어 풋 스프레드 (Bear Put Spread)',
      type: 'directional',
      legs: [
        { type: 'put', position: 'long', strike: 100, premium: 3, quantity: 1 },
        { type: 'put', position: 'short', strike: 95, premium: 1, quantity: 1 }
      ],
      maxProfit: 300,
      maxLoss: -200,
      breakeven: [98],
      description: '제한된 위험으로 하락 베팅'
    },
    'long-straddle': {
      name: '롱 스트래들 (Long Straddle)',
      type: 'volatility',
      legs: [
        { type: 'call', position: 'long', strike: 100, premium: 3, quantity: 1 },
        { type: 'put', position: 'long', strike: 100, premium: 3, quantity: 1 }
      ],
      maxProfit: null,
      maxLoss: -600,
      breakeven: [94, 106],
      description: '큰 변동성 예상 시 사용'
    },
    'iron-condor': {
      name: '아이언 콘도르 (Iron Condor)',
      type: 'neutral',
      legs: [
        { type: 'put', position: 'short', strike: 95, premium: 1, quantity: 1 },
        { type: 'put', position: 'long', strike: 90, premium: 0.5, quantity: 1 },
        { type: 'call', position: 'short', strike: 105, premium: 1, quantity: 1 },
        { type: 'call', position: 'long', strike: 110, premium: 0.5, quantity: 1 }
      ],
      maxProfit: 100,
      maxLoss: -400,
      breakeven: [94, 106],
      description: '횡보장에서 수익 창출'
    },
    'butterfly': {
      name: '버터플라이 (Butterfly)',
      type: 'neutral',
      legs: [
        { type: 'call', position: 'long', strike: 95, premium: 5, quantity: 1 },
        { type: 'call', position: 'short', strike: 100, premium: 2.5, quantity: 2 },
        { type: 'call', position: 'long', strike: 105, premium: 1, quantity: 1 }
      ],
      maxProfit: 300,
      maxLoss: -200,
      breakeven: [97, 103],
      description: '특정 가격대 도달 예상 시'
    },
    'calendar-spread': {
      name: '캘린더 스프레드 (Calendar Spread)',
      type: 'neutral',
      legs: [
        { type: 'call', position: 'short', strike: 100, premium: 2, quantity: 1 },
        { type: 'call', position: 'long', strike: 100, premium: 3.5, quantity: 1 }
      ],
      maxProfit: 150,
      maxLoss: -150,
      breakeven: [98.5, 101.5],
      description: '시간가치 차익 전략'
    }
  };

  useEffect(() => {
    setCurrentStrategy(strategies[selectedStrategy]);
  }, [selectedStrategy]);

  useEffect(() => {
    drawPayoffDiagram();
  }, [spotPrice, currentStrategy]);

  const drawPayoffDiagram = () => {
    const canvas = canvasRef.current;
    if (!canvas || !currentStrategy) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 40;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Set up coordinate system
    const priceRange = { min: 80, max: 120 };
    const profitRange = { min: -1000, max: 1000 };

    // Draw axes
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.stroke();

    // Draw payoff curve
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;
    ctx.beginPath();

    for (let price = priceRange.min; price <= priceRange.max; price += 0.5) {
      const profit = calculateProfit(price, currentStrategy);
      const x = padding + ((price - priceRange.min) / (priceRange.max - priceRange.min)) * (width - 2 * padding);
      const y = height - padding - ((profit - profitRange.min) / (profitRange.max - profitRange.min)) * (height - 2 * padding);

      if (price === priceRange.min) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw current spot price line
    const spotX = padding + ((spotPrice - priceRange.min) / (priceRange.max - priceRange.min)) * (width - 2 * padding);
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(spotX, padding);
    ctx.lineTo(spotX, height - padding);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw breakeven lines
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 1;
    currentStrategy.breakeven.forEach(be => {
      const x = padding + ((be - priceRange.min) / (priceRange.max - priceRange.min)) * (width - 2 * padding);
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();
    });
    ctx.setLineDash([]);

    // Draw labels
    ctx.fillStyle = '#333';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    
    // X-axis labels
    for (let price = priceRange.min; price <= priceRange.max; price += 10) {
      const x = padding + ((price - priceRange.min) / (priceRange.max - priceRange.min)) * (width - 2 * padding);
      ctx.fillText(price.toString(), x, height - padding + 20);
    }

    // Y-axis labels
    ctx.textAlign = 'right';
    for (let profit = profitRange.min; profit <= profitRange.max; profit += 500) {
      const y = height - padding - ((profit - profitRange.min) / (profitRange.max - profitRange.min)) * (height - 2 * padding);
      ctx.fillText(`$${profit}`, padding - 10, y + 5);
    }
  };

  const calculateProfit = (price: number, strategy: OptionStrategy): number => {
    let totalProfit = 0;

    strategy.legs.forEach(leg => {
      const intrinsicValue = leg.type === 'call' 
        ? Math.max(0, price - leg.strike)
        : Math.max(0, leg.strike - price);
      
      const profit = leg.position === 'long'
        ? (intrinsicValue - leg.premium) * 100 * leg.quantity
        : (leg.premium - intrinsicValue) * 100 * leg.quantity;
      
      totalProfit += profit;
    });

    // Add stock position for covered call and protective put
    if (strategy.name.includes('커버드 콜') || strategy.name.includes('보호적 풋')) {
      totalProfit += (price - 100) * 100; // Assuming stock bought at 100
    }

    return totalProfit;
  };

  const calculateGreeks = () => {
    // Simplified Greeks calculation
    const daysToExpiry = 30;
    const iv = 0.25; // 25% implied volatility
    
    return {
      delta: Math.random() * 0.8 + 0.1,
      gamma: Math.random() * 0.05,
      theta: -(Math.random() * 0.1 + 0.02),
      vega: Math.random() * 0.3,
      rho: Math.random() * 0.1
    };
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <TrendingUp className="w-8 h-8 text-blue-500" />
          옵션 전략 분석기
        </h2>
      </div>

      {/* Strategy Selector */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        {Object.entries(strategies).map(([key, strategy]) => (
          <button
            key={key}
            onClick={() => setSelectedStrategy(key)}
            className={`p-3 rounded-lg border-2 transition-all ${
              selectedStrategy === key
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
            }`}
          >
            <div className="text-sm font-semibold text-gray-900 dark:text-white">
              {strategy.name}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
              {strategy.type}
            </div>
          </button>
        ))}
      </div>

      {/* Current Strategy Details */}
      {currentStrategy && (
        <div className="grid grid-cols-3 gap-6 mb-6">
          {/* Payoff Diagram */}
          <div className="col-span-2 bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-4">손익 다이어그램</h3>
            <canvas
              ref={canvasRef}
              width={600}
              height={400}
              className="w-full border border-gray-200 dark:border-gray-700 rounded bg-white dark:bg-gray-800"
            />
            <div className="mt-4 flex items-center justify-between">
              <div className="text-sm text-gray-600 dark:text-gray-400">
                현재가: ${spotPrice}
              </div>
              <input
                type="range"
                min="80"
                max="120"
                value={spotPrice}
                onChange={(e) => setSpotPrice(Number(e.target.value))}
                className="w-48"
              />
            </div>
          </div>

          {/* Strategy Info */}
          <div className="space-y-4">
            {/* Key Metrics */}
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3">핵심 지표</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">최대 수익</span>
                  <span className={`text-sm font-semibold ${
                    currentStrategy.maxProfit === null ? 'text-green-600' : 'text-gray-900 dark:text-white'
                  }`}>
                    {currentStrategy.maxProfit === null ? '무제한' : `$${currentStrategy.maxProfit}`}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">최대 손실</span>
                  <span className={`text-sm font-semibold ${
                    currentStrategy.maxLoss === null ? 'text-red-600' : 'text-gray-900 dark:text-white'
                  }`}>
                    {currentStrategy.maxLoss === null ? '무제한' : `$${currentStrategy.maxLoss}`}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">손익분기점</span>
                  <span className="text-sm font-semibold text-gray-900 dark:text-white">
                    ${currentStrategy.breakeven.join(', $')}
                  </span>
                </div>
              </div>
            </div>

            {/* Greeks */}
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3">그릭스 (Greeks)</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Delta (Δ)</span>
                  <span className="text-sm font-semibold text-gray-900 dark:text-white">
                    {greeks.delta.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Gamma (Γ)</span>
                  <span className="text-sm font-semibold text-gray-900 dark:text-white">
                    {greeks.gamma.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Theta (Θ)</span>
                  <span className="text-sm font-semibold text-red-600">
                    {greeks.theta.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Vega (ν)</span>
                  <span className="text-sm font-semibold text-gray-900 dark:text-white">
                    {greeks.vega.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Rho (ρ)</span>
                  <span className="text-sm font-semibold text-gray-900 dark:text-white">
                    {greeks.rho.toFixed(3)}
                  </span>
                </div>
              </div>
            </div>

            {/* Strategy Description */}
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-2">전략 설명</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {currentStrategy.description}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Option Legs Detail */}
      {currentStrategy && (
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4">옵션 구성 (Option Legs)</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-2 px-4">유형</th>
                  <th className="text-left py-2 px-4">포지션</th>
                  <th className="text-left py-2 px-4">행사가</th>
                  <th className="text-left py-2 px-4">프리미엄</th>
                  <th className="text-left py-2 px-4">수량</th>
                  <th className="text-left py-2 px-4">현재 가치</th>
                </tr>
              </thead>
              <tbody>
                {currentStrategy.legs.map((leg, index) => {
                  const intrinsicValue = leg.type === 'call'
                    ? Math.max(0, spotPrice - leg.strike)
                    : Math.max(0, leg.strike - spotPrice);
                  
                  return (
                    <tr key={index} className="border-b border-gray-100 dark:border-gray-800">
                      <td className="py-2 px-4">
                        <span className={`font-semibold ${
                          leg.type === 'call' ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {leg.type === 'call' ? 'CALL' : 'PUT'}
                        </span>
                      </td>
                      <td className="py-2 px-4">
                        <span className={`${
                          leg.position === 'long' ? 'text-blue-600' : 'text-orange-600'
                        }`}>
                          {leg.position === 'long' ? '매수' : '매도'}
                        </span>
                      </td>
                      <td className="py-2 px-4">${leg.strike}</td>
                      <td className="py-2 px-4">${leg.premium.toFixed(2)}</td>
                      <td className="py-2 px-4">{leg.quantity}</td>
                      <td className="py-2 px-4">${intrinsicValue.toFixed(2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tips */}
      <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <div className="flex items-start gap-2">
          <AlertCircle className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
          <div className="text-sm text-blue-800 dark:text-blue-300">
            <p className="font-semibold mb-1">옵션 전략 활용 팁</p>
            <ul className="list-disc list-inside space-y-1 text-xs">
              <li>커버드 콜: 주식 보유 시 추가 수익 창출</li>
              <li>보호적 풋: 하락장 대비 보험 역할</li>
              <li>스프레드: 제한된 위험으로 방향성 베팅</li>
              <li>스트래들/스트랭글: 큰 변동성 예상 시</li>
              <li>아이언 콘도르: 횡보장에서 시간가치 수익</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}