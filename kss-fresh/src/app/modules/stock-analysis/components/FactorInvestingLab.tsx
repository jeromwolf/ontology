'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Microscope, TrendingUp, DollarSign, Shield, Activity, Zap, BarChart3, Target } from 'lucide-react';

interface Factor {
  name: string;
  description: string;
  currentScore: number;
  historicalReturn: number;
  sharpeRatio: number;
  icon: React.ElementType;
  color: string;
}

interface Stock {
  ticker: string;
  name: string;
  factorScores: {
    value: number;
    momentum: number;
    quality: number;
    lowVol: number;
    size: number;
  };
  compositeScore: number;
  expectedReturn: number;
}

export default function FactorInvestingLab() {
  const performanceChartRef = useRef<HTMLCanvasElement>(null);
  const factorCorrelationRef = useRef<HTMLCanvasElement>(null);
  
  const [selectedFactors, setSelectedFactors] = useState<string[]>(['value', 'momentum', 'quality']);
  const [backtestPeriod, setBacktestPeriod] = useState('5Y');
  
  const factors: Record<string, Factor> = {
    value: {
      name: '가치 (Value)',
      description: '저평가된 주식을 찾는 팩터 (PBR, PER, EV/EBITDA)',
      currentScore: 72,
      historicalReturn: 8.5,
      sharpeRatio: 0.65,
      icon: DollarSign,
      color: '#3b82f6'
    },
    momentum: {
      name: '모멘텀 (Momentum)',
      description: '상승 추세가 지속되는 주식 (6-12개월 수익률)',
      currentScore: 85,
      historicalReturn: 12.3,
      sharpeRatio: 0.82,
      icon: TrendingUp,
      color: '#10b981'
    },
    quality: {
      name: '퀄리티 (Quality)',
      description: '재무 건전성이 우수한 주식 (ROE, 부채비율)',
      currentScore: 78,
      historicalReturn: 10.2,
      sharpeRatio: 0.88,
      icon: Shield,
      color: '#8b5cf6'
    },
    lowVol: {
      name: '저변동성 (Low Volatility)',
      description: '변동성이 낮은 안정적인 주식',
      currentScore: 65,
      historicalReturn: 7.8,
      sharpeRatio: 0.95,
      icon: Activity,
      color: '#06b6d4'
    },
    size: {
      name: '소형주 (Size)',
      description: '시가총액이 작은 성장 가능성 높은 주식',
      currentScore: 70,
      historicalReturn: 9.5,
      sharpeRatio: 0.58,
      icon: Zap,
      color: '#f59e0b'
    }
  };

  const [topStocks] = useState<Stock[]>([
    {
      ticker: 'AAPL',
      name: 'Apple Inc.',
      factorScores: { value: 45, momentum: 92, quality: 95, lowVol: 78, size: 10 },
      compositeScore: 85,
      expectedReturn: 15.2
    },
    {
      ticker: 'BRK.B',
      name: 'Berkshire Hathaway',
      factorScores: { value: 88, momentum: 65, quality: 98, lowVol: 85, size: 5 },
      compositeScore: 82,
      expectedReturn: 12.8
    },
    {
      ticker: 'MSFT',
      name: 'Microsoft Corp.',
      factorScores: { value: 55, momentum: 88, quality: 92, lowVol: 75, size: 8 },
      compositeScore: 80,
      expectedReturn: 14.5
    },
    {
      ticker: 'JNJ',
      name: 'Johnson & Johnson',
      factorScores: { value: 78, momentum: 45, quality: 88, lowVol: 92, size: 15 },
      compositeScore: 76,
      expectedReturn: 10.2
    },
    {
      ticker: 'COST',
      name: 'Costco',
      factorScores: { value: 35, momentum: 78, quality: 85, lowVol: 70, size: 25 },
      compositeScore: 72,
      expectedReturn: 11.5
    }
  ]);

  const [backtestResults] = useState({
    annualReturn: 14.2,
    volatility: 12.5,
    sharpeRatio: 0.94,
    maxDrawdown: -18.5,
    winRate: 68,
    totalReturn: 95.3
  });

  useEffect(() => {
    drawPerformanceChart();
    drawFactorCorrelation();
  }, [selectedFactors]);

  const drawPerformanceChart = () => {
    const canvas = performanceChartRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Generate performance data
    const months = 60; // 5 years
    const data = {
      factorPortfolio: [100],
      benchmark: [100]
    };

    for (let i = 1; i <= months; i++) {
      const factorReturn = 1 + (Math.random() * 0.04 - 0.01); // -1% to 3% monthly
      const benchmarkReturn = 1 + (Math.random() * 0.03 - 0.01); // -1% to 2% monthly
      
      data.factorPortfolio.push(data.factorPortfolio[i-1] * factorReturn);
      data.benchmark.push(data.benchmark[i-1] * benchmarkReturn);
    }

    // Draw axes
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(50, 20);
    ctx.lineTo(50, height - 30);
    ctx.lineTo(width - 20, height - 30);
    ctx.stroke();

    // Draw performance lines
    const drawLine = (data: number[], color: string) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      data.forEach((value, index) => {
        const x = 50 + (index / (data.length - 1)) * (width - 70);
        const y = height - 30 - ((value - 80) / 120) * (height - 50);
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    };

    drawLine(data.factorPortfolio, '#3b82f6');
    drawLine(data.benchmark, '#9ca3af');

    // Draw labels
    ctx.fillStyle = '#666';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('수익률 비교 (5년)', width / 2, 15);

    // Legend
    ctx.fillStyle = '#3b82f6';
    ctx.fillRect(width - 150, 30, 20, 3);
    ctx.fillStyle = '#666';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('팩터 포트폴리오', width - 125, 35);

    ctx.fillStyle = '#9ca3af';
    ctx.fillRect(width - 150, 45, 20, 3);
    ctx.fillStyle = '#666';
    ctx.fillText('벤치마크', width - 125, 50);
  };

  const drawFactorCorrelation = () => {
    const canvas = factorCorrelationRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = 250;
    ctx.clearRect(0, 0, size, size);

    const factorList = Object.keys(factors);
    const cellSize = size / factorList.length;

    // Mock correlation matrix
    const correlations = [
      [1.00, -0.45, 0.25, 0.35, -0.15],
      [-0.45, 1.00, 0.15, -0.55, 0.30],
      [0.25, 0.15, 1.00, 0.65, -0.25],
      [0.35, -0.55, 0.65, 1.00, -0.40],
      [-0.15, 0.30, -0.25, -0.40, 1.00]
    ];

    // Draw correlation cells
    factorList.forEach((factor1, i) => {
      factorList.forEach((factor2, j) => {
        const correlation = correlations[i][j];
        const intensity = Math.abs(correlation);
        
        if (correlation > 0) {
          ctx.fillStyle = `rgba(59, 130, 246, ${intensity * 0.8})`;
        } else {
          ctx.fillStyle = `rgba(239, 68, 68, ${intensity * 0.8})`;
        }
        
        ctx.fillRect(j * cellSize + 1, i * cellSize + 1, cellSize - 2, cellSize - 2);
        
        // Draw value
        ctx.fillStyle = intensity > 0.5 ? 'white' : '#333';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(
          correlation.toFixed(2),
          j * cellSize + cellSize / 2,
          i * cellSize + cellSize / 2
        );
      });
    });
  };

  const toggleFactor = (factor: string) => {
    setSelectedFactors(prev => 
      prev.includes(factor)
        ? prev.filter(f => f !== factor)
        : [...prev, factor]
    );
  };

  const calculateCompositeScore = (stock: Stock) => {
    let totalScore = 0;
    let totalWeight = 0;
    
    selectedFactors.forEach(factor => {
      const score = stock.factorScores[factor as keyof typeof stock.factorScores];
      totalScore += score;
      totalWeight += 1;
    });
    
    return totalWeight > 0 ? totalScore / totalWeight : 0;
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <Microscope className="w-8 h-8 text-purple-500" />
          팩터 투자 연구소
        </h2>
        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-600 dark:text-gray-400">백테스트 기간:</span>
          <select
            value={backtestPeriod}
            onChange={(e) => setBacktestPeriod(e.target.value)}
            className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-sm"
          >
            <option value="1Y">1년</option>
            <option value="3Y">3년</option>
            <option value="5Y">5년</option>
            <option value="10Y">10년</option>
          </select>
        </div>
      </div>

      {/* Factor Selection */}
      <div className="grid grid-cols-5 gap-4 mb-6">
        {Object.entries(factors).map(([key, factor]) => {
          const IconComponent = factor.icon;
          const isSelected = selectedFactors.includes(key);
          
          return (
            <div
              key={key}
              onClick={() => toggleFactor(key)}
              className={`cursor-pointer p-4 rounded-lg border-2 transition-all ${
                isSelected
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <IconComponent className="w-6 h-6" style={{ color: factor.color }} />
                <div className={`w-4 h-4 rounded-full border-2 ${
                  isSelected
                    ? 'bg-blue-500 border-blue-500'
                    : 'bg-white dark:bg-gray-700 border-gray-300 dark:border-gray-600'
                }`}>
                  {isSelected && (
                    <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  )}
                </div>
              </div>
              <h3 className="font-semibold text-sm mb-1">{factor.name}</h3>
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">{factor.description}</p>
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">현재 스코어</span>
                  <span className="font-semibold">{factor.currentScore}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">연평균 수익</span>
                  <span className="font-semibold text-green-600">+{factor.historicalReturn}%</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">샤프 비율</span>
                  <span className="font-semibold">{factor.sharpeRatio}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Backtest Results */}
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4">백테스트 결과</h3>
          <canvas
            ref={performanceChartRef}
            width={400}
            height={200}
            className="w-full mb-4"
          />
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <span className="text-gray-600 dark:text-gray-400">연평균 수익률</span>
              <div className="font-semibold text-green-600">+{backtestResults.annualReturn}%</div>
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-400">변동성</span>
              <div className="font-semibold">{backtestResults.volatility}%</div>
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-400">샤프 비율</span>
              <div className="font-semibold text-blue-600">{backtestResults.sharpeRatio}</div>
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-400">최대 낙폭</span>
              <div className="font-semibold text-red-600">{backtestResults.maxDrawdown}%</div>
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-400">승률</span>
              <div className="font-semibold">{backtestResults.winRate}%</div>
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-400">누적 수익률</span>
              <div className="font-semibold text-green-600">+{backtestResults.totalReturn}%</div>
            </div>
          </div>
        </div>

        {/* Top Stocks */}
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4">팩터 기반 추천 종목</h3>
          <div className="space-y-2">
            {topStocks
              .sort((a, b) => calculateCompositeScore(b) - calculateCompositeScore(a))
              .map(stock => (
                <div key={stock.ticker} className="bg-white dark:bg-gray-800 rounded p-3">
                  <div className="flex items-center justify-between mb-2">
                    <div>
                      <span className="font-semibold">{stock.ticker}</span>
                      <span className="text-xs text-gray-500 ml-2">{stock.name}</span>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-bold text-blue-600">
                        {calculateCompositeScore(stock).toFixed(0)}점
                      </div>
                      <div className="text-xs text-green-600">
                        +{stock.expectedReturn}%
                      </div>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    {selectedFactors.map(factor => {
                      const score = stock.factorScores[factor as keyof typeof stock.factorScores];
                      return (
                        <div key={factor} className="flex-1">
                          <div className="text-xs text-gray-500 mb-1">
                            {factors[factor].name.split(' ')[0]}
                          </div>
                          <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                            <div
                              className="h-full"
                              style={{
                                width: `${score}%`,
                                backgroundColor: factors[factor].color
                              }}
                            />
                          </div>
                          <div className="text-xs text-center mt-1">{score}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
          </div>
        </div>

        {/* Factor Correlation */}
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4">팩터 상관관계</h3>
          <canvas
            ref={factorCorrelationRef}
            width={250}
            height={250}
            className="mx-auto mb-4"
          />
          <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
            <p>• Value ↔ Momentum: 음의 상관관계 (-0.45)</p>
            <p>• Quality ↔ Low Vol: 양의 상관관계 (+0.65)</p>
            <p>• Size ↔ Quality: 음의 상관관계 (-0.25)</p>
          </div>
        </div>
      </div>

      {/* Strategy Tips */}
      <div className="mt-6 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
        <div className="flex items-start gap-2">
          <Target className="w-5 h-5 text-purple-600 dark:text-purple-400 mt-0.5" />
          <div className="text-sm text-purple-800 dark:text-purple-300">
            <p className="font-semibold mb-1">멀티팩터 전략 활용 팁</p>
            <ul className="list-disc list-inside space-y-1 text-xs">
              <li>상관관계가 낮은 팩터들을 조합하여 분산 효과 극대화</li>
              <li>시장 상황에 따라 팩터 가중치를 동적으로 조정</li>
              <li>최소 3-5년 이상의 장기 투자 관점 유지</li>
              <li>정기적인 리밸런싱으로 팩터 노출도 관리</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}