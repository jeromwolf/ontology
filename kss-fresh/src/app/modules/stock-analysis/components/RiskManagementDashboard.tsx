'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Shield, AlertTriangle, TrendingDown, Activity, DollarSign, BarChart3, PieChart, AlertCircle } from 'lucide-react';

interface Portfolio {
  ticker: string;
  name: string;
  shares: number;
  avgPrice: number;
  currentPrice: number;
  weight: number;
  beta: number;
  volatility: number;
  sector: string;
}

interface RiskMetrics {
  portfolioValue: number;
  dailyVaR95: number;
  dailyVaR99: number;
  maxDrawdown: number;
  sharpeRatio: number;
  beta: number;
  standardDeviation: number;
  trackingError: number;
}

export default function RiskManagementDashboard() {
  const correlationCanvasRef = useRef<HTMLCanvasElement>(null);
  const varChartRef = useRef<HTMLCanvasElement>(null);
  
  const [portfolio] = useState<Portfolio[]>([
    { ticker: 'AAPL', name: 'Apple Inc.', shares: 100, avgPrice: 150, currentPrice: 178, weight: 0.25, beta: 1.2, volatility: 0.22, sector: 'Technology' },
    { ticker: 'MSFT', name: 'Microsoft Corp.', shares: 80, avgPrice: 280, currentPrice: 420, weight: 0.35, beta: 1.1, volatility: 0.20, sector: 'Technology' },
    { ticker: 'JPM', name: 'JP Morgan Chase', shares: 150, avgPrice: 120, currentPrice: 185, weight: 0.20, beta: 1.3, volatility: 0.25, sector: 'Finance' },
    { ticker: 'JNJ', name: 'Johnson & Johnson', shares: 120, avgPrice: 160, currentPrice: 155, weight: 0.15, beta: 0.7, volatility: 0.15, sector: 'Healthcare' },
    { ticker: 'XOM', name: 'Exxon Mobil', shares: 100, avgPrice: 90, currentPrice: 105, weight: 0.05, beta: 0.9, volatility: 0.28, sector: 'Energy' }
  ]);

  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics>({
    portfolioValue: 125000,
    dailyVaR95: -2500,
    dailyVaR99: -3800,
    maxDrawdown: -15.2,
    sharpeRatio: 1.45,
    beta: 1.08,
    standardDeviation: 0.18,
    trackingError: 0.05
  });

  const [scenarioResults] = useState([
    { name: '2008 금융위기', impact: -38.5, probability: 5 },
    { name: '코로나19 팬데믹', impact: -25.3, probability: 10 },
    { name: '금리 200bp 상승', impact: -12.5, probability: 30 },
    { name: '경기 침체', impact: -18.7, probability: 25 },
    { name: '기술주 조정', impact: -22.1, probability: 15 }
  ]);

  useEffect(() => {
    drawCorrelationMatrix();
    drawVaRChart();
  }, []);

  const drawCorrelationMatrix = () => {
    const canvas = correlationCanvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = 300;
    const cellSize = size / portfolio.length;
    
    ctx.clearRect(0, 0, size, size);

    // Mock correlation data
    const correlations = [
      [1.00, 0.85, 0.42, 0.15, 0.28],
      [0.85, 1.00, 0.38, 0.22, 0.31],
      [0.42, 0.38, 1.00, 0.05, 0.52],
      [0.15, 0.22, 0.05, 1.00, -0.12],
      [0.28, 0.31, 0.52, -0.12, 1.00]
    ];

    // Draw correlation cells
    portfolio.forEach((stock1, i) => {
      portfolio.forEach((stock2, j) => {
        const correlation = correlations[i][j];
        const intensity = Math.abs(correlation);
        
        if (correlation > 0) {
          ctx.fillStyle = `rgba(59, 130, 246, ${intensity})`;
        } else {
          ctx.fillStyle = `rgba(239, 68, 68, ${intensity})`;
        }
        
        ctx.fillRect(j * cellSize, i * cellSize, cellSize - 2, cellSize - 2);
        
        // Draw correlation value
        ctx.fillStyle = intensity > 0.5 ? 'white' : '#333';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(
          correlation.toFixed(2),
          j * cellSize + cellSize / 2,
          i * cellSize + cellSize / 2
        );
      });
    });

    // Draw labels
    ctx.fillStyle = '#666';
    ctx.font = '10px sans-serif';
    portfolio.forEach((stock, i) => {
      ctx.save();
      ctx.translate(i * cellSize + cellSize / 2, -5);
      ctx.rotate(-Math.PI / 4);
      ctx.textAlign = 'right';
      ctx.fillText(stock.ticker, 0, 0);
      ctx.restore();
      
      ctx.textAlign = 'right';
      ctx.fillText(stock.ticker, -5, i * cellSize + cellSize / 2);
    });
  };

  const drawVaRChart = () => {
    const canvas = varChartRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = 400;
    const height = 200;
    
    ctx.clearRect(0, 0, width, height);

    // Generate normal distribution
    const mean = 0;
    const stdDev = riskMetrics.standardDeviation * 100;
    const xMin = -4 * stdDev;
    const xMax = 4 * stdDev;
    
    // Draw distribution curve
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let x = xMin; x <= xMax; x += 1) {
      const y = (1 / (stdDev * Math.sqrt(2 * Math.PI))) * 
                Math.exp(-0.5 * Math.pow((x - mean) / stdDev, 2));
      const canvasX = ((x - xMin) / (xMax - xMin)) * width;
      const canvasY = height - (y * height * 1000);
      
      if (x === xMin) {
        ctx.moveTo(canvasX, canvasY);
      } else {
        ctx.lineTo(canvasX, canvasY);
      }
    }
    ctx.stroke();

    // Fill VaR areas
    // 95% VaR
    ctx.fillStyle = 'rgba(251, 191, 36, 0.3)';
    ctx.beginPath();
    const var95X = ((riskMetrics.dailyVaR95 / 100 * stdDev - xMin) / (xMax - xMin)) * width;
    ctx.moveTo(0, height);
    for (let x = xMin; x <= riskMetrics.dailyVaR95 / 100 * stdDev; x += 1) {
      const y = (1 / (stdDev * Math.sqrt(2 * Math.PI))) * 
                Math.exp(-0.5 * Math.pow((x - mean) / stdDev, 2));
      const canvasX = ((x - xMin) / (xMax - xMin)) * width;
      const canvasY = height - (y * height * 1000);
      ctx.lineTo(canvasX, canvasY);
    }
    ctx.lineTo(var95X, height);
    ctx.closePath();
    ctx.fill();

    // 99% VaR
    ctx.fillStyle = 'rgba(239, 68, 68, 0.3)';
    ctx.beginPath();
    const var99X = ((riskMetrics.dailyVaR99 / 100 * stdDev - xMin) / (xMax - xMin)) * width;
    ctx.moveTo(0, height);
    for (let x = xMin; x <= riskMetrics.dailyVaR99 / 100 * stdDev; x += 1) {
      const y = (1 / (stdDev * Math.sqrt(2 * Math.PI))) * 
                Math.exp(-0.5 * Math.pow((x - mean) / stdDev, 2));
      const canvasX = ((x - xMin) / (xMax - xMin)) * width;
      const canvasY = height - (y * height * 1000);
      ctx.lineTo(canvasX, canvasY);
    }
    ctx.lineTo(var99X, height);
    ctx.closePath();
    ctx.fill();

    // Labels
    ctx.fillStyle = '#666';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('수익률 분포', width / 2, 15);
    
    ctx.font = '10px sans-serif';
    ctx.fillText('VaR 95%', var95X, height - 5);
    ctx.fillText('VaR 99%', var99X, height - 5);
  };

  const calculatePortfolioReturn = () => {
    let totalCost = 0;
    let totalValue = 0;
    
    portfolio.forEach(stock => {
      totalCost += stock.shares * stock.avgPrice;
      totalValue += stock.shares * stock.currentPrice;
    });
    
    return ((totalValue - totalCost) / totalCost * 100).toFixed(2);
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <Shield className="w-8 h-8 text-blue-500" />
          통합 리스크 관리 대시보드
        </h2>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600 dark:text-gray-400">포트폴리오 가치:</span>
          <span className="text-xl font-bold text-gray-900 dark:text-white">
            ${riskMetrics.portfolioValue.toLocaleString()}
          </span>
        </div>
      </div>

      {/* Key Risk Metrics */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">일일 VaR (95%)</span>
            <AlertTriangle className="w-4 h-4 text-yellow-500" />
          </div>
          <div className="text-2xl font-bold text-yellow-600">
            ${Math.abs(riskMetrics.dailyVaR95).toLocaleString()}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            최대 예상 손실 (95% 신뢰수준)
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">최대 낙폭</span>
            <TrendingDown className="w-4 h-4 text-red-500" />
          </div>
          <div className="text-2xl font-bold text-red-600">
            {riskMetrics.maxDrawdown}%
          </div>
          <div className="text-xs text-gray-500 mt-1">
            과거 최고점 대비
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">샤프 비율</span>
            <Activity className="w-4 h-4 text-green-500" />
          </div>
          <div className="text-2xl font-bold text-green-600">
            {riskMetrics.sharpeRatio}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            위험 조정 수익률
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">포트폴리오 베타</span>
            <BarChart3 className="w-4 h-4 text-blue-500" />
          </div>
          <div className="text-2xl font-bold text-blue-600">
            {riskMetrics.beta}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            시장 민감도
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Portfolio Composition */}
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4">포트폴리오 구성</h3>
          <div className="space-y-2">
            {portfolio.map(stock => {
              const pnl = (stock.currentPrice - stock.avgPrice) * stock.shares;
              const pnlPercent = ((stock.currentPrice - stock.avgPrice) / stock.avgPrice * 100);
              
              return (
                <div key={stock.ticker} className="flex items-center justify-between p-2 bg-white dark:bg-gray-800 rounded">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-sm">{stock.ticker}</span>
                      <span className="text-xs text-gray-500">{stock.sector}</span>
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">
                      {stock.shares}주 @ ${stock.avgPrice}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`text-sm font-semibold ${pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      ${pnl >= 0 ? '+' : ''}{pnl.toFixed(0)}
                    </div>
                    <div className={`text-xs ${pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      ({pnlPercent >= 0 ? '+' : ''}{pnlPercent.toFixed(1)}%)
                    </div>
                  </div>
                  <div className="ml-4 text-right">
                    <div className="text-sm font-semibold">{(stock.weight * 100).toFixed(1)}%</div>
                    <div className="text-xs text-gray-500">β: {stock.beta}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Correlation Matrix */}
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4">상관관계 매트릭스</h3>
          <canvas
            ref={correlationCanvasRef}
            width={300}
            height={300}
            className="w-full max-w-[300px] mx-auto"
          />
          <div className="mt-2 text-xs text-gray-600 dark:text-gray-400 text-center">
            파랑: 양의 상관관계 | 빨강: 음의 상관관계
          </div>
        </div>

        {/* VaR Distribution */}
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4">VaR 분포</h3>
          <canvas
            ref={varChartRef}
            width={400}
            height={200}
            className="w-full"
          />
          <div className="mt-4 space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-600 dark:text-gray-400">95% VaR (일일)</span>
              <span className="font-semibold text-yellow-600">${Math.abs(riskMetrics.dailyVaR95).toLocaleString()}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-600 dark:text-gray-400">99% VaR (일일)</span>
              <span className="font-semibold text-red-600">${Math.abs(riskMetrics.dailyVaR99).toLocaleString()}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Scenario Analysis */}
      <div className="mt-6 bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-4">시나리오 분석</h3>
        <div className="grid grid-cols-5 gap-4">
          {scenarioResults.map(scenario => (
            <div key={scenario.name} className="bg-white dark:bg-gray-800 rounded-lg p-3">
              <div className="text-sm font-semibold text-gray-900 dark:text-white mb-2">
                {scenario.name}
              </div>
              <div className="text-2xl font-bold text-red-600 mb-1">
                {scenario.impact}%
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400">
                발생 확률: {scenario.probability}%
              </div>
              <div className="text-xs text-red-600 font-semibold mt-1">
                예상 손실: ${Math.abs(riskMetrics.portfolioValue * scenario.impact / 100).toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Risk Alerts */}
      <div className="mt-6 space-y-3">
        <div className="flex items-start gap-2 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
          <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5" />
          <div className="flex-1">
            <div className="text-sm font-semibold text-yellow-800 dark:text-yellow-200">
              섹터 집중도 경고
            </div>
            <div className="text-xs text-yellow-700 dark:text-yellow-300">
              기술주 비중이 60%를 초과했습니다. 섹터 분산을 고려하세요.
            </div>
          </div>
        </div>
        
        <div className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
          <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
          <div className="flex-1">
            <div className="text-sm font-semibold text-red-800 dark:text-red-200">
              변동성 증가
            </div>
            <div className="text-xs text-red-700 dark:text-red-300">
              포트폴리오 변동성이 지난주 대비 25% 증가했습니다.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}