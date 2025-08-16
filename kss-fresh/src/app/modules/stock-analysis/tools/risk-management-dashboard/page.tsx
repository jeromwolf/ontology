'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { 
  Activity, 
  AlertTriangle, 
  TrendingDown, 
  Shield, 
  BarChart3, 
  Calculator,
  Zap,
  PieChart,
  Eye,
  AlertCircle,
  Info,
  ChevronRight,
  ArrowUpRight,
  ArrowDownRight,
  Percent,
  DollarSign,
  Lock,
  Unlock,
  Target,
  Gauge
} from 'lucide-react';
import { Line, Bar, Radar, Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler,
  ArcElement
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler,
  ArcElement
);

interface Position {
  symbol: string;
  shares: number;
  avgCost: number;
  currentPrice: number;
  sector: string;
  beta: number;
  marketCap: string;
  optionGreeks?: {
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
    rho: number;
  };
}

interface RiskAlert {
  id: string;
  severity: 'high' | 'medium' | 'low';
  type: string;
  message: string;
  timestamp: Date;
  value?: number;
}

interface StressScenario {
  name: string;
  description: string;
  marketReturn: number;
  volatilityMultiplier: number;
  correlationShock: number;
  portfolioImpact?: number;
}

const RiskManagementDashboard: React.FC = () => {
  const [selectedTimeframe, setSelectedTimeframe] = useState('1M');
  const [confidenceLevel, setConfidenceLevel] = useState(95);
  const [riskMetrics, setRiskMetrics] = useState({
    portfolioVaR: 0,
    conditionalVaR: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    currentDrawdown: 0,
    beta: 0,
    treynorRatio: 0,
    informationRatio: 0
  });
  
  const [positions] = useState<Position[]>([
    {
      symbol: 'AAPL',
      shares: 100,
      avgCost: 150,
      currentPrice: 180,
      sector: 'Technology',
      beta: 1.2,
      marketCap: 'Large',
      optionGreeks: {
        delta: 0.6,
        gamma: 0.02,
        theta: -0.05,
        vega: 0.15,
        rho: 0.08
      }
    },
    {
      symbol: 'MSFT',
      shares: 50,
      avgCost: 300,
      currentPrice: 350,
      sector: 'Technology',
      beta: 1.1,
      marketCap: 'Large'
    },
    {
      symbol: 'JPM',
      shares: 75,
      avgCost: 140,
      currentPrice: 155,
      sector: 'Financial',
      beta: 1.3,
      marketCap: 'Large'
    },
    {
      symbol: 'AMZN',
      shares: 30,
      avgCost: 3200,
      currentPrice: 3400,
      sector: 'Consumer',
      beta: 1.4,
      marketCap: 'Large'
    },
    {
      symbol: 'XOM',
      shares: 120,
      avgCost: 90,
      currentPrice: 95,
      sector: 'Energy',
      beta: 0.9,
      marketCap: 'Large'
    }
  ]);

  const [riskAlerts] = useState<RiskAlert[]>([
    {
      id: '1',
      severity: 'high',
      type: 'Concentration Risk',
      message: 'Technology sector exceeds 40% of portfolio',
      timestamp: new Date(),
      value: 42.5
    },
    {
      id: '2',
      severity: 'medium',
      type: 'Volatility Spike',
      message: 'AMZN implied volatility increased 25% today',
      timestamp: new Date(),
      value: 25
    },
    {
      id: '3',
      severity: 'low',
      type: 'Liquidity',
      message: 'Average daily volume decreased for XOM',
      timestamp: new Date()
    }
  ]);

  const stressScenarios: StressScenario[] = [
    {
      name: '2008 Financial Crisis',
      description: 'Market crash scenario',
      marketReturn: -37,
      volatilityMultiplier: 2.5,
      correlationShock: 0.3,
      portfolioImpact: -28.5
    },
    {
      name: 'COVID-19 Crash',
      description: 'Pandemic shock',
      marketReturn: -34,
      volatilityMultiplier: 3.0,
      correlationShock: 0.4,
      portfolioImpact: -31.2
    },
    {
      name: 'Dot-com Bubble',
      description: 'Tech sector collapse',
      marketReturn: -49,
      volatilityMultiplier: 2.0,
      correlationShock: 0.2,
      portfolioImpact: -42.7
    },
    {
      name: 'Black Monday 1987',
      description: 'Single day crash',
      marketReturn: -22,
      volatilityMultiplier: 4.0,
      correlationShock: 0.5,
      portfolioImpact: -19.8
    },
    {
      name: 'Moderate Recession',
      description: 'Mild economic downturn',
      marketReturn: -20,
      volatilityMultiplier: 1.5,
      correlationShock: 0.15,
      portfolioImpact: -16.3
    }
  ];

  // Calculate portfolio metrics
  useEffect(() => {
    const calculateMetrics = () => {
      const totalValue = positions.reduce((sum, pos) => 
        sum + (pos.shares * pos.currentPrice), 0
      );
      
      const portfolioBeta = positions.reduce((sum, pos) => {
        const weight = (pos.shares * pos.currentPrice) / totalValue;
        return sum + (pos.beta * weight);
      }, 0);

      // Simulate VaR calculation
      const portfolioReturn = 0.08; // 8% annual return
      const portfolioVolatility = 0.16; // 16% annual volatility
      const zScore = confidenceLevel === 95 ? 1.645 : confidenceLevel === 99 ? 2.326 : 1.96;
      const timeHorizon = selectedTimeframe === '1D' ? 1/252 : 
                         selectedTimeframe === '1W' ? 5/252 : 
                         selectedTimeframe === '1M' ? 21/252 : 63/252;
      
      const portfolioVaR = totalValue * zScore * portfolioVolatility * Math.sqrt(timeHorizon);
      const conditionalVaR = portfolioVaR * 1.25; // Simplified CVaR

      setRiskMetrics({
        portfolioVaR: portfolioVaR,
        conditionalVaR: conditionalVaR,
        sharpeRatio: 1.45,
        maxDrawdown: 18.5,
        currentDrawdown: 7.2,
        beta: portfolioBeta,
        treynorRatio: 0.12,
        informationRatio: 0.85
      });
    };

    calculateMetrics();
  }, [positions, confidenceLevel, selectedTimeframe]);

  // Position sizing calculator
  const calculatePositionSize = (riskPercent: number, stopLoss: number) => {
    const totalValue = positions.reduce((sum, pos) => 
      sum + (pos.shares * pos.currentPrice), 0
    );
    const riskAmount = totalValue * (riskPercent / 100);
    const shares = Math.floor(riskAmount / stopLoss);
    return { shares, dollarAmount: shares * stopLoss };
  };

  // Correlation matrix data
  const correlationMatrix = {
    labels: positions.map(p => p.symbol),
    data: [
      [1.00, 0.82, 0.45, 0.76, 0.23],
      [0.82, 1.00, 0.52, 0.69, 0.31],
      [0.45, 0.52, 1.00, 0.38, 0.41],
      [0.76, 0.69, 0.38, 1.00, 0.29],
      [0.23, 0.31, 0.41, 0.29, 1.00]
    ]
  };

  // Factor analysis data
  const factorData = {
    labels: ['Market', 'Size', 'Value', 'Momentum', 'Quality', 'Low Vol'],
    datasets: [{
      label: 'Factor Exposure',
      data: [1.15, -0.3, 0.2, 0.8, 0.5, -0.4],
      backgroundColor: 'rgba(59, 130, 246, 0.2)',
      borderColor: 'rgb(59, 130, 246)',
      pointBackgroundColor: 'rgb(59, 130, 246)',
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: 'rgb(59, 130, 246)'
    }]
  };

  // VaR historical data
  const varHistoricalData = {
    labels: Array.from({length: 30}, (_, i) => `Day ${i + 1}`),
    datasets: [
      {
        label: 'Portfolio VaR',
        data: Array.from({length: 30}, () => 
          riskMetrics.portfolioVaR * (0.8 + Math.random() * 0.4)
        ),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        fill: true
      },
      {
        label: 'Actual Loss',
        data: Array.from({length: 30}, () => 
          Math.random() > 0.9 ? riskMetrics.portfolioVaR * (0.9 + Math.random() * 0.3) : 
          riskMetrics.portfolioVaR * Math.random() * 0.8
        ),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        type: 'scatter'
      }
    ]
  };

  // Drawdown chart data
  const drawdownData = {
    labels: Array.from({length: 252}, (_, i) => `Day ${i + 1}`),
    datasets: [{
      label: 'Drawdown %',
      data: Array.from({length: 252}, (_, i) => {
        const base = -Math.sin(i / 40) * 10;
        const noise = (Math.random() - 0.5) * 3;
        const trend = i > 150 ? -(i - 150) / 20 : 0;
        return Math.min(0, base + noise + trend);
      }),
      borderColor: 'rgb(239, 68, 68)',
      backgroundColor: 'rgba(239, 68, 68, 0.1)',
      fill: true
    }]
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Shield className="w-8 h-8 text-red-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                  Risk Management Dashboard
                </h1>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Institutional-grade portfolio risk monitoring
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <select 
                value={selectedTimeframe}
                onChange={(e) => setSelectedTimeframe(e.target.value)}
                className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-lg 
                         bg-white dark:bg-gray-700 text-sm"
              >
                <option value="1D">1 Day</option>
                <option value="1W">1 Week</option>
                <option value="1M">1 Month</option>
                <option value="3M">3 Months</option>
              </select>
              <select 
                value={confidenceLevel}
                onChange={(e) => setConfidenceLevel(Number(e.target.value))}
                className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-lg 
                         bg-white dark:bg-gray-700 text-sm"
              >
                <option value={95}>95% Confidence</option>
                <option value={99}>99% Confidence</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Risk Alerts */}
      {riskAlerts.length > 0 && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-3">
              <AlertTriangle className="w-5 h-5 text-red-600" />
              <h3 className="font-semibold text-red-900 dark:text-red-400">
                Active Risk Alerts
              </h3>
            </div>
            <div className="space-y-2">
              {riskAlerts.map(alert => (
                <div key={alert.id} className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`w-2 h-2 rounded-full ${
                      alert.severity === 'high' ? 'bg-red-500' :
                      alert.severity === 'medium' ? 'bg-yellow-500' : 'bg-blue-500'
                    }`} />
                    <span className="text-sm font-medium">{alert.type}</span>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {alert.message}
                    </span>
                  </div>
                  {alert.value && (
                    <span className="text-sm font-semibold text-red-600">
                      {alert.value}%
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Main Risk Metrics */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Portfolio VaR */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Portfolio VaR ({confidenceLevel}%)
              </h3>
              <Activity className="w-4 h-4 text-gray-400" />
            </div>
            <div className="text-2xl font-bold text-red-600">
              ${riskMetrics.portfolioVaR.toLocaleString('en-US', { maximumFractionDigits: 0 })}
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Potential loss in {selectedTimeframe}
            </p>
          </div>

          {/* CVaR */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Conditional VaR
              </h3>
              <TrendingDown className="w-4 h-4 text-gray-400" />
            </div>
            <div className="text-2xl font-bold text-orange-600">
              ${riskMetrics.conditionalVaR.toLocaleString('en-US', { maximumFractionDigits: 0 })}
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Expected loss if VaR breached
            </p>
          </div>

          {/* Maximum Drawdown */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Maximum Drawdown
              </h3>
              <BarChart3 className="w-4 h-4 text-gray-400" />
            </div>
            <div className="text-2xl font-bold text-purple-600">
              -{riskMetrics.maxDrawdown}%
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Current: -{riskMetrics.currentDrawdown}%
            </p>
          </div>

          {/* Sharpe Ratio */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Sharpe Ratio
              </h3>
              <Gauge className="w-4 h-4 text-gray-400" />
            </div>
            <div className="text-2xl font-bold text-green-600">
              {riskMetrics.sharpeRatio.toFixed(2)}
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Risk-adjusted returns
            </p>
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* VaR Analysis */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Value at Risk Analysis</h3>
          <div className="h-64">
            <Line 
              data={varHistoricalData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    position: 'top' as const,
                  },
                  tooltip: {
                    mode: 'index',
                    intersect: false,
                  }
                },
                scales: {
                  y: {
                    beginAtZero: true,
                    ticks: {
                      callback: function(value) {
                        return '$' + value.toLocaleString();
                      }
                    }
                  }
                }
              }}
            />
          </div>
        </div>

        {/* Factor Analysis */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Factor Exposure</h3>
          <div className="h-64">
            <Radar 
              data={factorData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  r: {
                    beginAtZero: true,
                    max: 2,
                    min: -1,
                    ticks: {
                      stepSize: 0.5
                    }
                  }
                }
              }}
            />
          </div>
        </div>

        {/* Stress Testing */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Stress Testing Scenarios</h3>
          <div className="space-y-3">
            {stressScenarios.map((scenario, index) => (
              <div key={index} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <h4 className="font-medium">{scenario.name}</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {scenario.description}
                    </p>
                  </div>
                  <div className="text-right">
                    <div className={`text-2xl font-bold ${
                      scenario.portfolioImpact! < -30 ? 'text-red-600' :
                      scenario.portfolioImpact! < -20 ? 'text-orange-600' :
                      'text-yellow-600'
                    }`}>
                      {scenario.portfolioImpact}%
                    </div>
                    <p className="text-xs text-gray-500">Portfolio Impact</p>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4 mt-3 text-sm">
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Market Return:</span>
                    <span className="ml-2 font-medium">{scenario.marketReturn}%</span>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Vol Multiplier:</span>
                    <span className="ml-2 font-medium">{scenario.volatilityMultiplier}x</span>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Correlation Shock:</span>
                    <span className="ml-2 font-medium">+{scenario.correlationShock}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Position Sizing Calculator */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Position Sizing</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Risk per Trade (%)
              </label>
              <input 
                type="number"
                defaultValue={2}
                min={0.5}
                max={5}
                step={0.5}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                         bg-white dark:bg-gray-700"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Stop Loss ($)
              </label>
              <input 
                type="number"
                defaultValue={5}
                min={1}
                max={50}
                step={1}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                         bg-white dark:bg-gray-700"
              />
            </div>
            <button className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 
                             transition-colors flex items-center justify-center space-x-2">
              <Calculator className="w-4 h-4" />
              <span>Calculate Position Size</span>
            </button>
            <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-sm space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Shares:</span>
                  <span className="font-medium">240</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Position Size:</span>
                  <span className="font-medium">$1,200</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Correlation Matrix */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Correlation Matrix</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr>
                  <th className="px-2 py-1"></th>
                  {correlationMatrix.labels.map(label => (
                    <th key={label} className="px-2 py-1 text-center font-medium">
                      {label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {correlationMatrix.data.map((row, i) => (
                  <tr key={i}>
                    <td className="px-2 py-1 font-medium">{correlationMatrix.labels[i]}</td>
                    {row.map((value, j) => (
                      <td key={j} className={`px-2 py-1 text-center ${
                        value === 1 ? 'bg-gray-100 dark:bg-gray-700' :
                        value > 0.7 ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400' :
                        value > 0.3 ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400' :
                        'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                      }`}>
                        {value.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-4 flex items-center space-x-4 text-xs">
            <div className="flex items-center space-x-1">
              <div className="w-3 h-3 bg-red-100 dark:bg-red-900/30 rounded"></div>
              <span>High (>0.7)</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-3 h-3 bg-yellow-100 dark:bg-yellow-900/30 rounded"></div>
              <span>Medium (0.3-0.7)</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-3 h-3 bg-green-100 dark:bg-green-900/30 rounded"></div>
              <span>Low (<0.3)</span>
            </div>
          </div>
        </div>

        {/* Greeks Display */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Options Greeks</h3>
          <div className="space-y-3">
            {positions.filter(p => p.optionGreeks).map(position => (
              <div key={position.symbol} className="border border-gray-200 dark:border-gray-700 rounded-lg p-3">
                <h4 className="font-medium mb-2">{position.symbol}</h4>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Δ Delta:</span>
                    <span className="font-medium">{position.optionGreeks?.delta.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Γ Gamma:</span>
                    <span className="font-medium">{position.optionGreeks?.gamma.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Θ Theta:</span>
                    <span className="font-medium text-red-600">{position.optionGreeks?.theta.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">ν Vega:</span>
                    <span className="font-medium">{position.optionGreeks?.vega.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">ρ Rho:</span>
                    <span className="font-medium">{position.optionGreeks?.rho.toFixed(3)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Drawdown Chart */}
        <div className="lg:col-span-3 bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Historical Drawdown</h3>
          <div className="h-64">
            <Line 
              data={drawdownData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    display: false
                  },
                  tooltip: {
                    callbacks: {
                      label: function(context) {
                        return `Drawdown: ${context.parsed.y.toFixed(2)}%`;
                      }
                    }
                  }
                },
                scales: {
                  y: {
                    max: 5,
                    min: -25,
                    ticks: {
                      callback: function(value) {
                        return value + '%';
                      }
                    }
                  }
                }
              }}
            />
          </div>
        </div>

        {/* Risk Attribution */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Risk Attribution by Sector</h3>
          <div className="space-y-3">
            {['Technology', 'Financial', 'Consumer', 'Energy'].map((sector, index) => {
              const sectorRisk = [45, 20, 25, 10][index];
              return (
                <div key={sector}>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-sm font-medium">{sector}</span>
                    <span className="text-sm text-gray-600 dark:text-gray-400">{sectorRisk}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${
                        sectorRisk > 40 ? 'bg-red-500' :
                        sectorRisk > 25 ? 'bg-yellow-500' :
                        'bg-green-500'
                      }`}
                      style={{ width: `${sectorRisk}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Liquidity Risk */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Liquidity Risk Assessment</h3>
          <div className="space-y-3">
            {positions.map(position => {
              const liquidityScore = Math.random() * 100;
              const avgVolume = Math.floor(Math.random() * 10000000) + 1000000;
              const daysToLiquidate = Math.ceil((position.shares * position.currentPrice) / (avgVolume * 0.1));
              
              return (
                <div key={position.symbol} className="border border-gray-200 dark:border-gray-700 rounded-lg p-3">
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="font-medium">{position.symbol}</h4>
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      liquidityScore > 80 ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                      liquidityScore > 50 ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' :
                      'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                    }`}>
                      {liquidityScore > 80 ? 'High' : liquidityScore > 50 ? 'Medium' : 'Low'} Liquidity
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs text-gray-600 dark:text-gray-400">
                    <div>
                      <span>Avg Volume:</span>
                      <span className="ml-1 font-medium">{(avgVolume / 1000000).toFixed(1)}M</span>
                    </div>
                    <div>
                      <span>Days to Liquidate:</span>
                      <span className="ml-1 font-medium">{daysToLiquidate}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Additional Metrics */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6 mb-8">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Additional Risk Metrics</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{riskMetrics.beta.toFixed(2)}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Portfolio Beta</div>
            </div>
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-2xl font-bold text-green-600">{riskMetrics.treynorRatio.toFixed(3)}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Treynor Ratio</div>
            </div>
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">{riskMetrics.informationRatio.toFixed(2)}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Information Ratio</div>
            </div>
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-2xl font-bold text-orange-600">0.74</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Sortino Ratio</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskManagementDashboard;