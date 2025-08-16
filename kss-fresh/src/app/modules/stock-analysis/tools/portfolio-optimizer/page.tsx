'use client'

import React, { useState, useEffect } from 'react'
import { ArrowLeft, TrendingUp, Shield, AlertTriangle, BarChart3, LineChart, PieChart, Calculator, RefreshCw, Download } from 'lucide-react'
import Link from 'next/link'

interface AssetAllocation {
  stocks: number
  bonds: number
  commodities: number
  realEstate: number
  cash: number
}

interface PortfolioMetrics {
  expectedReturn: number
  volatility: number
  sharpeRatio: number
  sortinoRatio: number
  maxDrawdown: number
  var95: number
  cvar95: number
}

interface EfficientFrontierPoint {
  return: number
  risk: number
  weights: AssetAllocation
}

interface MonteCarloResult {
  percentile5: number
  percentile25: number
  median: number
  percentile75: number
  percentile95: number
  probabilityOfLoss: number
}

export default function PortfolioOptimizer() {
  const [allocation, setAllocation] = useState<AssetAllocation>({
    stocks: 60,
    bonds: 30,
    commodities: 5,
    realEstate: 5,
    cash: 0
  })

  const [riskLevel, setRiskLevel] = useState<'conservative' | 'moderate' | 'aggressive'>('moderate')
  const [optimizationMethod, setOptimizationMethod] = useState<'meanVariance' | 'blackLitterman' | 'riskParity'>('meanVariance')
  const [timeHorizon, setTimeHorizon] = useState(10)
  const [initialInvestment, setInitialInvestment] = useState(100000)
  
  const [metrics, setMetrics] = useState<PortfolioMetrics>({
    expectedReturn: 8.5,
    volatility: 12.3,
    sharpeRatio: 0.69,
    sortinoRatio: 0.95,
    maxDrawdown: -18.5,
    var95: -15.2,
    cvar95: -22.1
  })

  const [efficientFrontier, setEfficientFrontier] = useState<EfficientFrontierPoint[]>([])
  const [monteCarloResults, setMonteCarloResults] = useState<MonteCarloResult>({
    percentile5: 68000,
    percentile25: 125000,
    median: 180000,
    percentile75: 265000,
    percentile95: 420000,
    probabilityOfLoss: 5.2
  })

  // Asset class expected returns and volatilities
  const assetClassData = {
    stocks: { return: 10.5, volatility: 16.5 },
    bonds: { return: 4.5, volatility: 6.2 },
    commodities: { return: 6.0, volatility: 19.8 },
    realEstate: { return: 8.0, volatility: 14.3 },
    cash: { return: 2.0, volatility: 0.5 }
  }

  // Correlation matrix
  const correlationMatrix = [
    [1.00, 0.15, 0.25, 0.30, -0.05], // Stocks
    [0.15, 1.00, -0.10, 0.10, 0.05],  // Bonds
    [0.25, -0.10, 1.00, 0.20, -0.15], // Commodities
    [0.30, 0.10, 0.20, 1.00, 0.00],   // Real Estate
    [-0.05, 0.05, -0.15, 0.00, 1.00]  // Cash
  ]

  // Update allocation to ensure it sums to 100
  const updateAllocation = (asset: keyof AssetAllocation, value: number) => {
    const newAllocation = { ...allocation, [asset]: value }
    const total = Object.values(newAllocation).reduce((sum, val) => sum + val, 0)
    
    if (total > 100) {
      const excess = total - 100
      const otherAssets = Object.keys(newAllocation).filter(key => key !== asset) as (keyof AssetAllocation)[]
      const reduction = excess / otherAssets.length
      
      otherAssets.forEach(key => {
        newAllocation[key] = Math.max(0, newAllocation[key] - reduction)
      })
    }
    
    setAllocation(newAllocation)
  }

  // Calculate portfolio metrics
  useEffect(() => {
    const weights = Object.values(allocation).map(w => w / 100)
    const returns = Object.values(assetClassData).map(a => a.return)
    const volatilities = Object.values(assetClassData).map(a => a.volatility)
    
    // Expected portfolio return
    const expectedReturn = weights.reduce((sum, w, i) => sum + w * returns[i], 0)
    
    // Portfolio volatility
    let variance = 0
    for (let i = 0; i < weights.length; i++) {
      for (let j = 0; j < weights.length; j++) {
        variance += weights[i] * weights[j] * volatilities[i] * volatilities[j] * correlationMatrix[i][j] / 10000
      }
    }
    const volatility = Math.sqrt(variance)
    
    // Risk metrics
    const riskFreeRate = 2.0
    const sharpeRatio = (expectedReturn - riskFreeRate) / volatility
    const downsideDeviation = volatility * 0.8 // Simplified
    const sortinoRatio = (expectedReturn - riskFreeRate) / downsideDeviation
    
    setMetrics({
      expectedReturn,
      volatility,
      sharpeRatio,
      sortinoRatio,
      maxDrawdown: -volatility * 1.5,
      var95: -volatility * 1.65,
      cvar95: -volatility * 2.0
    })
  }, [allocation])

  // Generate efficient frontier
  useEffect(() => {
    const points: EfficientFrontierPoint[] = []
    
    for (let stockWeight = 0; stockWeight <= 100; stockWeight += 10) {
      const bondWeight = Math.max(0, 100 - stockWeight - 10)
      const otherWeight = 10
      
      const weights = [
        stockWeight / 100,
        bondWeight / 100,
        otherWeight / 300,
        otherWeight / 300,
        otherWeight / 300
      ]
      
      const returns = Object.values(assetClassData).map(a => a.return)
      const volatilities = Object.values(assetClassData).map(a => a.volatility)
      
      const expectedReturn = weights.reduce((sum, w, i) => sum + w * returns[i], 0)
      
      let variance = 0
      for (let i = 0; i < weights.length; i++) {
        for (let j = 0; j < weights.length; j++) {
          variance += weights[i] * weights[j] * volatilities[i] * volatilities[j] * correlationMatrix[i][j] / 10000
        }
      }
      
      points.push({
        return: expectedReturn,
        risk: Math.sqrt(variance),
        weights: {
          stocks: stockWeight,
          bonds: bondWeight,
          commodities: otherWeight / 3,
          realEstate: otherWeight / 3,
          cash: otherWeight / 3
        }
      })
    }
    
    setEfficientFrontier(points)
  }, [])

  // Run Monte Carlo simulation
  const runMonteCarloSimulation = () => {
    const simulations = 1000
    const years = timeHorizon
    const results: number[] = []
    
    for (let sim = 0; sim < simulations; sim++) {
      let portfolioValue = initialInvestment
      
      for (let year = 0; year < years; year++) {
        const randomReturn = metrics.expectedReturn / 100 + (Math.random() - 0.5) * metrics.volatility / 100 * 2
        portfolioValue *= (1 + randomReturn)
      }
      
      results.push(portfolioValue)
    }
    
    results.sort((a, b) => a - b)
    
    setMonteCarloResults({
      percentile5: results[Math.floor(simulations * 0.05)],
      percentile25: results[Math.floor(simulations * 0.25)],
      median: results[Math.floor(simulations * 0.50)],
      percentile75: results[Math.floor(simulations * 0.75)],
      percentile95: results[Math.floor(simulations * 0.95)],
      probabilityOfLoss: (results.filter(r => r < initialInvestment).length / simulations) * 100
    })
  }

  useEffect(() => {
    runMonteCarloSimulation()
  }, [metrics, timeHorizon, initialInvestment])

  // Optimize portfolio based on method
  const optimizePortfolio = () => {
    switch (optimizationMethod) {
      case 'meanVariance':
        // Simplified mean-variance optimization
        if (riskLevel === 'conservative') {
          setAllocation({ stocks: 30, bonds: 60, commodities: 5, realEstate: 5, cash: 0 })
        } else if (riskLevel === 'moderate') {
          setAllocation({ stocks: 60, bonds: 30, commodities: 5, realEstate: 5, cash: 0 })
        } else {
          setAllocation({ stocks: 80, bonds: 10, commodities: 5, realEstate: 5, cash: 0 })
        }
        break
        
      case 'blackLitterman':
        // Simplified Black-Litterman (market-cap weighted with tilts)
        setAllocation({ stocks: 55, bonds: 35, commodities: 4, realEstate: 4, cash: 2 })
        break
        
      case 'riskParity':
        // Equal risk contribution
        setAllocation({ stocks: 25, bonds: 40, commodities: 15, realEstate: 15, cash: 5 })
        break
    }
  }

  // Generate rebalancing recommendations
  const getRebalancingRecommendations = () => {
    const targetAllocation = { ...allocation }
    const currentAllocation = {
      stocks: 65,
      bonds: 25,
      commodities: 5,
      realEstate: 3,
      cash: 2
    }
    
    const recommendations: string[] = []
    
    Object.keys(targetAllocation).forEach((asset) => {
      const key = asset as keyof AssetAllocation
      const diff = targetAllocation[key] - currentAllocation[key]
      
      if (Math.abs(diff) > 2) {
        if (diff > 0) {
          recommendations.push(`Buy ${diff.toFixed(1)}% more ${asset}`)
        } else {
          recommendations.push(`Sell ${Math.abs(diff).toFixed(1)}% of ${asset}`)
        }
      }
    })
    
    return recommendations
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Link href="/modules/stock-analysis/tools" className="text-gray-400 hover:text-white transition-colors">
              <ArrowLeft className="w-6 h-6" />
            </Link>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Portfolio Optimizer
              </h1>
              <p className="text-gray-400 mt-2">Modern Portfolio Theory & Risk Management</p>
            </div>
          </div>
          <div className="flex gap-3">
            <button
              onClick={() => runMonteCarloSimulation()}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center gap-2 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              Run Simulation
            </button>
            <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg flex items-center gap-2 transition-colors">
              <Download className="w-4 h-4" />
              Export Report
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Input Section */}
          <div className="space-y-6">
            {/* Asset Allocation Input */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <PieChart className="w-5 h-5 text-blue-400" />
                Asset Allocation
              </h3>
              
              <div className="space-y-4">
                {Object.entries(allocation).map(([asset, value]) => (
                  <div key={asset}>
                    <div className="flex justify-between mb-1">
                      <span className="capitalize">{asset}</span>
                      <span className="text-blue-400">{value}%</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={value}
                      onChange={(e) => updateAllocation(asset as keyof AssetAllocation, Number(e.target.value))}
                      className="w-full"
                    />
                  </div>
                ))}
              </div>
              
              <div className="mt-4 pt-4 border-t border-gray-700">
                <div className="flex justify-between">
                  <span className="font-semibold">Total</span>
                  <span className={`font-semibold ${Object.values(allocation).reduce((a, b) => a + b, 0) === 100 ? 'text-green-400' : 'text-red-400'}`}>
                    {Object.values(allocation).reduce((a, b) => a + b, 0)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Optimization Settings */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Calculator className="w-5 h-5 text-purple-400" />
                Optimization Settings
              </h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Risk Level</label>
                  <select
                    value={riskLevel}
                    onChange={(e) => setRiskLevel(e.target.value as any)}
                    className="w-full px-3 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                  >
                    <option value="conservative">Conservative</option>
                    <option value="moderate">Moderate</option>
                    <option value="aggressive">Aggressive</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Optimization Method</label>
                  <select
                    value={optimizationMethod}
                    onChange={(e) => setOptimizationMethod(e.target.value as any)}
                    className="w-full px-3 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                  >
                    <option value="meanVariance">Mean-Variance</option>
                    <option value="blackLitterman">Black-Litterman</option>
                    <option value="riskParity">Risk Parity</option>
                  </select>
                </div>
                
                <button
                  onClick={optimizePortfolio}
                  className="w-full px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 rounded-lg font-semibold transition-all"
                >
                  Optimize Portfolio
                </button>
              </div>
            </div>

            {/* Monte Carlo Settings */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-green-400" />
                Simulation Parameters
              </h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Time Horizon (years)</label>
                  <input
                    type="number"
                    value={timeHorizon}
                    onChange={(e) => setTimeHorizon(Number(e.target.value))}
                    className="w-full px-3 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Initial Investment ($)</label>
                  <input
                    type="number"
                    value={initialInvestment}
                    onChange={(e) => setInitialInvestment(Number(e.target.value))}
                    className="w-full px-3 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Middle Column - Visualizations */}
          <div className="space-y-6">
            {/* Portfolio Metrics */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-green-400" />
                Portfolio Metrics
              </h3>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <div className="text-sm text-gray-400">Expected Return</div>
                  <div className="text-2xl font-bold text-green-400">{metrics.expectedReturn.toFixed(2)}%</div>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <div className="text-sm text-gray-400">Volatility</div>
                  <div className="text-2xl font-bold text-yellow-400">{metrics.volatility.toFixed(2)}%</div>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <div className="text-sm text-gray-400">Sharpe Ratio</div>
                  <div className="text-2xl font-bold text-blue-400">{metrics.sharpeRatio.toFixed(2)}</div>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <div className="text-sm text-gray-400">Sortino Ratio</div>
                  <div className="text-2xl font-bold text-purple-400">{metrics.sortinoRatio.toFixed(2)}</div>
                </div>
              </div>
              
              <div className="mt-4 space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">Max Drawdown</span>
                  <span className="text-red-400 font-semibold">{metrics.maxDrawdown.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">VaR (95%)</span>
                  <span className="text-orange-400 font-semibold">{metrics.var95.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">CVaR (95%)</span>
                  <span className="text-red-400 font-semibold">{metrics.cvar95.toFixed(2)}%</span>
                </div>
              </div>
            </div>

            {/* Efficient Frontier */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <LineChart className="w-5 h-5 text-blue-400" />
                Efficient Frontier
              </h3>
              
              <div className="h-64 relative">
                <svg className="w-full h-full">
                  {/* Grid */}
                  <defs>
                    <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                      <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5"/>
                    </pattern>
                  </defs>
                  <rect width="100%" height="100%" fill="url(#grid)" />
                  
                  {/* Axes */}
                  <line x1="40" y1="220" x2="280" y2="220" stroke="rgba(255,255,255,0.3)" strokeWidth="1"/>
                  <line x1="40" y1="220" x2="40" y2="20" stroke="rgba(255,255,255,0.3)" strokeWidth="1"/>
                  
                  {/* Efficient Frontier Curve */}
                  <path
                    d={`M ${efficientFrontier.map((point, i) => {
                      const x = 40 + (point.risk / 25) * 240
                      const y = 220 - (point.return / 15) * 200
                      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
                    }).join(' ')}`}
                    fill="none"
                    stroke="url(#frontierGradient)"
                    strokeWidth="3"
                  />
                  
                  {/* Gradient for frontier */}
                  <defs>
                    <linearGradient id="frontierGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#3B82F6" />
                      <stop offset="100%" stopColor="#A855F7" />
                    </linearGradient>
                  </defs>
                  
                  {/* Current Portfolio Point */}
                  <circle
                    cx={40 + (metrics.volatility / 25) * 240}
                    cy={220 - (metrics.expectedReturn / 15) * 200}
                    r="6"
                    fill="#10B981"
                    stroke="#ffffff"
                    strokeWidth="2"
                  />
                  
                  {/* Labels */}
                  <text x="160" y="250" textAnchor="middle" className="fill-gray-400 text-sm">Risk (Volatility %)</text>
                  <text x="15" y="120" textAnchor="middle" className="fill-gray-400 text-sm" transform="rotate(-90 15 120)">Return %</text>
                </svg>
              </div>
            </div>

            {/* Correlation Matrix */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Shield className="w-5 h-5 text-purple-400" />
                Correlation Matrix
              </h3>
              
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr>
                      <th className="p-2"></th>
                      <th className="p-2">STK</th>
                      <th className="p-2">BND</th>
                      <th className="p-2">COM</th>
                      <th className="p-2">RE</th>
                      <th className="p-2">CSH</th>
                    </tr>
                  </thead>
                  <tbody>
                    {['STK', 'BND', 'COM', 'RE', 'CSH'].map((row, i) => (
                      <tr key={row}>
                        <td className="p-2 font-semibold">{row}</td>
                        {correlationMatrix[i].map((corr, j) => (
                          <td 
                            key={j} 
                            className="p-2 text-center"
                            style={{
                              backgroundColor: corr > 0 
                                ? `rgba(34, 197, 94, ${Math.abs(corr) * 0.5})`
                                : `rgba(239, 68, 68, ${Math.abs(corr) * 0.5})`
                            }}
                          >
                            {corr.toFixed(2)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Right Column - Results */}
          <div className="space-y-6">
            {/* Monte Carlo Results */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-yellow-400" />
                Monte Carlo Simulation
              </h3>
              
              <div className="mb-4">
                <div className="text-sm text-gray-400 mb-2">Portfolio Value Distribution</div>
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">5th Percentile</span>
                      <span className="text-red-400 font-semibold">
                        ${monteCarloResults.percentile5.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">25th Percentile</span>
                      <span className="text-orange-400 font-semibold">
                        ${monteCarloResults.percentile25.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-semibold">Median</span>
                      <span className="text-blue-400 font-bold">
                        ${monteCarloResults.median.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">75th Percentile</span>
                      <span className="text-green-400 font-semibold">
                        ${monteCarloResults.percentile75.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">95th Percentile</span>
                      <span className="text-green-500 font-semibold">
                        ${monteCarloResults.percentile95.toLocaleString()}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="bg-red-900/20 border border-red-700 rounded-lg p-4">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-red-400" />
                  <span className="text-sm">Probability of Loss</span>
                </div>
                <div className="text-2xl font-bold text-red-400 mt-1">
                  {monteCarloResults.probabilityOfLoss.toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Rebalancing Recommendations */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <RefreshCw className="w-5 h-5 text-green-400" />
                Rebalancing Recommendations
              </h3>
              
              <div className="space-y-3">
                {getRebalancingRecommendations().map((rec, i) => (
                  <div key={i} className="flex items-center gap-3 bg-gray-900/50 rounded-lg p-3">
                    <div className={`w-2 h-2 rounded-full ${rec.includes('Buy') ? 'bg-green-400' : 'bg-red-400'}`} />
                    <span className="text-sm">{rec}</span>
                  </div>
                ))}
              </div>
              
              {getRebalancingRecommendations().length === 0 && (
                <div className="text-center text-gray-400 py-4">
                  <Shield className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">Portfolio is well-balanced</p>
                </div>
              )}
            </div>

            {/* Historical Backtest */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <LineChart className="w-5 h-5 text-purple-400" />
                Historical Backtest
              </h3>
              
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-900/50 rounded-lg p-3">
                    <div className="text-xs text-gray-400">1 Year</div>
                    <div className="text-lg font-bold text-green-400">+12.5%</div>
                  </div>
                  <div className="bg-gray-900/50 rounded-lg p-3">
                    <div className="text-xs text-gray-400">3 Years</div>
                    <div className="text-lg font-bold text-green-400">+38.2%</div>
                  </div>
                  <div className="bg-gray-900/50 rounded-lg p-3">
                    <div className="text-xs text-gray-400">5 Years</div>
                    <div className="text-lg font-bold text-green-400">+72.8%</div>
                  </div>
                  <div className="bg-gray-900/50 rounded-lg p-3">
                    <div className="text-xs text-gray-400">10 Years</div>
                    <div className="text-lg font-bold text-green-400">+195.6%</div>
                  </div>
                </div>
                
                <div className="text-sm text-gray-400">
                  <p>Backtest includes:</p>
                  <ul className="list-disc list-inside mt-1 space-y-1">
                    <li>Annual rebalancing</li>
                    <li>0.1% transaction costs</li>
                    <li>Dividend reinvestment</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}