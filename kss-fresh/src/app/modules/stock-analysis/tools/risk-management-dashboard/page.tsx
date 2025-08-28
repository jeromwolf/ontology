'use client';

import React, { useState } from 'react';
import { Shield, AlertTriangle, TrendingUp, BarChart3, DollarSign } from 'lucide-react';

export default function RiskManagementDashboard() {
  const [selectedPortfolio, setSelectedPortfolio] = useState('main');

  // 샘플 포트폴리오 데이터
  const portfolios = {
    main: {
      name: '메인 포트폴리오',
      totalValue: 150000,
      dayChange: -2340,
      dayChangePercent: -1.54,
      positions: [
        { symbol: 'AAPL', value: 45000, risk: 'low', beta: 1.2 },
        { symbol: 'TSLA', value: 35000, risk: 'high', beta: 2.1 },
        { symbol: 'MSFT', value: 30000, risk: 'low', beta: 0.9 },
        { symbol: 'NVDA', value: 25000, risk: 'high', beta: 1.8 },
        { symbol: 'SPY', value: 15000, risk: 'low', beta: 1.0 },
      ]
    }
  };

  const portfolio = portfolios[selectedPortfolio as keyof typeof portfolios];

  // 리스크 메트릭 계산
  const riskMetrics = {
    var95: portfolio.totalValue * 0.023, // 95% VaR
    var99: portfolio.totalValue * 0.031, // 99% VaR
    expectedShortfall: portfolio.totalValue * 0.035,
    maxDrawdown: 8.5,
    sharpeRatio: 1.34,
    portfolioBeta: portfolio.positions.reduce((acc, pos) => acc + (pos.value / portfolio.totalValue) * pos.beta, 0)
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-200';
      case 'medium': return 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-200';
      case 'high': return 'bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-200';
      default: return 'bg-gray-100 dark:bg-gray-900/20 text-gray-800 dark:text-gray-200';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Shield className="text-blue-600 dark:text-blue-400" size={32} />
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                리스크 관리 대시보드
              </h1>
            </div>
            
            <select
              value={selectedPortfolio}
              onChange={(e) => setSelectedPortfolio(e.target.value)}
              className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="main">메인 포트폴리오</option>
            </select>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* 포트폴리오 개요 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
            {portfolio.name} 개요
          </h2>
          <div className="grid md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <DollarSign className="text-blue-600 dark:text-blue-400 mx-auto mb-2" size={24} />
              <p className="text-sm text-gray-600 dark:text-gray-400">총 자산</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                ${portfolio.totalValue.toLocaleString()}
              </p>
            </div>
            
            <div className="text-center p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
              <TrendingUp className="text-red-600 dark:text-red-400 mx-auto mb-2" size={24} />
              <p className="text-sm text-gray-600 dark:text-gray-400">일일 손익</p>
              <p className="text-2xl font-bold text-red-600 dark:text-red-400">
                ${portfolio.dayChange.toLocaleString()} ({portfolio.dayChangePercent.toFixed(2)}%)
              </p>
            </div>
            
            <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <BarChart3 className="text-green-600 dark:text-green-400 mx-auto mb-2" size={24} />
              <p className="text-sm text-gray-600 dark:text-gray-400">샤프 비율</p>
              <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                {riskMetrics.sharpeRatio}
              </p>
            </div>
            
            <div className="text-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <Shield className="text-purple-600 dark:text-purple-400 mx-auto mb-2" size={24} />
              <p className="text-sm text-gray-600 dark:text-gray-400">포트폴리오 베타</p>
              <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                {riskMetrics.portfolioBeta.toFixed(2)}
              </p>
            </div>
          </div>
        </div>

        {/* 리스크 메트릭 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">리스크 메트릭</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
              <h3 className="font-medium text-gray-900 dark:text-white mb-1">VaR (95%)</h3>
              <p className="text-lg font-bold text-orange-600 dark:text-orange-400">
                ${riskMetrics.var95.toFixed(0)}
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400">1일 최대 예상 손실</p>
            </div>
            
            <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
              <h3 className="font-medium text-gray-900 dark:text-white mb-1">VaR (99%)</h3>
              <p className="text-lg font-bold text-red-600 dark:text-red-400">
                ${riskMetrics.var99.toFixed(0)}
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400">극단적 시장 상황</p>
            </div>
            
            <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <h3 className="font-medium text-gray-900 dark:text-white mb-1">Expected Shortfall</h3>
              <p className="text-lg font-bold text-purple-600 dark:text-purple-400">
                ${riskMetrics.expectedShortfall.toFixed(0)}
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400">조건부 기댓값</p>
            </div>
            
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <h3 className="font-medium text-gray-900 dark:text-white mb-1">Max Drawdown</h3>
              <p className="text-lg font-bold text-blue-600 dark:text-blue-400">
                {riskMetrics.maxDrawdown.toFixed(1)}%
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400">최대 하락폭</p>
            </div>
          </div>
        </div>

        {/* 포지션별 리스크 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">포지션별 리스크 분석</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-3 px-4 font-medium text-gray-900 dark:text-white">종목</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-900 dark:text-white">금액</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-900 dark:text-white">비중</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-900 dark:text-white">베타</th>
                  <th className="text-center py-3 px-4 font-medium text-gray-900 dark:text-white">리스크</th>
                </tr>
              </thead>
              <tbody>
                {portfolio.positions.map((position, index) => (
                  <tr key={position.symbol} className={index % 2 === 0 ? 'bg-gray-50 dark:bg-gray-700/30' : ''}>
                    <td className="py-3 px-4 font-medium text-gray-900 dark:text-white">
                      {position.symbol}
                    </td>
                    <td className="text-right py-3 px-4 text-gray-900 dark:text-white">
                      ${position.value.toLocaleString()}
                    </td>
                    <td className="text-right py-3 px-4 text-gray-900 dark:text-white">
                      {((position.value / portfolio.totalValue) * 100).toFixed(1)}%
                    </td>
                    <td className="text-right py-3 px-4 text-gray-900 dark:text-white">
                      {position.beta.toFixed(1)}
                    </td>
                    <td className="text-center py-3 px-4">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(position.risk)}`}>
                        {position.risk.toUpperCase()}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* 리스크 알림 */}
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 rounded-lg p-4">
          <div className="flex items-center gap-3">
            <AlertTriangle className="text-yellow-600 dark:text-yellow-400" size={24} />
            <div>
              <h3 className="font-semibold text-yellow-800 dark:text-yellow-200">리스크 알림</h3>
              <p className="text-sm text-yellow-700 dark:text-yellow-300">
                TSLA와 NVDA의 비중이 40%를 초과합니다. 고변동성 종목의 비중 조절을 고려하세요.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}