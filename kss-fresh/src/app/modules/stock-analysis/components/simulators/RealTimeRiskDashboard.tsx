'use client';

import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, AlertTriangle, Shield, Activity, Bell, BarChart3, Eye } from 'lucide-react';

interface Position {
  symbol: string;
  name: string;
  quantity: number;
  avgCost: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPL: number;
  realizedPL: number;
  weight: number;
  beta: number;
  sector: string;
}

interface RiskMetrics {
  portfolioVaR: number;
  portfolioCVaR: number;
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
  currentDrawdown: number;
  beta: number;
  correlation: number;
  trackingError: number;
}

interface RiskLimit {
  metric: string;
  current: number;
  limit: number;
  usage: number;
  status: 'normal' | 'warning' | 'critical';
}

interface Alert {
  id: string;
  timestamp: Date;
  type: 'info' | 'warning' | 'critical';
  title: string;
  message: string;
  actionRequired: boolean;
}

interface ConcentrationRisk {
  type: 'sector' | 'position' | 'country' | 'currency';
  name: string;
  exposure: number;
  limit: number;
  status: 'normal' | 'warning' | 'critical';
}

// 모의 포지션 데이터
const mockPositions: Position[] = [
  { symbol: 'AAPL', name: 'Apple Inc.', quantity: 100, avgCost: 150, currentPrice: 189.95, marketValue: 18995, unrealizedPL: 3995, realizedPL: 1200, weight: 15.2, beta: 1.2, sector: 'Technology' },
  { symbol: 'MSFT', name: 'Microsoft', quantity: 50, avgCost: 320, currentPrice: 423.85, marketValue: 21192.5, unrealizedPL: 5192.5, realizedPL: 800, weight: 17.0, beta: 0.9, sector: 'Technology' },
  { symbol: 'NVDA', name: 'NVIDIA', quantity: 20, avgCost: 450, currentPrice: 875.28, marketValue: 17505.6, unrealizedPL: 8505.6, realizedPL: 0, weight: 14.0, beta: 1.8, sector: 'Technology' },
  { symbol: 'JPM', name: 'JPMorgan', quantity: 80, avgCost: 140, currentPrice: 195.42, marketValue: 15633.6, unrealizedPL: 4433.6, realizedPL: 500, weight: 12.5, beta: 1.1, sector: 'Financials' },
  { symbol: 'JNJ', name: 'Johnson & Johnson', quantity: 60, avgCost: 145, currentPrice: 155.20, marketValue: 9312, unrealizedPL: 612, realizedPL: 300, weight: 7.5, beta: 0.7, sector: 'Healthcare' },
  { symbol: 'XOM', name: 'Exxon Mobil', quantity: 100, avgCost: 85, currentPrice: 105.20, marketValue: 10520, unrealizedPL: 2020, realizedPL: 150, weight: 8.4, beta: 1.3, sector: 'Energy' },
  { symbol: 'GLD', name: 'Gold ETF', quantity: 50, avgCost: 170, currentPrice: 185.50, marketValue: 9275, unrealizedPL: 775, realizedPL: 0, weight: 7.4, beta: 0.2, sector: 'Commodity' }
];

export default function RealTimeRiskDashboard() {
  const [positions, setPositions] = useState<Position[]>(mockPositions);
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics>({
    portfolioVaR: -8.5,
    portfolioCVaR: -12.3,
    sharpeRatio: 1.45,
    sortinoRatio: 1.82,
    maxDrawdown: -15.2,
    currentDrawdown: -3.8,
    beta: 1.15,
    correlation: 0.85,
    trackingError: 4.2
  });
  
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [riskLimits, setRiskLimits] = useState<RiskLimit[]>([]);
  const [concentrationRisks, setConcentrationRisks] = useState<ConcentrationRisk[]>([]);
  const [viewMode, setViewMode] = useState<'overview' | 'positions' | 'limits' | 'alerts'>('overview');
  const [isLive, setIsLive] = useState(true);
  
  // 실시간 데이터 시뮬레이션
  useEffect(() => {
    if (!isLive) return;
    
    const interval = setInterval(() => {
      // 포지션 가격 업데이트
      setPositions(prev => prev.map(pos => {
        const priceChange = (Math.random() - 0.5) * 2;
        const newPrice = pos.currentPrice * (1 + priceChange / 100);
        const newMarketValue = newPrice * pos.quantity;
        const newUnrealizedPL = (newPrice - pos.avgCost) * pos.quantity;
        
        return {
          ...pos,
          currentPrice: newPrice,
          marketValue: newMarketValue,
          unrealizedPL: newUnrealizedPL
        };
      }));
      
      // 리스크 메트릭 업데이트
      setRiskMetrics(prev => ({
        ...prev,
        portfolioVaR: prev.portfolioVaR + (Math.random() - 0.5) * 0.5,
        currentDrawdown: prev.currentDrawdown + (Math.random() - 0.5) * 0.3,
        sharpeRatio: Math.max(0, prev.sharpeRatio + (Math.random() - 0.5) * 0.1)
      }));
    }, 3000);
    
    return () => clearInterval(interval);
  }, [isLive]);
  
  // 리스크 한도 계산
  useEffect(() => {
    const totalValue = positions.reduce((sum, pos) => sum + pos.marketValue, 0);
    
    const limits: RiskLimit[] = [
      {
        metric: 'Portfolio VaR (95%)',
        current: Math.abs(riskMetrics.portfolioVaR),
        limit: 10,
        usage: Math.abs(riskMetrics.portfolioVaR) / 10 * 100,
        status: Math.abs(riskMetrics.portfolioVaR) > 8 ? 'critical' : 
                Math.abs(riskMetrics.portfolioVaR) > 6 ? 'warning' : 'normal'
      },
      {
        metric: '최대 포지션 비중',
        current: Math.max(...positions.map(p => p.weight)),
        limit: 20,
        usage: Math.max(...positions.map(p => p.weight)) / 20 * 100,
        status: Math.max(...positions.map(p => p.weight)) > 18 ? 'critical' :
                Math.max(...positions.map(p => p.weight)) > 15 ? 'warning' : 'normal'
      },
      {
        metric: '섹터 집중도',
        current: 46.2, // Technology sector
        limit: 40,
        usage: 115.5,
        status: 'critical'
      },
      {
        metric: '포트폴리오 베타',
        current: riskMetrics.beta,
        limit: 1.3,
        usage: riskMetrics.beta / 1.3 * 100,
        status: riskMetrics.beta > 1.25 ? 'warning' : 'normal'
      },
      {
        metric: '현재 낙폭',
        current: Math.abs(riskMetrics.currentDrawdown),
        limit: 10,
        usage: Math.abs(riskMetrics.currentDrawdown) / 10 * 100,
        status: Math.abs(riskMetrics.currentDrawdown) > 8 ? 'critical' :
                Math.abs(riskMetrics.currentDrawdown) > 5 ? 'warning' : 'normal'
      }
    ];
    
    setRiskLimits(limits);
    
    // 집중 리스크 계산
    const sectorExposure = positions.reduce((acc, pos) => {
      acc[pos.sector] = (acc[pos.sector] || 0) + pos.weight;
      return acc;
    }, {} as Record<string, number>);
    
    const concentrations: ConcentrationRisk[] = Object.entries(sectorExposure).map(([sector, exposure]) => ({
      type: 'sector' as const,
      name: sector,
      exposure,
      limit: 35,
      status: exposure > 35 ? 'critical' : exposure > 30 ? 'warning' : 'normal'
    }));
    
    setConcentrationRisks(concentrations);
    
    // 알림 생성
    const newAlerts: Alert[] = [];
    
    limits.forEach(limit => {
      if (limit.status === 'critical' && Math.random() > 0.7) {
        newAlerts.push({
          id: `${Date.now()}-${limit.metric}`,
          timestamp: new Date(),
          type: 'critical',
          title: `${limit.metric} 한도 초과`,
          message: `현재 ${limit.current.toFixed(1)} / 한도 ${limit.limit}`,
          actionRequired: true
        });
      }
    });
    
    if (newAlerts.length > 0) {
      setAlerts(prev => [...newAlerts, ...prev].slice(0, 10));
    }
  }, [positions, riskMetrics]);
  
  const totalValue = positions.reduce((sum, pos) => sum + pos.marketValue, 0);
  const totalPL = positions.reduce((sum, pos) => sum + pos.unrealizedPL + pos.realizedPL, 0);
  const totalPLPercent = (totalPL / (totalValue - totalPL)) * 100;

  return (
    <div className="space-y-6">
      {/* 헤더 및 실시간 토글 */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">실시간 리스크 대시보드</h2>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            마지막 업데이트: {new Date().toLocaleTimeString()}
          </p>
        </div>
        <button
          onClick={() => setIsLive(!isLive)}
          className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${
            isLive 
              ? 'bg-green-500 text-white hover:bg-green-600' 
              : 'bg-gray-500 text-white hover:bg-gray-600'
          }`}
        >
          <Activity className={`w-4 h-4 ${isLive && 'animate-pulse'}`} />
          {isLive ? 'LIVE' : 'PAUSED'}
        </button>
      </div>

      {/* 탭 네비게이션 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-2 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex gap-2">
          {['overview', 'positions', 'limits', 'alerts'].map((mode) => (
            <button
              key={mode}
              onClick={() => setViewMode(mode as any)}
              className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                viewMode === mode
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700'
              }`}
            >
              {mode === 'overview' && '개요'}
              {mode === 'positions' && '포지션'}
              {mode === 'limits' && '리스크 한도'}
              {mode === 'alerts' && `알림 (${alerts.filter(a => a.type === 'critical').length})`}
            </button>
          ))}
        </div>
      </div>

      {viewMode === 'overview' && (
        <>
          {/* 포트폴리오 요약 */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">포트폴리오 가치</p>
              <p className="text-2xl font-bold">${totalValue.toLocaleString()}</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">총 손익</p>
              <p className={`text-2xl font-bold flex items-center gap-1 ${
                totalPL > 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {totalPL > 0 ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
                ${Math.abs(totalPL).toLocaleString()}
                <span className="text-sm">({totalPLPercent > 0 ? '+' : ''}{totalPLPercent.toFixed(1)}%)</span>
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">VaR (95%)</p>
              <p className="text-2xl font-bold text-red-600">
                {riskMetrics.portfolioVaR.toFixed(1)}%
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">샤프 비율</p>
              <p className="text-2xl font-bold">{riskMetrics.sharpeRatio.toFixed(2)}</p>
            </div>
          </div>

          {/* 리스크 메트릭 그리드 */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">CVaR (95%)</span>
                <AlertTriangle className="w-4 h-4 text-orange-500" />
              </div>
              <p className="text-xl font-bold text-orange-600">{riskMetrics.portfolioCVaR.toFixed(1)}%</p>
              <p className="text-xs text-gray-500">꼬리 리스크</p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">현재 낙폭</span>
                <TrendingDown className="w-4 h-4 text-red-500" />
              </div>
              <p className="text-xl font-bold text-red-600">{riskMetrics.currentDrawdown.toFixed(1)}%</p>
              <p className="text-xs text-gray-500">최고점 대비</p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">포트폴리오 베타</span>
                <Activity className="w-4 h-4 text-blue-500" />
              </div>
              <p className="text-xl font-bold">{riskMetrics.beta.toFixed(2)}</p>
              <p className="text-xs text-gray-500">시장 민감도</p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">소르티노 비율</span>
                <Shield className="w-4 h-4 text-green-500" />
              </div>
              <p className="text-xl font-bold text-green-600">{riskMetrics.sortinoRatio.toFixed(2)}</p>
              <p className="text-xs text-gray-500">하방 리스크 조정</p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">상관계수</span>
                <BarChart3 className="w-4 h-4 text-purple-500" />
              </div>
              <p className="text-xl font-bold">{riskMetrics.correlation.toFixed(2)}</p>
              <p className="text-xs text-gray-500">벤치마크 대비</p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">추적 오차</span>
                <Eye className="w-4 h-4 text-gray-500" />
              </div>
              <p className="text-xl font-bold">{riskMetrics.trackingError.toFixed(1)}%</p>
              <p className="text-xs text-gray-500">연율화</p>
            </div>
          </div>

          {/* 집중 리스크 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">집중 리스크</h3>
            
            <div className="space-y-3">
              {concentrationRisks.map((risk, idx) => (
                <div key={idx} className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium">{risk.name}</span>
                      <span className={`text-sm font-medium ${
                        risk.status === 'critical' ? 'text-red-600' :
                        risk.status === 'warning' ? 'text-yellow-600' : 'text-green-600'
                      }`}>
                        {risk.exposure.toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          risk.status === 'critical' ? 'bg-red-500' :
                          risk.status === 'warning' ? 'bg-yellow-500' : 'bg-green-500'
                        }`}
                        style={{ width: `${Math.min(100, (risk.exposure / risk.limit) * 100)}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {viewMode === 'positions' && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-4">포지션 상세</h3>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-2">종목</th>
                  <th className="text-right p-2">수량</th>
                  <th className="text-right p-2">평균단가</th>
                  <th className="text-right p-2">현재가</th>
                  <th className="text-right p-2">평가액</th>
                  <th className="text-right p-2">손익</th>
                  <th className="text-right p-2">비중</th>
                  <th className="text-right p-2">베타</th>
                  <th className="text-left p-2">섹터</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((pos) => (
                  <tr key={pos.symbol} className="border-b border-gray-100 dark:border-gray-900">
                    <td className="p-2 font-medium">{pos.symbol}</td>
                    <td className="text-right p-2">{pos.quantity}</td>
                    <td className="text-right p-2">${pos.avgCost.toFixed(2)}</td>
                    <td className="text-right p-2">${pos.currentPrice.toFixed(2)}</td>
                    <td className="text-right p-2">${pos.marketValue.toLocaleString()}</td>
                    <td className="text-right p-2">
                      <span className={pos.unrealizedPL > 0 ? 'text-green-600' : 'text-red-600'}>
                        ${pos.unrealizedPL.toFixed(0)}
                        <span className="text-xs ml-1">
                          ({pos.unrealizedPL > 0 ? '+' : ''}{((pos.unrealizedPL / (pos.avgCost * pos.quantity)) * 100).toFixed(1)}%)
                        </span>
                      </span>
                    </td>
                    <td className="text-right p-2">{pos.weight.toFixed(1)}%</td>
                    <td className="text-right p-2">{pos.beta.toFixed(2)}</td>
                    <td className="p-2 text-xs">
                      <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded">
                        {pos.sector}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {viewMode === 'limits' && (
        <div className="space-y-4">
          {riskLimits.map((limit, idx) => (
            <div key={idx} className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold">{limit.metric}</h4>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  limit.status === 'critical' ? 'bg-red-100 text-red-700 dark:bg-red-900/20' :
                  limit.status === 'warning' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/20' :
                  'bg-green-100 text-green-700 dark:bg-green-900/20'
                }`}>
                  {limit.status === 'critical' ? '위험' :
                   limit.status === 'warning' ? '주의' : '정상'}
                </span>
              </div>
              
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  현재: {limit.current.toFixed(1)} / 한도: {limit.limit}
                </span>
                <span className="text-sm font-medium">
                  {limit.usage.toFixed(0)}% 사용
                </span>
              </div>
              
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                <div
                  className={`h-3 rounded-full transition-all ${
                    limit.status === 'critical' ? 'bg-red-500' :
                    limit.status === 'warning' ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${Math.min(100, limit.usage)}%` }}
                />
              </div>
              
              {limit.status !== 'normal' && (
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                  {limit.status === 'critical' 
                    ? '즉시 조치가 필요합니다. 포지션 축소를 검토하세요.'
                    : '한도에 근접하고 있습니다. 모니터링을 강화하세요.'}
                </p>
              )}
            </div>
          ))}
        </div>
      )}

      {viewMode === 'alerts' && (
        <div className="space-y-3">
          {alerts.length === 0 ? (
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700 text-center">
              <Bell className="w-12 h-12 mx-auto text-gray-400 mb-3" />
              <p className="text-gray-600 dark:text-gray-400">현재 활성 알림이 없습니다</p>
            </div>
          ) : (
            alerts.map((alert) => (
              <div
                key={alert.id}
                className={`p-4 rounded-lg border-2 ${
                  alert.type === 'critical' ? 'bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800' :
                  alert.type === 'warning' ? 'bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20 dark:border-yellow-800' :
                  'bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-800'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className={`w-5 h-5 flex-shrink-0 ${
                      alert.type === 'critical' ? 'text-red-600' :
                      alert.type === 'warning' ? 'text-yellow-600' : 'text-blue-600'
                    }`} />
                    <div>
                      <h4 className="font-semibold">{alert.title}</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        {alert.message}
                      </p>
                      {alert.actionRequired && (
                        <p className="text-sm font-medium mt-2 text-red-600">
                          조치 필요
                        </p>
                      )}
                    </div>
                  </div>
                  <span className="text-xs text-gray-500">
                    {alert.timestamp.toLocaleTimeString()}
                  </span>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* 리스크 대시보드 가이드 */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Shield className="w-5 h-5" />
          실시간 리스크 관리 가이드
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-3">핵심 모니터링 지표</h4>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-blue-500">✓</span>
                <span><strong>VaR/CVaR</strong>: 일일 점검, 한도의 80% 초과 시 경고</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">✓</span>
                <span><strong>집중도</strong>: 단일 포지션 20%, 섹터 35% 한도</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">✓</span>
                <span><strong>낙폭</strong>: -10% 도달 시 리스크 감축 검토</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">✓</span>
                <span><strong>베타</strong>: 1.3 초과 시 시장 리스크 과다</span>
              </li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-3">대응 프로토콜</h4>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>한도 90% 도달: 신규 포지션 중단, 기존 포지션 검토</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>한도 초과: 즉시 포지션 축소, 헤징 실행</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>복수 경고: 포트폴리오 전면 재검토</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>시스템 알림: 24시간 내 대응 완료</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}