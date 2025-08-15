'use client';

import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, Shield, AlertCircle, Info, Target, Activity, PieChart } from 'lucide-react';

interface AssetClass {
  name: string;
  symbol: string;
  category: 'equity' | 'bond' | 'commodity' | 'real_estate' | 'currency';
  currentWeight: number;
  targetWeight: number;
  volatility: number;
  expectedReturn: number;
  sharpeRatio: number;
  correlation: { [key: string]: number };
  riskContribution: number;
}

interface PortfolioMetrics {
  totalRisk: number;
  expectedReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  diversificationRatio: number;
  riskParityScore: number; // 0-100, 100이 완벽한 리스크 패리티
}

// 자산 클래스 데이터
const assetClasses: AssetClass[] = [
  {
    name: '미국 주식',
    symbol: 'SPY',
    category: 'equity',
    currentWeight: 60,
    targetWeight: 25,
    volatility: 16.5,
    expectedReturn: 10.2,
    sharpeRatio: 0.62,
    correlation: { 'SPY': 1, 'AGG': -0.1, 'GLD': 0.15, 'VNQ': 0.75, 'DXY': -0.3 },
    riskContribution: 0
  },
  {
    name: '미국 채권',
    symbol: 'AGG',
    category: 'bond',
    currentWeight: 30,
    targetWeight: 40,
    volatility: 3.8,
    expectedReturn: 3.5,
    sharpeRatio: 0.92,
    correlation: { 'SPY': -0.1, 'AGG': 1, 'GLD': 0.2, 'VNQ': 0.1, 'DXY': 0.05 },
    riskContribution: 0
  },
  {
    name: '금',
    symbol: 'GLD',
    category: 'commodity',
    currentWeight: 5,
    targetWeight: 15,
    volatility: 15.2,
    expectedReturn: 7.5,
    sharpeRatio: 0.49,
    correlation: { 'SPY': 0.15, 'AGG': 0.2, 'GLD': 1, 'VNQ': 0.25, 'DXY': -0.5 },
    riskContribution: 0
  },
  {
    name: '부동산',
    symbol: 'VNQ',
    category: 'real_estate',
    currentWeight: 5,
    targetWeight: 15,
    volatility: 19.8,
    expectedReturn: 8.7,
    sharpeRatio: 0.44,
    correlation: { 'SPY': 0.75, 'AGG': 0.1, 'GLD': 0.25, 'VNQ': 1, 'DXY': -0.2 },
    riskContribution: 0
  },
  {
    name: '달러 인덱스',
    symbol: 'DXY',
    category: 'currency',
    currentWeight: 0,
    targetWeight: 5,
    volatility: 7.5,
    expectedReturn: 2.0,
    sharpeRatio: 0.27,
    correlation: { 'SPY': -0.3, 'AGG': 0.05, 'GLD': -0.5, 'VNQ': -0.2, 'DXY': 1 },
    riskContribution: 0
  }
];

export default function RiskParityPortfolio() {
  const [assets, setAssets] = useState<AssetClass[]>(assetClasses);
  const [portfolioType, setPortfolioType] = useState<'current' | 'riskParity' | 'custom'>('current');
  const [rebalanceFrequency, setRebalanceFrequency] = useState<'monthly' | 'quarterly' | 'annually'>('quarterly');
  const [showDetails, setShowDetails] = useState(false);
  const [targetVolatility, setTargetVolatility] = useState(10);
  
  // 포트폴리오 메트릭 계산
  const calculatePortfolioMetrics = (weights: number[]): PortfolioMetrics => {
    let portfolioVariance = 0;
    let expectedReturn = 0;
    
    // 포트폴리오 분산 계산
    for (let i = 0; i < assets.length; i++) {
      for (let j = 0; j < assets.length; j++) {
        const correlation = assets[i].correlation[assets[j].symbol] || 0;
        portfolioVariance += weights[i] * weights[j] * 
                           (assets[i].volatility / 100) * (assets[j].volatility / 100) * 
                           correlation;
      }
      expectedReturn += weights[i] * assets[i].expectedReturn;
    }
    
    const totalRisk = Math.sqrt(portfolioVariance) * 100;
    const sharpeRatio = expectedReturn / totalRisk;
    
    // 리스크 기여도 계산
    const riskContributions = assets.map((asset, i) => {
      let contribution = 0;
      for (let j = 0; j < assets.length; j++) {
        const correlation = asset.correlation[assets[j].symbol] || 0;
        contribution += weights[j] * (asset.volatility / 100) * (assets[j].volatility / 100) * correlation;
      }
      return (weights[i] * contribution) / (totalRisk / 100);
    });
    
    // 리스크 패리티 점수 계산 (표준편차가 낮을수록 점수가 높음)
    const avgContribution = 1 / assets.length;
    const variance = riskContributions.reduce((sum, rc) => 
      sum + Math.pow(rc - avgContribution, 2), 0) / assets.length;
    const riskParityScore = Math.max(0, 100 - Math.sqrt(variance) * 1000);
    
    // 다각화 비율
    const sumVolatility = assets.reduce((sum, asset, i) => 
      sum + weights[i] * asset.volatility, 0);
    const diversificationRatio = sumVolatility / totalRisk;
    
    return {
      totalRisk,
      expectedReturn: expectedReturn / 100,
      sharpeRatio,
      maxDrawdown: -totalRisk * 2.5, // 근사치
      diversificationRatio,
      riskParityScore
    };
  };
  
  // 리스크 패리티 가중치 계산
  const calculateRiskParityWeights = (): number[] => {
    const n = assets.length;
    let weights = new Array(n).fill(1 / n);
    
    // Newton-Raphson 방법으로 리스크 패리티 가중치 찾기
    for (let iter = 0; iter < 100; iter++) {
      const metrics = calculatePortfolioMetrics(weights);
      const totalRisk = metrics.totalRisk / 100;
      
      // 각 자산의 한계 리스크 기여도 계산
      const marginalRisks = assets.map((asset, i) => {
        let marginal = 0;
        for (let j = 0; j < n; j++) {
          const correlation = asset.correlation[assets[j].symbol] || 0;
          marginal += weights[j] * (asset.volatility / 100) * (assets[j].volatility / 100) * correlation;
        }
        return marginal / totalRisk;
      });
      
      // 가중치 업데이트
      const sumInverse = marginalRisks.reduce((sum, mr) => sum + 1 / mr, 0);
      weights = marginalRisks.map(mr => (1 / mr) / sumInverse);
    }
    
    return weights;
  };
  
  // 현재 가중치 가져오기
  const getCurrentWeights = () => {
    if (portfolioType === 'current') {
      return assets.map(a => a.currentWeight / 100);
    } else if (portfolioType === 'riskParity') {
      return calculateRiskParityWeights();
    } else {
      return assets.map(a => a.targetWeight / 100);
    }
  };
  
  const weights = getCurrentWeights();
  const metrics = calculatePortfolioMetrics(weights);
  
  // 리스크 기여도 업데이트
  useEffect(() => {
    const updatedAssets = assets.map((asset, i) => {
      let contribution = 0;
      for (let j = 0; j < assets.length; j++) {
        const correlation = asset.correlation[assets[j].symbol] || 0;
        contribution += weights[j] * (asset.volatility / 100) * (assets[j].volatility / 100) * correlation;
      }
      return {
        ...asset,
        riskContribution: (weights[i] * contribution) / (metrics.totalRisk / 100) * 100
      };
    });
    setAssets(updatedAssets);
  }, [portfolioType]);
  
  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'equity': return 'bg-blue-500';
      case 'bond': return 'bg-green-500';
      case 'commodity': return 'bg-yellow-500';
      case 'real_estate': return 'bg-purple-500';
      case 'currency': return 'bg-gray-500';
      default: return 'bg-gray-400';
    }
  };

  return (
    <div className="space-y-6">
      {/* 포트폴리오 타입 선택 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-2 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex gap-2">
          <button
            onClick={() => setPortfolioType('current')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              portfolioType === 'current'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            현재 포트폴리오
          </button>
          <button
            onClick={() => setPortfolioType('riskParity')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              portfolioType === 'riskParity'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            리스크 패리티
          </button>
          <button
            onClick={() => setPortfolioType('custom')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              portfolioType === 'custom'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            커스텀 설정
          </button>
        </div>
      </div>

      {/* 포트폴리오 메트릭 */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-600 dark:text-gray-400">예상 수익률</p>
          <p className="text-2xl font-bold text-green-600">{metrics.expectedReturn.toFixed(2)}%</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-600 dark:text-gray-400">포트폴리오 변동성</p>
          <p className="text-2xl font-bold">{metrics.totalRisk.toFixed(2)}%</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-600 dark:text-gray-400">샤프 비율</p>
          <p className="text-2xl font-bold">{metrics.sharpeRatio.toFixed(3)}</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-600 dark:text-gray-400">최대 낙폭</p>
          <p className="text-2xl font-bold text-red-600">{metrics.maxDrawdown.toFixed(1)}%</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-600 dark:text-gray-400">다각화 비율</p>
          <p className="text-2xl font-bold">{metrics.diversificationRatio.toFixed(2)}</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-600 dark:text-gray-400">리스크 패리티 점수</p>
          <p className="text-2xl font-bold text-blue-600">{metrics.riskParityScore.toFixed(0)}/100</p>
        </div>
      </div>

      {/* 자산 배분 및 리스크 기여도 */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* 자산 배분 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <PieChart className="w-5 h-5" />
            자산 배분
          </h3>
          
          <div className="space-y-3">
            {assets.map((asset, idx) => (
              <div key={asset.symbol} className="flex items-center gap-3">
                <div className={`w-4 h-4 rounded ${getCategoryColor(asset.category)}`} />
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium">{asset.name}</span>
                    <span className="text-sm font-medium">{(weights[idx] * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${getCategoryColor(asset.category)}`}
                      style={{ width: `${weights[idx] * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {portfolioType === 'custom' && (
            <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
              >
                가중치 조정 {showDetails ? '숨기기' : '보기'}
              </button>
            </div>
          )}
        </div>

        {/* 리스크 기여도 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Shield className="w-5 h-5" />
            리스크 기여도
          </h3>
          
          <div className="space-y-3">
            {assets.map((asset) => (
              <div key={asset.symbol} className="flex items-center gap-3">
                <div className={`w-4 h-4 rounded ${getCategoryColor(asset.category)}`} />
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium">{asset.name}</span>
                    <span className="text-sm font-medium">{asset.riskContribution.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${
                        Math.abs(asset.riskContribution - 100/assets.length) < 5
                          ? 'bg-green-500'
                          : 'bg-orange-500'
                      }`}
                      style={{ width: `${asset.riskContribution}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <p className="text-sm">
              {portfolioType === 'riskParity' 
                ? '✓ 각 자산이 동일한 리스크를 기여하도록 최적화됨'
                : '⚠ 리스크 기여도가 불균형함 - 리스크 패리티 전략 고려'}
            </p>
          </div>
        </div>
      </div>

      {/* 상관관계 매트릭스 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4">자산 간 상관관계</h3>
        
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr>
                <th className="text-left p-2"></th>
                {assets.map(asset => (
                  <th key={asset.symbol} className="text-center p-2 font-medium">
                    {asset.symbol}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {assets.map((asset1) => (
                <tr key={asset1.symbol}>
                  <td className="font-medium p-2">{asset1.symbol}</td>
                  {assets.map((asset2) => {
                    const correlation = asset1.correlation[asset2.symbol] || 0;
                    const color = correlation > 0.5 
                      ? 'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-300'
                      : correlation < -0.3
                      ? 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-300'
                      : 'bg-gray-100 dark:bg-gray-900';
                    return (
                      <td key={asset2.symbol} className={`text-center p-2 ${color}`}>
                        {correlation.toFixed(2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        <div className="mt-4 flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-100 dark:bg-green-900/20 rounded" />
            <span>음의 상관관계 (분산 효과 ↑)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-gray-100 dark:bg-gray-900 rounded" />
            <span>낮은 상관관계</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-100 dark:bg-red-900/20 rounded" />
            <span>높은 상관관계 (분산 효과 ↓)</span>
          </div>
        </div>
      </div>

      {/* 리밸런싱 전략 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          리밸런싱 전략
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-3">리밸런싱 주기</h4>
            <div className="space-y-2">
              {['monthly', 'quarterly', 'annually'].map((freq) => (
                <label key={freq} className="flex items-center gap-3">
                  <input
                    type="radio"
                    value={freq}
                    checked={rebalanceFrequency === freq}
                    onChange={(e) => setRebalanceFrequency(e.target.value as any)}
                    className="text-blue-600"
                  />
                  <span className="capitalize">
                    {freq === 'monthly' && '월간 리밸런싱'}
                    {freq === 'quarterly' && '분기별 리밸런싱 (권장)'}
                    {freq === 'annually' && '연간 리밸런싱'}
                  </span>
                </label>
              ))}
            </div>
          </div>
          
          <div>
            <h4 className="font-medium mb-3">목표 변동성 설정</h4>
            <div className="space-y-3">
              <input
                type="range"
                min="5"
                max="20"
                value={targetVolatility}
                onChange={(e) => setTargetVolatility(Number(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm">
                <span>보수적 (5%)</span>
                <span className="font-medium text-lg">{targetVolatility}%</span>
                <span>공격적 (20%)</span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                현재 포트폴리오 변동성: {metrics.totalRisk.toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* 리스크 패리티 가이드 */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Info className="w-5 h-5" />
          리스크 패리티 전략 가이드
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-3">리스크 패리티란?</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              각 자산이 포트폴리오 전체 리스크에 동일하게 기여하도록 가중치를 배분하는 전략입니다.
            </p>
            <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
              <li>• 전통적인 60/40 포트폴리오보다 안정적</li>
              <li>• 다양한 시장 환경에서 일관된 성과</li>
              <li>• 하방 리스크 감소</li>
              <li>• 샤프 비율 개선</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-3">구현 시 고려사항</h4>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>레버리지 사용 가능 시 목표 변동성까지 확대 가능</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>거래 비용을 고려한 리밸런싱 임계값 설정</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>상관관계와 변동성의 시간적 변화 모니터링</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>극단적 시장 상황에 대한 스트레스 테스트</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}