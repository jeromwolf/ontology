'use client';

import React, { useState, useEffect } from 'react';
import { TrendingDown, AlertTriangle, Zap, BarChart3, Info, Play, Shield, Target } from 'lucide-react';

interface Asset {
  symbol: string;
  name: string;
  type: 'stock' | 'bond' | 'commodity' | 'currency' | 'crypto';
  currentPrice: number;
  weight: number;
  beta: number;
  correlation: { [key: string]: number };
}

interface Scenario {
  id: string;
  name: string;
  description: string;
  severity: 'moderate' | 'severe' | 'extreme';
  probability: number;
  impacts: {
    equity: number;
    bond: number;
    commodity: number;
    currency: number;
    crypto: number;
  };
  duration: number; // months
  historicalExample?: string;
}

interface StressTestResult {
  scenario: Scenario;
  portfolioLoss: number;
  assetLosses: { asset: Asset; loss: number; newValue: number }[];
  recovery: {
    months: number;
    path: number[];
  };
  recommendations: string[];
  hedgingStrategies: string[];
}

interface PortfolioMetrics {
  totalValue: number;
  expectedLoss: number;
  worstCaseLoss: number;
  stressScore: number;
  vulnerabilities: string[];
}

// 역사적 시나리오 데이터
const scenarios: Scenario[] = [
  {
    id: '2008-crisis',
    name: '2008년 금융위기',
    description: '서브프라임 모기지 위기로 시작된 글로벌 금융시스템 붕괴',
    severity: 'extreme',
    probability: 5,
    impacts: {
      equity: -55,
      bond: 5,
      commodity: -70,
      currency: 15,
      crypto: 0
    },
    duration: 18,
    historicalExample: 'S&P 500: -56.8%, 리먼 브라더스 파산'
  },
  {
    id: 'covid-crash',
    name: '코로나19 팬데믹',
    description: '전 세계적 봉쇄와 경제 활동 중단으로 인한 급격한 시장 하락',
    severity: 'severe',
    probability: 10,
    impacts: {
      equity: -34,
      bond: 8,
      commodity: -50,
      currency: 5,
      crypto: -50
    },
    duration: 3,
    historicalExample: 'S&P 500: -33.9% (23일간), 역사상 가장 빠른 하락'
  },
  {
    id: 'dotcom-bubble',
    name: '닷컴 버블 붕괴',
    description: '기술주 과대평가 버블 붕괴와 9/11 테러의 복합 충격',
    severity: 'severe',
    probability: 7,
    impacts: {
      equity: -49,
      bond: 12,
      commodity: -20,
      currency: 0,
      crypto: 0
    },
    duration: 30,
    historicalExample: 'NASDAQ: -78%, 수많은 닷컴 기업 파산'
  },
  {
    id: 'inflation-shock',
    name: '급격한 인플레이션',
    description: '1970년대 스타일 스태그플레이션, 금리 급등',
    severity: 'moderate',
    probability: 20,
    impacts: {
      equity: -25,
      bond: -30,
      commodity: 40,
      currency: -10,
      crypto: -40
    },
    duration: 24,
    historicalExample: '1970년대: CPI 13.5%, 금리 20%'
  },
  {
    id: 'china-crisis',
    name: '중국 경제 경착륙',
    description: '중국 부동산 버블 붕괴와 금융시스템 위기',
    severity: 'severe',
    probability: 15,
    impacts: {
      equity: -40,
      bond: 10,
      commodity: -45,
      currency: 8,
      crypto: -60
    },
    duration: 12,
    historicalExample: '예상 시나리오: GDP 성장률 급락, 위안화 평가절하'
  },
  {
    id: 'cyber-attack',
    name: '대규모 사이버 공격',
    description: '주요 금융 인프라에 대한 조직적 사이버 공격',
    severity: 'moderate',
    probability: 25,
    impacts: {
      equity: -15,
      bond: 5,
      commodity: 0,
      currency: 3,
      crypto: -70
    },
    duration: 1,
    historicalExample: '2017 WannaCry, 2021 Colonial Pipeline'
  },
  {
    id: 'sovereign-debt',
    name: '국가 부채 위기',
    description: '주요국 디폴트 위험과 신용등급 하락',
    severity: 'severe',
    probability: 12,
    impacts: {
      equity: -30,
      bond: -40,
      commodity: -10,
      currency: 20,
      crypto: -45
    },
    duration: 18,
    historicalExample: '2010-2012 유럽 재정위기, 그리스 디폴트'
  },
  {
    id: 'geopolitical',
    name: '지정학적 분쟁 확대',
    description: '주요 지역 군사 충돌과 무역 중단',
    severity: 'moderate',
    probability: 30,
    impacts: {
      equity: -20,
      bond: 15,
      commodity: 30,
      currency: 5,
      crypto: -25
    },
    duration: 6,
    historicalExample: '2022 러시아-우크라이나 전쟁'
  }
];

// 포트폴리오 예시
const samplePortfolio: Asset[] = [
  { symbol: 'SPY', name: 'S&P 500 ETF', type: 'stock', currentPrice: 450, weight: 40, beta: 1.0, correlation: { SPY: 1, AGG: -0.2, GLD: 0.1, DXY: -0.3, BTC: 0.4 } },
  { symbol: 'AGG', name: '채권 ETF', type: 'bond', currentPrice: 105, weight: 30, beta: 0.2, correlation: { SPY: -0.2, AGG: 1, GLD: 0.2, DXY: 0.1, BTC: -0.1 } },
  { symbol: 'GLD', name: '금', type: 'commodity', currentPrice: 180, weight: 10, beta: 0.3, correlation: { SPY: 0.1, AGG: 0.2, GLD: 1, DXY: -0.5, BTC: 0.3 } },
  { symbol: 'DXY', name: '달러 인덱스', type: 'currency', currentPrice: 105, weight: 10, beta: -0.2, correlation: { SPY: -0.3, AGG: 0.1, GLD: -0.5, DXY: 1, BTC: -0.4 } },
  { symbol: 'BTC', name: '비트코인', type: 'crypto', currentPrice: 65000, weight: 10, beta: 2.5, correlation: { SPY: 0.4, AGG: -0.1, GLD: 0.3, DXY: -0.4, BTC: 1 } }
];

export default function StressTestScenarios() {
  const [portfolio, setPortfolio] = useState<Asset[]>(samplePortfolio);
  const [selectedScenarios, setSelectedScenarios] = useState<string[]>(['2008-crisis', 'covid-crash']);
  const [testResults, setTestResults] = useState<StressTestResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [viewMode, setViewMode] = useState<'scenarios' | 'results' | 'hedging'>('scenarios');
  const [portfolioMetrics, setPortfolioMetrics] = useState<PortfolioMetrics | null>(null);
  
  // 스트레스 테스트 실행
  const runStressTest = () => {
    setIsRunning(true);
    
    const results: StressTestResult[] = [];
    const totalValue = portfolio.reduce((sum, asset) => sum + asset.currentPrice * asset.weight, 0);
    
    selectedScenarios.forEach(scenarioId => {
      const scenario = scenarios.find(s => s.id === scenarioId);
      if (!scenario) return;
      
      // 각 자산별 손실 계산
      const assetLosses = portfolio.map(asset => {
        const baseImpact = scenario.impacts[asset.type];
        
        // 베타와 상관관계를 고려한 조정
        let adjustedImpact = baseImpact * asset.beta;
        
        // 상관관계 기반 추가 영향
        portfolio.forEach(otherAsset => {
          if (otherAsset.symbol !== asset.symbol) {
            const correlation = asset.correlation[otherAsset.symbol] || 0;
            const otherImpact = scenario.impacts[otherAsset.type];
            adjustedImpact += correlation * otherImpact * 0.3;
          }
        });
        
        const loss = adjustedImpact;
        const newValue = asset.currentPrice * (1 + loss / 100);
        
        return { asset, loss, newValue };
      });
      
      // 포트폴리오 전체 손실
      const portfolioLoss = assetLosses.reduce((sum, al) => 
        sum + (al.loss * al.asset.weight / 100), 0
      );
      
      // 회복 경로 시뮬레이션
      const recoveryPath: number[] = [portfolioLoss];
      const recoveryRate = scenario.severity === 'extreme' ? 0.05 : 
                          scenario.severity === 'severe' ? 0.08 : 0.12;
      
      let currentLoss = portfolioLoss;
      let months = 0;
      
      while (currentLoss < -5 && months < 60) {
        currentLoss = currentLoss * (1 - recoveryRate);
        recoveryPath.push(currentLoss);
        months++;
      }
      
      // 권장사항 생성
      const recommendations: string[] = [];
      if (Math.abs(portfolioLoss) > 30) {
        recommendations.push('포트폴리오 다각화 강화 필요');
        recommendations.push('방어적 자산 비중 확대 고려');
      }
      if (portfolio.find(a => a.type === 'crypto' && a.weight > 5)) {
        recommendations.push('암호화폐 비중 축소 검토');
      }
      if (portfolio.filter(a => a.type === 'bond').reduce((sum, a) => sum + a.weight, 0) < 20) {
        recommendations.push('채권 비중 확대로 변동성 완화');
      }
      
      // 헤징 전략
      const hedgingStrategies: string[] = [];
      if (scenario.impacts.equity < -30) {
        hedgingStrategies.push('Put 옵션 매수로 하방 보호');
        hedgingStrategies.push('VIX 관련 상품으로 변동성 헤지');
      }
      if (scenario.impacts.currency !== 0) {
        hedgingStrategies.push('통화 선물로 환리스크 헤지');
      }
      hedgingStrategies.push('금과 같은 안전자산 비중 확대');
      
      results.push({
        scenario,
        portfolioLoss,
        assetLosses,
        recovery: {
          months,
          path: recoveryPath
        },
        recommendations,
        hedgingStrategies
      });
    });
    
    setTestResults(results);
    
    // 포트폴리오 메트릭 계산
    const expectedLoss = results.reduce((sum, r) => 
      sum + r.portfolioLoss * r.scenario.probability / 100, 0
    );
    const worstCaseLoss = Math.min(...results.map(r => r.portfolioLoss));
    
    const vulnerabilities: string[] = [];
    if (portfolio.find(a => a.type === 'stock' && a.weight > 60)) {
      vulnerabilities.push('주식 집중도 과다');
    }
    if (portfolio.filter(a => a.type === 'bond').reduce((sum, a) => sum + a.weight, 0) < 20) {
      vulnerabilities.push('채권 비중 부족');
    }
    if (portfolio.find(a => a.type === 'crypto' && a.weight > 10)) {
      vulnerabilities.push('암호화폐 리스크 과다');
    }
    
    setPortfolioMetrics({
      totalValue,
      expectedLoss,
      worstCaseLoss,
      stressScore: Math.abs(worstCaseLoss) > 40 ? 3 : Math.abs(worstCaseLoss) > 25 ? 2 : 1,
      vulnerabilities
    });
    
    setIsRunning(false);
    setViewMode('results');
  };
  
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'extreme': return 'text-red-600 bg-red-100 dark:bg-red-900/20';
      case 'severe': return 'text-orange-600 bg-orange-100 dark:bg-orange-900/20';
      case 'moderate': return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20';
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-900/20';
    }
  };

  return (
    <div className="space-y-6">
      {/* 탭 네비게이션 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-2 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex gap-2">
          <button
            onClick={() => setViewMode('scenarios')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === 'scenarios'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            시나리오 선택
          </button>
          <button
            onClick={() => setViewMode('results')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === 'results'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            } ${testResults.length === 0 && 'opacity-50 cursor-not-allowed'}`}
            disabled={testResults.length === 0}
          >
            테스트 결과
          </button>
          <button
            onClick={() => setViewMode('hedging')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === 'hedging'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            } ${testResults.length === 0 && 'opacity-50 cursor-not-allowed'}`}
            disabled={testResults.length === 0}
          >
            헤징 전략
          </button>
        </div>
      </div>

      {viewMode === 'scenarios' && (
        <>
          {/* 포트폴리오 구성 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">현재 포트폴리오</h3>
            
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {portfolio.map((asset) => (
                <div key={asset.symbol} className="text-center">
                  <div className="text-2xl font-bold">{asset.weight}%</div>
                  <div className="text-sm font-medium">{asset.symbol}</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">{asset.name}</div>
                  <div className={`text-xs mt-1 px-2 py-1 rounded ${
                    asset.type === 'stock' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/20' :
                    asset.type === 'bond' ? 'bg-green-100 text-green-700 dark:bg-green-900/20' :
                    asset.type === 'commodity' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/20' :
                    asset.type === 'currency' ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/20' :
                    'bg-orange-100 text-orange-700 dark:bg-orange-900/20'
                  }`}>
                    {asset.type === 'stock' ? '주식' :
                     asset.type === 'bond' ? '채권' :
                     asset.type === 'commodity' ? '원자재' :
                     asset.type === 'currency' ? '통화' : '암호화폐'}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 시나리오 목록 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">스트레스 테스트 시나리오</h3>
            
            <div className="space-y-3">
              {scenarios.map((scenario) => (
                <div
                  key={scenario.id}
                  className={`p-4 rounded-lg border-2 transition-colors cursor-pointer ${
                    selectedScenarios.includes(scenario.id)
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                  onClick={() => {
                    if (selectedScenarios.includes(scenario.id)) {
                      setSelectedScenarios(selectedScenarios.filter(id => id !== scenario.id));
                    } else {
                      setSelectedScenarios([...selectedScenarios, scenario.id]);
                    }
                  }}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h4 className="font-semibold">{scenario.name}</h4>
                        <span className={`px-2 py-1 text-xs rounded-full ${getSeverityColor(scenario.severity)}`}>
                          {scenario.severity === 'extreme' ? '극심' :
                           scenario.severity === 'severe' ? '심각' : '중간'}
                        </span>
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                          발생 확률: {scenario.probability}%
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        {scenario.description}
                      </p>
                      {scenario.historicalExample && (
                        <p className="text-xs text-gray-500 italic">
                          역사적 사례: {scenario.historicalExample}
                        </p>
                      )}
                    </div>
                    <div className="ml-4 text-right">
                      <div className="text-sm font-medium mb-1">예상 영향</div>
                      <div className="text-xs space-y-1">
                        <div className={scenario.impacts.equity < 0 ? 'text-red-600' : 'text-green-600'}>
                          주식: {scenario.impacts.equity > 0 ? '+' : ''}{scenario.impacts.equity}%
                        </div>
                        <div className={scenario.impacts.bond < 0 ? 'text-red-600' : 'text-green-600'}>
                          채권: {scenario.impacts.bond > 0 ? '+' : ''}{scenario.impacts.bond}%
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            
            <button
              onClick={runStressTest}
              disabled={selectedScenarios.length === 0 || isRunning}
              className="mt-6 w-full px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isRunning ? (
                <>
                  <Zap className="w-4 h-4 animate-pulse" />
                  스트레스 테스트 실행 중...
                </>
              ) : (
                <>
                  <AlertTriangle className="w-4 h-4" />
                  스트레스 테스트 실행
                </>
              )}
            </button>
          </div>
        </>
      )}

      {viewMode === 'results' && testResults.length > 0 && portfolioMetrics && (
        <>
          {/* 전체 요약 */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">기대 손실</p>
              <p className="text-2xl font-bold text-red-600">
                {portfolioMetrics.expectedLoss.toFixed(1)}%
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">최악 시나리오</p>
              <p className="text-2xl font-bold text-red-600">
                {portfolioMetrics.worstCaseLoss.toFixed(1)}%
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">스트레스 점수</p>
              <p className={`text-2xl font-bold ${
                portfolioMetrics.stressScore === 3 ? 'text-red-600' :
                portfolioMetrics.stressScore === 2 ? 'text-yellow-600' : 'text-green-600'
              }`}>
                {portfolioMetrics.stressScore}/3
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">취약점</p>
              <p className="text-2xl font-bold text-orange-600">
                {portfolioMetrics.vulnerabilities.length}개
              </p>
            </div>
          </div>

          {/* 시나리오별 결과 */}
          <div className="space-y-4">
            {testResults.map((result, idx) => (
              <div key={idx} className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">{result.scenario.name}</h3>
                  <span className={`text-2xl font-bold ${
                    result.portfolioLoss < -30 ? 'text-red-600' :
                    result.portfolioLoss < -15 ? 'text-orange-600' : 'text-yellow-600'
                  }`}>
                    {result.portfolioLoss.toFixed(1)}% 손실
                  </span>
                </div>
                
                {/* 자산별 영향 */}
                <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
                  {result.assetLosses.map((al) => (
                    <div key={al.asset.symbol} className="text-center p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                      <div className="font-medium">{al.asset.symbol}</div>
                      <div className={`text-lg font-bold ${
                        al.loss < -20 ? 'text-red-600' :
                        al.loss < 0 ? 'text-orange-600' : 'text-green-600'
                      }`}>
                        {al.loss > 0 ? '+' : ''}{al.loss.toFixed(1)}%
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">
                        ${al.newValue.toFixed(2)}
                      </div>
                    </div>
                  ))}
                </div>
                
                {/* 회복 기간 */}
                <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <p className="text-sm">
                    예상 회복 기간: <span className="font-semibold">{result.recovery.months}개월</span>
                  </p>
                </div>
              </div>
            ))}
          </div>

          {/* 취약점 분석 */}
          {portfolioMetrics.vulnerabilities.length > 0 && (
            <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5" />
                포트폴리오 취약점
              </h3>
              <ul className="space-y-2">
                {portfolioMetrics.vulnerabilities.map((vulnerability, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-sm">
                    <span className="text-yellow-600">•</span>
                    <span>{vulnerability}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}

      {viewMode === 'hedging' && testResults.length > 0 && (
        <div className="space-y-6">
          {/* 헤징 전략 요약 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Shield className="w-5 h-5" />
              권장 헤징 전략
            </h3>
            
            {testResults.map((result, idx) => (
              <div key={idx} className="mb-6 last:mb-0">
                <h4 className="font-medium mb-3">{result.scenario.name} 대비</h4>
                
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h5 className="text-sm font-medium mb-2 text-gray-600 dark:text-gray-400">
                      헤징 전략
                    </h5>
                    <ul className="space-y-1">
                      {result.hedgingStrategies.map((strategy, sIdx) => (
                        <li key={sIdx} className="text-sm flex items-start gap-2">
                          <Target className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" />
                          <span>{strategy}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  <div>
                    <h5 className="text-sm font-medium mb-2 text-gray-600 dark:text-gray-400">
                      포트폴리오 조정 권고
                    </h5>
                    <ul className="space-y-1">
                      {result.recommendations.map((rec, rIdx) => (
                        <li key={rIdx} className="text-sm flex items-start gap-2">
                          <Info className="w-4 h-4 text-green-500 flex-shrink-0 mt-0.5" />
                          <span>{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* 구체적 헤징 상품 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">구체적 헤징 상품 예시</h3>
            
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <h4 className="font-medium mb-2">VIX 옵션/ETF</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  시장 변동성 급등 시 수익 창출
                </p>
                <p className="text-xs">추천: VXX, VIXY ETF</p>
              </div>
              
              <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <h4 className="font-medium mb-2">Put 옵션</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  포트폴리오 하방 보호
                </p>
                <p className="text-xs">추천: SPY/QQQ 3-6개월 OTM Put</p>
              </div>
              
              <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <h4 className="font-medium mb-2">금/은 ETF</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  안전자산 비중 확대
                </p>
                <p className="text-xs">추천: GLD, SLV, IAU</p>
              </div>
              
              <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <h4 className="font-medium mb-2">역상관 ETF</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  시장 하락 시 수익
                </p>
                <p className="text-xs">추천: SH, PSQ, DOG</p>
              </div>
              
              <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <h4 className="font-medium mb-2">장기 국채</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  위기 시 안전자산 선호
                </p>
                <p className="text-xs">추천: TLT, IEF, GOVT</p>
              </div>
              
              <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <h4 className="font-medium mb-2">통화 헤지</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  환율 변동 리스크 관리
                </p>
                <p className="text-xs">추천: UUP, FXY, FXE</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 스트레스 테스트 가이드 */}
      <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Info className="w-5 h-5" />
          스트레스 테스트 가이드
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-3">스트레스 테스트의 중요성</h4>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-red-500">✓</span>
                <span>극단적 시장 상황에서의 포트폴리오 취약점 파악</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-red-500">✓</span>
                <span>역사적 위기를 통한 미래 리스크 예측</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-red-500">✓</span>
                <span>사전적 리스크 관리 전략 수립</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-red-500">✓</span>
                <span>정기적 실행으로 포트폴리오 건전성 유지</span>
              </li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-3">효과적인 활용법</h4>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>분기별 1회 이상 정기적으로 실행</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>복수 시나리오를 동시에 테스트</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>테스트 결과에 따라 실제 포트폴리오 조정</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>헤징 비용과 보호 수준의 균형 고려</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}