'use client';

import React, { useState, useEffect } from 'react';
import { Calculator, TrendingUp, TrendingDown, Shield, AlertTriangle, DollarSign, Info } from 'lucide-react';

interface InvestmentData {
  investmentAmount: number;
  stockReturn: number;
  initialExchangeRate: number;
  finalExchangeRate: number;
  hedgeRatio: number;
}

interface CalculationResult {
  // 환노출 (No Hedge)
  stockReturnUSD: number;
  exchangeReturn: number;
  totalReturnKRW_NoHedge: number;
  finalValueKRW_NoHedge: number;
  
  // 환헤지 (Hedged)
  effectiveExchangeReturn: number;
  totalReturnKRW_Hedged: number;
  finalValueKRW_Hedged: number;
  hedgeCost: number;
  
  // 비교
  difference: number;
  betterStrategy: 'hedge' | 'no-hedge';
}

export default function CurrencyImpactAnalyzer() {
  const [investmentData, setInvestmentData] = useState<InvestmentData>({
    investmentAmount: 10000000, // 1천만원
    stockReturn: 10, // 10% 주식 수익률
    initialExchangeRate: 1320,
    finalExchangeRate: 1250,
    hedgeRatio: 100 // 100% 헤지
  });

  const [result, setResult] = useState<CalculationResult | null>(null);
  const [scenarioMode, setScenarioMode] = useState(false);

  // 계산 로직
  const calculateReturns = (data: InvestmentData): CalculationResult => {
    const { investmentAmount, stockReturn, initialExchangeRate, finalExchangeRate, hedgeRatio } = data;
    
    // USD 투자금액
    const investmentUSD = investmentAmount / initialExchangeRate;
    
    // 주식 수익률 적용
    const stockReturnUSD = investmentUSD * (1 + stockReturn / 100);
    
    // 환율 변동률
    const exchangeReturn = ((finalExchangeRate - initialExchangeRate) / initialExchangeRate) * 100;
    
    // 환노출 시 최종 가치
    const finalValueKRW_NoHedge = stockReturnUSD * finalExchangeRate;
    const totalReturnKRW_NoHedge = ((finalValueKRW_NoHedge - investmentAmount) / investmentAmount) * 100;
    
    // 환헤지 시 계산
    const hedgeRatioDecimal = hedgeRatio / 100;
    const hedgedPortion = investmentUSD * hedgeRatioDecimal;
    const unhedgedPortion = investmentUSD * (1 - hedgeRatioDecimal);
    
    // 헤지 비용 (연 1% 가정)
    const hedgeCost = hedgedPortion * 0.01;
    
    // 헤지된 부분은 초기 환율 적용, 헤지되지 않은 부분은 최종 환율 적용
    const hedgedValueKRW = (hedgedPortion - hedgeCost) * (1 + stockReturn / 100) * initialExchangeRate;
    const unhedgedValueKRW = unhedgedPortion * (1 + stockReturn / 100) * finalExchangeRate;
    const finalValueKRW_Hedged = hedgedValueKRW + unhedgedValueKRW;
    
    const totalReturnKRW_Hedged = ((finalValueKRW_Hedged - investmentAmount) / investmentAmount) * 100;
    const effectiveExchangeReturn = ((finalValueKRW_Hedged / stockReturnUSD - initialExchangeRate) / initialExchangeRate) * 100;
    
    const difference = totalReturnKRW_NoHedge - totalReturnKRW_Hedged;
    const betterStrategy = totalReturnKRW_NoHedge > totalReturnKRW_Hedged ? 'no-hedge' : 'hedge';
    
    return {
      stockReturnUSD: stockReturnUSD,
      exchangeReturn,
      totalReturnKRW_NoHedge,
      finalValueKRW_NoHedge,
      effectiveExchangeReturn,
      totalReturnKRW_Hedged,
      finalValueKRW_Hedged,
      hedgeCost: hedgeCost * initialExchangeRate,
      difference: Math.abs(difference),
      betterStrategy
    };
  };

  useEffect(() => {
    setResult(calculateReturns(investmentData));
  }, [investmentData]);

  // 시나리오 데이터
  const scenarios = [
    { name: '강달러 시나리오', exchangeChange: 10, description: '달러 강세로 환율 10% 상승' },
    { name: '약달러 시나리오', exchangeChange: -10, description: '달러 약세로 환율 10% 하락' },
    { name: '급격한 원화 강세', exchangeChange: -20, description: '원화 강세로 환율 20% 하락' },
    { name: '금융위기 시나리오', exchangeChange: 30, description: '위기로 환율 30% 급등' },
  ];

  const applyScenario = (exchangeChange: number) => {
    const newExchangeRate = investmentData.initialExchangeRate * (1 + exchangeChange / 100);
    setInvestmentData({
      ...investmentData,
      finalExchangeRate: Math.round(newExchangeRate)
    });
  };

  const formatCurrency = (amount: number) => {
    return amount.toLocaleString('ko-KR', { maximumFractionDigits: 0 });
  };

  const formatPercent = (percent: number) => {
    return `${percent >= 0 ? '+' : ''}${percent.toFixed(2)}%`;
  };

  return (
    <div className="space-y-6">
      {/* 입력 섹션 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Calculator className="w-5 h-5" />
          투자 정보 입력
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">투자 금액 (원)</label>
              <input
                type="number"
                value={investmentData.investmentAmount}
                onChange={(e) => setInvestmentData({...investmentData, investmentAmount: Number(e.target.value)})}
                className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                step="1000000"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">예상 주식 수익률 (%)</label>
              <input
                type="number"
                value={investmentData.stockReturn}
                onChange={(e) => setInvestmentData({...investmentData, stockReturn: Number(e.target.value)})}
                className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                step="1"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">환헤지 비율 (%)</label>
              <input
                type="range"
                value={investmentData.hedgeRatio}
                onChange={(e) => setInvestmentData({...investmentData, hedgeRatio: Number(e.target.value)})}
                className="w-full"
                min="0"
                max="100"
                step="10"
              />
              <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400">
                <span>0% (환노출)</span>
                <span className="font-medium text-lg">{investmentData.hedgeRatio}%</span>
                <span>100% (완전헤지)</span>
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">초기 환율 (USD/KRW)</label>
              <input
                type="number"
                value={investmentData.initialExchangeRate}
                onChange={(e) => setInvestmentData({...investmentData, initialExchangeRate: Number(e.target.value)})}
                className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                step="10"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">최종 환율 (USD/KRW)</label>
              <input
                type="number"
                value={investmentData.finalExchangeRate}
                onChange={(e) => setInvestmentData({...investmentData, finalExchangeRate: Number(e.target.value)})}
                className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                step="10"
              />
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
              <div className="flex items-center justify-between">
                <span className="text-sm">환율 변동</span>
                <span className={`font-medium ${result && result.exchangeReturn < 0 ? 'text-blue-600' : 'text-red-600'}`}>
                  {result && formatPercent(result.exchangeReturn)}
                </span>
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                {result && result.exchangeReturn < 0 ? '원화 강세 (달러 약세)' : '원화 약세 (달러 강세)'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* 시나리오 분석 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <AlertTriangle className="w-5 h-5" />
            환율 시나리오 분석
          </h3>
          <button
            onClick={() => setScenarioMode(!scenarioMode)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              scenarioMode 
                ? 'bg-blue-500 text-white' 
                : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600'
            }`}
          >
            {scenarioMode ? '시나리오 모드 ON' : '시나리오 모드 OFF'}
          </button>
        </div>
        
        <div className="grid md:grid-cols-2 gap-4">
          {scenarios.map((scenario) => (
            <button
              key={scenario.name}
              onClick={() => applyScenario(scenario.exchangeChange)}
              className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors text-left"
            >
              <h4 className="font-medium mb-1">{scenario.name}</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">{scenario.description}</p>
              <p className={`text-lg font-bold mt-2 ${scenario.exchangeChange > 0 ? 'text-red-600' : 'text-blue-600'}`}>
                환율 {scenario.exchangeChange > 0 ? '+' : ''}{scenario.exchangeChange}%
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* 결과 비교 */}
      {result && (
        <div className="grid md:grid-cols-2 gap-6">
          {/* 환노출 전략 */}
          <div className={`bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border-2 ${
            result.betterStrategy === 'no-hedge' 
              ? 'border-green-500' 
              : 'border-gray-200 dark:border-gray-700'
          }`}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">환노출 전략</h3>
              {result.betterStrategy === 'no-hedge' && (
                <span className="px-3 py-1 bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 rounded-full text-sm font-medium">
                  유리함
                </span>
              )}
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600 dark:text-gray-400">주식 수익률</span>
                <span className="font-medium">{formatPercent(investmentData.stockReturn)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600 dark:text-gray-400">환율 수익률</span>
                <span className={`font-medium ${result.exchangeReturn < 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {formatPercent(result.exchangeReturn)}
                </span>
              </div>
              <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                <div className="flex justify-between items-center">
                  <span className="font-medium">총 수익률</span>
                  <span className={`text-lg font-bold ${result.totalReturnKRW_NoHedge < 0 ? 'text-red-600' : 'text-green-600'}`}>
                    {formatPercent(result.totalReturnKRW_NoHedge)}
                  </span>
                </div>
                <div className="flex justify-between items-center mt-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">최종 평가액</span>
                  <span className="font-medium">₩{formatCurrency(result.finalValueKRW_NoHedge)}</span>
                </div>
              </div>
            </div>
          </div>

          {/* 환헤지 전략 */}
          <div className={`bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border-2 ${
            result.betterStrategy === 'hedge' 
              ? 'border-green-500' 
              : 'border-gray-200 dark:border-gray-700'
          }`}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <Shield className="w-5 h-5" />
                환헤지 전략 ({investmentData.hedgeRatio}%)
              </h3>
              {result.betterStrategy === 'hedge' && (
                <span className="px-3 py-1 bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 rounded-full text-sm font-medium">
                  유리함
                </span>
              )}
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600 dark:text-gray-400">주식 수익률</span>
                <span className="font-medium">{formatPercent(investmentData.stockReturn)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600 dark:text-gray-400">헤지 비용</span>
                <span className="text-red-600">-₩{formatCurrency(result.hedgeCost)}</span>
              </div>
              <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                <div className="flex justify-between items-center">
                  <span className="font-medium">총 수익률</span>
                  <span className={`text-lg font-bold ${result.totalReturnKRW_Hedged < 0 ? 'text-red-600' : 'text-green-600'}`}>
                    {formatPercent(result.totalReturnKRW_Hedged)}
                  </span>
                </div>
                <div className="flex justify-between items-center mt-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">최종 평가액</span>
                  <span className="font-medium">₩{formatCurrency(result.finalValueKRW_Hedged)}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 전략 비교 요약 */}
      {result && (
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">전략 비교 요약</h3>
          
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">수익률 차이</p>
              <p className="text-2xl font-bold">{result.difference.toFixed(2)}%p</p>
              <p className="text-sm mt-1">
                {result.betterStrategy === 'no-hedge' ? '환노출' : '환헤지'}이 유리
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">금액 차이</p>
              <p className="text-2xl font-bold">
                ₩{formatCurrency(Math.abs(result.finalValueKRW_NoHedge - result.finalValueKRW_Hedged))}
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">추천 전략</p>
              <p className="text-lg font-bold">
                {result.exchangeReturn < 0 
                  ? '환헤지 추천 (원화 강세)' 
                  : '환노출 추천 (원화 약세)'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* 환헤지 가이드 */}
      <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Info className="w-5 h-5" />
          환헤지 전략 가이드
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-2">환헤지가 유리한 경우</h4>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• 원화 강세가 예상될 때</li>
              <li>• 단기 투자 (1년 미만)</li>
              <li>• 안정적 수익 추구</li>
              <li>• 환율 변동성이 클 때</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-2">환노출이 유리한 경우</h4>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• 달러 강세가 예상될 때</li>
              <li>• 장기 투자 (3년 이상)</li>
              <li>• 높은 수익 추구</li>
              <li>• 달러 자산 축적 목적</li>
            </ul>
          </div>
        </div>
        
        <div className="mt-4 p-4 bg-white dark:bg-gray-800 rounded-lg">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            <strong>💡 Tip:</strong> 완전 헤지(100%)보다는 부분 헤지(50-70%)를 고려하세요. 
            환율 방향을 정확히 예측하기 어렵기 때문에 적절한 비율로 리스크를 분산하는 것이 현명합니다.
          </p>
        </div>
      </div>

      {/* 포트폴리오 환율 민감도 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <DollarSign className="w-5 h-5" />
          포트폴리오 환율 민감도 분석
        </h3>
        
        <div className="space-y-4">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            환율이 1% 변동할 때 포트폴리오 가치 변화
          </p>
          
          <div className="grid grid-cols-5 gap-2">
            {[0, 25, 50, 75, 100].map((hedgeRatio) => {
              const sensitivity = 1 - (hedgeRatio / 100);
              return (
                <div key={hedgeRatio} className="text-center">
                  <div 
                    className={`p-4 rounded-lg ${
                      hedgeRatio === investmentData.hedgeRatio 
                        ? 'bg-blue-100 dark:bg-blue-900 border-2 border-blue-500' 
                        : 'bg-gray-50 dark:bg-gray-900'
                    }`}
                  >
                    <p className="text-sm font-medium mb-1">헤지 {hedgeRatio}%</p>
                    <p className="text-lg font-bold">{(sensitivity).toFixed(2)}%</p>
                  </div>
                </div>
              );
            })}
          </div>
          
          <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
            <p className="text-sm">
              현재 설정 (헤지 {investmentData.hedgeRatio}%)에서는 환율이 10% 변동하면 
              포트폴리오 가치가 약 <span className="font-bold">
                {((1 - investmentData.hedgeRatio / 100) * 10).toFixed(1)}%
              </span> 변동합니다.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}