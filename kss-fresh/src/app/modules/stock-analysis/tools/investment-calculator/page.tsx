'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Calculator, TrendingUp, DollarSign, Calendar, PieChart, Info, RefreshCw, Download, Target, Percent, Clock } from 'lucide-react';

interface CalculationResult {
  finalAmount: number;
  totalReturn: number;
  totalReturnPercent: number;
  annualizedReturn: number;
  totalInvested: number;
  totalProfit: number;
  monthlyIncome?: number;
  breakEvenMonth?: number;
}

export default function InvestmentCalculatorPage() {
  // 기본 계산기 상태
  const [calcType, setCalcType] = useState<'compound' | 'regular' | 'target'>('compound');
  const [initialAmount, setInitialAmount] = useState(100); // 100만원
  const [monthlyAmount, setMonthlyAmount] = useState(10); // 10만원
  const [annualReturn, setAnnualReturn] = useState(8);
  const [dividendYield, setDividendYield] = useState(3); // 배당 수익률
  const [investmentPeriod, setInvestmentPeriod] = useState(10);
  const [targetAmount, setTargetAmount] = useState(10000); // 1억원
  const [inflationRate, setInflationRate] = useState(2);
  const [taxRate, setTaxRate] = useState(15.4);
  
  // 계산 결과
  const [result, setResult] = useState<CalculationResult | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // 복리 계산
  const calculateCompoundReturn = () => {
    const monthlyRate = annualReturn / 100 / 12;
    const months = investmentPeriod * 12;
    
    // 만원 단위를 원 단위로 변환
    const initialInWon = initialAmount * 10000;
    const monthlyInWon = monthlyAmount * 10000;
    
    // 초기 투자금의 복리 성장
    const initialGrowth = initialInWon * Math.pow(1 + monthlyRate, months);
    
    // 적립식 투자금의 복리 성장
    const regularGrowth = monthlyInWon * ((Math.pow(1 + monthlyRate, months) - 1) / monthlyRate);
    
    const finalAmount = initialGrowth + regularGrowth;
    const totalInvested = initialInWon + (monthlyInWon * months);
    const totalProfit = finalAmount - totalInvested;
    const totalReturnPercent = (totalProfit / totalInvested) * 100;
    const annualizedReturn = Math.pow(finalAmount / totalInvested, 1 / investmentPeriod) - 1;
    
    // 세후 수익 계산
    const afterTaxProfit = totalProfit * (1 - taxRate / 100);
    const afterTaxFinalAmount = totalInvested + afterTaxProfit;
    
    // 인플레이션 조정
    const realFinalAmount = afterTaxFinalAmount / Math.pow(1 + inflationRate / 100, investmentPeriod);
    
    setResult({
      finalAmount: showAdvanced ? realFinalAmount : finalAmount,
      totalReturn: totalProfit,
      totalReturnPercent,
      annualizedReturn: annualizedReturn * 100,
      totalInvested,
      totalProfit: showAdvanced ? afterTaxProfit : totalProfit
    });
  };

  // 목표 수익 달성 기간 계산
  const calculateTargetPeriod = () => {
    const monthlyRate = annualReturn / 100 / 12;
    
    // 만원 단위를 원 단위로 변환
    const initialInWon = initialAmount * 10000;
    const monthlyInWon = monthlyAmount * 10000;
    const targetInWon = targetAmount * 10000;
    
    if (monthlyAmount === 0) {
      // 일시금만 있는 경우
      const months = Math.log(targetInWon / initialInWon) / Math.log(1 + monthlyRate);
      const years = months / 12;
      
      setResult({
        finalAmount: targetInWon,
        totalReturn: targetInWon - initialInWon,
        totalReturnPercent: ((targetInWon - initialInWon) / initialInWon) * 100,
        annualizedReturn: annualReturn,
        totalInvested: initialInWon,
        totalProfit: targetInWon - initialInWon,
        breakEvenMonth: Math.ceil(months)
      });
    } else {
      // 적립식 투자 포함
      let balance = initialInWon;
      let months = 0;
      let totalInvested = initialInWon;
      
      while (balance < targetInWon && months < 600) { // 최대 50년
        balance = balance * (1 + monthlyRate) + monthlyInWon;
        totalInvested += monthlyInWon;
        months++;
      }
      
      setResult({
        finalAmount: balance,
        totalReturn: balance - totalInvested,
        totalReturnPercent: ((balance - totalInvested) / totalInvested) * 100,
        annualizedReturn: annualReturn,
        totalInvested,
        totalProfit: balance - totalInvested,
        breakEvenMonth: months
      });
    }
  };

  // 은퇴 후 월소득 계산 (최종 자산을 배당주에 투자했을 때)
  const calculateMonthlyIncome = () => {
    const monthlyRate = annualReturn / 100 / 12;
    const months = investmentPeriod * 12;
    
    // 만원 단위를 원 단위로 변환
    const initialInWon = initialAmount * 10000;
    const monthlyInWon = monthlyAmount * 10000;
    
    // 투자 기간 후 최종 자산 계산
    const initialGrowth = initialInWon * Math.pow(1 + monthlyRate, months);
    const regularGrowth = monthlyInWon * ((Math.pow(1 + monthlyRate, months) - 1) / monthlyRate);
    const finalAmount = initialGrowth + regularGrowth;
    
    // 은퇴 후 월 소득 계산
    // 최종 자산을 배당주에 투자했을 때의 월 배당금
    const monthlyIncome = (finalAmount * dividendYield / 100) / 12;
    const totalInvested = initialInWon + (monthlyInWon * months);
    
    setResult({
      finalAmount,
      totalReturn: finalAmount - totalInvested,
      totalReturnPercent: ((finalAmount - totalInvested) / totalInvested) * 100,
      annualizedReturn: annualReturn,
      totalInvested,
      totalProfit: finalAmount - totalInvested,
      monthlyIncome
    });
  };

  const handleCalculate = () => {
    switch (calcType) {
      case 'compound':
        calculateCompoundReturn();
        break;
      case 'regular':
        calculateMonthlyIncome();
        break;
      case 'target':
        calculateTargetPeriod();
        break;
    }
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('ko-KR').format(Math.round(num));
  };

  const formatCurrency = (num: number) => {
    if (num >= 100000000) {
      return `${(num / 100000000).toFixed(2)}억원`;
    } else if (num >= 10000) {
      return `${(num / 10000).toFixed(0)}만원`;
    }
    return `${formatNumber(num)}원`;
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link 
                href="/modules/stock-analysis/tools"
                className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>도구 목록</span>
              </Link>
              <div className="h-6 w-px bg-gray-300 dark:bg-gray-700" />
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">투자 수익률 계산기</h1>
              <span className="px-2 py-1 bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 rounded text-xs font-medium">
                Free
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Panel - Input */}
          <div className="space-y-6">
            {/* Calculator Type Selection */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">계산 유형 선택</h2>
              <div className="grid grid-cols-3 gap-3">
                <button
                  onClick={() => setCalcType('compound')}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    calcType === 'compound'
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                  }`}
                >
                  <TrendingUp className={`w-6 h-6 mx-auto mb-2 ${
                    calcType === 'compound' ? 'text-blue-600' : 'text-gray-500'
                  }`} />
                  <p className={`text-sm font-medium ${
                    calcType === 'compound' ? 'text-blue-600' : 'text-gray-700 dark:text-gray-300'
                  }`}>
                    복리 수익률
                  </p>
                </button>
                
                <button
                  onClick={() => setCalcType('regular')}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    calcType === 'regular'
                      ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                  }`}
                >
                  <DollarSign className={`w-6 h-6 mx-auto mb-2 ${
                    calcType === 'regular' ? 'text-green-600' : 'text-gray-500'
                  }`} />
                  <p className={`text-sm font-medium ${
                    calcType === 'regular' ? 'text-green-600' : 'text-gray-700 dark:text-gray-300'
                  }`}>
                    은퇴 후 월소득
                  </p>
                </button>
                
                <button
                  onClick={() => setCalcType('target')}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    calcType === 'target'
                      ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Target className={`w-6 h-6 mx-auto mb-2 ${
                    calcType === 'target' ? 'text-purple-600' : 'text-gray-500'
                  }`} />
                  <p className={`text-sm font-medium ${
                    calcType === 'target' ? 'text-purple-600' : 'text-gray-700 dark:text-gray-300'
                  }`}>
                    목표 달성
                  </p>
                </button>
              </div>
            </div>

            {/* Input Fields */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 space-y-4">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">투자 정보 입력</h2>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  초기 투자금 (만원)
                </label>
                <div className="relative">
                  <input
                    type="number"
                    value={initialAmount}
                    onChange={(e) => setInitialAmount(Number(e.target.value))}
                    className="w-full px-4 py-3 pr-16 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="100"
                  />
                  <span className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-500">만원</span>
                </div>
                <p className="text-xs text-gray-500 mt-1">현재: {formatCurrency(initialAmount * 10000)}</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  월 적립금 (만원)
                </label>
                <div className="relative">
                  <input
                    type="number"
                    value={monthlyAmount}
                    onChange={(e) => setMonthlyAmount(Number(e.target.value))}
                    className="w-full px-4 py-3 pr-16 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="10"
                  />
                  <span className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-500">만원</span>
                </div>
                <p className="text-xs text-gray-500 mt-1">현재: {formatCurrency(monthlyAmount * 10000)}</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  연 수익률
                </label>
                <div className="relative">
                  <input
                    type="number"
                    value={annualReturn}
                    onChange={(e) => setAnnualReturn(Number(e.target.value))}
                    step="0.1"
                    className="w-full px-4 py-3 pr-12 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  <span className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-500">%</span>
                </div>
              </div>

              {calcType === 'regular' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    배당 수익률
                  </label>
                  <div className="relative">
                    <input
                      type="number"
                      value={dividendYield}
                      onChange={(e) => setDividendYield(Number(e.target.value))}
                      step="0.1"
                      className="w-full px-4 py-3 pr-12 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <span className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-500">%</span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">은퇴 후 투자할 배당주의 연간 배당률</p>
                </div>
              )}

              {calcType !== 'target' ? (
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    투자 기간
                  </label>
                  <div className="relative">
                    <input
                      type="number"
                      value={investmentPeriod}
                      onChange={(e) => setInvestmentPeriod(Number(e.target.value))}
                      className="w-full px-4 py-3 pr-12 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <span className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-500">년</span>
                  </div>
                </div>
              ) : (
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    목표 금액 (만원)
                  </label>
                  <div className="relative">
                    <input
                      type="number"
                      value={targetAmount}
                      onChange={(e) => setTargetAmount(Number(e.target.value))}
                      className="w-full px-4 py-3 pr-16 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="10000"
                    />
                    <span className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-500">만원</span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">현재: {formatCurrency(targetAmount * 10000)}</p>
                </div>
              )}

              {/* Advanced Options */}
              <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700"
                >
                  <Info className="w-4 h-4" />
                  고급 옵션 {showAdvanced ? '숨기기' : '보기'}
                </button>
                
                {showAdvanced && (
                  <div className="mt-4 space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        세율
                      </label>
                      <div className="relative">
                        <input
                          type="number"
                          value={taxRate}
                          onChange={(e) => setTaxRate(Number(e.target.value))}
                          step="0.1"
                          className="w-full px-4 py-3 pr-12 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                        <span className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-500">%</span>
                      </div>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        물가상승률
                      </label>
                      <div className="relative">
                        <input
                          type="number"
                          value={inflationRate}
                          onChange={(e) => setInflationRate(Number(e.target.value))}
                          step="0.1"
                          className="w-full px-4 py-3 pr-12 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                        <span className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-500">%</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <button
                onClick={handleCalculate}
                className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                <Calculator className="w-5 h-5" />
                계산하기
              </button>
            </div>
          </div>

          {/* Right Panel - Results */}
          <div className="space-y-6">
            {result && (
              <>
                {/* Main Result */}
                <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl shadow-lg p-6 text-white">
                  <h2 className="text-lg font-semibold mb-4">계산 결과</h2>
                  
                  <div className="space-y-4">
                    <div>
                      <p className="text-blue-100 text-sm">최종 자산</p>
                      <p className="text-3xl font-bold">{formatCurrency(result.finalAmount)}</p>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-blue-100 text-sm">총 투자금</p>
                        <p className="text-xl font-semibold">{formatCurrency(result.totalInvested)}</p>
                      </div>
                      <div>
                        <p className="text-blue-100 text-sm">총 수익</p>
                        <p className="text-xl font-semibold">{formatCurrency(result.totalProfit)}</p>
                      </div>
                    </div>
                    
                    {result.monthlyIncome && (
                      <div className="pt-4 border-t border-blue-400">
                        <p className="text-blue-100 text-sm">은퇴 후 예상 월 소득</p>
                        <p className="text-2xl font-bold">{formatCurrency(result.monthlyIncome)}</p>
                        <p className="text-xs text-blue-200 mt-1">
                          최종 자산을 연 {dividendYield}% 배당주에 투자 시
                        </p>
                      </div>
                    )}
                    
                    {result.breakEvenMonth && (
                      <div className="pt-4 border-t border-blue-400">
                        <p className="text-blue-100 text-sm">목표 달성 기간</p>
                        <p className="text-2xl font-bold">
                          {Math.floor(result.breakEvenMonth / 12)}년 {result.breakEvenMonth % 12}개월
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Detailed Breakdown */}
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">상세 분석</h3>
                  
                  <div className="space-y-4">
                    <div className="flex items-center justify-between py-3 border-b border-gray-200 dark:border-gray-700">
                      <div className="flex items-center gap-2">
                        <Percent className="w-5 h-5 text-gray-500" />
                        <span className="text-gray-700 dark:text-gray-300">총 수익률</span>
                      </div>
                      <span className="font-semibold text-gray-900 dark:text-white">
                        {result.totalReturnPercent.toFixed(2)}%
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between py-3 border-b border-gray-200 dark:border-gray-700">
                      <div className="flex items-center gap-2">
                        <TrendingUp className="w-5 h-5 text-gray-500" />
                        <span className="text-gray-700 dark:text-gray-300">연평균 수익률</span>
                      </div>
                      <span className="font-semibold text-gray-900 dark:text-white">
                        {result.annualizedReturn.toFixed(2)}%
                      </span>
                    </div>
                    
                    {showAdvanced && (
                      <>
                        <div className="flex items-center justify-between py-3 border-b border-gray-200 dark:border-gray-700">
                          <div className="flex items-center gap-2">
                            <DollarSign className="w-5 h-5 text-gray-500" />
                            <span className="text-gray-700 dark:text-gray-300">세후 실질 가치</span>
                          </div>
                          <span className="font-semibold text-gray-900 dark:text-white">
                            {formatCurrency(result.finalAmount)}
                          </span>
                        </div>
                      </>
                    )}
                  </div>
                </div>

                {/* Tips */}
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-3">
                    💡 계산 방법 설명
                  </h3>
                  <div className="space-y-3 text-sm text-blue-800 dark:text-blue-200">
                    <div>
                      <p className="font-semibold mb-1">📈 복리 수익률</p>
                      <p className="text-xs">투자 기간 동안 원금과 수익을 재투자하여 얻는 총 수익을 계산합니다.</p>
                    </div>
                    <div>
                      <p className="font-semibold mb-1">💰 은퇴 후 월소득</p>
                      <p className="text-xs">투자 기간 후 모은 자산을 배당주에 투자했을 때 받을 수 있는 월 배당금을 계산합니다. 예: 2억원을 연 6% 배당주에 투자 시 월 100만원 수령</p>
                    </div>
                    <div>
                      <p className="font-semibold mb-1">🎯 목표 달성</p>
                      <p className="text-xs">원하는 목표 금액에 도달하는데 필요한 기간을 계산합니다.</p>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}