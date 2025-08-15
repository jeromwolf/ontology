'use client';

import React, { useState, useEffect } from 'react';
import { Calculator, DollarSign, TrendingDown, Shield, AlertCircle, FileText, Info, ChevronDown } from 'lucide-react';

interface TaxCalculation {
  // 투자 정보
  investmentAmount: number;
  purchasePrice: number;
  sellPrice: number;
  shares: number;
  holdingPeriod: number; // 개월
  
  // 배당 정보
  totalDividends: number;
  
  // 세금 정보
  countryCode: 'US' | 'KR';
  annualIncome: number; // 연소득 (한국 세율 결정용)
  
  // 계산 결과
  capitalGain: number;
  capitalGainTax: number;
  dividendTax: number;
  totalTax: number;
  netProfit: number;
  effectiveTaxRate: number;
  
  // 절세 전략
  taxSavings: {
    strategy: string;
    amount: number;
    description: string;
  }[];
}

// 미국 세금 브라켓 (2024년 기준)
const usTaxBrackets = {
  single: [
    { min: 0, max: 11600, rate: 0.10 },
    { min: 11600, max: 47150, rate: 0.12 },
    { min: 47150, max: 100525, rate: 0.22 },
    { min: 100525, max: 191950, rate: 0.24 },
    { min: 191950, max: 243725, rate: 0.32 },
    { min: 243725, max: 609350, rate: 0.35 },
    { min: 609350, max: Infinity, rate: 0.37 }
  ],
  capitalGains: {
    longTerm: [
      { min: 0, max: 47025, rate: 0 },
      { min: 47025, max: 518900, rate: 0.15 },
      { min: 518900, max: Infinity, rate: 0.20 }
    ],
    shortTerm: 'ordinary' // 일반 소득세율 적용
  }
};

// 한국 양도소득세율
const krTaxRates = {
  capitalGains: {
    basic: 0.22, // 기본세율 22%
    major: 0.25, // 대주주 25%
    shortTerm: 0.40, // 1년 미만 단기 40%
  },
  dividends: {
    separate: 0.154, // 분리과세 15.4%
    comprehensive: [0.06, 0.38] // 종합과세 6~38%
  }
};

export default function TaxOptimizationCalculator() {
  const [activeTab, setActiveTab] = useState<'US' | 'KR'>('US');
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  const [usInvestment, setUsInvestment] = useState({
    investmentAmount: 10000,
    purchasePrice: 100,
    sellPrice: 150,
    shares: 100,
    holdingPeriod: 18,
    totalDividends: 200,
    annualIncome: 75000,
    filingStatus: 'single' as 'single' | 'married'
  });
  
  const [krInvestment, setKrInvestment] = useState({
    investmentAmount: 10000000,
    purchasePrice: 50000,
    sellPrice: 75000,
    shares: 200,
    holdingPeriod: 6,
    totalDividends: 500000,
    annualIncome: 50000000,
    isMajorShareholder: false
  });
  
  const [usResult, setUsResult] = useState<TaxCalculation | null>(null);
  const [krResult, setKrResult] = useState<TaxCalculation | null>(null);

  // 미국 세금 계산
  const calculateUSTax = () => {
    const { investmentAmount, purchasePrice, sellPrice, shares, holdingPeriod, totalDividends, annualIncome } = usInvestment;
    
    const capitalGain = (sellPrice - purchasePrice) * shares;
    const isLongTerm = holdingPeriod >= 12;
    
    let capitalGainTax = 0;
    
    if (isLongTerm) {
      // 장기 양도소득세
      const brackets = usTaxBrackets.capitalGains.longTerm;
      for (const bracket of brackets) {
        if (annualIncome > bracket.min) {
          const taxableInThisBracket = Math.min(capitalGain, annualIncome - bracket.min);
          capitalGainTax += taxableInThisBracket * bracket.rate;
        }
      }
    } else {
      // 단기 양도소득세 (일반 소득세율)
      const brackets = usTaxBrackets.single;
      let remainingGain = capitalGain;
      let currentIncome = annualIncome;
      
      for (const bracket of brackets) {
        if (currentIncome > bracket.max) continue;
        
        const roomInBracket = bracket.max - currentIncome;
        const taxableInThisBracket = Math.min(remainingGain, roomInBracket);
        
        capitalGainTax += taxableInThisBracket * bracket.rate;
        remainingGain -= taxableInThisBracket;
        currentIncome += taxableInThisBracket;
        
        if (remainingGain <= 0) break;
      }
    }
    
    // 배당소득세 (일반적으로 15% 또는 20%)
    const dividendTax = annualIncome > 518900 ? totalDividends * 0.20 : totalDividends * 0.15;
    
    const totalTax = capitalGainTax + dividendTax;
    const netProfit = capitalGain + totalDividends - totalTax;
    const effectiveTaxRate = (totalTax / (capitalGain + totalDividends)) * 100;
    
    // 절세 전략
    const taxSavings = [];
    
    if (!isLongTerm) {
      const potentialLongTermTax = capitalGain * 0.15; // 추정
      taxSavings.push({
        strategy: '장기 보유 전략',
        amount: capitalGainTax - potentialLongTermTax,
        description: `${12 - holdingPeriod}개월 더 보유 시 약 $${(capitalGainTax - potentialLongTermTax).toFixed(0)} 절세`
      });
    }
    
    taxSavings.push({
      strategy: '세금 우대 계좌 활용',
      amount: totalTax * 0.3,
      description: 'IRA, 401(k) 등 은퇴계좌 활용 시 세금 이연 가능'
    });
    
    taxSavings.push({
      strategy: '손실 상계 (Tax Loss Harvesting)',
      amount: Math.min(capitalGain * 0.15, 3000),
      description: '다른 투자 손실과 상계하여 연간 최대 $3,000 공제'
    });
    
    setUsResult({
      investmentAmount,
      purchasePrice,
      sellPrice,
      shares,
      holdingPeriod,
      totalDividends,
      countryCode: 'US',
      annualIncome,
      capitalGain,
      capitalGainTax,
      dividendTax,
      totalTax,
      netProfit,
      effectiveTaxRate,
      taxSavings
    });
  };
  
  // 한국 세금 계산
  const calculateKRTax = () => {
    const { investmentAmount, purchasePrice, sellPrice, shares, holdingPeriod, totalDividends, annualIncome, isMajorShareholder } = krInvestment;
    
    const capitalGain = (sellPrice - purchasePrice) * shares;
    let capitalGainTax = 0;
    
    // 양도소득세 계산
    if (holdingPeriod < 12) {
      // 1년 미만 단기
      capitalGainTax = capitalGain * krTaxRates.capitalGains.shortTerm;
    } else if (isMajorShareholder) {
      // 대주주
      capitalGainTax = capitalGain * krTaxRates.capitalGains.major;
    } else {
      // 일반
      capitalGainTax = capitalGain * krTaxRates.capitalGains.basic;
    }
    
    // 배당소득세 (분리과세 가정)
    const dividendTax = totalDividends * krTaxRates.dividends.separate;
    
    const totalTax = capitalGainTax + dividendTax;
    const netProfit = capitalGain + totalDividends - totalTax;
    const effectiveTaxRate = (totalTax / (capitalGain + totalDividends)) * 100;
    
    // 절세 전략
    const taxSavings = [];
    
    if (holdingPeriod < 12) {
      const potentialLongTermTax = capitalGain * krTaxRates.capitalGains.basic;
      taxSavings.push({
        strategy: '장기 보유 전략',
        amount: capitalGainTax - potentialLongTermTax,
        description: `1년 이상 보유 시 약 ${((capitalGainTax - potentialLongTermTax) / 1000000).toFixed(1)}백만원 절세`
      });
    }
    
    taxSavings.push({
      strategy: 'ISA 계좌 활용',
      amount: Math.min(totalTax * 0.5, 2000000),
      description: 'ISA 계좌 활용 시 연간 200만원까지 비과세'
    });
    
    taxSavings.push({
      strategy: '국내 상장주식 비과세',
      amount: capitalGain * 0.22,
      description: '국내 상장주식 투자 시 양도차익 비과세 (일반 투자자)'
    });
    
    setKrResult({
      investmentAmount,
      purchasePrice,
      sellPrice,
      shares,
      holdingPeriod,
      totalDividends,
      countryCode: 'KR',
      annualIncome,
      capitalGain,
      capitalGainTax,
      dividendTax,
      totalTax,
      netProfit,
      effectiveTaxRate,
      taxSavings
    });
  };
  
  useEffect(() => {
    if (activeTab === 'US') {
      calculateUSTax();
    } else {
      calculateKRTax();
    }
  }, [usInvestment, krInvestment, activeTab]);
  
  const formatUSD = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };
  
  const formatKRW = (amount: number) => {
    return new Intl.NumberFormat('ko-KR', {
      style: 'currency',
      currency: 'KRW',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };

  return (
    <div className="space-y-6">
      {/* 국가 선택 탭 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-2 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex gap-2">
          <button
            onClick={() => setActiveTab('US')}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-colors ${
              activeTab === 'US'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            🇺🇸 미국 주식 세금 계산
          </button>
          <button
            onClick={() => setActiveTab('KR')}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-colors ${
              activeTab === 'KR'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            🇰🇷 한국 주식 세금 계산
          </button>
        </div>
      </div>

      {/* 미국 주식 세금 계산 */}
      {activeTab === 'US' && (
        <>
          {/* 입력 폼 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Calculator className="w-5 h-5" />
              투자 정보 입력
            </h3>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">매수 가격 (주당)</label>
                  <input
                    type="number"
                    value={usInvestment.purchasePrice}
                    onChange={(e) => setUsInvestment({...usInvestment, purchasePrice: Number(e.target.value)})}
                    className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                    step="0.01"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">매도 가격 (주당)</label>
                  <input
                    type="number"
                    value={usInvestment.sellPrice}
                    onChange={(e) => setUsInvestment({...usInvestment, sellPrice: Number(e.target.value)})}
                    className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                    step="0.01"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">주식 수</label>
                  <input
                    type="number"
                    value={usInvestment.shares}
                    onChange={(e) => setUsInvestment({...usInvestment, shares: Number(e.target.value)})}
                    className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                  />
                </div>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">보유 기간 (개월)</label>
                  <input
                    type="number"
                    value={usInvestment.holdingPeriod}
                    onChange={(e) => setUsInvestment({...usInvestment, holdingPeriod: Number(e.target.value)})}
                    className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    {usInvestment.holdingPeriod >= 12 ? '장기 보유 (세율 우대)' : '단기 보유 (일반 소득세율)'}
                  </p>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">총 배당금</label>
                  <input
                    type="number"
                    value={usInvestment.totalDividends}
                    onChange={(e) => setUsInvestment({...usInvestment, totalDividends: Number(e.target.value)})}
                    className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                    step="0.01"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">연간 소득</label>
                  <input
                    type="number"
                    value={usInvestment.annualIncome}
                    onChange={(e) => setUsInvestment({...usInvestment, annualIncome: Number(e.target.value)})}
                    className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                    step="1000"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    세율 결정에 사용됩니다
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* 계산 결과 */}
          {usResult && (
            <>
              <div className="grid md:grid-cols-2 gap-6">
                {/* 세금 계산 결과 */}
                <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <DollarSign className="w-5 h-5" />
                    세금 계산 결과
                  </h3>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">양도차익</span>
                      <span className="font-medium text-green-600">{formatUSD(usResult.capitalGain)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">배당소득</span>
                      <span className="font-medium text-green-600">{formatUSD(usResult.totalDividends)}</span>
                    </div>
                    <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">양도소득세</span>
                        <span className="font-medium text-red-600">-{formatUSD(usResult.capitalGainTax)}</span>
                      </div>
                      <div className="flex justify-between items-center mt-2">
                        <span className="text-sm text-gray-600 dark:text-gray-400">배당소득세</span>
                        <span className="font-medium text-red-600">-{formatUSD(usResult.dividendTax)}</span>
                      </div>
                    </div>
                    <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                      <div className="flex justify-between items-center">
                        <span className="font-medium">총 세금</span>
                        <span className="text-lg font-bold text-red-600">-{formatUSD(usResult.totalTax)}</span>
                      </div>
                      <div className="flex justify-between items-center mt-2">
                        <span className="font-medium">세후 순수익</span>
                        <span className="text-lg font-bold text-green-600">{formatUSD(usResult.netProfit)}</span>
                      </div>
                      <div className="flex justify-between items-center mt-2">
                        <span className="text-sm text-gray-600 dark:text-gray-400">실효세율</span>
                        <span className="font-medium">{usResult.effectiveTaxRate.toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* 수익률 분석 */}
                <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <TrendingDown className="w-5 h-5" />
                    수익률 분석
                  </h3>
                  
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm">세전 수익률</span>
                        <span className="font-medium">
                          {(((usResult.capitalGain + usResult.totalDividends) / (usResult.purchasePrice * usResult.shares)) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full"
                          style={{ width: `${Math.min(100, ((usResult.capitalGain + usResult.totalDividends) / (usResult.purchasePrice * usResult.shares)) * 100)}%` }}
                        />
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm">세후 수익률</span>
                        <span className="font-medium">
                          {((usResult.netProfit / (usResult.purchasePrice * usResult.shares)) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-green-500 h-2 rounded-full"
                          style={{ width: `${Math.min(100, (usResult.netProfit / (usResult.purchasePrice * usResult.shares)) * 100)}%` }}
                        />
                      </div>
                    </div>
                    
                    <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        세금으로 인해 수익률이 <span className="font-medium text-red-600">
                          {(((usResult.capitalGain + usResult.totalDividends - usResult.netProfit) / (usResult.capitalGain + usResult.totalDividends)) * 100).toFixed(1)}%
                        </span> 감소했습니다.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* 절세 전략 */}
              <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Shield className="w-5 h-5" />
                  절세 전략
                </h3>
                
                <div className="grid md:grid-cols-3 gap-4">
                  {usResult.taxSavings.map((strategy, index) => (
                    <div key={index} className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                      <h4 className="font-medium mb-2">{strategy.strategy}</h4>
                      <p className="text-2xl font-bold text-green-600 mb-2">
                        {formatUSD(strategy.amount)}
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {strategy.description}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </>
      )}

      {/* 한국 주식 세금 계산 */}
      {activeTab === 'KR' && (
        <>
          {/* 입력 폼 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Calculator className="w-5 h-5" />
              투자 정보 입력
            </h3>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">매수 가격 (주당)</label>
                  <input
                    type="number"
                    value={krInvestment.purchasePrice}
                    onChange={(e) => setKrInvestment({...krInvestment, purchasePrice: Number(e.target.value)})}
                    className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                    step="100"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">매도 가격 (주당)</label>
                  <input
                    type="number"
                    value={krInvestment.sellPrice}
                    onChange={(e) => setKrInvestment({...krInvestment, sellPrice: Number(e.target.value)})}
                    className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                    step="100"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">주식 수</label>
                  <input
                    type="number"
                    value={krInvestment.shares}
                    onChange={(e) => setKrInvestment({...krInvestment, shares: Number(e.target.value)})}
                    className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                  />
                </div>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">보유 기간 (개월)</label>
                  <input
                    type="number"
                    value={krInvestment.holdingPeriod}
                    onChange={(e) => setKrInvestment({...krInvestment, holdingPeriod: Number(e.target.value)})}
                    className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    {krInvestment.holdingPeriod >= 12 ? '장기 보유' : '단기 보유 (세율 40%)'}
                  </p>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">총 배당금</label>
                  <input
                    type="number"
                    value={krInvestment.totalDividends}
                    onChange={(e) => setKrInvestment({...krInvestment, totalDividends: Number(e.target.value)})}
                    className="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600"
                    step="10000"
                  />
                </div>
                
                <div>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={krInvestment.isMajorShareholder}
                      onChange={(e) => setKrInvestment({...krInvestment, isMajorShareholder: e.target.checked})}
                      className="rounded border-gray-300 dark:border-gray-600"
                    />
                    <span className="text-sm font-medium">대주주 여부</span>
                  </label>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    지분율 1% 이상 또는 시가총액 10억원 이상
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* 계산 결과 */}
          {krResult && (
            <>
              <div className="grid md:grid-cols-2 gap-6">
                {/* 세금 계산 결과 */}
                <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <DollarSign className="w-5 h-5" />
                    세금 계산 결과
                  </h3>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">양도차익</span>
                      <span className="font-medium text-green-600">{formatKRW(krResult.capitalGain)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">배당소득</span>
                      <span className="font-medium text-green-600">{formatKRW(krResult.totalDividends)}</span>
                    </div>
                    <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">양도소득세</span>
                        <span className="font-medium text-red-600">-{formatKRW(krResult.capitalGainTax)}</span>
                      </div>
                      <div className="flex justify-between items-center mt-2">
                        <span className="text-sm text-gray-600 dark:text-gray-400">배당소득세</span>
                        <span className="font-medium text-red-600">-{formatKRW(krResult.dividendTax)}</span>
                      </div>
                    </div>
                    <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                      <div className="flex justify-between items-center">
                        <span className="font-medium">총 세금</span>
                        <span className="text-lg font-bold text-red-600">-{formatKRW(krResult.totalTax)}</span>
                      </div>
                      <div className="flex justify-between items-center mt-2">
                        <span className="font-medium">세후 순수익</span>
                        <span className="text-lg font-bold text-green-600">{formatKRW(krResult.netProfit)}</span>
                      </div>
                      <div className="flex justify-between items-center mt-2">
                        <span className="text-sm text-gray-600 dark:text-gray-400">실효세율</span>
                        <span className="font-medium">{krResult.effectiveTaxRate.toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* 수익률 분석 */}
                <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <TrendingDown className="w-5 h-5" />
                    수익률 분석
                  </h3>
                  
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm">세전 수익률</span>
                        <span className="font-medium">
                          {(((krResult.capitalGain + krResult.totalDividends) / (krResult.purchasePrice * krResult.shares)) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full"
                          style={{ width: `${Math.min(100, ((krResult.capitalGain + krResult.totalDividends) / (krResult.purchasePrice * krResult.shares)) * 100)}%` }}
                        />
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm">세후 수익률</span>
                        <span className="font-medium">
                          {((krResult.netProfit / (krResult.purchasePrice * krResult.shares)) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-green-500 h-2 rounded-full"
                          style={{ width: `${Math.min(100, (krResult.netProfit / (krResult.purchasePrice * krResult.shares)) * 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* 절세 전략 */}
              <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Shield className="w-5 h-5" />
                  절세 전략
                </h3>
                
                <div className="grid md:grid-cols-3 gap-4">
                  {krResult.taxSavings.map((strategy, index) => (
                    <div key={index} className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                      <h4 className="font-medium mb-2">{strategy.strategy}</h4>
                      <p className="text-2xl font-bold text-green-600 mb-2">
                        {formatKRW(strategy.amount)}
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {strategy.description}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </>
      )}

      {/* 세금 가이드 */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Info className="w-5 h-5" />
          주식 투자 세금 가이드
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-3">🇺🇸 미국 주식 세금</h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• <strong>장기 양도소득세</strong>: 1년 이상 보유 시 0%, 15%, 20%</li>
              <li>• <strong>단기 양도소득세</strong>: 1년 미만 보유 시 일반 소득세율 (10~37%)</li>
              <li>• <strong>배당소득세</strong>: 일반적으로 15% (고소득자 20%)</li>
              <li>• <strong>원천징수</strong>: 한미 조세조약에 따라 배당 15%, 양도차익 면제</li>
              <li>• <strong>손실 상계</strong>: 연간 $3,000까지 일반 소득에서 공제 가능</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-3">🇰🇷 한국 주식 세금</h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• <strong>국내 상장주식</strong>: 일반 투자자 양도차익 비과세</li>
              <li>• <strong>해외 주식</strong>: 양도소득세 22% (1년 미만 40%)</li>
              <li>• <strong>대주주</strong>: 양도소득세 25% (지분 1% 또는 10억원 이상)</li>
              <li>• <strong>배당소득세</strong>: 분리과세 15.4% 또는 종합과세 선택</li>
              <li>• <strong>ISA 계좌</strong>: 연간 200만원까지 비과세 혜택</li>
            </ul>
          </div>
        </div>
        
        <div className="mt-4 p-4 bg-white dark:bg-gray-800 rounded-lg">
          <p className="text-sm flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-yellow-600 mt-0.5 flex-shrink-0" />
            <span>
              본 계산기는 일반적인 세금 계산을 위한 참고용입니다. 
              실제 세금은 개인별 상황에 따라 다를 수 있으므로, 정확한 세금 계산을 위해서는 세무 전문가와 상담하시기 바랍니다.
            </span>
          </p>
        </div>
      </div>
    </div>
  );
}