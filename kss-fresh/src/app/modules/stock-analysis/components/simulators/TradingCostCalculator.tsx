'use client';

import React, { useState, useEffect } from 'react';
import { Calculator, DollarSign, TrendingUp, AlertCircle, Info, BarChart3, Zap, Target } from 'lucide-react';

interface TradeDetails {
  symbol: string;
  quantity: number;
  price: number;
  orderType: 'market' | 'limit' | 'stop';
  venue: 'exchange' | 'darkpool' | 'otc';
  country: 'us' | 'kr' | 'jp' | 'hk' | 'eu';
  accountType: 'cash' | 'margin';
  holdingPeriod: number; // days
}

interface CostBreakdown {
  commission: number;
  spread: number;
  slippage: number;
  marketImpact: number;
  tax: number;
  regulatoryFees: number;
  exchangeFees: number;
  clearingFees: number;
  borrowingCost: number;
  currencyConversion: number;
  totalCost: number;
  effectivePrice: number;
  costPercentage: number;
}

interface BrokerProfile {
  name: string;
  commissionRate: number;
  minCommission: number;
  maxCommission: number;
  spreadMarkup: number;
  features: string[];
}

interface OptimizationSuggestion {
  category: string;
  description: string;
  potentialSaving: number;
  implementation: string;
}

// 브로커 프로필
const brokerProfiles: BrokerProfile[] = [
  {
    name: 'Interactive Brokers',
    commissionRate: 0.0035,
    minCommission: 0.35,
    maxCommission: 1,
    spreadMarkup: 0,
    features: ['최저 수수료', '다양한 시장 접근', 'API 지원']
  },
  {
    name: 'Charles Schwab',
    commissionRate: 0,
    minCommission: 0,
    maxCommission: 0,
    spreadMarkup: 0.01,
    features: ['무료 수수료', '리서치 제공', '초보자 친화적']
  },
  {
    name: '한국투자증권',
    commissionRate: 0.25,
    minCommission: 5,
    maxCommission: 0,
    spreadMarkup: 0,
    features: ['국내 최대 규모', '해외주식 지원', '모바일 앱']
  },
  {
    name: 'Robinhood',
    commissionRate: 0,
    minCommission: 0,
    maxCommission: 0,
    spreadMarkup: 0.02,
    features: ['무료 거래', '간편한 UI', 'Payment for Order Flow']
  }
];

export default function TradingCostCalculator() {
  const [tradeDetails, setTradeDetails] = useState<TradeDetails>({
    symbol: 'AAPL',
    quantity: 100,
    price: 189.95,
    orderType: 'market',
    venue: 'exchange',
    country: 'us',
    accountType: 'cash',
    holdingPeriod: 30
  });
  
  const [selectedBroker, setSelectedBroker] = useState<BrokerProfile>(brokerProfiles[0]);
  const [costBreakdown, setCostBreakdown] = useState<CostBreakdown | null>(null);
  const [viewMode, setViewMode] = useState<'calculator' | 'comparison' | 'optimization'>('calculator');
  const [annualTrades, setAnnualTrades] = useState(50);
  const [avgTradeSize, setAvgTradeSize] = useState(10000);
  
  // 거래 비용 계산
  const calculateTradingCosts = () => {
    const tradeValue = tradeDetails.quantity * tradeDetails.price;
    
    // 수수료 계산
    let commission = 0;
    if (selectedBroker.commissionRate > 0) {
      commission = Math.max(
        selectedBroker.minCommission,
        Math.min(
          tradeValue * selectedBroker.commissionRate,
          selectedBroker.maxCommission || Infinity
        )
      );
    }
    
    // 스프레드 계산
    let spread = 0;
    if (tradeDetails.orderType === 'market') {
      // 기본 스프레드 + 브로커 마크업
      const baseSpread = tradeDetails.venue === 'exchange' ? 0.01 : 0.02;
      spread = tradeValue * (baseSpread + selectedBroker.spreadMarkup);
    }
    
    // 슬리피지 계산
    let slippage = 0;
    if (tradeDetails.orderType === 'market') {
      // 거래량에 따른 슬리피지
      const slippageFactor = tradeDetails.quantity > 1000 ? 0.002 : 
                           tradeDetails.quantity > 500 ? 0.001 : 0.0005;
      slippage = tradeValue * slippageFactor;
    }
    
    // 시장 충격 계산
    let marketImpact = 0;
    if (tradeDetails.quantity > 1000) {
      // 대량 거래의 시장 충격
      const impactFactor = Math.log10(tradeDetails.quantity / 100) * 0.001;
      marketImpact = tradeValue * impactFactor;
    }
    
    // 세금 계산
    let tax = 0;
    if (tradeDetails.country === 'us') {
      // 미국 주식 양도세 (장단기)
      const gainRate = 0.2; // 가정: 20% 수익
      const gain = tradeValue * gainRate;
      const taxRate = tradeDetails.holdingPeriod > 365 ? 0.15 : 0.35;
      tax = gain * taxRate;
    } else if (tradeDetails.country === 'kr') {
      // 한국 주식 거래세
      tax = tradeValue * 0.0023; // 거래세 0.23%
    }
    
    // 규제 수수료
    const regulatoryFees = tradeDetails.country === 'us' ? 
      Math.max(0.01, tradeValue * 0.0000278) : 0; // SEC + FINRA fees
    
    // 거래소 수수료
    const exchangeFees = tradeDetails.venue === 'exchange' ? 
      tradeValue * 0.00003 : 0;
    
    // 청산 수수료
    const clearingFees = tradeValue * 0.00002;
    
    // 차입 비용 (마진 거래)
    let borrowingCost = 0;
    if (tradeDetails.accountType === 'margin') {
      const marginRate = 0.08; // 연 8%
      borrowingCost = (tradeValue * 0.5) * (marginRate / 365) * tradeDetails.holdingPeriod;
    }
    
    // 환전 수수료
    let currencyConversion = 0;
    if (tradeDetails.country !== 'us' && tradeDetails.country !== 'kr') {
      currencyConversion = tradeValue * 0.005; // 0.5% 환전 스프레드
    }
    
    const totalCost = commission + spread + slippage + marketImpact + 
                     tax + regulatoryFees + exchangeFees + clearingFees + 
                     borrowingCost + currencyConversion;
    
    const effectivePrice = tradeDetails.price + (totalCost / tradeDetails.quantity);
    const costPercentage = (totalCost / tradeValue) * 100;
    
    setCostBreakdown({
      commission,
      spread,
      slippage,
      marketImpact,
      tax,
      regulatoryFees,
      exchangeFees,
      clearingFees,
      borrowingCost,
      currencyConversion,
      totalCost,
      effectivePrice,
      costPercentage
    });
  };
  
  useEffect(() => {
    calculateTradingCosts();
  }, [tradeDetails, selectedBroker]);
  
  // 연간 비용 계산
  const calculateAnnualCosts = (broker: BrokerProfile): number => {
    const costPerTrade = avgTradeSize * (broker.commissionRate + broker.spreadMarkup);
    const minCostPerTrade = Math.max(broker.minCommission, costPerTrade);
    return minCostPerTrade * annualTrades * 2; // 매수 + 매도
  };
  
  // 최적화 제안 생성
  const getOptimizationSuggestions = (): OptimizationSuggestion[] => {
    if (!costBreakdown) return [];
    
    const suggestions: OptimizationSuggestion[] = [];
    
    if (costBreakdown.spread > costBreakdown.totalCost * 0.3) {
      suggestions.push({
        category: '주문 유형',
        description: '지정가 주문 사용으로 스프레드 비용 절감',
        potentialSaving: costBreakdown.spread * 0.8,
        implementation: '시장가 대신 지정가 주문 사용, 미체결 위험 감수'
      });
    }
    
    if (costBreakdown.commission > 0) {
      suggestions.push({
        category: '브로커 변경',
        description: '무료 수수료 브로커로 전환',
        potentialSaving: costBreakdown.commission,
        implementation: 'Charles Schwab, Robinhood 등 검토'
      });
    }
    
    if (costBreakdown.marketImpact > 0) {
      suggestions.push({
        category: '주문 분할',
        description: 'VWAP/TWAP 전략으로 시장 충격 최소화',
        potentialSaving: costBreakdown.marketImpact * 0.6,
        implementation: '대량 주문을 여러 개의 작은 주문으로 분할'
      });
    }
    
    if (costBreakdown.tax > costBreakdown.totalCost * 0.4) {
      suggestions.push({
        category: '세금 최적화',
        description: '장기 보유로 세율 절감',
        potentialSaving: costBreakdown.tax * 0.5,
        implementation: '1년 이상 보유 시 장기 양도세율 적용'
      });
    }
    
    if (costBreakdown.borrowingCost > 0) {
      suggestions.push({
        category: '자금 관리',
        description: '현금 계좌 사용으로 차입 비용 제거',
        potentialSaving: costBreakdown.borrowingCost,
        implementation: '마진 사용 최소화, 자기 자본으로 거래'
      });
    }
    
    return suggestions.sort((a, b) => b.potentialSaving - a.potentialSaving);
  };

  return (
    <div className="space-y-6">
      {/* 탭 네비게이션 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-2 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex gap-2">
          <button
            onClick={() => setViewMode('calculator')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === 'calculator'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            비용 계산기
          </button>
          <button
            onClick={() => setViewMode('comparison')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === 'comparison'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            브로커 비교
          </button>
          <button
            onClick={() => setViewMode('optimization')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === 'optimization'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            비용 최적화
          </button>
        </div>
      </div>

      {viewMode === 'calculator' && (
        <>
          {/* 거래 상세 입력 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">거래 상세 정보</h3>
            
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">종목 코드</label>
                <input
                  type="text"
                  value={tradeDetails.symbol}
                  onChange={(e) => setTradeDetails({ ...tradeDetails, symbol: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">수량</label>
                <input
                  type="number"
                  value={tradeDetails.quantity}
                  onChange={(e) => setTradeDetails({ ...tradeDetails, quantity: Number(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">가격</label>
                <input
                  type="number"
                  value={tradeDetails.price}
                  onChange={(e) => setTradeDetails({ ...tradeDetails, price: Number(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                  step="0.01"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">주문 유형</label>
                <select
                  value={tradeDetails.orderType}
                  onChange={(e) => setTradeDetails({ ...tradeDetails, orderType: e.target.value as any })}
                  className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                >
                  <option value="market">시장가</option>
                  <option value="limit">지정가</option>
                  <option value="stop">스톱</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">거래 장소</label>
                <select
                  value={tradeDetails.venue}
                  onChange={(e) => setTradeDetails({ ...tradeDetails, venue: e.target.value as any })}
                  className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                >
                  <option value="exchange">거래소</option>
                  <option value="darkpool">다크풀</option>
                  <option value="otc">장외거래</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">국가</label>
                <select
                  value={tradeDetails.country}
                  onChange={(e) => setTradeDetails({ ...tradeDetails, country: e.target.value as any })}
                  className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                >
                  <option value="us">미국</option>
                  <option value="kr">한국</option>
                  <option value="jp">일본</option>
                  <option value="hk">홍콩</option>
                  <option value="eu">유럽</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">계좌 유형</label>
                <select
                  value={tradeDetails.accountType}
                  onChange={(e) => setTradeDetails({ ...tradeDetails, accountType: e.target.value as any })}
                  className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                >
                  <option value="cash">현금</option>
                  <option value="margin">마진</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">보유 기간 (일)</label>
                <input
                  type="number"
                  value={tradeDetails.holdingPeriod}
                  onChange={(e) => setTradeDetails({ ...tradeDetails, holdingPeriod: Number(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">브로커</label>
                <select
                  value={selectedBroker.name}
                  onChange={(e) => setSelectedBroker(brokerProfiles.find(b => b.name === e.target.value) || brokerProfiles[0])}
                  className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                >
                  {brokerProfiles.map(broker => (
                    <option key={broker.name} value={broker.name}>
                      {broker.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {costBreakdown && (
            <>
              {/* 거래 요약 */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
                  <p className="text-sm text-gray-600 dark:text-gray-400">거래 금액</p>
                  <p className="text-2xl font-bold">
                    ${(tradeDetails.quantity * tradeDetails.price).toLocaleString()}
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
                  <p className="text-sm text-gray-600 dark:text-gray-400">총 비용</p>
                  <p className="text-2xl font-bold text-red-600">
                    ${costBreakdown.totalCost.toFixed(2)}
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
                  <p className="text-sm text-gray-600 dark:text-gray-400">실효 가격</p>
                  <p className="text-2xl font-bold">
                    ${costBreakdown.effectivePrice.toFixed(2)}
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
                  <p className="text-sm text-gray-600 dark:text-gray-400">비용 비율</p>
                  <p className="text-2xl font-bold text-orange-600">
                    {costBreakdown.costPercentage.toFixed(3)}%
                  </p>
                </div>
              </div>

              {/* 비용 상세 분석 */}
              <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold mb-4">비용 상세 분석</h3>
                
                <div className="space-y-3">
                  {[
                    { label: '수수료', value: costBreakdown.commission, icon: DollarSign },
                    { label: '스프레드', value: costBreakdown.spread, icon: Activity },
                    { label: '슬리피지', value: costBreakdown.slippage, icon: TrendingUp },
                    { label: '시장 충격', value: costBreakdown.marketImpact, icon: Zap },
                    { label: '세금', value: costBreakdown.tax, icon: Calculator },
                    { label: '규제 수수료', value: costBreakdown.regulatoryFees, icon: Shield },
                    { label: '거래소 수수료', value: costBreakdown.exchangeFees, icon: BarChart3 },
                    { label: '청산 수수료', value: costBreakdown.clearingFees, icon: Target },
                    { label: '차입 비용', value: costBreakdown.borrowingCost, icon: TrendingUp },
                    { label: '환전 수수료', value: costBreakdown.currencyConversion, icon: DollarSign }
                  ].map((item, idx) => {
                    const Icon = item.icon;
                    const percentage = (item.value / costBreakdown.totalCost) * 100;
                    
                    if (item.value === 0) return null;
                    
                    return (
                      <div key={idx} className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <Icon className="w-5 h-5 text-gray-400" />
                          <span className="font-medium">{item.label}</span>
                        </div>
                        <div className="flex items-center gap-4">
                          <div className="w-32 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div
                              className="h-2 rounded-full bg-blue-500"
                              style={{ width: `${percentage}%` }}
                            />
                          </div>
                          <span className="text-sm font-medium w-20 text-right">
                            ${item.value.toFixed(2)}
                          </span>
                          <span className="text-sm text-gray-600 dark:text-gray-400 w-12 text-right">
                            {percentage.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          )}
        </>
      )}

      {viewMode === 'comparison' && (
        <>
          {/* 연간 거래 설정 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">연간 거래 프로필</h3>
            
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  연간 거래 횟수
                </label>
                <input
                  type="number"
                  value={annualTrades}
                  onChange={(e) => setAnnualTrades(Number(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">
                  평균 거래 규모
                </label>
                <input
                  type="number"
                  value={avgTradeSize}
                  onChange={(e) => setAvgTradeSize(Number(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                  step="1000"
                />
              </div>
            </div>
          </div>

          {/* 브로커 비교 테이블 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">브로커별 연간 비용 비교</h3>
            
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <th className="text-left p-2">브로커</th>
                    <th className="text-right p-2">수수료율</th>
                    <th className="text-right p-2">최소 수수료</th>
                    <th className="text-right p-2">스프레드 마크업</th>
                    <th className="text-right p-2">연간 예상 비용</th>
                    <th className="text-left p-2">특징</th>
                  </tr>
                </thead>
                <tbody>
                  {brokerProfiles
                    .map(broker => ({
                      ...broker,
                      annualCost: calculateAnnualCosts(broker)
                    }))
                    .sort((a, b) => a.annualCost - b.annualCost)
                    .map((broker, idx) => (
                      <tr key={broker.name} className="border-b border-gray-100 dark:border-gray-900">
                        <td className="p-2 font-medium">{broker.name}</td>
                        <td className="text-right p-2">
                          {broker.commissionRate > 0 
                            ? `${(broker.commissionRate * 100).toFixed(3)}%`
                            : '무료'}
                        </td>
                        <td className="text-right p-2">
                          ${broker.minCommission.toFixed(2)}
                        </td>
                        <td className="text-right p-2">
                          {(broker.spreadMarkup * 100).toFixed(2)}%
                        </td>
                        <td className="text-right p-2">
                          <span className={`font-bold ${
                            idx === 0 ? 'text-green-600' : 
                            idx === brokerProfiles.length - 1 ? 'text-red-600' : ''
                          }`}>
                            ${broker.annualCost.toLocaleString()}
                          </span>
                        </td>
                        <td className="p-2">
                          <div className="flex flex-wrap gap-1">
                            {broker.features.map((feature, fIdx) => (
                              <span 
                                key={fIdx}
                                className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded"
                              >
                                {feature}
                              </span>
                            ))}
                          </div>
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
            
            <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <p className="text-sm">
                💡 최저 비용 브로커 대비 최고 비용 브로커 사용 시 
                <span className="font-bold text-red-600 mx-1">
                  연간 ${(calculateAnnualCosts(brokerProfiles[3]) - calculateAnnualCosts(brokerProfiles[0])).toLocaleString()}
                </span>
                추가 비용 발생
              </p>
            </div>
          </div>
        </>
      )}

      {viewMode === 'optimization' && costBreakdown && (
        <div className="space-y-4">
          {/* 최적화 제안 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5" />
              비용 최적화 제안
            </h3>
            
            <div className="space-y-4">
              {getOptimizationSuggestions().map((suggestion, idx) => (
                <div key={idx} className="border-l-4 border-blue-500 pl-4 py-3">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="font-medium">{suggestion.category}</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        {suggestion.description}
                      </p>
                      <p className="text-sm mt-2">
                        <strong>실행 방법:</strong> {suggestion.implementation}
                      </p>
                    </div>
                    <div className="ml-4 text-right">
                      <p className="text-sm text-gray-600 dark:text-gray-400">절감 가능액</p>
                      <p className="text-lg font-bold text-green-600">
                        ${suggestion.potentialSaving.toFixed(2)}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            
            {getOptimizationSuggestions().length > 0 && (
              <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <p className="text-sm">
                  ✓ 모든 최적화 적용 시 총 
                  <span className="font-bold text-green-600 mx-1">
                    ${getOptimizationSuggestions()
                      .reduce((sum, s) => sum + s.potentialSaving, 0)
                      .toFixed(2)}
                  </span>
                  절감 가능 ({(getOptimizationSuggestions()
                    .reduce((sum, s) => sum + s.potentialSaving, 0) / costBreakdown.totalCost * 100
                  ).toFixed(1)}%)
                </p>
              </div>
            )}
          </div>

          {/* 거래 실행 체크리스트 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">거래 실행 체크리스트</h3>
            
            <div className="space-y-2">
              {[
                '거래 시간대 확인 - 개장 직후/마감 직전 피하기',
                '유동성 확인 - 일평균 거래량 대비 주문 크기',
                '스프레드 확인 - Bid-Ask 차이 0.05% 이내',
                '뉴스/이벤트 확인 - 변동성 증가 요인',
                '주문 유형 결정 - 시장가 vs 지정가',
                '거래 분할 고려 - 대량 주문 시 VWAP',
                '세금 영향 검토 - 보유 기간별 세율',
                '환율 확인 - 해외 주식 거래 시'
              ].map((item, idx) => (
                <label key={idx} className="flex items-center gap-3 p-2 hover:bg-gray-50 dark:hover:bg-gray-700 rounded">
                  <input type="checkbox" className="w-4 h-4 text-blue-600" />
                  <span className="text-sm">{item}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* 거래 비용 가이드 */}
      <div className="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Info className="w-5 h-5" />
          거래 비용 최소화 가이드
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-3">숨겨진 비용 주의</h4>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span><strong>Payment for Order Flow</strong>: 무료 브로커의 숨은 비용</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span><strong>환율 스프레드</strong>: 표시 환율과 실제 적용 환율 차이</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span><strong>기회비용</strong>: 미체결로 인한 수익 기회 상실</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span><strong>세금 drag</strong>: 빈번한 매매로 인한 단기 세율 적용</span>
              </li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-3">비용 절감 전략</h4>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>거래 빈도 최소화 - 장기 투자 전략 수립</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>적정 브로커 선택 - 거래 스타일에 맞는 브로커</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>세금 효율적 계좌 활용 - IRA, 401(k) 등</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>지정가 주문 활용 - 스프레드 비용 제어</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}