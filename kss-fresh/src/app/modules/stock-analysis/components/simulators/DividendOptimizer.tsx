'use client';

import React, { useState, useEffect } from 'react';
import { TrendingUp, Calendar, DollarSign, PieChart, BarChart3, AlertCircle, Info, Target } from 'lucide-react';

interface DividendStock {
  symbol: string;
  name: string;
  sector: string;
  price: number;
  dividendYield: number;
  dividendGrowth5yr: number;
  payoutRatio: number;
  consecutiveYears: number;
  exDividendDate: string;
  paymentFrequency: 'Monthly' | 'Quarterly' | 'Semi-Annual' | 'Annual';
  lastDividend: number;
  forwardDividend: number;
  fcfYield: number;
  debtToEquity: number;
  dividendScore: number;
}

interface PortfolioMetrics {
  totalValue: number;
  annualDividend: number;
  averageYield: number;
  monthlyIncome: number;
  yieldOnCost: number;
  growthRate: number;
  diversificationScore: number;
  safetyScore: number;
}

interface DividendCalendar {
  month: string;
  stocks: { symbol: string; amount: number; date: string }[];
  totalAmount: number;
}

// 모의 배당주 데이터
const dividendStocks: DividendStock[] = [
  {
    symbol: 'JNJ',
    name: 'Johnson & Johnson',
    sector: 'Healthcare',
    price: 155.20,
    dividendYield: 3.05,
    dividendGrowth5yr: 5.8,
    payoutRatio: 58,
    consecutiveYears: 61,
    exDividendDate: '2024-02-20',
    paymentFrequency: 'Quarterly',
    lastDividend: 1.19,
    forwardDividend: 4.76,
    fcfYield: 5.2,
    debtToEquity: 0.65,
    dividendScore: 92
  },
  {
    symbol: 'KO',
    name: 'Coca-Cola',
    sector: 'Consumer Staples',
    price: 60.15,
    dividendYield: 3.08,
    dividendGrowth5yr: 3.5,
    payoutRatio: 72,
    consecutiveYears: 61,
    exDividendDate: '2024-03-15',
    paymentFrequency: 'Quarterly',
    lastDividend: 0.46,
    forwardDividend: 1.84,
    fcfYield: 4.1,
    debtToEquity: 1.82,
    dividendScore: 88
  },
  {
    symbol: 'PG',
    name: 'Procter & Gamble',
    sector: 'Consumer Staples',
    price: 152.30,
    dividendYield: 2.42,
    dividendGrowth5yr: 5.2,
    payoutRatio: 61,
    consecutiveYears: 67,
    exDividendDate: '2024-01-19',
    paymentFrequency: 'Quarterly',
    lastDividend: 0.94,
    forwardDividend: 3.76,
    fcfYield: 4.8,
    debtToEquity: 0.71,
    dividendScore: 94
  },
  {
    symbol: 'O',
    name: 'Realty Income',
    sector: 'Real Estate',
    price: 57.85,
    dividendYield: 5.53,
    dividendGrowth5yr: 3.9,
    payoutRatio: 74,
    consecutiveYears: 29,
    exDividendDate: '2024-01-31',
    paymentFrequency: 'Monthly',
    lastDividend: 0.256,
    forwardDividend: 3.072,
    fcfYield: 5.8,
    debtToEquity: 0.85,
    dividendScore: 85
  },
  {
    symbol: 'ABBV',
    name: 'AbbVie',
    sector: 'Healthcare',
    price: 168.45,
    dividendYield: 3.54,
    dividendGrowth5yr: 18.2,
    payoutRatio: 45,
    consecutiveYears: 11,
    exDividendDate: '2024-01-15',
    paymentFrequency: 'Quarterly',
    lastDividend: 1.48,
    forwardDividend: 5.92,
    fcfYield: 8.2,
    debtToEquity: 2.15,
    dividendScore: 82
  },
  {
    symbol: 'T',
    name: 'AT&T',
    sector: 'Communication Services',
    price: 16.25,
    dividendYield: 6.82,
    dividendGrowth5yr: -4.2,
    payoutRatio: 42,
    consecutiveYears: 38,
    exDividendDate: '2024-01-10',
    paymentFrequency: 'Quarterly',
    lastDividend: 0.2775,
    forwardDividend: 1.11,
    fcfYield: 9.5,
    debtToEquity: 1.25,
    dividendScore: 68
  },
  {
    symbol: 'XOM',
    name: 'Exxon Mobil',
    sector: 'Energy',
    price: 105.20,
    dividendYield: 3.31,
    dividendGrowth5yr: 2.1,
    payoutRatio: 38,
    consecutiveYears: 41,
    exDividendDate: '2024-02-09',
    paymentFrequency: 'Quarterly',
    lastDividend: 0.91,
    forwardDividend: 3.64,
    fcfYield: 7.8,
    debtToEquity: 0.28,
    dividendScore: 86
  },
  {
    symbol: 'VZ',
    name: 'Verizon',
    sector: 'Communication Services',
    price: 39.85,
    dividendYield: 6.71,
    dividendGrowth5yr: 2.2,
    payoutRatio: 48,
    consecutiveYears: 20,
    exDividendDate: '2024-01-10',
    paymentFrequency: 'Quarterly',
    lastDividend: 0.665,
    forwardDividend: 2.66,
    fcfYield: 8.9,
    debtToEquity: 1.75,
    dividendScore: 75
  }
];

export default function DividendOptimizer() {
  const [portfolio, setPortfolio] = useState<{ stock: DividendStock; shares: number; weight: number }[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<'high-yield' | 'growth' | 'aristocrats' | 'monthly'>('high-yield');
  const [investmentAmount, setInvestmentAmount] = useState(100000);
  const [targetYield, setTargetYield] = useState(4);
  const [showCalendar, setShowCalendar] = useState(false);
  const [sortBy, setSortBy] = useState<'yield' | 'growth' | 'safety' | 'score'>('yield');
  
  // 포트폴리오 메트릭 계산
  const calculatePortfolioMetrics = (): PortfolioMetrics => {
    if (portfolio.length === 0) {
      return {
        totalValue: 0,
        annualDividend: 0,
        averageYield: 0,
        monthlyIncome: 0,
        yieldOnCost: 0,
        growthRate: 0,
        diversificationScore: 0,
        safetyScore: 0
      };
    }
    
    const totalValue = portfolio.reduce((sum, p) => sum + (p.stock.price * p.shares), 0);
    const annualDividend = portfolio.reduce((sum, p) => sum + (p.stock.forwardDividend * p.shares), 0);
    const averageYield = (annualDividend / totalValue) * 100;
    const monthlyIncome = annualDividend / 12;
    
    // 가중평균 성장률
    const growthRate = portfolio.reduce((sum, p) => 
      sum + (p.stock.dividendGrowth5yr * p.weight / 100), 0);
    
    // 섹터 다각화 점수
    const sectors = [...new Set(portfolio.map(p => p.stock.sector))];
    const diversificationScore = Math.min(sectors.length * 20, 100);
    
    // 안전성 점수 (배당 점수 가중평균)
    const safetyScore = portfolio.reduce((sum, p) => 
      sum + (p.stock.dividendScore * p.weight / 100), 0);
    
    return {
      totalValue,
      annualDividend,
      averageYield,
      monthlyIncome,
      yieldOnCost: averageYield,
      growthRate,
      diversificationScore,
      safetyScore
    };
  };
  
  // 전략별 포트폴리오 구성
  const buildPortfolio = () => {
    let selectedStocks: DividendStock[] = [];
    
    switch (selectedStrategy) {
      case 'high-yield':
        selectedStocks = [...dividendStocks]
          .filter(s => s.dividendYield >= targetYield)
          .sort((a, b) => b.dividendYield - a.dividendYield)
          .slice(0, 6);
        break;
        
      case 'growth':
        selectedStocks = [...dividendStocks]
          .filter(s => s.dividendGrowth5yr > 5)
          .sort((a, b) => b.dividendGrowth5yr - a.dividendGrowth5yr)
          .slice(0, 5);
        break;
        
      case 'aristocrats':
        selectedStocks = [...dividendStocks]
          .filter(s => s.consecutiveYears >= 25)
          .sort((a, b) => b.dividendScore - a.dividendScore)
          .slice(0, 5);
        break;
        
      case 'monthly':
        // 월배당 + 분기배당 조합
        const monthlyStocks = dividendStocks.filter(s => s.paymentFrequency === 'Monthly');
        const quarterlyStocks = dividendStocks
          .filter(s => s.paymentFrequency === 'Quarterly')
          .sort((a, b) => b.dividendScore - a.dividendScore)
          .slice(0, 4);
        selectedStocks = [...monthlyStocks, ...quarterlyStocks];
        break;
    }
    
    // 동일 가중 포트폴리오 구성
    const equalWeight = 100 / selectedStocks.length;
    const newPortfolio = selectedStocks.map(stock => {
      const investment = investmentAmount * (equalWeight / 100);
      const shares = Math.floor(investment / stock.price);
      return {
        stock,
        shares,
        weight: equalWeight
      };
    });
    
    setPortfolio(newPortfolio);
  };
  
  // 배당 캘린더 생성
  const generateDividendCalendar = (): DividendCalendar[] => {
    const calendar: DividendCalendar[] = [];
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    
    months.forEach((month, idx) => {
      const monthStocks = portfolio
        .filter(p => {
          if (p.stock.paymentFrequency === 'Monthly') return true;
          if (p.stock.paymentFrequency === 'Quarterly') {
            const exMonth = new Date(p.stock.exDividendDate).getMonth();
            return (idx - exMonth) % 3 === 0;
          }
          return false;
        })
        .map(p => ({
          symbol: p.stock.symbol,
          amount: p.stock.paymentFrequency === 'Monthly' 
            ? p.stock.lastDividend * p.shares
            : p.stock.lastDividend * p.shares,
          date: `2024-${String(idx + 1).padStart(2, '0')}-15`
        }));
      
      calendar.push({
        month,
        stocks: monthStocks,
        totalAmount: monthStocks.reduce((sum, s) => sum + s.amount, 0)
      });
    });
    
    return calendar;
  };
  
  useEffect(() => {
    buildPortfolio();
  }, [selectedStrategy, investmentAmount, targetYield]);
  
  const metrics = calculatePortfolioMetrics();
  const calendar = generateDividendCalendar();
  
  // 정렬된 주식 목록
  const sortedStocks = [...dividendStocks].sort((a, b) => {
    switch (sortBy) {
      case 'yield': return b.dividendYield - a.dividendYield;
      case 'growth': return b.dividendGrowth5yr - a.dividendGrowth5yr;
      case 'safety': return b.consecutiveYears - a.consecutiveYears;
      case 'score': return b.dividendScore - a.dividendScore;
      default: return 0;
    }
  });

  return (
    <div className="space-y-6">
      {/* 전략 선택 및 설정 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4">배당 전략 선택</h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <div className="grid grid-cols-2 gap-2 mb-4">
              {[
                { id: 'high-yield', name: '고배당주', desc: '높은 현재 수익률' },
                { id: 'growth', name: '배당성장주', desc: '높은 성장률' },
                { id: 'aristocrats', name: '배당귀족주', desc: '25년+ 연속 증배' },
                { id: 'monthly', name: '월배당 포트폴리오', desc: '매월 배당 수령' }
              ].map((strategy) => (
                <button
                  key={strategy.id}
                  onClick={() => setSelectedStrategy(strategy.id as any)}
                  className={`p-3 rounded-lg border text-left transition-colors ${
                    selectedStrategy === strategy.id
                      ? 'bg-blue-500 text-white border-blue-500'
                      : 'bg-gray-50 dark:bg-gray-700 border-gray-200 dark:border-gray-600'
                  }`}
                >
                  <div className="font-medium">{strategy.name}</div>
                  <div className="text-xs opacity-75">{strategy.desc}</div>
                </button>
              ))}
            </div>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">투자 금액</label>
              <input
                type="number"
                value={investmentAmount}
                onChange={(e) => setInvestmentAmount(Number(e.target.value))}
                className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
                step="10000"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">
                목표 수익률: {targetYield}%
              </label>
              <input
                type="range"
                min="2"
                max="8"
                step="0.5"
                value={targetYield}
                onChange={(e) => setTargetYield(Number(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </div>
      </div>

      {/* 포트폴리오 메트릭 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-600 dark:text-gray-400">포트폴리오 가치</p>
          <p className="text-2xl font-bold">${metrics.totalValue.toLocaleString()}</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-600 dark:text-gray-400">연간 배당금</p>
          <p className="text-2xl font-bold text-green-600">
            ${metrics.annualDividend.toLocaleString()}
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-600 dark:text-gray-400">평균 수익률</p>
          <p className="text-2xl font-bold">{metrics.averageYield.toFixed(2)}%</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-600 dark:text-gray-400">월 예상 수입</p>
          <p className="text-2xl font-bold text-blue-600">
            ${metrics.monthlyIncome.toFixed(0)}
          </p>
        </div>
      </div>

      {/* 추가 메트릭 */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-4 h-4 text-green-500" />
            <p className="text-sm text-gray-600 dark:text-gray-400">배당 성장률</p>
          </div>
          <p className="text-xl font-bold">{metrics.growthRate.toFixed(1)}%</p>
          <p className="text-xs text-gray-500">5년 평균</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2 mb-2">
            <PieChart className="w-4 h-4 text-purple-500" />
            <p className="text-sm text-gray-600 dark:text-gray-400">다각화 점수</p>
          </div>
          <p className="text-xl font-bold">{metrics.diversificationScore}/100</p>
          <p className="text-xs text-gray-500">섹터 분산도</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2 mb-2">
            <Target className="w-4 h-4 text-blue-500" />
            <p className="text-sm text-gray-600 dark:text-gray-400">안전성 점수</p>
          </div>
          <p className="text-xl font-bold">{metrics.safetyScore.toFixed(0)}/100</p>
          <p className="text-xs text-gray-500">배당 지속가능성</p>
        </div>
      </div>

      {/* 포트폴리오 구성 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">포트폴리오 구성</h3>
          <button
            onClick={() => setShowCalendar(!showCalendar)}
            className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
          >
            {showCalendar ? '포트폴리오 보기' : '배당 캘린더 보기'}
          </button>
        </div>
        
        {!showCalendar ? (
          <div className="space-y-3">
            {portfolio.map((item) => (
              <div key={item.stock.symbol} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    <span className="font-semibold">{item.stock.symbol}</span>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {item.stock.name}
                    </span>
                    <span className="text-xs px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded">
                      {item.stock.sector}
                    </span>
                  </div>
                  <div className="flex items-center gap-4 mt-1 text-sm">
                    <span>주식수: {item.shares}</span>
                    <span>비중: {item.weight.toFixed(1)}%</span>
                    <span className="text-green-600">
                      연 배당: ${(item.stock.forwardDividend * item.shares).toFixed(0)}
                    </span>
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-semibold">{item.stock.dividendYield.toFixed(2)}%</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    ${item.stock.price.toFixed(2)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-3 md:grid-cols-4 gap-3">
            {calendar.map((month) => (
              <div key={month.month} className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
                <div className="font-medium mb-2">{month.month}</div>
                <div className="text-lg font-bold text-green-600">
                  ${month.totalAmount.toFixed(0)}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                  {month.stocks.length} 종목
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 전체 배당주 리스트 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">배당주 스크리너</h3>
          <div className="flex gap-2">
            {['yield', 'growth', 'safety', 'score'].map((sort) => (
              <button
                key={sort}
                onClick={() => setSortBy(sort as any)}
                className={`px-3 py-1 text-sm rounded-lg transition-colors ${
                  sortBy === sort
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-700'
                }`}
              >
                {sort === 'yield' && '수익률'}
                {sort === 'growth' && '성장률'}
                {sort === 'safety' && '안전성'}
                {sort === 'score' && '종합점수'}
              </button>
            ))}
          </div>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="text-left p-2">종목</th>
                <th className="text-right p-2">주가</th>
                <th className="text-right p-2">수익률</th>
                <th className="text-right p-2">5년 성장률</th>
                <th className="text-right p-2">연속 증배</th>
                <th className="text-right p-2">배당성향</th>
                <th className="text-right p-2">점수</th>
              </tr>
            </thead>
            <tbody>
              {sortedStocks.map((stock) => (
                <tr key={stock.symbol} className="border-b border-gray-100 dark:border-gray-900">
                  <td className="p-2">
                    <div>
                      <span className="font-medium">{stock.symbol}</span>
                      <span className="text-gray-600 dark:text-gray-400 ml-2 text-xs">
                        {stock.name}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500">{stock.sector}</div>
                  </td>
                  <td className="text-right p-2">${stock.price.toFixed(2)}</td>
                  <td className="text-right p-2 font-medium text-green-600">
                    {stock.dividendYield.toFixed(2)}%
                  </td>
                  <td className="text-right p-2">
                    <span className={stock.dividendGrowth5yr > 0 ? 'text-green-600' : 'text-red-600'}>
                      {stock.dividendGrowth5yr > 0 ? '+' : ''}{stock.dividendGrowth5yr.toFixed(1)}%
                    </span>
                  </td>
                  <td className="text-right p-2">{stock.consecutiveYears}년</td>
                  <td className="text-right p-2">{stock.payoutRatio}%</td>
                  <td className="text-right p-2">
                    <span className={`font-medium ${
                      stock.dividendScore >= 90 ? 'text-green-600' :
                      stock.dividendScore >= 80 ? 'text-blue-600' :
                      stock.dividendScore >= 70 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {stock.dividendScore}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 배당 투자 가이드 */}
      <div className="bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-900/20 dark:to-green-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Info className="w-5 h-5" />
          배당 투자 전략 가이드
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-3">배당주 선택 기준</h4>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>배당성향 60% 이하 - 지속가능한 배당</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>FCF 수익률 > 배당 수익률 - 현금창출력</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>5년 이상 연속 증배 - 안정적 성장</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>부채비율 1.0 이하 - 재무 건전성</span>
              </li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-3">주의사항</h4>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>과도한 고배당(8%+)은 배당 삭감 위험</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>단일 섹터 집중은 위험 - 최소 5개 섹터 분산</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>배당락일 전후 단기 매매 금지</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>세금 효과 고려 - 해외주식 이중과세</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}