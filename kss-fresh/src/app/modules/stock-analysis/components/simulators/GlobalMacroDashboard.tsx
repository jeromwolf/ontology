'use client';

import React, { useState, useEffect } from 'react';
import { Globe, TrendingUp, TrendingDown, DollarSign, Percent, Activity, AlertTriangle, BarChart3, Calendar } from 'lucide-react';

interface MacroIndicator {
  name: string;
  value: number;
  change: number;
  unit: string;
  importance: 'high' | 'medium' | 'low';
  trend: 'up' | 'down' | 'stable';
  description: string;
}

interface CountryData {
  country: string;
  code: string;
  flag: string;
  gdpGrowth: number;
  inflation: number;
  interestRate: number;
  unemployment: number;
  currency: string;
  exchangeRate: number;
  exchangeChange: number;
  stockIndex: {
    name: string;
    value: number;
    change: number;
  };
}

interface CommodityData {
  name: string;
  symbol: string;
  price: number;
  change: number;
  unit: string;
  category: 'energy' | 'metals' | 'agriculture';
}

// 글로벌 매크로 지표
const globalIndicators: MacroIndicator[] = [
  {
    name: '미국 GDP 성장률',
    value: 2.1,
    change: 0.3,
    unit: '%',
    importance: 'high',
    trend: 'up',
    description: '전분기 대비 연율 환산'
  },
  {
    name: '미국 CPI',
    value: 3.2,
    change: -0.2,
    unit: '%',
    importance: 'high',
    trend: 'down',
    description: '전년 동월 대비'
  },
  {
    name: '연준 기준금리',
    value: 5.5,
    change: 0,
    unit: '%',
    importance: 'high',
    trend: 'stable',
    description: 'Federal Funds Rate'
  },
  {
    name: '미국 실업률',
    value: 3.9,
    change: 0.1,
    unit: '%',
    importance: 'medium',
    trend: 'up',
    description: '계절조정'
  },
  {
    name: 'DXY 달러 인덱스',
    value: 104.5,
    change: 0.8,
    unit: '',
    importance: 'high',
    trend: 'up',
    description: '주요 6개국 통화 대비'
  },
  {
    name: '10년물 미국채 금리',
    value: 4.25,
    change: -0.05,
    unit: '%',
    importance: 'high',
    trend: 'down',
    description: '장단기 금리차 주목'
  },
  {
    name: 'VIX 변동성 지수',
    value: 13.5,
    change: -1.2,
    unit: '',
    importance: 'medium',
    trend: 'down',
    description: '공포지수'
  },
  {
    name: '유로존 CPI',
    value: 2.4,
    change: -0.3,
    unit: '%',
    importance: 'medium',
    trend: 'down',
    description: '전년 동월 대비'
  }
];

// 주요국 데이터
const countryData: CountryData[] = [
  {
    country: '미국',
    code: 'US',
    flag: '🇺🇸',
    gdpGrowth: 2.1,
    inflation: 3.2,
    interestRate: 5.5,
    unemployment: 3.9,
    currency: 'USD',
    exchangeRate: 1,
    exchangeChange: 0,
    stockIndex: {
      name: 'S&P 500',
      value: 4783,
      change: 0.8
    }
  },
  {
    country: '중국',
    code: 'CN',
    flag: '🇨🇳',
    gdpGrowth: 5.2,
    inflation: 0.3,
    interestRate: 3.45,
    unemployment: 5.1,
    currency: 'CNY',
    exchangeRate: 7.24,
    exchangeChange: 0.15,
    stockIndex: {
      name: '상해종합',
      value: 3052,
      change: -0.5
    }
  },
  {
    country: '유로존',
    code: 'EU',
    flag: '🇪🇺',
    gdpGrowth: 0.5,
    inflation: 2.4,
    interestRate: 4.5,
    unemployment: 6.4,
    currency: 'EUR',
    exchangeRate: 0.92,
    exchangeChange: -0.3,
    stockIndex: {
      name: 'STOXX 600',
      value: 472,
      change: 0.3
    }
  },
  {
    country: '일본',
    code: 'JP',
    flag: '🇯🇵',
    gdpGrowth: 1.3,
    inflation: 3.1,
    interestRate: -0.1,
    unemployment: 2.5,
    currency: 'JPY',
    exchangeRate: 150.2,
    exchangeChange: 0.5,
    stockIndex: {
      name: '닛케이 225',
      value: 33445,
      change: 1.2
    }
  },
  {
    country: '한국',
    code: 'KR',
    flag: '🇰🇷',
    gdpGrowth: 1.4,
    inflation: 3.3,
    interestRate: 3.5,
    unemployment: 2.7,
    currency: 'KRW',
    exchangeRate: 1320,
    exchangeChange: 5.2,
    stockIndex: {
      name: 'KOSPI',
      value: 2505,
      change: -0.2
    }
  },
  {
    country: '영국',
    code: 'UK',
    flag: '🇬🇧',
    gdpGrowth: 0.3,
    inflation: 4.0,
    interestRate: 5.25,
    unemployment: 4.2,
    currency: 'GBP',
    exchangeRate: 0.79,
    exchangeChange: -0.2,
    stockIndex: {
      name: 'FTSE 100',
      value: 7732,
      change: 0.1
    }
  }
];

// 원자재 데이터
const commodityData: CommodityData[] = [
  { name: 'WTI 원유', symbol: 'CL', price: 78.45, change: -1.23, unit: '$/배럴', category: 'energy' },
  { name: '천연가스', symbol: 'NG', price: 2.85, change: 2.45, unit: '$/MMBtu', category: 'energy' },
  { name: '금', symbol: 'GC', price: 2052.30, change: 0.45, unit: '$/온스', category: 'metals' },
  { name: '은', symbol: 'SI', price: 23.45, change: -0.89, unit: '$/온스', category: 'metals' },
  { name: '구리', symbol: 'HG', price: 3.89, change: 1.23, unit: '$/파운드', category: 'metals' },
  { name: '밀', symbol: 'ZW', price: 615.25, change: -2.34, unit: '센트/부셸', category: 'agriculture' },
  { name: '옥수수', symbol: 'ZC', price: 475.50, change: -1.56, unit: '센트/부셸', category: 'agriculture' },
  { name: '대두', symbol: 'ZS', price: 1245.75, change: 0.89, unit: '센트/부셸', category: 'agriculture' }
];

export default function GlobalMacroDashboard() {
  const [selectedView, setSelectedView] = useState<'overview' | 'countries' | 'commodities'>('overview');
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);
  const [alertLevel, setAlertLevel] = useState<'low' | 'medium' | 'high'>('medium');
  
  // 경고 레벨 계산
  useEffect(() => {
    const highImportanceIndicators = globalIndicators.filter(i => i.importance === 'high');
    const negativeCount = highImportanceIndicators.filter(i => 
      (i.name.includes('CPI') && i.value > 3) || 
      (i.name.includes('실업률') && i.trend === 'up') ||
      (i.name.includes('VIX') && i.value > 20)
    ).length;
    
    if (negativeCount >= 3) setAlertLevel('high');
    else if (negativeCount >= 1) setAlertLevel('medium');
    else setAlertLevel('low');
  }, []);
  
  const getAlertColor = () => {
    switch (alertLevel) {
      case 'high': return 'text-red-600 bg-red-100 dark:bg-red-900';
      case 'medium': return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900';
      case 'low': return 'text-green-600 bg-green-100 dark:bg-green-900';
    }
  };
  
  const getAlertMessage = () => {
    switch (alertLevel) {
      case 'high': return '글로벌 경제 리스크 높음 - 방어적 투자 권장';
      case 'medium': return '일부 경계 신호 - 선별적 투자 필요';
      case 'low': return '양호한 투자 환경 - 적극적 투자 가능';
    }
  };
  
  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'down': return <TrendingDown className="w-4 h-4 text-red-500" />;
      default: return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* 경고 배너 */}
      <div className={`rounded-lg p-4 flex items-center gap-3 ${getAlertColor()}`}>
        <AlertTriangle className="w-5 h-5" />
        <div>
          <p className="font-medium">매크로 환경 평가: {getAlertMessage()}</p>
          <p className="text-sm opacity-80">
            주요 지표 종합 분석 결과 ({new Date().toLocaleDateString('ko-KR')} 기준)
          </p>
        </div>
      </div>

      {/* 뷰 선택 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-2 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex gap-2">
          <button
            onClick={() => setSelectedView('overview')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              selectedView === 'overview'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            주요 지표
          </button>
          <button
            onClick={() => setSelectedView('countries')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              selectedView === 'countries'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            국가별 현황
          </button>
          <button
            onClick={() => setSelectedView('commodities')}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              selectedView === 'commodities'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}
          >
            원자재
          </button>
        </div>
      </div>

      {/* 주요 지표 뷰 */}
      {selectedView === 'overview' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {globalIndicators.map((indicator) => (
            <div
              key={indicator.name}
              className={`bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border ${
                indicator.importance === 'high' 
                  ? 'border-blue-200 dark:border-blue-700' 
                  : 'border-gray-200 dark:border-gray-700'
              }`}
            >
              <div className="flex items-start justify-between mb-2">
                <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  {indicator.name}
                </h4>
                {getTrendIcon(indicator.trend)}
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold">
                  {indicator.value}{indicator.unit}
                </span>
                <span className={`text-sm font-medium ${
                  indicator.change > 0 ? 'text-green-600' : indicator.change < 0 ? 'text-red-600' : 'text-gray-600'
                }`}>
                  {indicator.change > 0 ? '+' : ''}{indicator.change}{indicator.unit}
                </span>
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                {indicator.description}
              </p>
              {indicator.importance === 'high' && (
                <span className="inline-block mt-2 px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 text-xs rounded">
                  핵심 지표
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* 국가별 현황 */}
      {selectedView === 'countries' && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {countryData.map((country) => (
              <div
                key={country.code}
                className={`bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border cursor-pointer transition-all ${
                  selectedCountry === country.code
                    ? 'border-blue-500 shadow-lg'
                    : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                }`}
                onClick={() => setSelectedCountry(country.code)}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">{country.flag}</span>
                    <h3 className="font-semibold">{country.country}</h3>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-medium">{country.stockIndex.name}</p>
                    <p className={`text-sm ${country.stockIndex.change > 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {country.stockIndex.value.toLocaleString()} ({country.stockIndex.change > 0 ? '+' : ''}{country.stockIndex.change}%)
                    </p>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="bg-gray-50 dark:bg-gray-900 rounded p-2">
                    <p className="text-gray-600 dark:text-gray-400">GDP 성장률</p>
                    <p className="font-medium">{country.gdpGrowth}%</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-900 rounded p-2">
                    <p className="text-gray-600 dark:text-gray-400">인플레이션</p>
                    <p className="font-medium">{country.inflation}%</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-900 rounded p-2">
                    <p className="text-gray-600 dark:text-gray-400">기준금리</p>
                    <p className="font-medium">{country.interestRate}%</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-900 rounded p-2">
                    <p className="text-gray-600 dark:text-gray-400">실업률</p>
                    <p className="font-medium">{country.unemployment}%</p>
                  </div>
                </div>
                
                {country.code !== 'US' && (
                  <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        1 USD = {country.exchangeRate} {country.currency}
                      </span>
                      <span className={`text-sm font-medium ${
                        country.exchangeChange > 0 ? 'text-red-600' : 'text-green-600'
                      }`}>
                        {country.exchangeChange > 0 ? '+' : ''}{country.exchangeChange}%
                      </span>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
          
          {selectedCountry && (
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
              <h4 className="font-medium mb-2">투자 시사점</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                {selectedCountry === 'US' && '미국: 금리 인하 기대감으로 주식시장 긍정적, 달러 약세 가능성 주목'}
                {selectedCountry === 'CN' && '중국: 디플레이션 우려와 부동산 리스크 지속, 정부 부양책 기대'}
                {selectedCountry === 'EU' && '유로존: 경기 둔화 우려 속 ECB 금리 인하 가능성, 유로화 약세 전망'}
                {selectedCountry === 'JP' && '일본: 금융정책 정상화 진행 중, 엔화 강세 및 주식시장 상승 기대'}
                {selectedCountry === 'KR' && '한국: 수출 회복세와 반도체 업황 개선, 원화 약세는 부담 요인'}
                {selectedCountry === 'UK' && '영국: 높은 인플레이션과 경기 침체 우려, 파운드화 변동성 확대'}
              </p>
            </div>
          )}
        </div>
      )}

      {/* 원자재 뷰 */}
      {selectedView === 'commodities' && (
        <div className="space-y-4">
          {['energy', 'metals', 'agriculture'].map((category) => (
            <div key={category} className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4 capitalize">
                {category === 'energy' && '에너지'}
                {category === 'metals' && '금속'}
                {category === 'agriculture' && '농산물'}
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {commodityData
                  .filter(c => c.category === category)
                  .map((commodity) => (
                    <div key={commodity.symbol} className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium">{commodity.name}</h4>
                        <span className="text-xs text-gray-500">{commodity.symbol}</span>
                      </div>
                      <p className="text-xl font-bold mb-1">
                        {commodity.price.toLocaleString()}
                      </p>
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-gray-500">{commodity.unit}</span>
                        <span className={`text-sm font-medium ${
                          commodity.change > 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {commodity.change > 0 ? '+' : ''}{commodity.change}%
                        </span>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* 투자 전략 제안 */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5" />
          매크로 기반 투자 전략
        </h3>
        
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-medium mb-2 text-green-600 dark:text-green-400">추천 포지션</h4>
            <ul className="text-sm space-y-1">
              <li>• 미국 기술주 (금리 인하 수혜)</li>
              <li>• 일본 주식 (엔화 강세 활용)</li>
              <li>• 금 (안전자산 선호)</li>
              <li>• 단기 채권 (금리 하락 대비)</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-medium mb-2 text-yellow-600 dark:text-yellow-400">주의 포지션</h4>
            <ul className="text-sm space-y-1">
              <li>• 중국 주식 (경제 불확실성)</li>
              <li>• 유럽 은행주 (경기 둔화)</li>
              <li>• 신흥국 통화 (달러 강세)</li>
              <li>• 장기 채권 (변동성 확대)</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-medium mb-2 text-red-600 dark:text-red-400">회피 포지션</h4>
            <ul className="text-sm space-y-1">
              <li>• 부동산 리츠 (금리 부담)</li>
              <li>• 고배당주 (성장주 선호)</li>
              <li>• 원자재 (수요 둔화)</li>
              <li>• 고위험 채권 (스프레드 확대)</li>
            </ul>
          </div>
        </div>
      </div>

      {/* 업데이트 정보 */}
      <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
        <div className="flex items-center gap-2">
          <Calendar className="w-4 h-4" />
          <span>마지막 업데이트: {new Date().toLocaleString('ko-KR')}</span>
        </div>
        <p>* 실시간 데이터는 API 연동 후 제공 예정</p>
      </div>
    </div>
  );
}