'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ArrowLeft, Globe, TrendingUp, TrendingDown, BarChart3, Calendar, DollarSign, Factory, Building2, Users, Zap, Activity, Target, AlertTriangle, Clock, ChevronRight, RefreshCw, Award } from 'lucide-react';
import ChapterNavigation from '../../components/ChapterNavigation';

interface MacroData {
  gdp: number;
  inflation: number;
  unemployment: number;
  interestRate: number;
  exchange: number;
}

interface EconomicScenario {
  id: string;
  title: string;
  description: string;
  data: MacroData;
  implications: {
    stocks: 'bullish' | 'bearish' | 'neutral';
    bonds: 'bullish' | 'bearish' | 'neutral';
    currency: 'strong' | 'weak' | 'neutral';
    sectors: { [key: string]: 'positive' | 'negative' | 'neutral' };
  };
  investmentStrategy: string[];
  riskFactors: string[];
}

function MacroEconomicAnalysis() {
  const [selectedScenario, setSelectedScenario] = useState<string>('recovery');
  const [userAnalysis, setUserAnalysis] = useState({
    stockOutlook: '',
    bondOutlook: '',
    currencyOutlook: '',
    recommendedSectors: [] as string[],
    strategy: '',
    risks: ''
  });
  const [showExpertAnalysis, setShowExpertAnalysis] = useState(false);
  
  const scenarios: { [key: string]: EconomicScenario } = {
    recovery: {
      id: 'recovery',
      title: '🌱 경기 회복 국면',
      description: '코로나19 이후 경기가 회복세를 보이며, GDP 성장률이 상승하고 있습니다. 그러나 인플레이션도 함께 상승하고 있어 중앙은행의 통화정책 변화가 예상됩니다.',
      data: {
        gdp: 3.2,
        inflation: 4.1,
        unemployment: 3.5,
        interestRate: 2.5,
        exchange: 1320
      },
      implications: {
        stocks: 'bullish',
        bonds: 'bearish',
        currency: 'strong',
        sectors: {
          '금융': 'positive',
          '소비재': 'positive',
          '기술': 'positive',
          '부동산': 'neutral',
          '유틸리티': 'negative'
        }
      },
      investmentStrategy: [
        '성장주 및 경기민감주 비중 확대',
        '금융주 투자 기회 포착',
        '장기 채권 비중 축소',
        '소비관련주 선별적 투자'
      ],
      riskFactors: [
        '인플레이션 가속화 위험',
        '급격한 금리 인상 가능성',
        '공급망 불안정 지속',
        '부동산 버블 우려'
      ]
    },
    stagflation: {
      id: 'stagflation',
      title: '⚠️ 스태그플레이션 우려',
      description: '경제 성장은 둔화되고 있지만 인플레이션은 높은 수준을 유지하고 있습니다. 중앙은행은 딜레마에 빠져 있으며, 투자자들은 불확실성에 직면해 있습니다.',
      data: {
        gdp: 1.1,
        inflation: 6.8,
        unemployment: 5.2,
        interestRate: 4.0,
        exchange: 1380
      },
      implications: {
        stocks: 'bearish',
        bonds: 'bearish',
        currency: 'weak',
        sectors: {
          '에너지': 'positive',
          '원자재': 'positive',
          '금융': 'neutral',
          '기술': 'negative',
          '소비재': 'negative'
        }
      },
      investmentStrategy: [
        '실물자산(부동산, 원자재) 비중 확대',
        '인플레이션 헤지 자산 투자',
        '현금 비중 축소',
        '배당주 중심 포트폴리오'
      ],
      riskFactors: [
        '경기침체 장기화 위험',
        '실질 소득 감소',
        '기업 수익성 악화',
        '사회적 불안 증가'
      ]
    },
    deflation: {
      id: 'deflation',
      title: '❄️ 디플레이션 압력',
      description: '경기 침체가 지속되며 물가가 하락하고 있습니다. 중앙은행은 제로 금리 정책을 유지하고 있으며, 양적완화를 검토하고 있습니다.',
      data: {
        gdp: -1.8,
        inflation: -0.5,
        unemployment: 8.1,
        interestRate: 0.25,
        exchange: 1420
      },
      implications: {
        stocks: 'bearish',
        bonds: 'bullish',
        currency: 'weak',
        sectors: {
          '유틸리티': 'positive',
          '생필품': 'positive',
          '부동산': 'negative',
          '기술': 'negative',
          '산업재': 'negative'
        }
      },
      investmentStrategy: [
        '장기 국채 투자 확대',
        '방어주 중심 포트폴리오',
        '현금 및 현금성 자산 비중 확대',
        '고배당주 선별적 투자'
      ],
      riskFactors: [
        '디플레이션 스파이럴 위험',
        '기업 부채 부담 증가',
        '소비 위축 지속',
        '자산 가치 하락'
      ]
    },
    overheating: {
      id: 'overheating',
      title: '🔥 경기 과열',
      description: '경제가 급속히 성장하며 완전고용에 근접했습니다. 인플레이션 압력이 높아지고 있어 중앙은행의 긴축 정책이 예상됩니다.',
      data: {
        gdp: 5.8,
        inflation: 7.2,
        unemployment: 2.1,
        interestRate: 1.5,
        exchange: 1280
      },
      implications: {
        stocks: 'neutral',
        bonds: 'bearish',
        currency: 'strong',
        sectors: {
          '금융': 'positive',
          '원자재': 'positive',
          '기술': 'negative',
          '부동산': 'negative',
          '소비재': 'neutral'
        }
      },
      investmentStrategy: [
        '금리 상승 수혜주 투자',
        '단기 채권 중심 운용',
        '인플레이션 헤지 자산 확대',
        '성장주 비중 축소'
      ],
      riskFactors: [
        '급격한 금리 인상 위험',
        '자산 버블 형성',
        '경기 하강 전환점',
        '소득 불평등 심화'
      ]
    }
  };

  const sectors = ['금융', '기술', '소비재', '에너지', '부동산', '유틸리티', '원자재', '생필품', '산업재'];

  const handleSectorToggle = (sector: string) => {
    setUserAnalysis(prev => ({
      ...prev,
      recommendedSectors: prev.recommendedSectors.includes(sector)
        ? prev.recommendedSectors.filter(s => s !== sector)
        : [...prev.recommendedSectors, sector]
    }));
  };

  const getIndicatorColor = (value: number, type: string) => {
    switch (type) {
      case 'gdp':
        return value > 3 ? 'text-green-600' : value > 0 ? 'text-yellow-600' : 'text-red-600';
      case 'inflation':
        return value > 5 ? 'text-red-600' : value > 2 ? 'text-yellow-600' : value > 0 ? 'text-green-600' : 'text-blue-600';
      case 'unemployment':
        return value < 3 ? 'text-green-600' : value < 6 ? 'text-yellow-600' : 'text-red-600';
      case 'interest':
        return value > 4 ? 'text-red-600' : value > 2 ? 'text-yellow-600' : 'text-green-600';
      case 'exchange':
        return value < 1300 ? 'text-green-600' : value < 1400 ? 'text-yellow-600' : 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish':
      case 'positive':
      case 'strong':
        return '📈';
      case 'bearish':
      case 'negative':
      case 'weak':
        return '📉';
      default:
        return '➖';
    }
  };

  const scenario = scenarios[selectedScenario];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-blue-900">
      {/* Header */}
      <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-b border-gray-200 dark:border-gray-700 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link
                href="/modules/stock-analysis"
                className="inline-flex items-center text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 transition-colors"
              >
                <ArrowLeft size={20} className="mr-2" />
                주식 분석
              </Link>
              <div className="h-6 w-px bg-gray-300 dark:bg-gray-600" />
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                거시경제 분석 실습
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Foundation Program 9/9 단계 • 완료 예정
              </div>
              <div className="w-32 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div className="bg-gradient-to-r from-blue-500 to-indigo-600 h-2 rounded-full" style={{ width: '100%' }}></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* Introduction */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-8 mb-8 shadow-lg">
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full mb-4">
              <Globe className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
              거시경제 분석 실습
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              실제 거시경제 상황을 분석하고 투자 전략을 수립해보세요. 
              경제 지표를 종합적으로 해석하고, 시장 상황에 맞는 포트폴리오 구성 능력을 기릅니다.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg">
              <BarChart3 className="w-12 h-12 text-green-600 dark:text-green-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">경제 지표 해석</h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                GDP, 인플레이션, 실업률 등 핵심 지표 분석
              </p>
            </div>
            <div className="text-center p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg">
              <Target className="w-12 h-12 text-blue-600 dark:text-blue-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">투자 전략 수립</h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                거시 환경에 따른 자산배분 및 섹터 선택
              </p>
            </div>
            <div className="text-center p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg">
              <AlertTriangle className="w-12 h-12 text-purple-600 dark:text-purple-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">리스크 관리</h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                경제 상황별 투자 위험 요소 파악
              </p>
            </div>
          </div>
        </div>

        {/* Scenario Selection */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-8 mb-8 shadow-lg">
          <h2 className="text-2xl font-bold mb-6 flex items-center">
            <Calendar className="mr-3 text-blue-600" />
            경제 시나리오 선택
          </h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            {Object.values(scenarios).map((s) => (
              <button
                key={s.id}
                onClick={() => {
                  setSelectedScenario(s.id);
                  setShowExpertAnalysis(false);
                  setUserAnalysis({
                    stockOutlook: '',
                    bondOutlook: '',
                    currencyOutlook: '',
                    recommendedSectors: [],
                    strategy: '',
                    risks: ''
                  });
                }}
                className={`p-4 rounded-lg text-left transition-all ${
                  selectedScenario === s.id
                    ? 'bg-gradient-to-br from-blue-500 to-indigo-600 text-white shadow-lg'
                    : 'bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600'
                }`}
              >
                <h3 className="font-semibold mb-2">{s.title}</h3>
                <p className={`text-sm ${selectedScenario === s.id ? 'text-blue-100' : 'text-gray-600 dark:text-gray-300'}`}>
                  {s.description.substring(0, 80)}...
                </p>
              </button>
            ))}
          </div>

          {/* Economic Indicators */}
          <div className="bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-700 dark:to-blue-900/20 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <BarChart3 className="mr-2 text-blue-600" />
              {scenario.title} - 경제 지표 현황
            </h3>
            
            <div className="grid md:grid-cols-5 gap-4 mb-6">
              <div className="text-center">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <TrendingUp className="w-6 h-6 mx-auto mb-2 text-green-600" />
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">GDP 성장률</div>
                  <div className={`text-xl font-bold ${getIndicatorColor(scenario.data.gdp, 'gdp')}`}>
                    {scenario.data.gdp > 0 ? '+' : ''}{scenario.data.gdp}%
                  </div>
                </div>
              </div>
              
              <div className="text-center">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <Activity className="w-6 h-6 mx-auto mb-2 text-red-600" />
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">인플레이션</div>
                  <div className={`text-xl font-bold ${getIndicatorColor(scenario.data.inflation, 'inflation')}`}>
                    {scenario.data.inflation > 0 ? '+' : ''}{scenario.data.inflation}%
                  </div>
                </div>
              </div>
              
              <div className="text-center">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <Users className="w-6 h-6 mx-auto mb-2 text-orange-600" />
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">실업률</div>
                  <div className={`text-xl font-bold ${getIndicatorColor(scenario.data.unemployment, 'unemployment')}`}>
                    {scenario.data.unemployment}%
                  </div>
                </div>
              </div>
              
              <div className="text-center">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <DollarSign className="w-6 h-6 mx-auto mb-2 text-blue-600" />
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">기준금리</div>
                  <div className={`text-xl font-bold ${getIndicatorColor(scenario.data.interestRate, 'interest')}`}>
                    {scenario.data.interestRate}%
                  </div>
                </div>
              </div>
              
              <div className="text-center">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <Globe className="w-6 h-6 mx-auto mb-2 text-purple-600" />
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">USD/KRW</div>
                  <div className={`text-xl font-bold ${getIndicatorColor(scenario.data.exchange, 'exchange')}`}>
                    {scenario.data.exchange.toLocaleString()}
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
              <p className="text-sm text-gray-700 dark:text-gray-300">
                <strong>상황 설명:</strong> {scenario.description}
              </p>
            </div>
          </div>
        </div>

        {/* User Analysis Section */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-8 mb-8 shadow-lg">
          <h2 className="text-2xl font-bold mb-6 flex items-center">
            <Target className="mr-3 text-green-600" />
            나의 분석 및 전략 수립
          </h2>
          
          <div className="grid lg:grid-cols-2 gap-8">
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium mb-2">주식시장 전망</label>
                <select
                  value={userAnalysis.stockOutlook}
                  onChange={(e) => setUserAnalysis(prev => ({ ...prev, stockOutlook: e.target.value }))}
                  className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700"
                >
                  <option value="">전망을 선택하세요</option>
                  <option value="bullish">📈 강세 (상승 전망)</option>
                  <option value="bearish">📉 약세 (하락 전망)</option>
                  <option value="neutral">➖ 중립 (횡보 전망)</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">채권시장 전망</label>
                <select
                  value={userAnalysis.bondOutlook}
                  onChange={(e) => setUserAnalysis(prev => ({ ...prev, bondOutlook: e.target.value }))}
                  className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700"
                >
                  <option value="">전망을 선택하세요</option>
                  <option value="bullish">📈 강세 (채권가격 상승)</option>
                  <option value="bearish">📉 약세 (채권가격 하락)</option>
                  <option value="neutral">➖ 중립</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">원화 전망</label>
                <select
                  value={userAnalysis.currencyOutlook}
                  onChange={(e) => setUserAnalysis(prev => ({ ...prev, currencyOutlook: e.target.value }))}
                  className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700"
                >
                  <option value="">전망을 선택하세요</option>
                  <option value="strong">💪 원화 강세 (환율 하락)</option>
                  <option value="weak">📉 원화 약세 (환율 상승)</option>
                  <option value="neutral">➖ 현 수준 유지</option>
                </select>
              </div>
            </div>
            
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium mb-3">추천 투자 섹터 (복수 선택 가능)</label>
                <div className="grid grid-cols-3 gap-2">
                  {sectors.map(sector => (
                    <button
                      key={sector}
                      onClick={() => handleSectorToggle(sector)}
                      className={`p-2 rounded-lg text-sm font-medium transition-all ${
                        userAnalysis.recommendedSectors.includes(sector)
                          ? 'bg-green-500 text-white'
                          : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                      }`}
                    >
                      {sector}
                    </button>
                  ))}
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">투자 전략 요약</label>
                <textarea
                  value={userAnalysis.strategy}
                  onChange={(e) => setUserAnalysis(prev => ({ ...prev, strategy: e.target.value }))}
                  rows={3}
                  className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700"
                  placeholder="이 상황에서 어떤 투자 전략을 취하시겠습니까?"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">주요 리스크 요인</label>
                <textarea
                  value={userAnalysis.risks}
                  onChange={(e) => setUserAnalysis(prev => ({ ...prev, risks: e.target.value }))}
                  rows={3}
                  className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700"
                  placeholder="어떤 리스크를 주의해야 할까요?"
                />
              </div>
            </div>
          </div>
          
          <div className="mt-6 text-center">
            <button
              onClick={() => setShowExpertAnalysis(true)}
              disabled={!userAnalysis.stockOutlook || !userAnalysis.bondOutlook || !userAnalysis.currencyOutlook}
              className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white px-8 py-3 rounded-lg font-semibold hover:from-blue-600 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              전문가 분석과 비교하기
            </button>
          </div>
        </div>

        {/* Expert Analysis Comparison */}
        {showExpertAnalysis && (
          <div className="bg-white dark:bg-gray-800 rounded-xl p-8 mb-8 shadow-lg">
            <h2 className="text-2xl font-bold mb-6 flex items-center">
              <Award className="mr-3 text-yellow-600" />
              전문가 분석 비교
            </h2>
            
            <div className="grid lg:grid-cols-2 gap-8">
              {/* User Analysis */}
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                <h3 className="text-lg font-semibold mb-4 text-blue-800 dark:text-blue-300">나의 분석</h3>
                
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span>주식시장:</span>
                    <span className="font-semibold">
                      {userAnalysis.stockOutlook === 'bullish' ? '📈 강세' : 
                       userAnalysis.stockOutlook === 'bearish' ? '📉 약세' : '➖ 중립'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>채권시장:</span>
                    <span className="font-semibold">
                      {userAnalysis.bondOutlook === 'bullish' ? '📈 강세' : 
                       userAnalysis.bondOutlook === 'bearish' ? '📉 약세' : '➖ 중립'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>원화:</span>
                    <span className="font-semibold">
                      {userAnalysis.currencyOutlook === 'strong' ? '💪 강세' : 
                       userAnalysis.currencyOutlook === 'weak' ? '📉 약세' : '➖ 중립'}
                    </span>
                  </div>
                  <div className="pt-2">
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">추천 섹터:</div>
                    <div className="flex flex-wrap gap-1">
                      {userAnalysis.recommendedSectors.map(sector => (
                        <span key={sector} className="bg-blue-200 dark:bg-blue-800 text-blue-800 dark:text-blue-200 px-2 py-1 rounded text-xs">
                          {sector}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Expert Analysis */}
              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
                <h3 className="text-lg font-semibold mb-4 text-green-800 dark:text-green-300">전문가 분석</h3>
                
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span>주식시장:</span>
                    <span className="font-semibold">
                      {getSentimentIcon(scenario.implications.stocks)} {
                        scenario.implications.stocks === 'bullish' ? '강세' :
                        scenario.implications.stocks === 'bearish' ? '약세' : '중립'
                      }
                      {userAnalysis.stockOutlook === scenario.implications.stocks && 
                        <span className="text-green-600 ml-2">✓</span>
                      }
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>채권시장:</span>
                    <span className="font-semibold">
                      {getSentimentIcon(scenario.implications.bonds)} {
                        scenario.implications.bonds === 'bullish' ? '강세' :
                        scenario.implications.bonds === 'bearish' ? '약세' : '중립'
                      }
                      {userAnalysis.bondOutlook === scenario.implications.bonds && 
                        <span className="text-green-600 ml-2">✓</span>
                      }
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>원화:</span>
                    <span className="font-semibold">
                      {getSentimentIcon(scenario.implications.currency)} {
                        scenario.implications.currency === 'strong' ? '강세' :
                        scenario.implications.currency === 'weak' ? '약세' : '중립'
                      }
                      {userAnalysis.currencyOutlook === scenario.implications.currency && 
                        <span className="text-green-600 ml-2">✓</span>
                      }
                    </span>
                  </div>
                  <div className="pt-2">
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">유망 섹터:</div>
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(scenario.implications.sectors)
                        .filter(([_, sentiment]) => sentiment === 'positive')
                        .map(([sector, _]) => (
                          <span 
                            key={sector} 
                            className={`px-2 py-1 rounded text-xs ${
                              userAnalysis.recommendedSectors.includes(sector)
                                ? 'bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200'
                                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                            }`}
                          >
                            {sector} {userAnalysis.recommendedSectors.includes(sector) && '✓'}
                          </span>
                        ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Expert Strategy & Risks */}
            <div className="mt-8 grid lg:grid-cols-2 gap-8">
              <div className="bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 rounded-lg p-6">
                <h4 className="font-semibold mb-3 text-indigo-800 dark:text-indigo-300">전문가 추천 전략</h4>
                <ul className="space-y-2">
                  {scenario.investmentStrategy.map((strategy, index) => (
                    <li key={index} className="flex items-start">
                      <ChevronRight size={16} className="text-indigo-600 mt-0.5 mr-2 flex-shrink-0" />
                      <span className="text-sm text-gray-700 dark:text-gray-300">{strategy}</span>
                    </li>
                  ))}
                </ul>
              </div>
              
              <div className="bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-lg p-6">
                <h4 className="font-semibold mb-3 text-red-800 dark:text-red-300">주요 리스크 요인</h4>
                <ul className="space-y-2">
                  {scenario.riskFactors.map((risk, index) => (
                    <li key={index} className="flex items-start">
                      <AlertTriangle size={16} className="text-red-600 mt-0.5 mr-2 flex-shrink-0" />
                      <span className="text-sm text-gray-700 dark:text-gray-300">{risk}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
            
            {/* Performance Score */}
            <div className="mt-6 text-center">
              <div className="bg-gradient-to-r from-yellow-400 to-orange-500 text-white rounded-lg p-4 inline-block">
                <h4 className="font-semibold mb-2">분석 정확도</h4>
                <div className="text-2xl font-bold">
                  {(() => {
                    let score = 0;
                    if (userAnalysis.stockOutlook === scenario.implications.stocks) score += 30;
                    if (userAnalysis.bondOutlook === scenario.implications.bonds) score += 30;
                    if (userAnalysis.currencyOutlook === scenario.implications.currency) score += 20;
                    
                    const correctSectors = userAnalysis.recommendedSectors.filter(sector => 
                      scenario.implications.sectors[sector] === 'positive'
                    ).length;
                    const totalCorrectSectors = Object.values(scenario.implications.sectors).filter(s => s === 'positive').length;
                    score += Math.round((correctSectors / totalCorrectSectors) * 20);
                    
                    return Math.min(score, 100);
                  })()}점
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Summary and Next Steps */}
        <div className="bg-gradient-to-br from-green-50 to-emerald-100 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold mb-6 flex items-center text-green-800 dark:text-green-300">
            <Award className="mr-3" />
            Foundation Program 단계 완료! 🎉
          </h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-semibold mb-4">학습 완료 내용</h3>
              <ul className="space-y-2">
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                  <span className="text-sm">시장 구조와 참가자 이해</span>
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                  <span className="text-sm">거래 시스템과 주문 방식</span>
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                  <span className="text-sm">투자 심리와 행동 편향</span>
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                  <span className="text-sm">리스크 관리와 포지션 사이징</span>
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                  <span className="text-sm">경제 지표와 통화정책 분석</span>
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                  <span className="text-sm">거시경제 분석 실습</span>
                </li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold mb-4">다음 단계: Young Eagle</h3>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
                <h4 className="font-semibold mb-3 text-blue-600 dark:text-blue-400">기술적 분석 마스터</h4>
                <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                  <li>• 차트 패턴 인식과 해석</li>
                  <li>• 기술적 지표 활용법</li>
                  <li>• 트레이딩 전략 개발</li>
                  <li>• 백테스팅과 성과 평가</li>
                </ul>
                <div className="mt-4">
                  <Link
                    href="/modules/stock-analysis/young-eagle"
                    className="inline-flex items-center text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 font-medium"
                  >
                    Young Eagle 시작하기
                    <ChevronRight size={16} className="ml-1" />
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Quiz Component
function MacroAnalysisQuiz() {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswers, setSelectedAnswers] = useState<string[]>([]);
  const [showResults, setShowResults] = useState(false);

  const questions = [
    {
      question: "경기 회복 국면에서 가장 적절한 투자 전략은?",
      options: [
        "현금 비중을 늘리고 방어적 자산에 투자",
        "성장주와 경기민감주 비중을 확대",
        "장기 채권 비중을 늘리고 주식 비중 축소",
        "금과 같은 안전자산에만 투자"
      ],
      correct: 1,
      explanation: "경기 회복 국면에서는 기업 실적 개선이 예상되므로 성장주와 경기민감주에 투자하는 것이 유리합니다."
    },
    {
      question: "인플레이션이 상승하고 있을 때 가장 부정적인 영향을 받는 자산은?",
      options: [
        "부동산",
        "원자재",
        "장기 채권",
        "에너지주"
      ],
      correct: 2,
      explanation: "인플레이션 상승 시 금리 인상 압력이 높아져 장기 채권 가격이 가장 큰 타격을 받습니다."
    },
    {
      question: "스태그플레이션 상황에서 투자자가 고려해야 할 주요 전략은?",
      options: [
        "성장주 집중 투자",
        "실물자산과 인플레이션 헤지 자산 투자",
        "기술주 비중 확대",
        "장기 채권 투자 확대"
      ],
      correct: 1,
      explanation: "스태그플레이션 시에는 부동산, 원자재 등 실물자산과 인플레이션에 대한 헤지가 가능한 자산에 투자하는 것이 중요합니다."
    }
  ];

  const handleAnswerSelect = (answerIndex: number) => {
    const newAnswers = [...selectedAnswers];
    newAnswers[currentQuestion] = answerIndex.toString();
    setSelectedAnswers(newAnswers);
  };

  const handleNextQuestion = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
    } else {
      setShowResults(true);
    }
  };

  const calculateScore = () => {
    return selectedAnswers.reduce((score, answer, index) => {
      return score + (parseInt(answer) === questions[index].correct ? 1 : 0);
    }, 0);
  };

  if (showResults) {
    const score = calculateScore();
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-center">퀴즈 결과</h2>
        <div className="text-center mb-8">
          <div className="text-4xl font-bold text-blue-600 dark:text-blue-400 mb-2">
            {score}/{questions.length}
          </div>
          <p className="text-gray-600 dark:text-gray-300">
            {score === questions.length ? "완벽합니다! 🎉" :
             score >= questions.length * 0.7 ? "잘했습니다! 👏" : "더 공부가 필요해요 📚"}
          </p>
        </div>
        
        <div className="space-y-6">
          {questions.map((q, index) => (
            <div key={index} className="border-l-4 border-blue-500 pl-4">
              <h3 className="font-semibold mb-2">{q.question}</h3>
              <p className={`text-sm mb-2 ${
                parseInt(selectedAnswers[index]) === q.correct ? 'text-green-600' : 'text-red-600'
              }`}>
                선택한 답: {q.options[parseInt(selectedAnswers[index])]}
              </p>
              {parseInt(selectedAnswers[index]) !== q.correct && (
                <p className="text-sm text-green-600 mb-2">
                  정답: {q.options[q.correct]}
                </p>
              )}
              <p className="text-sm text-gray-600 dark:text-gray-400">{q.explanation}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  const currentQ = questions[currentQuestion];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">거시경제 분석 퀴즈</h2>
          <span className="text-sm text-gray-600 dark:text-gray-400">
            {currentQuestion + 1} / {questions.length}
          </span>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <div 
            className="bg-blue-500 h-2 rounded-full transition-all duration-300"
            style={{ width: `${((currentQuestion + 1) / questions.length) * 100}%` }}
          ></div>
        </div>
      </div>

      <div className="mb-8">
        <h3 className="text-lg font-semibold mb-6">{currentQ.question}</h3>
        <div className="space-y-3">
          {currentQ.options.map((option, index) => (
            <button
              key={index}
              onClick={() => handleAnswerSelect(index)}
              className={`w-full p-4 text-left rounded-lg border-2 transition-all ${
                selectedAnswers[currentQuestion] === index.toString()
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
              }`}
            >
              <span className="font-medium mr-3">{String.fromCharCode(65 + index)}.</span>
              {option}
            </button>
          ))}
        </div>
      </div>

      <div className="flex justify-end">
        <button
          onClick={handleNextQuestion}
          disabled={!selectedAnswers[currentQuestion]}
          className="bg-blue-500 hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed text-white px-6 py-2 rounded-lg font-medium transition-all"
        >
          {currentQuestion < questions.length - 1 ? '다음 문제' : '결과 보기'}
        </button>
      </div>
    </div>
  );
}

export default function MacroPracticePage() {
  const [showQuiz, setShowQuiz] = useState(false);

  return (
    <div>
      <MacroEconomicAnalysis />
      
      {/* Quiz Section */}
      <div className="max-w-6xl mx-auto px-6 py-8">
        <div className="text-center mb-8">
          <button
            onClick={() => setShowQuiz(!showQuiz)}
            className="bg-gradient-to-r from-purple-500 to-pink-600 text-white px-8 py-3 rounded-lg font-semibold hover:from-purple-600 hover:to-pink-700 transition-all"
          >
            {showQuiz ? '퀴즈 숨기기' : '📝 학습 확인 퀴즈'}
          </button>
        </div>
        
        {showQuiz && <MacroAnalysisQuiz />}
      </div>

      {/* Chapter Navigation */}
      <ChapterNavigation currentChapterId="macro-practice" programType="foundation" />
    </div>
  );
}