'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ArrowLeft, TrendingUp, TrendingDown, BarChart3, Globe, DollarSign, Calendar, AlertCircle, Target, Clock, ChevronRight, Zap, Building2, Users, Activity } from 'lucide-react';

function EconomicIndicatorSimulator() {
  const [selectedIndicator, setSelectedIndicator] = useState('gdp');
  const [indicatorValue, setIndicatorValue] = useState(2.5);
  const [direction, setDirection] = useState<'up' | 'down'>('up');
  const [marketImpact, setMarketImpact] = useState({ stocks: 0, bonds: 0, currency: 0 });

  const indicators = [
    {
      id: 'gdp',
      name: 'GDP 성장률',
      unit: '%',
      description: '국내총생산 전년 동기 대비 성장률',
      current: 2.5,
      range: [0, 6],
      positiveFor: ['주식시장', '통화가치'],
      negativeFor: ['채권가격'],
      color: 'blue'
    },
    {
      id: 'inflation',
      name: '소비자물가상승률 (CPI)',
      unit: '%',
      description: '전년 동월 대비 소비자물가 상승률',
      current: 3.2,
      range: [0, 8],
      positiveFor: ['금리', '실물자산'],
      negativeFor: ['채권가격', '성장주'],
      color: 'red'
    },
    {
      id: 'interest',
      name: '기준금리',
      unit: '%',
      description: '중앙은행이 설정하는 기준이 되는 금리',
      current: 3.5,
      range: [0, 8],
      positiveFor: ['통화가치', '금융주', '채권수익률'],
      negativeFor: ['성장주', '부동산'],
      color: 'green'
    },
    {
      id: 'unemployment',
      name: '실업률',
      unit: '%',
      description: '경제활동인구 중 실업자가 차지하는 비율',
      current: 3.8,
      range: [2, 12],
      positiveFor: [], // 실업률 상승은 대체로 부정적
      negativeFor: ['주식시장', '소비', '경제성장'],
      color: 'orange',
      inverse: true // 값이 낮을수록 긍정적
    },
    {
      id: 'exchange',
      name: '환율 (USD/KRW)',
      unit: '원',
      description: '달러 대비 원화 환율',
      current: 1340,
      range: [1200, 1500],
      positiveFor: ['수출기업', '외국인관광'],
      negativeFor: ['수입기업', '인플레이션'],
      color: 'purple'
    }
  ];

  const calculateMarketImpact = (indicatorId: string, value: number) => {
    const indicator = indicators.find(i => i.id === indicatorId);
    if (!indicator) return { stocks: 0, bonds: 0, currency: 0 };

    // 자동으로 상승/하락 판단
    const isUp = value > indicator.current;
    const intensity = Math.abs(value - indicator.current) / indicator.current * 100;
    const multiplier = isUp ? 1 : -1;
    
    switch (indicatorId) {
      case 'gdp':
        return {
          stocks: multiplier * intensity * 2,
          bonds: -multiplier * intensity * 1,
          currency: multiplier * intensity * 1.5
        };
      case 'inflation':
        return {
          stocks: -multiplier * intensity * 1.5,
          bonds: -multiplier * intensity * 2,
          currency: -multiplier * intensity * 1
        };
      case 'interest':
        return {
          stocks: -multiplier * intensity * 1.8,
          bonds: multiplier * intensity * 1.5,
          currency: multiplier * intensity * 2
        };
      case 'unemployment':
        return {
          stocks: -multiplier * intensity * 1.5,
          bonds: multiplier * intensity * 1,
          currency: -multiplier * intensity * 1
        };
      case 'exchange':
        return {
          stocks: multiplier * intensity * 1.2, // 원화 약세시 수출주 상승
          bonds: 0,
          currency: -multiplier * intensity * 2
        };
      default:
        return { stocks: 0, bonds: 0, currency: 0 };
    }
  };

  useEffect(() => {
    const impact = calculateMarketImpact(selectedIndicator, indicatorValue);
    setMarketImpact(impact);
    
    // 자동으로 방향 업데이트
    const indicator = indicators.find(i => i.id === selectedIndicator);
    if (indicator) {
      const newDirection = indicatorValue > indicator.current ? 'up' : 'down';
      setDirection(newDirection);
    }
  }, [selectedIndicator, indicatorValue]);

  const currentIndicator = indicators.find(i => i.id === selectedIndicator)!;
  const isPositive = currentIndicator.inverse ? direction === 'down' : direction === 'up';

  return (
    <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-8">
      <h2 className="text-2xl font-bold mb-6">📊 경제지표 영향 분석기</h2>
      
      <div className="space-y-8">
        {/* 경제지표 선택 */}
        <div className="bg-white dark:bg-gray-700 p-6 rounded-lg">
          <h3 className="text-lg font-semibold mb-4">📋 경제지표 선택</h3>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
            {indicators.map((indicator) => (
              <button
                key={indicator.id}
                onClick={() => {
                  setSelectedIndicator(indicator.id);
                  setIndicatorValue(indicator.current);
                }}
                className={`p-3 text-left rounded-lg border-2 transition-all ${
                  selectedIndicator === indicator.id
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-200 dark:border-gray-600 hover:border-gray-300'
                }`}
              >
                <div className="font-medium">{indicator.name}</div>
                <div className="text-sm text-gray-500 mt-1">{indicator.description}</div>
                <div className="text-sm text-blue-600 dark:text-blue-400 mt-1">
                  현재: {indicator.current}{indicator.unit}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* 시나리오 설정 */}
        <div className="bg-white dark:bg-gray-700 p-6 rounded-lg">
          <h3 className="text-lg font-semibold mb-4">⚙️ 시나리오 설정</h3>
          
          <div className="space-y-4">
            <div>
              <div className="flex justify-between items-center mb-3">
                <label className="text-sm font-medium">
                  {currentIndicator.name}
                </label>
                <div className="text-right">
                  <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
                    {indicatorValue}{currentIndicator.unit}
                  </div>
                  <div className="text-xs text-gray-500">
                    기준: {currentIndicator.current}{currentIndicator.unit}
                  </div>
                </div>
              </div>
              
              <input
                type="range"
                min={currentIndicator.range[0]}
                max={currentIndicator.range[1]}
                step={currentIndicator.id === 'exchange' ? 10 : 0.1}
                value={indicatorValue}
                onChange={(e) => setIndicatorValue(Number(e.target.value))}
                className="w-full mb-2"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>{currentIndicator.range[0]}{currentIndicator.unit}</span>
                <span className="font-medium bg-blue-100 dark:bg-blue-900/20 px-2 py-1 rounded">
                  기준: {currentIndicator.current}{currentIndicator.unit}
                </span>
                <span>{currentIndicator.range[1]}{currentIndicator.unit}</span>
              </div>
              
              {/* 변화 방향 자동 표시 */}
              <div className="mt-3">
                <div className={`inline-flex items-center px-3 py-2 rounded-lg text-sm font-medium ${
                  indicatorValue > currentIndicator.current
                    ? 'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-400'
                    : indicatorValue < currentIndicator.current
                    ? 'bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                }`}>
                  {indicatorValue > currentIndicator.current ? (
                    <>
                      <TrendingUp className="w-4 h-4 mr-2" />
                      <span>
                        <strong>{Math.abs(indicatorValue - currentIndicator.current).toFixed(1)}{currentIndicator.unit}</strong> 상승
                        <span className="ml-2 text-xs opacity-75">
                          ({((indicatorValue - currentIndicator.current) / currentIndicator.current * 100).toFixed(1)}%)
                        </span>
                      </span>
                    </>
                  ) : indicatorValue < currentIndicator.current ? (
                    <>
                      <TrendingDown className="w-4 h-4 mr-2" />
                      <span>
                        <strong>{Math.abs(indicatorValue - currentIndicator.current).toFixed(1)}{currentIndicator.unit}</strong> 하락
                        <span className="ml-2 text-xs opacity-75">
                          ({((currentIndicator.current - indicatorValue) / currentIndicator.current * 100).toFixed(1)}%)
                        </span>
                      </span>
                    </>
                  ) : (
                    <>
                      <div className="w-4 h-4 mr-2 rounded-full bg-gray-400"></div>
                      현재 수준 유지
                    </>
                  )}
                </div>
              </div>
            </div>

            {indicatorValue !== currentIndicator.current && (
              <div className={`p-4 rounded-lg mt-4 ${
                isPositive
                  ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300'
                  : 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300'
              }`}>
                <div className="font-medium mb-2">
                  {isPositive ? '📈 경제에 긍정적 영향' : '📉 경제에 부정적 영향'}
                </div>
                <div className="text-sm">
                  <strong>{currentIndicator.name}</strong>이 현재 <strong>{currentIndicator.current}{currentIndicator.unit}</strong>에서 
                  <strong className="mx-1">{indicatorValue}{currentIndicator.unit}</strong>로 {direction === 'up' ? '상승' : '하락'}하면,
                  일반적으로 다음 분야에 영향을 줍니다:
                </div>
                <div className="mt-3 space-y-1">
                  {isPositive && (
                    <div className="text-sm">
                      <span className="font-medium text-green-600 dark:text-green-400">긍정적 영향: </span>
                      {currentIndicator.positiveFor.join(', ')}
                    </div>
                  )}
                  {currentIndicator.negativeFor.length > 0 && (
                    <div className="text-sm">
                      <span className="font-medium text-red-600 dark:text-red-400">부정적 영향: </span>
                      {currentIndicator.negativeFor.join(', ')}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* 시장 영향 예측 */}
        <div className="bg-white dark:bg-gray-700 p-6 rounded-lg">
          <h3 className="text-lg font-semibold mb-4">📈 시장 영향 예측</h3>
            
          <div className="space-y-4">
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium">주식시장</span>
                <span className={`text-sm font-bold ${
                  marketImpact.stocks > 0 
                    ? 'text-green-600 dark:text-green-400' 
                    : marketImpact.stocks < 0 
                    ? 'text-red-600 dark:text-red-400'
                    : 'text-gray-500'
                }`}>
                  {marketImpact.stocks > 0 ? '+' : ''}{marketImpact.stocks.toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-600 h-2 rounded">
                <div 
                  className={`h-2 rounded transition-all ${
                    marketImpact.stocks > 0 ? 'bg-green-500' : 'bg-red-500'
                  }`}
                  style={{ 
                    width: `${Math.abs(marketImpact.stocks) * 2}%`,
                    maxWidth: '100%' 
                  }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium">채권시장</span>
                <span className={`text-sm font-bold ${
                  marketImpact.bonds > 0 
                    ? 'text-green-600 dark:text-green-400' 
                    : marketImpact.bonds < 0 
                    ? 'text-red-600 dark:text-red-400'
                    : 'text-gray-500'
                }`}>
                  {marketImpact.bonds > 0 ? '+' : ''}{marketImpact.bonds.toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-600 h-2 rounded">
                <div 
                  className={`h-2 rounded transition-all ${
                    marketImpact.bonds > 0 ? 'bg-green-500' : 'bg-red-500'
                  }`}
                  style={{ 
                    width: `${Math.abs(marketImpact.bonds) * 2}%`,
                    maxWidth: '100%' 
                  }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium">환율/통화</span>
                <span className={`text-sm font-bold ${
                  marketImpact.currency > 0 
                    ? 'text-green-600 dark:text-green-400' 
                    : marketImpact.currency < 0 
                    ? 'text-red-600 dark:text-red-400'
                    : 'text-gray-500'
                }`}>
                  {marketImpact.currency > 0 ? '+' : ''}{marketImpact.currency.toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-600 h-2 rounded">
                <div 
                  className={`h-2 rounded transition-all ${
                    marketImpact.currency > 0 ? 'bg-green-500' : 'bg-red-500'
                  }`}
                  style={{ 
                    width: `${Math.abs(marketImpact.currency) * 2}%`,
                    maxWidth: '100%' 
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function QuizSection() {
  const [answers, setAnswers] = useState<{ q1: string; q2: string; q3: string }>({ q1: '', q2: '', q3: '' });
  const [showResults, setShowResults] = useState(false);
  
  const correctAnswers = {
    q1: 'q1-2', // 금리 상승은 성장주에 부정적, 금융주에 긍정적 영향을 미친다
    q2: 'q2-3', // 환율 상승(원화 약세)은 수출기업에 긍정적이다
    q3: 'q3-1'  // GDP 성장률이 예상보다 높으면 주식시장은 상승할 가능성이 크다
  };
  
  const handleAnswerChange = (question: string, value: string) => {
    if (!showResults) {
      setAnswers(prev => ({ ...prev, [question]: value }));
    }
  };
  
  const checkAnswers = () => {
    if (answers.q1 && answers.q2 && answers.q3) {
      setShowResults(true);
    } else {
      alert('모든 문제에 답해주세요.');
    }
  };
  
  const resetQuiz = () => {
    setAnswers({ q1: '', q2: '', q3: '' });
    setShowResults(false);
  };
  
  const getResultStyle = (question: 'q1' | 'q2' | 'q3', optionValue: string) => {
    if (!showResults) return '';
    
    const userAnswer = answers[question];
    const correctAnswer = correctAnswers[question];
    
    if (optionValue === correctAnswer) {
      return 'text-green-600 dark:text-green-400 font-medium';
    } else if (optionValue === userAnswer && optionValue !== correctAnswer) {
      return 'text-red-600 dark:text-red-400 font-medium';
    }
    return '';
  };

  const score = Object.keys(correctAnswers).reduce((acc, key) => {
    return acc + (answers[key as keyof typeof answers] === correctAnswers[key as keyof typeof correctAnswers] ? 1 : 0);
  }, 0);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
      <h2 className="text-2xl font-bold mb-6">📝 경제지표 이해도 퀴즈</h2>
      
      {!showResults ? (
        <div className="space-y-8">
          {/* Question 1 */}
          <div className="border-b border-gray-200 dark:border-gray-700 pb-6">
            <h3 className="text-lg font-semibold mb-4">1. 기준금리 상승이 시장에 미치는 영향으로 가장 적절한 것은?</h3>
            <div className="space-y-2">
              {[
                { value: 'q1-1', text: '모든 주식에 긍정적 영향을 미친다' },
                { value: 'q1-2', text: '성장주에는 부정적, 금융주에는 긍정적 영향을 미친다' },
                { value: 'q1-3', text: '채권 투자 매력도가 떨어진다' },
                { value: 'q1-4', text: '부동산 투자가 더욱 활성화된다' }
              ].map(option => (
                <label key={option.value} className="flex items-center space-x-3 cursor-pointer">
                  <input
                    type="radio"
                    name="q1"
                    value={option.value}
                    checked={answers.q1 === option.value}
                    onChange={(e) => handleAnswerChange('q1', e.target.value)}
                    className="text-blue-600"
                  />
                  <span className={getResultStyle('q1', option.value)}>{option.text}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Question 2 */}
          <div className="border-b border-gray-200 dark:border-gray-700 pb-6">
            <h3 className="text-lg font-semibold mb-4">2. 원/달러 환율이 상승(원화 약세)할 때의 영향은?</h3>
            <div className="space-y-2">
              {[
                { value: 'q2-1', text: '수입 물가가 하락한다' },
                { value: 'q2-2', text: '내수 기업에 긍정적 영향을 미친다' },
                { value: 'q2-3', text: '수출 기업의 경쟁력이 향상된다' },
                { value: 'q2-4', text: '외국인 관광객이 감소한다' }
              ].map(option => (
                <label key={option.value} className="flex items-center space-x-3 cursor-pointer">
                  <input
                    type="radio"
                    name="q2"
                    value={option.value}
                    checked={answers.q2 === option.value}
                    onChange={(e) => handleAnswerChange('q2', e.target.value)}
                    className="text-blue-600"
                  />
                  <span className={getResultStyle('q2', option.value)}>{option.text}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Question 3 */}
          <div className="pb-6">
            <h3 className="text-lg font-semibold mb-4">3. GDP 성장률이 시장 예상치를 상회했을 때 일반적으로 나타나는 현상은?</h3>
            <div className="space-y-2">
              {[
                { value: 'q3-1', text: '주식시장 상승 가능성이 높아진다' },
                { value: 'q3-2', text: '중앙은행이 금리를 즉시 인하한다' },
                { value: 'q3-3', text: '채권 가격이 상승한다' },
                { value: 'q3-4', text: '실업률이 급격히 상승한다' }
              ].map(option => (
                <label key={option.value} className="flex items-center space-x-3 cursor-pointer">
                  <input
                    type="radio"
                    name="q3"
                    value={option.value}
                    checked={answers.q3 === option.value}
                    onChange={(e) => handleAnswerChange('q3', e.target.value)}
                    className="text-blue-600"
                  />
                  <span className={getResultStyle('q3', option.value)}>{option.text}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="flex justify-center">
            <button
              onClick={checkAnswers}
              className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-lg font-medium"
            >
              결과 확인하기
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="text-center">
            <div className="text-4xl font-bold text-blue-600 dark:text-blue-400 mb-2">
              {score}/3
            </div>
            <p className="text-gray-600 dark:text-gray-300">
              {score === 3 ? "완벽합니다! 🎉" : score >= 2 ? "잘했습니다! 👏" : "더 공부가 필요해요 📚"}
            </p>
          </div>

          <div className="space-y-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">해설</h4>
              <div className="space-y-2 text-sm text-blue-700 dark:text-blue-200">
                <p><strong>1번:</strong> 금리 상승 시 성장주는 할인율 상승으로 부담, 금융주는 이자마진 개선으로 수혜</p>
                <p><strong>2번:</strong> 원화 약세 시 수출품의 달러 가격 경쟁력 향상으로 수출 기업에 긍정적</p>
                <p><strong>3번:</strong> GDP 성장률 상승은 기업 실적 개선 기대로 이어져 주식시장 상승 요인</p>
              </div>
            </div>
          </div>

          <div className="flex justify-center">
            <button
              onClick={resetQuiz}
              className="bg-gray-500 hover:bg-gray-600 text-white px-6 py-3 rounded-lg font-medium"
            >
              다시 풀기
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function EconomicIndicatorsPage() {
  const [showQuiz, setShowQuiz] = useState(false);

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
                주요 경제지표의 이해
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Baby Chick 7/9 단계
              </div>
              <div className="w-32 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div className="bg-gradient-to-r from-yellow-400 to-orange-500 h-2 rounded-full" style={{ width: '78%' }}></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full mb-6">
            <BarChart3 className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            주요 경제지표의 이해
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto mb-8">
            GDP, 인플레이션, 금리, 실업률, 환율 등 핵심 경제지표가 주식시장과 투자에 미치는 영향을 실습을 통해 학습합니다.
          </p>
        </div>

        {/* Main Content */}
        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-8 mb-8">
          {/* Section 1: Understanding Economic Indicators */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              1️⃣ 5가지 핵심 경제지표
            </h2>
            
            <div className="grid gap-6">
              <div className="bg-blue-50 dark:bg-blue-900/10 border border-blue-200 dark:border-blue-800 rounded-xl p-6">
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0">
                    <div className="w-12 h-12 bg-blue-500 rounded-lg flex items-center justify-center">
                      <TrendingUp className="w-6 h-6 text-white" />
                    </div>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-lg font-bold text-blue-900 dark:text-blue-100 mb-2">
                      GDP 성장률 (Gross Domestic Product)
                    </h3>
                    <p className="text-blue-800 dark:text-blue-200 mb-3">
                      국내에서 생산된 모든 재화와 서비스의 시장가치를 나타내는 지표로, 경제 전반의 건강상태를 보여줍니다.
                    </p>
                    <div className="grid md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="font-semibold text-green-600 dark:text-green-400 mb-1">긍정적 영향</h4>
                        <p className="text-sm text-blue-700 dark:text-blue-300">주식시장, 통화가치, 고용</p>
                      </div>
                      <div>
                        <h4 className="font-semibold text-red-600 dark:text-red-400 mb-1">부정적 영향</h4>
                        <p className="text-sm text-blue-700 dark:text-blue-300">채권가격 (금리 상승 압력)</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-red-50 dark:bg-red-900/10 border border-red-200 dark:border-red-800 rounded-xl p-6">
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0">
                    <div className="w-12 h-12 bg-red-500 rounded-lg flex items-center justify-center">
                      <Activity className="w-6 h-6 text-white" />
                    </div>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-lg font-bold text-red-900 dark:text-red-100 mb-2">
                      소비자물가상승률 (CPI)
                    </h3>
                    <p className="text-red-800 dark:text-red-200 mb-3">
                      일반 소비자가 구입하는 재화와 서비스의 평균적인 가격 변동을 측정하는 인플레이션 지표입니다.
                    </p>
                    <div className="grid md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="font-semibold text-green-600 dark:text-green-400 mb-1">긍정적 영향</h4>
                        <p className="text-sm text-red-700 dark:text-red-300">실물자산, 금리 (적정 수준시)</p>
                      </div>
                      <div>
                        <h4 className="font-semibold text-red-600 dark:text-red-400 mb-1">부정적 영향</h4>
                        <p className="text-sm text-red-700 dark:text-red-300">채권가격, 성장주, 구매력</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-green-50 dark:bg-green-900/10 border border-green-200 dark:border-green-800 rounded-xl p-6">
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0">
                    <div className="w-12 h-12 bg-green-500 rounded-lg flex items-center justify-center">
                      <DollarSign className="w-6 h-6 text-white" />
                    </div>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-lg font-bold text-green-900 dark:text-green-100 mb-2">
                      기준금리
                    </h3>
                    <p className="text-green-800 dark:text-green-200 mb-3">
                      중앙은행이 설정하는 정책금리로, 모든 금리의 기준이 되어 경제 전반에 광범위한 영향을 미칩니다.
                    </p>
                    <div className="grid md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="font-semibold text-green-600 dark:text-green-400 mb-1">긍정적 영향</h4>
                        <p className="text-sm text-green-700 dark:text-green-300">통화가치, 금융주, 채권수익률</p>
                      </div>
                      <div>
                        <h4 className="font-semibold text-red-600 dark:text-red-400 mb-1">부정적 영향</h4>
                        <p className="text-sm text-green-700 dark:text-green-300">성장주, 부동산, 대출 활동</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-orange-50 dark:bg-orange-900/10 border border-orange-200 dark:border-orange-800 rounded-xl p-6">
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0">
                    <div className="w-12 h-12 bg-orange-500 rounded-lg flex items-center justify-center">
                      <Users className="w-6 h-6 text-white" />
                    </div>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-lg font-bold text-orange-900 dark:text-orange-100 mb-2">
                      실업률
                    </h3>
                    <p className="text-orange-800 dark:text-orange-200 mb-3">
                      경제활동인구 중 일자리를 찾고 있지만 구하지 못한 사람들의 비율로, 경제의 노동시장 건강도를 나타냅니다.
                    </p>
                    <div className="bg-orange-100 dark:bg-orange-900/20 p-3 rounded-lg">
                      <p className="text-sm text-orange-800 dark:text-orange-200">
                        <strong>특징:</strong> 실업률은 낮을수록 경제에 긍정적입니다. (역지표)
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/10 border border-purple-200 dark:border-purple-800 rounded-xl p-6">
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0">
                    <div className="w-12 h-12 bg-purple-500 rounded-lg flex items-center justify-center">
                      <Globe className="w-6 h-6 text-white" />
                    </div>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-lg font-bold text-purple-900 dark:text-purple-100 mb-2">
                      환율 (USD/KRW)
                    </h3>
                    <p className="text-purple-800 dark:text-purple-200 mb-3">
                      한국 원화와 미국 달러 간의 교환비율로, 수출입과 외국인 투자에 직접적인 영향을 미칩니다.
                    </p>
                    <div className="grid md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="font-semibold text-green-600 dark:text-green-400 mb-1">원화 약세 수혜</h4>
                        <p className="text-sm text-purple-700 dark:text-purple-300">수출기업, 해외매출 기업</p>
                      </div>
                      <div>
                        <h4 className="font-semibold text-red-600 dark:text-red-400 mb-1">원화 약세 피해</h4>
                        <p className="text-sm text-purple-700 dark:text-purple-300">수입기업, 인플레이션</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Section 2: Interactive Simulator */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              2️⃣ 경제지표 영향 시뮬레이터
            </h2>
            
            <EconomicIndicatorSimulator />
          </section>

          {/* Section 3: Quiz */}
          <section>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                3️⃣ 학습 확인 퀴즈
              </h2>
              <button
                onClick={() => setShowQuiz(!showQuiz)}
                className="bg-gradient-to-r from-purple-500 to-pink-600 text-white px-6 py-2 rounded-lg font-medium hover:from-purple-600 hover:to-pink-700 transition-all"
              >
                {showQuiz ? '퀴즈 숨기기' : '퀴즈 시작하기'}
              </button>
            </div>

            {showQuiz && <QuizSection />}
          </section>
        </div>

        {/* Next Steps */}
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-8 text-center">
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">다음 학습</h3>
          <p className="text-gray-600 dark:text-gray-300 mb-6">
            경제지표 분석을 마스터했다면 이제 FOMC와 한국은행의 통화정책을 학습해보세요
          </p>
          <div className="flex justify-center space-x-4">
            <Link
              href="/modules/stock-analysis/chapters/fomc-analysis"
              className="inline-flex items-center bg-gradient-to-r from-blue-500 to-indigo-600 text-white px-6 py-3 rounded-lg font-medium hover:from-blue-600 hover:to-indigo-700 transition-all"
            >
              다음: FOMC와 통화정책
              <ChevronRight size={20} className="ml-2" />
            </Link>
          </div>
        </div>

        {/* Progress Indicator */}
        <div className="mt-8 pt-6 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
            <span>Baby Chick 진행률</span>
            <span>7/9 완료</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div className="bg-gradient-to-r from-yellow-400 to-orange-500 h-2 rounded-full" style={{ width: '78%' }}></div>
          </div>
        </div>
      </div>
    </div>
  );
}