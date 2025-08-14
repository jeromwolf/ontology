'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ArrowLeft, Play, RefreshCw, TrendingUp, TrendingDown, Brain, AlertTriangle, Target, Clock, ChevronRight, BarChart3, DollarSign, Users, Zap, Shield } from 'lucide-react';

interface SimulationState {
  day: number;
  portfolio: {
    cash: number;
    stocks: { [key: string]: { shares: number; avgPrice: number } };
    totalValue: number;
    dailyReturn: number;
  };
  marketCondition: 'bull' | 'bear' | 'volatile' | 'crash';
  emotions: {
    fear: number;
    greed: number;
    confidence: number;
  };
  news: string[];
  decisions: Array<{
    day: number;
    action: string;
    reason: string;
    outcome: 'good' | 'bad' | 'neutral';
    emotionalBias?: string;
  }>;
}

function MarketSimulation() {
  const [simulation, setSimulation] = useState<SimulationState>({
    day: 1,
    portfolio: {
      cash: 1000000,
      stocks: {},
      totalValue: 1000000,
      dailyReturn: 0
    },
    marketCondition: 'bull',
    emotions: { fear: 30, greed: 40, confidence: 60 },
    news: [],
    decisions: []
  });

  const [isRunning, setIsRunning] = useState(false);
  const [selectedAction, setSelectedAction] = useState<string>('');
  const [showResults, setShowResults] = useState(false);

  const stocks = [
    { symbol: 'TECH', name: '테크 대장주', price: 50000, volatility: 0.3 },
    { symbol: 'BANK', name: '금융 안정주', price: 30000, volatility: 0.15 },
    { symbol: 'GROWTH', name: '성장 중소형주', price: 20000, volatility: 0.4 }
  ];

  const scenarios = [
    {
      day: 3,
      condition: 'bull',
      title: '🚀 3일 연속 급등',
      description: '시장이 3일 연속 5% 이상 상승하고 있습니다. SNS와 뉴스에서는 "이번이 마지막 기회"라는 말이 넘쳐납니다.',
      news: [
        '📈 개인투자자 신규 계좌 개설 급증',
        '💰 "곧 10만전자 간다" 전문가 전망',
        '🔥 커뮤니티: "지금 안 사면 후회"'
      ],
      emotions: { fear: 10, greed: 80, confidence: 90 },
      choices: [
        {
          id: 'buy-more',
          text: '🚀 추가 매수 - "놓치면 안 돼!"',
          bias: '군중심리',
          outcome: 'bad',
          explanation: '고점에서의 추격 매수는 위험합니다.'
        },
        {
          id: 'hold',
          text: '⏸️ 보유 유지 - "계획대로 가자"',
          bias: null,
          outcome: 'good',
          explanation: '감정에 휘둘리지 않은 현명한 판단입니다.'
        },
        {
          id: 'sell-some',
          text: '💰 일부 차익실현 - "이익은 챙기자"',
          bias: null,
          outcome: 'good',
          explanation: '고점에서 차익실현하는 좋은 전략입니다.'
        }
      ]
    },
    {
      day: 7,
      condition: 'volatile',
      title: '📰 충격적인 뉴스',
      description: '금리 인상 가능성에 대한 중앙은행 총재 발언으로 시장이 혼란에 빠졌습니다. 하루 만에 -8% 급락했습니다.',
      news: [
        '🏛️ 중앙은행: "인플레이션 우려, 금리 인상 검토"',
        '📉 코스피 -8% 급락, 개인 패닉 매도',
        '😰 전문가: "조정은 시작일 뿐"'
      ],
      emotions: { fear: 85, greed: 20, confidence: 25 },
      choices: [
        {
          id: 'panic-sell',
          text: '🏃 패닉 매도 - "더 떨어지기 전에!"',
          bias: '손실회피',
          outcome: 'bad',
          explanation: '공포에 휩쓸린 저점 매도는 큰 실수입니다.'
        },
        {
          id: 'buy-dip',
          text: '💎 물타기 - "싸게 더 사자!"',
          bias: '앵커링',
          outcome: 'neutral',
          explanation: '하락 이유를 분석하지 않은 무분별한 매수는 위험할 수 있습니다.'
        },
        {
          id: 'wait-analyze',
          text: '🔍 관망 후 분석 - "상황을 파악해보자"',
          bias: null,
          outcome: 'good',
          explanation: '감정적 대응 대신 냉정한 분석을 선택한 현명한 판단입니다.'
        }
      ]
    },
    {
      day: 12,
      condition: 'bear',
      title: '📉 지속적인 하락장',
      description: '시장이 2주째 하락세를 지속하고 있습니다. 보유 종목들이 평균 -20% 손실 상태입니다.',
      news: [
        '📊 외국인 5일 연속 순매도',
        '💸 개인투자자 손실 확대',
        '🔻 애널리스트들 목표가 하향 조정'
      ],
      emotions: { fear: 70, greed: 15, confidence: 30 },
      choices: [
        {
          id: 'cut-loss',
          text: '✂️ 손절매 - "더 이상 못 버텨!"',
          bias: '손실회피',
          outcome: 'neutral',
          explanation: '손절 자체는 필요하지만, 감정적 판단보다는 계획에 따른 손절이 중요합니다.'
        },
        {
          id: 'hold-hope',
          text: '🙏 보유 지속 - "언젠가는 올라갈 거야"',
          bias: '확증편향',
          outcome: 'bad',
          explanation: '희망적 사고로 인한 무분별한 보유는 더 큰 손실로 이어질 수 있습니다.'
        },
        {
          id: 'rebalance',
          text: '⚖️ 포트폴리오 리밸런싱',
          bias: null,
          outcome: 'good',
          explanation: '시장 상황에 맞는 체계적인 포트폴리오 조정입니다.'
        }
      ]
    }
  ];

  const getCurrentScenario = () => {
    return scenarios.find(s => s.day === simulation.day);
  };

  const advanceDay = (action?: string) => {
    const scenario = getCurrentScenario();
    if (scenario && action) {
      const choice = scenario.choices.find(c => c.id === action);
      if (choice) {
        setSimulation(prev => ({
          ...prev,
          day: prev.day + 1,
          emotions: scenario.emotions,
          news: scenario.news,
          decisions: [...prev.decisions, {
            day: prev.day,
            action: choice.text,
            reason: choice.explanation,
            outcome: choice.outcome,
            emotionalBias: choice.bias
          }]
        }));
      }
    } else {
      setSimulation(prev => ({
        ...prev,
        day: prev.day + 1
      }));
    }

    if (simulation.day >= 15) {
      setShowResults(true);
    }
  };

  const resetSimulation = () => {
    setSimulation({
      day: 1,
      portfolio: {
        cash: 1000000,
        stocks: {},
        totalValue: 1000000,
        dailyReturn: 0
      },
      marketCondition: 'bull',
      emotions: { fear: 30, greed: 40, confidence: 60 },
      news: [],
      decisions: []
    });
    setShowResults(false);
    setSelectedAction('');
  };

  const getBiasCount = () => {
    return simulation.decisions.filter(d => d.emotionalBias).length;
  };

  const getGoodDecisions = () => {
    return simulation.decisions.filter(d => d.outcome === 'good').length;
  };

  if (showResults) {
    const biasCount = getBiasCount();
    const goodDecisions = getGoodDecisions();
    const totalDecisions = simulation.decisions.length;

    return (
      <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-8">
        <h2 className="text-2xl font-bold mb-6">📊 투자 심리 시뮬레이션 결과</h2>
        
        <div className="grid md:grid-cols-3 gap-6 mb-6">
          <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg text-center">
            <div className="text-3xl font-bold text-green-600 dark:text-green-400">
              {goodDecisions}/{totalDecisions}
            </div>
            <div className="text-sm text-green-700 dark:text-green-300">합리적 결정</div>
          </div>
          
          <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-lg text-center">
            <div className="text-3xl font-bold text-red-600 dark:text-red-400">
              {biasCount}
            </div>
            <div className="text-sm text-red-700 dark:text-red-300">편향된 결정</div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg text-center">
            <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
              {Math.round(goodDecisions / totalDecisions * 100)}%
            </div>
            <div className="text-sm text-blue-700 dark:text-blue-300">성공률</div>
          </div>
        </div>

        <div className={`p-6 rounded-lg mb-6 ${
          goodDecisions >= 2 ? 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-300'
          : goodDecisions >= 1 ? 'bg-yellow-100 dark:bg-yellow-900/10 text-yellow-700 dark:text-yellow-300'
          : 'bg-red-100 dark:bg-red-900/10 text-red-700 dark:text-red-300'
        }`}>
          <h3 className="font-bold text-lg mb-2">
            {goodDecisions >= 2 ? '🎉 우수한 심리 통제!' 
             : goodDecisions >= 1 ? '😊 괜찮은 편입니다!' 
             : '😰 감정 통제 연습이 필요해요!'}
          </h3>
          <p>
            {goodDecisions >= 2
              ? '극한 상황에서도 감정에 휩쓸리지 않고 합리적 판단을 내렸습니다. 실전 투자에서도 좋은 성과를 기대할 수 있어요.'
              : goodDecisions >= 1
              ? '어려운 상황에서 부분적으로 합리적 판단을 보였습니다. 조금 더 연습하면 감정 통제력이 향상될 거예요.'
              : '감정적 투자 결정이 많았습니다. 투자 규칙을 미리 정하고 엄격히 지키는 연습이 필요합니다.'}
          </p>
        </div>

        <div className="space-y-4 mb-6">
          <h3 className="text-lg font-semibold">결정 분석</h3>
          {simulation.decisions.map((decision, index) => (
            <div key={index} className="bg-white dark:bg-gray-700 p-4 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">Day {decision.day}</span>
                <div className="flex items-center gap-2">
                  {decision.emotionalBias && (
                    <span className="text-xs px-2 py-1 bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400 rounded">
                      {decision.emotionalBias}
                    </span>
                  )}
                  <span className={`text-xs px-2 py-1 rounded ${
                    decision.outcome === 'good' 
                      ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                      : decision.outcome === 'bad'
                      ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                      : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                  }`}>
                    {decision.outcome === 'good' ? '좋음' : decision.outcome === 'bad' ? '나쁨' : '보통'}
                  </span>
                </div>
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                선택: {decision.action}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-500">
                {decision.reason}
              </div>
            </div>
          ))}
        </div>

        <button
          onClick={resetSimulation}
          className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
        >
          다시 시뮬레이션 하기
        </button>
      </div>
    );
  }

  const currentScenario = getCurrentScenario();

  return (
    <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold">🎮 실전 투자 심리 시뮬레이션</h2>
        <div className="text-sm text-gray-500">
          Day {simulation.day} / 15
        </div>
      </div>

      <div className="mb-6">
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mb-4">
          <div 
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${(simulation.day / 15) * 100}%` }}
          ></div>
        </div>
      </div>

      {currentScenario ? (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-700 p-6 rounded-lg">
            <h3 className="text-xl font-bold mb-4">{currentScenario.title}</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              {currentScenario.description}
            </p>

            <div className="mb-4">
              <h4 className="font-semibold mb-2">📰 시장 뉴스</h4>
              <div className="space-y-1">
                {currentScenario.news.map((news, index) => (
                  <div key={index} className="text-sm text-gray-600 dark:text-gray-400">
                    {news}
                  </div>
                ))}
              </div>
            </div>

            <div className="mb-6">
              <h4 className="font-semibold mb-3">💭 현재 감정 상태</h4>
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-2xl mb-1">😨</div>
                  <div className="text-sm text-red-600 dark:text-red-400">
                    공포 {currentScenario.emotions.fear}%
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl mb-1">🤑</div>
                  <div className="text-sm text-green-600 dark:text-green-400">
                    탐욕 {currentScenario.emotions.greed}%
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl mb-1">💪</div>
                  <div className="text-sm text-blue-600 dark:text-blue-400">
                    자신감 {currentScenario.emotions.confidence}%
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <h4 className="font-semibold">🤔 어떻게 행동하시겠습니까?</h4>
              {currentScenario.choices.map((choice) => (
                <button
                  key={choice.id}
                  onClick={() => setSelectedAction(choice.id)}
                  className={`w-full p-4 text-left rounded-lg border-2 transition-all ${
                    selectedAction === choice.id
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className={`w-5 h-5 rounded-full border-2 mt-0.5 ${
                      selectedAction === choice.id
                        ? 'border-blue-500 bg-blue-500'
                        : 'border-gray-300 dark:border-gray-600'
                    }`}>
                      {selectedAction === choice.id && (
                        <div className="w-1.5 h-1.5 bg-white rounded-full mx-auto mt-1"></div>
                      )}
                    </div>
                    <div>
                      <span className="text-gray-900 dark:text-gray-100">{choice.text}</span>
                      {choice.bias && (
                        <div className="text-xs text-red-500 dark:text-red-400 mt-1">
                          ⚠️ {choice.bias} 편향 위험
                        </div>
                      )}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="flex justify-end">
            <button
              onClick={() => advanceDay(selectedAction)}
              disabled={!selectedAction}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
            >
              결정하기
            </button>
          </div>
        </div>
      ) : (
        <div className="bg-white dark:bg-gray-700 p-6 rounded-lg text-center">
          <div className="text-4xl mb-4">📈</div>
          <h3 className="text-xl font-bold mb-2">평상시 거래일</h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            특별한 이슈가 없는 평범한 하루입니다. 시장은 보합세를 유지하고 있습니다.
          </p>
          <button
            onClick={() => advanceDay()}
            className="px-6 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors"
          >
            다음 날로
          </button>
        </div>
      )}
    </div>
  );
}

function QuizSection() {
  const [answers, setAnswers] = useState<{ q1: string; q2: string; q3: string }>({ q1: '', q2: '', q3: '' });
  const [showResults, setShowResults] = useState(false);
  
  const correctAnswers = {
    q1: 'q1-3', // 미리 세운 투자 규칙을 엄격히 따른다
    q2: 'q2-2', // 다양한 관점의 정보를 균형있게 검토한다
    q3: 'q3-1'  // 감정이 격해질 때 24시간 대기 후 결정한다
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
      return 'text-red-600 dark:text-red-400';
    }
    return 'text-gray-400';
  };
  
  const getResultIcon = (question: 'q1' | 'q2' | 'q3', optionValue: string) => {
    if (!showResults) return '';
    
    const userAnswer = answers[question];
    const correctAnswer = correctAnswers[question];
    
    if (optionValue === correctAnswer) {
      return ' ✓';
    } else if (optionValue === userAnswer && optionValue !== correctAnswer) {
      return ' ✗';
    }
    return '';
  };
  
  const score = showResults 
    ? Object.keys(correctAnswers).filter(q => answers[q as keyof typeof answers] === correctAnswers[q as keyof typeof correctAnswers]).length
    : 0;
  
  return (
    <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-8">
      <h2 className="text-2xl font-bold mb-6">🧠 이해도 체크</h2>
      
      {showResults && (
        <div className={`mb-6 p-4 rounded-lg ${
          score === 3 ? 'bg-green-100 dark:bg-green-900/10 text-green-700 dark:text-green-300' 
          : score === 2 ? 'bg-yellow-100 dark:bg-yellow-900/10 text-yellow-700 dark:text-yellow-300'
          : score === 1 ? 'bg-orange-100 dark:bg-orange-900/10 text-orange-700 dark:text-orange-300'
          : 'bg-red-100 dark:bg-red-900/10 text-red-700 dark:text-red-300'
        }`}>
          <p className="font-semibold">
            {score === 3 ? '🎉 완벽합니다!' : score === 2 ? '😊 잘하셨어요!' : score === 1 ? '💪 조금 더 공부해보세요!' : '📚 다시 학습해보세요!'}
            {` ${score}/3 문제를 맞추셨습니다.`}
          </p>
        </div>
      )}
      
      <div className="space-y-6">
        <div>
          <h3 className="font-semibold mb-3">Q1. 극한 상황에서 감정적 투자를 피하는 가장 효과적인 방법은?</h3>
          <div className="space-y-2 ml-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q1" 
                value="q1-1"
                checked={answers.q1 === 'q1-1'}
                onChange={(e) => handleAnswerChange('q1', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q1', 'q1-1')}>
                감정을 완전히 무시하고 투자한다{getResultIcon('q1', 'q1-1')}
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q1" 
                value="q1-2"
                checked={answers.q1 === 'q1-2'}
                onChange={(e) => handleAnswerChange('q1', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q1', 'q1-2')}>
                시장 상황에 따라 유연하게 대응한다{getResultIcon('q1', 'q1-2')}
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q1" 
                value="q1-3"
                checked={answers.q1 === 'q1-3'}
                onChange={(e) => handleAnswerChange('q1', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q1', 'q1-3')}>
                미리 세운 투자 규칙을 엄격히 따른다{getResultIcon('q1', 'q1-3')}
              </span>
            </label>
          </div>
        </div>

        <div>
          <h3 className="font-semibold mb-3">Q2. 확증편향을 극복하는 올바른 방법은?</h3>
          <div className="space-y-2 ml-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q2" 
                value="q2-1"
                checked={answers.q2 === 'q2-1'}
                onChange={(e) => handleAnswerChange('q2', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className="text-wrap" className={getResultStyle('q2', 'q2-1')}>
                자신의 투자 논리를 뒷받침하는 자료만 찾는다{getResultIcon('q2', 'q2-1')}
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q2" 
                value="q2-2"
                checked={answers.q2 === 'q2-2'}
                onChange={(e) => handleAnswerChange('q2', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q2', 'q2-2')}>
                다양한 관점의 정보를 균형있게 검토한다{getResultIcon('q2', 'q2-2')}
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q2" 
                value="q2-3"
                checked={answers.q2 === 'q2-3'}
                onChange={(e) => handleAnswerChange('q2', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q2', 'q2-3')}>
                부정적인 정보는 무시하고 긍정적 정보만 본다{getResultIcon('q2', 'q2-3')}
              </span>
            </label>
          </div>
        </div>

        <div>
          <h3 className="font-semibold mb-3">Q3. 투자 심리 시뮬레이션에서 배운 핵심 교훈은?</h3>
          <div className="space-y-2 ml-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q3" 
                value="q3-1"
                checked={answers.q3 === 'q3-1'}
                onChange={(e) => handleAnswerChange('q3', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q3', 'q3-1')}>
                감정이 격해질 때 24시간 대기 후 결정한다{getResultIcon('q3', 'q3-1')}
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q3" 
                value="q3-2"
                checked={answers.q3 === 'q3-2'}
                onChange={(e) => handleAnswerChange('q3', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q3', 'q3-2')}>
                시장 분위기에 맞춰 빠르게 대응한다{getResultIcon('q3', 'q3-2')}
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q3" 
                value="q3-3"
                checked={answers.q3 === 'q3-3'}
                onChange={(e) => handleAnswerChange('q3', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q3', 'q3-3')}>
                전문가 의견을 무조건 따라한다{getResultIcon('q3', 'q3-3')}
              </span>
            </label>
          </div>
        </div>
      </div>

      <div className="flex gap-3 mt-8">
        {!showResults ? (
          <button
            onClick={checkAnswers}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
          >
            정답 확인하기
          </button>
        ) : (
          <button
            onClick={resetQuiz}
            className="px-6 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors"
          >
            다시 풀기
          </button>
        )}
      </div>
    </div>
  );
}

export default function PsychologySimulationPage() {
  const [currentTip, setCurrentTip] = useState(0);

  const psychologyTips = [
    {
      title: '24시간 룰',
      icon: Clock,
      description: '강한 감정이 들 때는 24시간 기다린 후 결정하기',
      detail: '분노, 공포, 흥분 상태에서 내린 결정은 90% 이상 후회하게 됩니다.'
    },
    {
      title: '매매일지 작성',
      icon: Target,
      description: '모든 투자 결정의 이유와 감정 상태를 기록',
      detail: '객관적 기록을 통해 자신의 편향 패턴을 파악할 수 있습니다.'
    },
    {
      title: '역발상 사고',
      icon: RefreshCw,
      description: '모두가 같은 방향을 볼 때 반대를 생각해보기',
      detail: '군중이 극도로 낙관적이거나 비관적일 때가 기회인 경우가 많습니다.'
    },
    {
      title: '손절선 자동화',
      icon: Shield,
      description: '미리 정한 손절선을 기계적으로 실행',
      detail: '감정이 개입할 여지를 아예 차단하는 것이 가장 확실한 방법입니다.'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <Link 
            href="/modules/stock-analysis"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Stock Analysis로 돌아가기</span>
          </Link>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Chapter Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-16 h-16 bg-yellow-100 dark:bg-yellow-900/30 rounded-full flex items-center justify-center">
              <Play className="w-8 h-8 text-yellow-600 dark:text-yellow-400" />
            </div>
            <div className="text-left">
              <div className="text-sm text-gray-500 mb-1">Baby Chick • Chapter 6</div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                투자 심리 시뮬레이션
              </h1>
            </div>
          </div>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
            실제 시장 상황을 재현한 시뮬레이션으로 극한 상황에서의 심리적 함정을 체험하고 극복하는 훈련을 해보세요.
          </p>
        </div>

        {/* Learning Objectives */}
        <div className="bg-blue-50 dark:bg-blue-900/10 rounded-xl p-6 mb-8">
          <h2 className="text-xl font-bold text-blue-900 dark:text-blue-300 mb-4">
            📚 학습 목표
          </h2>
          <ul className="space-y-2 text-blue-800 dark:text-blue-300">
            <li className="flex items-start gap-2">
              <span className="font-bold mt-1">•</span>
              <span>극한 상황에서 감정 통제 경험 쌓기</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold mt-1">•</span>
              <span>다양한 시장 상황별 대응 방법 학습</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold mt-1">•</span>
              <span>심리적 편향의 실제 영향력 체감</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold mt-1">•</span>
              <span>냉정한 판단력 기르는 실전 훈련</span>
            </li>
          </ul>
        </div>

        {/* Main Content */}
        <div className="space-y-12">
          {/* Section 1: Why Simulation */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              1️⃣ 왜 시뮬레이션 훈련이 필요할까?
            </h2>
            
            <div className="bg-red-50 dark:bg-red-900/10 p-6 rounded-xl mb-8">
              <div className="flex items-center gap-3 mb-4">
                <AlertTriangle className="w-8 h-8 text-red-600 dark:text-red-400" />
                <div>
                  <h3 className="text-xl font-bold text-red-800 dark:text-red-300">실전에서는 돈이 걸려 있다</h3>
                  <p className="text-red-600 dark:text-red-400">진짜 돈이 오가는 상황에서는 평소보다 10배 더 감정적이 됩니다.</p>
                </div>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
              <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-xl">
                <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                  <Brain className="inline w-5 h-5 mr-2" />
                  이론 학습의 한계
                </h3>
                
                <div className="space-y-3 text-sm text-gray-600 dark:text-gray-400">
                  <div className="flex items-start gap-2">
                    <span className="text-red-500 mt-1">❌</span>
                    <span>"손실회피 편향을 조심해야 한다" (머리로만 이해)</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-red-500 mt-1">❌</span>
                    <span>실제 상황에서는 감정이 이성을 압도</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-red-500 mt-1">❌</span>
                    <span>책으로 배운 것과 현실은 완전히 다름</span>
                  </div>
                </div>
              </div>

              <div className="bg-green-50 dark:bg-green-900/10 p-6 rounded-xl">
                <h3 className="text-lg font-bold text-green-800 dark:text-green-300 mb-4">
                  <Zap className="inline w-5 h-5 mr-2" />
                  시뮬레이션의 효과
                </h3>
                
                <div className="space-y-3 text-sm text-green-700 dark:text-green-300">
                  <div className="flex items-start gap-2">
                    <span className="text-green-500 mt-1">✅</span>
                    <span>안전한 환경에서 위험한 상황 체험</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-green-500 mt-1">✅</span>
                    <span>실수의 결과를 직접 경험하고 학습</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-green-500 mt-1">✅</span>
                    <span>감정 통제 능력의 점진적 향상</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/10 p-6 rounded-xl mt-8">
              <h3 className="text-lg font-bold text-yellow-800 dark:text-yellow-300 mb-4">
                📊 실제 연구 결과
              </h3>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-gray-700 p-4 rounded-lg">
                  <h4 className="font-semibold text-yellow-700 dark:text-yellow-400 mb-2">스탠포드 대학 연구</h4>
                  <p className="text-sm text-yellow-600 dark:text-yellow-300">
                    시뮬레이션 훈련을 받은 투자자는 실전에서 <strong>감정적 실수를 67% 덜</strong> 범했습니다.
                  </p>
                </div>
                
                <div className="bg-white dark:bg-gray-700 p-4 rounded-lg">
                  <h4 className="font-semibold text-yellow-700 dark:text-yellow-400 mb-2">하버드 비즈니스 스쿨</h4>
                  <p className="text-sm text-yellow-600 dark:text-yellow-300">
                    체험 학습은 이론 학습 대비 <strong>5배 더 오래 기억</strong>되며 실제 행동 변화를 이끌어냅니다.
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* Section 2: Interactive Simulation */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              2️⃣ 실전 시장 상황 시뮬레이션
            </h2>
            
            <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/10 dark:to-pink-900/10 p-6 rounded-xl mb-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center">
                  <Play className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white">15일간의 극한 체험</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">급등, 급락, 패닉 상황에서의 심리적 반응을 테스트하세요</p>
                </div>
              </div>
            </div>

            <MarketSimulation />
          </section>

          {/* Section 3: Psychology Tips */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              3️⃣ 실전 심리 통제 기법
            </h2>
            
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {psychologyTips.map((tip, index) => {
                const Icon = tip.icon;
                return (
                  <div 
                    key={index}
                    className={`p-6 rounded-xl cursor-pointer transition-all duration-300 ${
                      index === currentTip 
                        ? 'bg-blue-50 dark:bg-blue-900/20 ring-2 ring-blue-500' 
                        : 'bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700'
                    }`}
                    onClick={() => setCurrentTip(index)}
                  >
                    <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center mx-auto mb-4">
                      <Icon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                    </div>
                    
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2 text-center">
                      {tip.title}
                    </h3>
                    
                    <p className="text-sm text-gray-600 dark:text-gray-400 text-center mb-3">
                      {tip.description}
                    </p>
                    
                    {index === currentTip && (
                      <div className="mt-4 p-3 bg-blue-100 dark:bg-blue-900/30 rounded text-xs text-blue-700 dark:text-blue-300">
                        💡 <strong>전문가 팁:</strong> {tip.detail}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </section>

          {/* Section 4: Real Practice */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              4️⃣ 실전 적용 가이드
            </h2>
            
            <div className="bg-green-50 dark:bg-green-900/10 p-6 rounded-xl">
              <h3 className="text-lg font-bold text-green-800 dark:text-green-300 mb-6">
                🎯 시뮬레이션에서 실전으로
              </h3>
              
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h4 className="font-semibold text-green-700 dark:text-green-400 mb-4">실전 준비 체크리스트</h4>
                  <div className="space-y-3">
                    <div className="flex items-start gap-2">
                      <input type="checkbox" className="mt-1" />
                      <span className="text-sm text-green-600 dark:text-green-300">
                        투자 규칙서 작성 (언제 사고 팔지)
                      </span>
                    </div>
                    <div className="flex items-start gap-2">
                      <input type="checkbox" className="mt-1" />
                      <span className="text-sm text-green-600 dark:text-green-300">
                        리스크 허용 한도 설정 (계좌의 1-2%)
                      </span>
                    </div>
                    <div className="flex items-start gap-2">
                      <input type="checkbox" className="mt-1" />
                      <span className="text-sm text-green-600 dark:text-green-300">
                        매매일지 양식 준비
                      </span>
                    </div>
                    <div className="flex items-start gap-2">
                      <input type="checkbox" className="mt-1" />
                      <span className="text-sm text-green-600 dark:text-green-300">
                        감정 통제 기법 숙지 (호흡법, 대기법)
                      </span>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-semibold text-green-700 dark:text-green-400 mb-4">첫 달 실전 계획</h4>
                  <div className="space-y-3 text-sm text-green-600 dark:text-green-300">
                    <div className="bg-white dark:bg-gray-700 p-3 rounded">
                      <strong>1주차:</strong> 소액(10만원)으로 시작, 매매 과정 체험
                    </div>
                    <div className="bg-white dark:bg-gray-700 p-3 rounded">
                      <strong>2주차:</strong> 매매일지 작성 습관화, 감정 상태 모니터링  
                    </div>
                    <div className="bg-white dark:bg-gray-700 p-3 rounded">
                      <strong>3주차:</strong> 첫 손실 경험과 대응, 규칙 준수 점검
                    </div>
                    <div className="bg-white dark:bg-gray-700 p-3 rounded">
                      <strong>4주차:</strong> 한 달 성과 분석, 개선점 도출
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="mt-6 p-4 bg-green-100 dark:bg-green-900/20 rounded-lg">
                <p className="text-sm text-green-700 dark:text-green-300">
                  <strong>💡 핵심 포인트:</strong> 시뮬레이션에서 배운 것을 실전에서도 그대로 적용하세요. 
                  처음에는 작은 금액으로 시작해서 감정 통제 능력을 점검하고, 
                  충분히 숙련된 후에 투자 금액을 늘려가는 것이 안전합니다.
                </p>
              </div>
            </div>
          </section>

          {/* Interactive Quiz */}
          <section>
            <QuizSection />
          </section>
        </div>

        {/* Next Steps */}
        <div className="mt-16 bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/10 dark:to-orange-900/10 rounded-2xl p-8">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 bg-white dark:bg-gray-700 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-2xl">📊</span>
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                다음 단계로 진행
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                심리 훈련을 마쳤다면 이제 경제지표 분석 방법을 배워보세요.
              </p>
            </div>
          </div>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                📊 Chapter 7: 주요 경제지표의 이해
              </h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">
                GDP, 금리, 환율, 인플레이션이 주식시장에 미치는 영향을 분석하는 방법을 학습합니다.
              </p>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <Clock className="w-4 h-4" />
                  <span>예상 학습시간: 60분</span>
                </div>
                <Link
                  href="/modules/stock-analysis/chapters/economic-indicators"
                  className="inline-flex items-center gap-1 text-sm font-medium text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors"
                >
                  <span>시작하기</span>
                  <ChevronRight className="w-4 h-4" />
                </Link>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                🏛️ 전체 커리큘럼 보기
              </h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">
                Baby Chick 단계의 전체 학습 경로를 확인하고 나만의 학습 계획을 세워보세요.
              </p>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <Target className="w-4 h-4" />
                  <span>총 9개 챕터</span>
                </div>
                <Link
                  href="/modules/stock-analysis/stages/baby-chick"
                  className="inline-flex items-center gap-1 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                >
                  <span>전체 보기</span>
                  <ChevronRight className="w-4 h-4" />
                </Link>
              </div>
            </div>
          </div>

          {/* Progress Indicator */}
          <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
              <span>Baby Chick 진행률</span>
              <span>6/9 완료</span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div className="bg-gradient-to-r from-yellow-400 to-orange-500 h-2 rounded-full" style={{ width: '67%' }}></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}