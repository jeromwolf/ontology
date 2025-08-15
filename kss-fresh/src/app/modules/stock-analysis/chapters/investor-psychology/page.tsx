'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ArrowLeft, Brain, AlertTriangle, TrendingDown, TrendingUp, Users, Target, Eye, Zap, BarChart3, Clock, ChevronRight, Play, RefreshCw } from 'lucide-react';

function PsychologySimulator() {
  const [scenario, setScenario] = useState(0);
  const [userChoices, setUserChoices] = useState<string[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [currentPrice, setCurrentPrice] = useState(50000);
  const [portfolio, setPortfolio] = useState({ cash: 1000000, shares: 20, avgPrice: 50000 });
  
  const scenarios = [
    {
      id: 'loss-aversion',
      title: '손실 회피 편향',
      situation: '보유한 주식이 매수가 대비 -15% 하락했습니다. 최근 뉴스에서는 해당 기업의 실적 부진이 지속될 것이라고 보도했습니다.',
      currentValue: 42500,
      options: [
        { id: 'hold', text: '더 기다려보자. 언젠가는 오를 것이다', type: 'bias' },
        { id: 'sell', text: '손실을 인정하고 매도한다', type: 'rational' },
        { id: 'buy-more', text: '더 싸게 살 기회다. 추가 매수한다', type: 'bias' }
      ],
      explanation: '손실 회피 편향으로 인해 손실을 확정하기를 거부하고 계속 보유하려는 경향이 있습니다.'
    },
    {
      id: 'confirmation-bias',
      title: '확증 편향',
      situation: '관심 있던 종목을 분석 중입니다. 여러 애널리스트 보고서가 있는데, 일부는 긍정적, 일부는 부정적입니다.',
      currentValue: 55000,
      options: [
        { id: 'positive-only', text: '긍정적인 보고서만 더 자세히 읽어본다', type: 'bias' },
        { id: 'all-reports', text: '긍정적/부정적 보고서를 모두 균형있게 검토한다', type: 'rational' },
        { id: 'ignore-negative', text: '부정적 보고서는 과장된 것 같아서 무시한다', type: 'bias' }
      ],
      explanation: '자신의 기존 믿음을 뒷받침하는 정보만 선택적으로 수집하고 받아들이는 편향입니다.'
    },
    {
      id: 'herd-mentality',
      title: '군중 심리',
      situation: 'SNS와 커뮤니티에서 특정 종목에 대한 매수 추천이 폭발적으로 늘어나고 있습니다. 주가도 3일 연속 상승 중입니다.',
      currentValue: 65000,
      options: [
        { id: 'follow-crowd', text: '모두가 사고 있으니 나도 매수한다', type: 'bias' },
        { id: 'research-first', text: '독자적으로 분석한 후 결정한다', type: 'rational' },
        { id: 'contrarian', text: '너무 많은 사람이 관심을 보이니 오히려 의심스럽다', type: 'rational' }
      ],
      explanation: '다수가 하는 행동을 따라하려는 심리로, 버블 형성의 주요 원인 중 하나입니다.'
    }
  ];

  const handleChoice = (choiceId: string) => {
    const newChoices = [...userChoices];
    newChoices[scenario] = choiceId;
    setUserChoices(newChoices);
  };

  const nextScenario = () => {
    if (scenario < scenarios.length - 1) {
      setScenario(scenario + 1);
    } else {
      setShowResults(true);
    }
  };

  const resetSimulation = () => {
    setScenario(0);
    setUserChoices([]);
    setShowResults(false);
    setPortfolio({ cash: 1000000, shares: 20, avgPrice: 50000 });
  };

  const getRationalChoices = () => {
    return userChoices.filter((choice, index) => {
      const scenario = scenarios[index];
      const option = scenario.options.find(opt => opt.id === choice);
      return option?.type === 'rational';
    }).length;
  };

  if (showResults) {
    const rationalCount = getRationalChoices();
    const biasCount = scenarios.length - rationalCount;
    
    return (
      <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-8">
        <h2 className="text-2xl font-bold mb-6">📊 투자 심리 분석 결과</h2>
        
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-green-50 dark:bg-green-900/10 p-6 rounded-lg">
            <h3 className="text-lg font-bold text-green-700 dark:text-green-400 mb-2">
              합리적 판단
            </h3>
            <div className="text-3xl font-bold text-green-600 dark:text-green-300 mb-2">
              {rationalCount}/3
            </div>
            <p className="text-sm text-green-600 dark:text-green-300">
              감정에 휘둘리지 않고 객관적으로 판단했습니다.
            </p>
          </div>
          
          <div className="bg-red-50 dark:bg-red-900/10 p-6 rounded-lg">
            <h3 className="text-lg font-bold text-red-700 dark:text-red-400 mb-2">
              편향적 판단
            </h3>
            <div className="text-3xl font-bold text-red-600 dark:text-red-300 mb-2">
              {biasCount}/3
            </div>
            <p className="text-sm text-red-600 dark:text-red-300">
              심리적 편향에 영향을 받은 판단이었습니다.
            </p>
          </div>
        </div>

        <div className={`p-6 rounded-lg mb-6 ${
          rationalCount === 3 ? 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-300'
          : rationalCount === 2 ? 'bg-yellow-100 dark:bg-yellow-900/10 text-yellow-700 dark:text-yellow-300'
          : 'bg-red-100 dark:bg-red-900/10 text-red-700 dark:text-red-300'
        }`}>
          <h3 className="font-bold text-lg mb-2">
            {rationalCount === 3 ? '🎉 탁월한 투자 심리!' 
             : rationalCount === 2 ? '😊 양호한 편입니다!' 
             : '⚠️ 주의가 필요합니다!'}
          </h3>
          <p>
            {rationalCount === 3 
              ? '감정 통제가 뛰어나고 객관적 판단력을 갖추고 있습니다. 성공적인 투자를 위한 좋은 기반을 가지고 있습니다.'
              : rationalCount === 2
              ? '대체로 합리적이지만 가끔 감정에 휘둘릴 수 있습니다. 투자 전 항상 한 번 더 생각해보세요.'
              : '감정적 투자로 인한 손실 위험이 높습니다. 투자 규칙을 정하고 엄격히 따르는 연습이 필요합니다.'}
          </p>
        </div>

        <div className="space-y-4 mb-6">
          {scenarios.map((s, index) => (
            <div key={s.id} className="bg-white dark:bg-gray-700 p-4 rounded-lg">
              <h4 className="font-semibold mb-2">{s.title}</h4>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                <p>선택: {s.options.find(opt => opt.id === userChoices[index])?.text}</p>
                <p className="mt-1 text-gray-500 dark:text-gray-400">해설: {s.explanation}</p>
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

  const currentScenario = scenarios[scenario];

  return (
    <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold">🧠 투자 심리 테스트</h2>
        <div className="text-sm text-gray-500">
          {scenario + 1} / {scenarios.length}
        </div>
      </div>

      <div className="mb-6">
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mb-4">
          <div 
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${((scenario + 1) / scenarios.length) * 100}%` }}
          ></div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-700 p-6 rounded-lg mb-6">
        <h3 className="text-xl font-bold mb-4">{currentScenario.title}</h3>
        <div className="bg-gray-50 dark:bg-gray-600 p-4 rounded-lg mb-6">
          <p className="text-gray-700 dark:text-gray-300">{currentScenario.situation}</p>
          
          <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
            <div className="text-center">
              <div className="font-semibold text-gray-500">보유 수량</div>
              <div className="text-lg font-bold">{portfolio.shares}주</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-gray-500">매수 평단</div>
              <div className="text-lg font-bold">{portfolio.avgPrice.toLocaleString()}원</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-gray-500">현재가</div>
              <div className={`text-lg font-bold ${
                currentScenario.currentValue > portfolio.avgPrice ? 'text-red-600' : 'text-blue-600'
              }`}>
                {currentScenario.currentValue.toLocaleString()}원
              </div>
            </div>
          </div>

          <div className="mt-2 text-center">
            <div className={`text-sm font-medium ${
              currentScenario.currentValue > portfolio.avgPrice ? 'text-red-600' : 'text-blue-600'
            }`}>
              {currentScenario.currentValue > portfolio.avgPrice ? '+' : ''}
              {((currentScenario.currentValue - portfolio.avgPrice) / portfolio.avgPrice * 100).toFixed(1)}%
              ({(currentScenario.currentValue - portfolio.avgPrice).toLocaleString()}원)
            </div>
          </div>
        </div>

        <div className="space-y-3">
          {currentScenario.options.map((option) => (
            <button
              key={option.id}
              onClick={() => handleChoice(option.id)}
              className={`w-full p-4 text-left rounded-lg border-2 transition-all ${
                userChoices[scenario] === option.id
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
              }`}
            >
              <div className="flex items-start gap-3">
                <div className={`w-5 h-5 rounded-full border-2 mt-0.5 ${
                  userChoices[scenario] === option.id
                    ? 'border-blue-500 bg-blue-500'
                    : 'border-gray-300 dark:border-gray-600'
                }`}>
                  {userChoices[scenario] === option.id && (
                    <div className="w-1.5 h-1.5 bg-white rounded-full mx-auto mt-1"></div>
                  )}
                </div>
                <span className="text-gray-900 dark:text-gray-100">{option.text}</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="flex justify-end">
        <button
          onClick={nextScenario}
          disabled={!userChoices[scenario]}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
        >
          {scenario < scenarios.length - 1 ? '다음 상황' : '결과 보기'}
        </button>
      </div>
    </div>
  );
}

function QuizSection() {
  const [answers, setAnswers] = useState<{ q1: string; q2: string; q3: string }>({ q1: '', q2: '', q3: '' });
  const [showResults, setShowResults] = useState(false);
  
  const correctAnswers = {
    q1: 'q1-2', // 손실을 회피하려는 심리가 더 강하다
    q2: 'q2-3', // 투자 규칙을 미리 정하고 엄격히 지킨다
    q3: 'q3-1'  // 다수의 의견과 반대로 투자하는 전략
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
          <h3 className="font-semibold mb-3">Q1. 행동재무학에서 말하는 '손실 회피 편향'이란?</h3>
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
                이익을 얻으려는 심리가 손실을 회피하려는 심리보다 강하다{getResultIcon('q1', 'q1-1')}
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
                손실을 회피하려는 심리가 이익을 추구하는 심리보다 강하다{getResultIcon('q1', 'q1-2')}
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
                이익과 손실에 대한 심리적 반응이 동일하다{getResultIcon('q1', 'q1-3')}
              </span>
            </label>
          </div>
        </div>

        <div>
          <h3 className="font-semibold mb-3">Q2. 투자 심리의 함정을 극복하는 가장 효과적인 방법은?</h3>
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
              <span className={getResultStyle('q2', 'q2-1')}>
                감정을 완전히 배제하고 투자한다{getResultIcon('q2', 'q2-1')}
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
                직감을 믿고 빠르게 결정한다{getResultIcon('q2', 'q2-2')}
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
                투자 규칙을 미리 정하고 엄격히 지킨다{getResultIcon('q2', 'q2-3')}
              </span>
            </label>
          </div>
        </div>

        <div>
          <h3 className="font-semibold mb-3">Q3. '역발상 투자(Contrarian Investing)'란?</h3>
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
                다수의 의견과 반대로 투자하는 전략{getResultIcon('q3', 'q3-1')}
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
                시장 흐름을 따라가는 순응 전략{getResultIcon('q3', 'q3-2')}
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
                기술적 분석을 중시하는 전략{getResultIcon('q3', 'q3-3')}
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

export default function InvestorPsychologyPage() {
  const [currentBias, setCurrentBias] = useState(0);

  const biases = [
    {
      id: 'loss-aversion',
      title: '손실 회피 편향 (Loss Aversion)',
      icon: TrendingDown,
      color: 'red',
      description: '같은 크기의 이익보다 손실을 2배 더 크게 느끼는 심리',
      example: '100만원 수익보다 100만원 손실이 심리적으로 2배 더 큰 충격',
      symptoms: [
        '손실 종목을 계속 보유하며 회복을 기다림',
        '수익 종목은 빨리 팔고 싶어함',
        '손절매를 하지 못하고 물타기를 반복'
      ],
      solutions: [
        '손절선을 미리 정하고 기계적으로 실행',
        '포트폴리오 전체 관점에서 판단',
        '매매일지 작성으로 객관적 분석'
      ]
    },
    {
      id: 'confirmation-bias',
      title: '확증 편향 (Confirmation Bias)',
      icon: Eye,
      color: 'orange',
      description: '자신의 믿음을 뒷받침하는 정보만 선택적으로 수집하고 해석',
      example: '보유 종목에 대한 긍정적 뉴스만 찾고 부정적 소식은 무시',
      symptoms: [
        '반대 의견을 무시하거나 폄하',
        '같은 생각을 가진 사람들과만 소통',
        '부정적 신호를 합리화하려고 함'
      ],
      solutions: [
        '의도적으로 반대 의견 찾아보기',
        '투자 논리의 약점 먼저 분석',
        '다양한 관점의 전문가 의견 수집'
      ]
    },
    {
      id: 'herd-mentality',
      title: '군중 심리 (Herd Mentality)',
      icon: Users,
      color: 'purple',
      description: '다수가 하는 행동을 따라하려는 강한 심리적 충동',
      example: '주변에서 모두 특정 종목을 매수한다고 하면 나도 따라 매수',
      symptoms: [
        'SNS나 커뮤니티 추천 종목 맹신',
        '유명인이나 전문가 말만 듣고 투자',
        '시장 분위기에 쉽게 휩쓸림'
      ],
      solutions: [
        '독립적 사고와 분석 능력 기르기',
        '시장 과열시 오히려 신중해지기',
        '자신만의 투자 철학 확립'
      ]
    },
    {
      id: 'anchoring-bias',
      title: '앵커링 편향 (Anchoring Bias)',
      icon: Target,
      color: 'blue',
      description: '처음 접한 정보나 가격을 기준점으로 고착되는 현상',
      example: '52주 최고가를 기준으로 현재 주가를 저평가된 것으로 착각',
      symptoms: [
        '매수 가격에 집착해서 판단 흐림',
        '과거 고점 대비로만 주가 평가',
        '첫 정보에만 의존하는 성향'
      ],
      solutions: [
        '현재 기업 가치로만 판단하기',
        '과거 가격 정보 의도적으로 무시',
        '여러 밸류에이션 방법 동시 적용'
      ]
    },
    {
      id: 'overconfidence',
      title: '과신 편향 (Overconfidence Bias)',
      icon: Zap,
      color: 'green',
      description: '자신의 능력이나 판단을 과도하게 신뢰하는 경향',
      example: '몇 번의 성공 경험으로 자신이 투자 천재라고 착각',
      symptoms: [
        '리스크를 과소평가',
        '과도한 레버리지 사용',
        '빈번한 매매로 수수료 증가'
      ],
      solutions: [
        '겸손한 마음가짐 유지',
        '분산투자로 리스크 관리',
        '실수와 실패로부터 학습'
      ]
    }
  ];

  const getColorClasses = (color: string) => {
    const colors = {
      red: 'border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/10',
      orange: 'border-orange-200 dark:border-orange-800 bg-orange-50 dark:bg-orange-900/10',
      purple: 'border-purple-200 dark:border-purple-800 bg-purple-50 dark:bg-purple-900/10',
      blue: 'border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/10',
      green: 'border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/10'
    };
    return colors[color as keyof typeof colors] || colors.blue;
  };

  const getIconColorClasses = (color: string) => {
    const colors = {
      red: 'text-red-600 dark:text-red-400',
      orange: 'text-orange-600 dark:text-orange-400',
      purple: 'text-purple-600 dark:text-purple-400',
      blue: 'text-blue-600 dark:text-blue-400',
      green: 'text-green-600 dark:text-green-400'
    };
    return colors[color as keyof typeof colors] || colors.blue;
  };

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
              <Brain className="w-8 h-8 text-yellow-600 dark:text-yellow-400" />
            </div>
            <div className="text-left">
              <div className="text-sm text-gray-500 mb-1">Baby Chick • Chapter 4</div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                투자자 심리의 함정
              </h1>
            </div>
          </div>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
            인간의 심리적 편향이 투자에 미치는 영향을 이해하고, 감정적 결정을 피하는 방법을 학습합니다.
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
              <span>행동재무학의 주요 심리적 편향 이해</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold mt-1">•</span>
              <span>투자 실패의 심리적 원인 파악</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold mt-1">•</span>
              <span>감정적 투자를 피하는 실전 방법 학습</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold mt-1">•</span>
              <span>체계적이고 규칙적인 투자 습관 형성</span>
            </li>
          </ul>
        </div>

        {/* Main Content */}
        <div className="space-y-12">
          {/* Introduction */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              1️⃣ 왜 90%의 개인투자자가 실패할까?
            </h2>
            
            <div className="bg-red-50 dark:bg-red-900/10 p-6 rounded-xl mb-8">
              <h3 className="text-xl font-bold text-red-800 dark:text-red-300 mb-4">
                <AlertTriangle className="inline w-6 h-6 mr-2" />
                충격적인 통계
              </h3>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold text-red-700 dark:text-red-400 mb-2">개인투자자 수익률</h4>
                  <ul className="space-y-2 text-sm text-red-600 dark:text-red-300">
                    <li>• 연평균 수익률: <strong>-1.5%</strong></li>
                    <li>• 시장 평균 대비: <strong>-4.2%p</strong> 저조</li>
                    <li>• 10년 이상 보유 비율: <strong>5% 미만</strong></li>
                    <li>• 손실 경험률: <strong>87%</strong></li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold text-red-700 dark:text-red-400 mb-2">주요 실패 원인</h4>
                  <ul className="space-y-2 text-sm text-red-600 dark:text-red-300">
                    <li>• <strong>감정적 거래:</strong> 두려움과 탐욕</li>
                    <li>• <strong>타이밍 오판:</strong> 고점 매수, 저점 매도</li>
                    <li>• <strong>정보 편향:</strong> 선택적 정보 수집</li>
                    <li>• <strong>과도한 거래:</strong> 빈번한 매매</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="prose prose-lg dark:prose-invert max-w-none">
              <p>
                투자의 실패는 대부분 <strong>기술적 무능력</strong>보다는 <strong>심리적 편향</strong>에서 비롯됩니다. 
                노벨경제학상을 수상한 다니엘 카네만과 아모스 트베르스키의 연구에 따르면, 
                인간은 투자 결정을 내릴 때 합리적이지 못한 여러 편향을 보입니다.
              </p>

              <p>
                행동재무학(Behavioral Finance)은 이러한 심리적 편향이 금융 시장에 미치는 영향을 연구하는 학문입니다. 
                이 학문이 밝혀낸 핵심 통찰은 "<strong>시장은 효율적이지 않으며, 
                투자자들의 비합리적 행동이 시장 가격을 왜곡시킨다</strong>"는 것입니다.
              </p>
            </div>
          </section>

          {/* Psychology Biases */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              2️⃣ 5대 투자 심리 편향
            </h2>
            
            <div className="grid gap-6">
              {biases.map((bias, index) => {
                const Icon = bias.icon;
                return (
                  <div 
                    key={bias.id} 
                    className={`border-2 rounded-xl p-6 transition-all duration-300 ${getColorClasses(bias.color)} ${
                      index === currentBias ? 'ring-2 ring-blue-500' : ''
                    }`}
                  >
                    <div className="flex items-start gap-4 mb-4">
                      <div className="w-12 h-12 bg-white dark:bg-gray-700 rounded-full flex items-center justify-center shadow-md">
                        <Icon className={`w-6 h-6 ${getIconColorClasses(bias.color)}`} />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                          {bias.title}
                        </h3>
                        <p className="text-gray-600 dark:text-gray-400">
                          {bias.description}
                        </p>
                      </div>
                      <button
                        onClick={() => setCurrentBias(index === currentBias ? -1 : index)}
                        className="p-2 hover:bg-white dark:hover:bg-gray-700 rounded-lg transition-colors"
                      >
                        <ChevronRight className={`w-5 h-5 transition-transform ${
                          index === currentBias ? 'rotate-90' : ''
                        }`} />
                      </button>
                    </div>

                    {index === currentBias && (
                      <div className="border-t border-gray-200 dark:border-gray-600 pt-6 space-y-4">
                        <div className="bg-white dark:bg-gray-700 p-4 rounded-lg">
                          <h4 className="font-semibold text-gray-900 dark:text-white mb-2">📝 실제 사례</h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            {bias.example}
                          </p>
                        </div>

                        <div className="grid md:grid-cols-2 gap-4">
                          <div className="bg-white dark:bg-gray-700 p-4 rounded-lg">
                            <h4 className="font-semibold text-red-700 dark:text-red-400 mb-3">⚠️ 주요 증상</h4>
                            <ul className="space-y-1 text-sm">
                              {bias.symptoms.map((symptom, idx) => (
                                <li key={idx} className="text-gray-600 dark:text-gray-400">
                                  • {symptom}
                                </li>
                              ))}
                            </ul>
                          </div>

                          <div className="bg-white dark:bg-gray-700 p-4 rounded-lg">
                            <h4 className="font-semibold text-green-700 dark:text-green-400 mb-3">💡 극복 방법</h4>
                            <ul className="space-y-1 text-sm">
                              {bias.solutions.map((solution, idx) => (
                                <li key={idx} className="text-gray-600 dark:text-gray-400">
                                  • {solution}
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </section>

          {/* Interactive Simulation */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              3️⃣ 심리 편향 체험 시뮬레이션
            </h2>
            
            <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/10 dark:to-pink-900/10 p-6 rounded-xl mb-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center">
                  <Play className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white">실전 상황 체험</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">실제 투자 상황에서 어떤 선택을 하는지 테스트해보세요</p>
                </div>
              </div>
            </div>

            <PsychologySimulator />
          </section>

          {/* Practical Solutions */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              4️⃣ 심리적 함정을 극복하는 실전 전략
            </h2>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-green-50 dark:bg-green-900/10 p-6 rounded-xl">
                <h3 className="text-lg font-bold text-green-800 dark:text-green-300 mb-4">
                  <BarChart3 className="inline w-5 h-5 mr-2" />
                  체계적 투자 규칙 수립
                </h3>
                
                <div className="space-y-3 text-sm text-green-700 dark:text-green-300">
                  <div className="bg-white dark:bg-gray-700 p-3 rounded">
                    <strong>1. 투자 목적 명확화</strong>
                    <p className="text-xs mt-1 text-gray-600 dark:text-gray-400">
                      "언제까지, 얼마의 수익을 목표로 하는가?"
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-700 p-3 rounded">
                    <strong>2. 리스크 허용범위 설정</strong>
                    <p className="text-xs mt-1 text-gray-600 dark:text-gray-400">
                      "최대 몇 %까지 손실을 감당할 수 있는가?"
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-700 p-3 rounded">
                    <strong>3. 매수/매도 기준 수립</strong>
                    <p className="text-xs mt-1 text-gray-600 dark:text-gray-400">
                      "어떤 조건에서 사고 팔 것인가?"
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/10 p-6 rounded-xl">
                <h3 className="text-lg font-bold text-blue-800 dark:text-blue-300 mb-4">
                  <RefreshCw className="inline w-5 h-5 mr-2" />
                  감정 제어 기법
                </h3>
                
                <div className="space-y-3 text-sm text-blue-700 dark:text-blue-300">
                  <div className="bg-white dark:bg-gray-700 p-3 rounded">
                    <strong>6-3-5 호흡법</strong>
                    <p className="text-xs mt-1 text-gray-600 dark:text-gray-400">
                      6초 흡입 - 3초 멈춤 - 5초 호흡으로 감정 진정
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-700 p-3 rounded">
                    <strong>24시간 대기 원칙</strong>
                    <p className="text-xs mt-1 text-gray-600 dark:text-gray-400">
                      충동적 결정 대신 하루 후 재검토
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-700 p-3 rounded">
                    <strong>매매일지 작성</strong>
                    <p className="text-xs mt-1 text-gray-600 dark:text-gray-400">
                      결정 이유와 감정 상태 기록으로 객관화
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/10 p-6 rounded-xl mt-6">
              <h3 className="text-lg font-bold text-yellow-800 dark:text-yellow-300 mb-4">
                🎯 성공하는 투자자의 5가지 습관
              </h3>
              
              <div className="grid md:grid-cols-5 gap-4 text-sm">
                <div className="text-center">
                  <div className="w-12 h-12 bg-yellow-200 dark:bg-yellow-900/30 rounded-full flex items-center justify-center mx-auto mb-2">
                    <span className="text-xl">📋</span>
                  </div>
                  <div className="font-semibold text-yellow-700 dark:text-yellow-400">계획적 투자</div>
                  <p className="text-xs text-yellow-600 dark:text-yellow-300 mt-1">
                    사전에 세운 규칙 엄격히 준수
                  </p>
                </div>
                
                <div className="text-center">
                  <div className="w-12 h-12 bg-yellow-200 dark:bg-yellow-900/30 rounded-full flex items-center justify-center mx-auto mb-2">
                    <span className="text-xl">🔄</span>
                  </div>
                  <div className="font-semibold text-yellow-700 dark:text-yellow-400">분산 투자</div>
                  <p className="text-xs text-yellow-600 dark:text-yellow-300 mt-1">
                    한 곳에 모든 계란을 담지 않기
                  </p>
                </div>
                
                <div className="text-center">
                  <div className="w-12 h-12 bg-yellow-200 dark:bg-yellow-900/30 rounded-full flex items-center justify-center mx-auto mb-2">
                    <span className="text-xl">⏰</span>
                  </div>
                  <div className="font-semibold text-yellow-700 dark:text-yellow-400">장기 관점</div>
                  <p className="text-xs text-yellow-600 dark:text-yellow-300 mt-1">
                    단기 변동에 흔들리지 않기
                  </p>
                </div>
                
                <div className="text-center">
                  <div className="w-12 h-12 bg-yellow-200 dark:bg-yellow-900/30 rounded-full flex items-center justify-center mx-auto mb-2">
                    <span className="text-xl">📚</span>
                  </div>
                  <div className="font-semibold text-yellow-700 dark:text-yellow-400">지속적 학습</div>
                  <p className="text-xs text-yellow-600 dark:text-yellow-300 mt-1">
                    실수로부터 배우고 개선
                  </p>
                </div>
                
                <div className="text-center">
                  <div className="w-12 h-12 bg-yellow-200 dark:bg-yellow-900/30 rounded-full flex items-center justify-center mx-auto mb-2">
                    <span className="text-xl">🧘</span>
                  </div>
                  <div className="font-semibold text-yellow-700 dark:text-yellow-400">감정 절제</div>
                  <p className="text-xs text-yellow-600 dark:text-yellow-300 mt-1">
                    탐욕과 공포에서 자유롭기
                  </p>
                </div>
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
              <span className="text-2xl">🛡️</span>
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                다음 단계로 진행
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                투자 심리를 이해했다면 이제 리스크 관리의 기초를 배워보세요.
              </p>
            </div>
          </div>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                🛡️ Chapter 5: 리스크 관리 기초
              </h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">
                손절선 설정, 포지션 사이징, 분산투자 등 투자 리스크를 체계적으로 관리하는 방법을 학습합니다.
              </p>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <Clock className="w-4 h-4" />
                  <span>예상 학습시간: 45분</span>
                </div>
                <Link
                  href="/modules/stock-analysis/chapters/risk-management-basics"
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
                  href="/modules/stock-analysis/stages/foundation"
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
              <span>4/9 완료</span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div className="bg-gradient-to-r from-yellow-400 to-orange-500 h-2 rounded-full" style={{ width: '44%' }}></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}