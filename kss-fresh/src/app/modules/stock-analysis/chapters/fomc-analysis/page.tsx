'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ArrowLeft, Building2, Globe, TrendingUp, TrendingDown, Calendar, Clock, Target, ChevronRight, AlertTriangle, DollarSign, BarChart3, Users, Zap, Activity } from 'lucide-react';
import ChapterNavigation from '../../components/ChapterNavigation';

interface PolicyScenario {
  title: string;
  description: string;
  fedAction: 'raise' | 'hold' | 'cut';
  bokAction: 'follow' | 'diverge' | 'wait';
  marketImpact: {
    stocks: number;
    bonds: number;
    currency: number;
    sectors: { [key: string]: number };
  };
}

function MonetaryPolicySimulator() {
  const [selectedScenario, setSelectedScenario] = useState<string>('aggressive_hike');
  const [timeHorizon, setTimeHorizon] = useState<'immediate' | 'short' | 'long'>('immediate');
  
  const scenarios: { [key: string]: PolicyScenario } = {
    aggressive_hike: {
      title: '🔥 공격적 금리 인상',
      description: '인플레이션 억제를 위한 연준의 0.75%p 대폭 인상',
      fedAction: 'raise',
      bokAction: 'follow',
      marketImpact: {
        stocks: -8.5,
        bonds: 3.2,
        currency: 12.5,
        sectors: {
          '금융': 15.2,
          '기술': -12.8,
          '부동산': -18.5,
          '에너지': -5.2,
          '소비재': -8.1
        }
      }
    },
    gradual_hike: {
      title: '📈 점진적 금리 인상',
      description: '경제 안정성을 고려한 연준의 0.25%p 소폭 인상',
      fedAction: 'raise',
      bokAction: 'follow',
      marketImpact: {
        stocks: -3.2,
        bonds: 1.8,
        currency: 4.5,
        sectors: {
          '금융': 8.5,
          '기술': -4.2,
          '부동산': -6.8,
          '에너지': -1.5,
          '소비재': -2.1
        }
      }
    },
    hold_pattern: {
      title: '⏸️ 금리 동결',
      description: '경기 상황 관망을 위한 연준의 현 수준 유지',
      fedAction: 'hold',
      bokAction: 'wait',
      marketImpact: {
        stocks: 1.2,
        bonds: -0.5,
        currency: -1.8,
        sectors: {
          '금융': -2.1,
          '기술': 3.5,
          '부동산': 2.8,
          '에너지': 1.2,
          '소비재': 1.8
        }
      }
    },
    emergency_cut: {
      title: '🚨 긴급 금리 인하',
      description: '경기 침체 우려로 인한 연준의 0.5%p 긴급 인하',
      fedAction: 'cut',
      bokAction: 'follow',
      marketImpact: {
        stocks: 12.8,
        bonds: -2.5,
        currency: -8.2,
        sectors: {
          '금융': -8.5,
          '기술': 18.2,
          '부동산': 25.1,
          '에너지': 8.5,
          '소비재': 12.1
        }
      }
    },
    divergence: {
      title: '🌏 통화정책 분기',
      description: '연준은 인상, 한국은행은 현 수준 유지로 정책 분기',
      fedAction: 'raise',
      bokAction: 'diverge',
      marketImpact: {
        stocks: -5.8,
        bonds: 2.1,
        currency: -3.5, // 원화 상대적 약세
        sectors: {
          '금융': 3.2,
          '기술': -8.5,
          '부동산': -12.1,
          '에너지': 5.8, // 수출 유리
          '소비재': -3.2
        }
      }
    }
  };

  const currentScenario = scenarios[selectedScenario];
  
  const getTimeImpactMultiplier = () => {
    switch (timeHorizon) {
      case 'immediate': return 1.0;
      case 'short': return 0.7;
      case 'long': return 0.4;
      default: return 1.0;
    }
  };

  const adjustedImpact = {
    ...currentScenario.marketImpact,
    stocks: currentScenario.marketImpact.stocks * getTimeImpactMultiplier(),
    bonds: currentScenario.marketImpact.bonds * getTimeImpactMultiplier(),
    currency: currentScenario.marketImpact.currency * getTimeImpactMultiplier()
  };

  return (
    <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-8">
      <h2 className="text-2xl font-bold mb-6">🏛️ 통화정책 영향 분석기</h2>
      
      <div className="grid lg:grid-cols-2 gap-8">
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-700 p-6 rounded-lg">
            <h3 className="text-lg font-semibold mb-4">정책 시나리오 선택</h3>
            
            <div className="space-y-3">
              {Object.entries(scenarios).map(([key, scenario]) => (
                <button
                  key={key}
                  onClick={() => setSelectedScenario(key)}
                  className={`w-full p-4 text-left rounded-lg border-2 transition-all ${
                    selectedScenario === key
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-600 hover:border-gray-300'
                  }`}
                >
                  <div className="font-medium mb-1">{scenario.title}</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    {scenario.description}
                  </div>
                  <div className="flex items-center gap-4 mt-2 text-xs">
                    <span className={`px-2 py-1 rounded ${
                      scenario.fedAction === 'raise' ? 'bg-red-100 text-red-700' :
                      scenario.fedAction === 'cut' ? 'bg-green-100 text-green-700' :
                      'bg-gray-100 text-gray-700'
                    }`}>
                      Fed: {scenario.fedAction === 'raise' ? '인상' : scenario.fedAction === 'cut' ? '인하' : '동결'}
                    </span>
                    <span className={`px-2 py-1 rounded ${
                      scenario.bokAction === 'follow' ? 'bg-blue-100 text-blue-700' :
                      scenario.bokAction === 'diverge' ? 'bg-orange-100 text-orange-700' :
                      'bg-gray-100 text-gray-700'
                    }`}>
                      BOK: {scenario.bokAction === 'follow' ? '동조' : scenario.bokAction === 'diverge' ? '분기' : '관망'}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="bg-white dark:bg-gray-700 p-6 rounded-lg">
            <h3 className="text-lg font-semibold mb-4">분석 기간</h3>
            
            <div className="grid grid-cols-3 gap-2">
              {[
                { key: 'immediate', label: '즉시 반응', desc: '발표 후 1주일' },
                { key: 'short', label: '단기 영향', desc: '발표 후 1개월' },
                { key: 'long', label: '장기 영향', desc: '발표 후 3개월' }
              ].map((period) => (
                <button
                  key={period.key}
                  onClick={() => setTimeHorizon(period.key as any)}
                  className={`p-3 rounded-lg text-center transition-all ${
                    timeHorizon === period.key
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-100 dark:bg-gray-600 hover:bg-gray-200 dark:hover:bg-gray-500'
                  }`}
                >
                  <div className="text-sm font-medium">{period.label}</div>
                  <div className="text-xs opacity-80">{period.desc}</div>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-700 p-6 rounded-lg">
            <h3 className="text-lg font-semibold mb-4">시장 영향 예측</h3>
            
            <div className="space-y-4">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium flex items-center gap-2">
                    <BarChart3 className="w-4 h-4" />
                    주식시장 (KOSPI)
                  </span>
                  <span className={`text-sm font-bold ${
                    adjustedImpact.stocks > 0 
                      ? 'text-red-600 dark:text-red-400' 
                      : adjustedImpact.stocks < 0 
                      ? 'text-blue-600 dark:text-blue-400'
                      : 'text-gray-500'
                  }`}>
                    {adjustedImpact.stocks > 0 ? '+' : ''}{adjustedImpact.stocks.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-600 h-3 rounded">
                  <div 
                    className={`h-3 rounded transition-all ${
                      adjustedImpact.stocks > 0 ? 'bg-red-500' : 'bg-blue-500'
                    }`}
                    style={{ 
                      width: `${Math.min(Math.abs(adjustedImpact.stocks) * 3, 100)}%` 
                    }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium flex items-center gap-2">
                    <Activity className="w-4 h-4" />
                    채권시장 (국고채 10년)
                  </span>
                  <span className={`text-sm font-bold ${
                    adjustedImpact.bonds > 0 
                      ? 'text-red-600 dark:text-red-400' 
                      : adjustedImpact.bonds < 0 
                      ? 'text-blue-600 dark:text-blue-400'
                      : 'text-gray-500'
                  }`}>
                    {adjustedImpact.bonds > 0 ? '+' : ''}{adjustedImpact.bonds.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-600 h-3 rounded">
                  <div 
                    className={`h-3 rounded transition-all ${
                      adjustedImpact.bonds > 0 ? 'bg-red-500' : 'bg-blue-500'
                    }`}
                    style={{ 
                      width: `${Math.min(Math.abs(adjustedImpact.bonds) * 8, 100)}%` 
                    }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium flex items-center gap-2">
                    <DollarSign className="w-4 h-4" />
                    원달러 환율
                  </span>
                  <span className={`text-sm font-bold ${
                    adjustedImpact.currency > 0 
                      ? 'text-red-600 dark:text-red-400' 
                      : adjustedImpact.currency < 0 
                      ? 'text-blue-600 dark:text-blue-400'
                      : 'text-gray-500'
                  }`}>
                    {adjustedImpact.currency > 0 ? '+' : ''}{adjustedImpact.currency.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-600 h-3 rounded">
                  <div 
                    className={`h-3 rounded transition-all ${
                      adjustedImpact.currency > 0 ? 'bg-red-500' : 'bg-blue-500'
                    }`}
                    style={{ 
                      width: `${Math.min(Math.abs(adjustedImpact.currency) * 2, 100)}%` 
                    }}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-700 p-6 rounded-lg">
            <h3 className="text-lg font-semibold mb-4">섹터별 영향</h3>
            
            <div className="space-y-2">
              {Object.entries(currentScenario.marketImpact.sectors).map(([sector, impact]) => (
                <div key={sector} className="flex justify-between items-center py-1">
                  <span className="text-sm">{sector}</span>
                  <span className={`text-sm font-medium ${
                    impact > 0 ? 'text-red-600 dark:text-red-400' : 'text-blue-600 dark:text-blue-400'
                  }`}>
                    {impact > 0 ? '+' : ''}{impact.toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className={`p-4 rounded-lg ${
            adjustedImpact.stocks > 0 
              ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300'
              : 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300'
          }`}>
            <h4 className="font-semibold mb-2">💡 투자 전략</h4>
            <p className="text-sm">
              {selectedScenario === 'aggressive_hike' && 
                "공격적 인상 시기에는 은행주 매수, 성장주 비중 축소를 고려하세요."}
              {selectedScenario === 'gradual_hike' && 
                "점진적 인상은 금융주에 유리하되, 급락보다는 조정 관점에서 접근하세요."}
              {selectedScenario === 'hold_pattern' && 
                "금리 동결 시에는 현 포지션 유지하며 다음 정책 방향을 관망하세요."}
              {selectedScenario === 'emergency_cut' && 
                "긴급 인하는 성장주 대상승의 기회. 부동산 리츠도 주목하세요."}
              {selectedScenario === 'divergence' && 
                "정책 분기 시에는 환율 변동성 확대에 주의하며 수출주를 주목하세요."}
            </p>
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
    q1: 'q1-2', // FOMC는 연 8회 개최되며, 금리 결정과 향후 정책 방향을 제시한다
    q2: 'q2-3', // 한국은행이 미국과 다른 정책을 펼칠 때 환율 변동성이 확대된다
    q3: 'q3-1'  // 연준의 매파적 발언은 금리 인상 가능성을 시사하며 주식시장에 부담을 준다
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
          <h3 className="font-semibold mb-3">Q1. FOMC(연방공개시장위원회)에 대한 설명으로 옳은 것은?</h3>
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
                매월 개최되며 미국 경제정책을 전반적으로 결정한다{getResultIcon('q1', 'q1-1')}
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
                연 8회 개최되며, 금리 결정과 향후 정책 방향을 제시한다{getResultIcon('q1', 'q1-2')}
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
                분기별로 개최되며 주로 은행 규제를 담당한다{getResultIcon('q1', 'q1-3')}
              </span>
            </label>
          </div>
        </div>

        <div>
          <h3 className="font-semibold mb-3">Q2. 한미 통화정책 분기 시 나타나는 현상으로 가장 적절한 것은?</h3>
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
                양국 주식시장이 동조화 현상을 보인다{getResultIcon('q2', 'q2-1')}
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
                환율이 안정화되고 자본 유출입이 줄어든다{getResultIcon('q2', 'q2-2')}
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
                환율 변동성이 확대되고 자본 이동이 활발해진다{getResultIcon('q2', 'q2-3')}
              </span>
            </label>
          </div>
        </div>

        <div>
          <h3 className="font-semibold mb-3">Q3. 연준 의장의 '매파적(Hawkish)' 발언이 시장에 미치는 영향은?</h3>
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
                금리 인상 가능성 ↑ → 주식 부담 ↑ → 달러 강세{getResultIcon('q3', 'q3-1')}
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
                금리 인하 기대 ↑ → 주식 상승 → 달러 약세{getResultIcon('q3', 'q3-2')}
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
                시장에 별다른 영향을 미치지 않는다{getResultIcon('q3', 'q3-3')}
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

export default function FomcAnalysisPage() {
  const [selectedInstitution, setSelectedInstitution] = useState<'fed' | 'bok'>('fed');

  const fedInfo = {
    fullName: 'Federal Reserve System',
    chairman: '제롬 파월 (Jerome Powell)',
    established: '1913년',
    mandate: '물가안정과 완전고용',
    keyRate: 'Federal Funds Rate',
    meetings: '연 8회 (6주마다)',
    influence: '글로벌 기축통화국으로 전세계 금융시장에 막대한 영향',
    recentActions: [
      { date: '2024.03', action: '5.25-5.50% 동결', impact: '인플레이션 둔화 관망' },
      { date: '2024.01', action: '5.25-5.50% 동결', impact: '고금리 정책 지속' },
      { date: '2023.12', action: '5.25-5.50% 동결', impact: '매파적 스탠스 유지' }
    ]
  };

  const bokInfo = {
    fullName: '한국은행 (Bank of Korea)',
    chairman: '이창용 총재',
    established: '1950년',
    mandate: '물가안정을 통한 국민경제 건전한 발전',
    keyRate: '기준금리 (Base Rate)',
    meetings: '연 8회 통화정책회의',
    influence: '한국 금융시장과 원화 가치에 직접적 영향',
    recentActions: [
      { date: '2024.02', action: '3.50% 동결', impact: '경기 둔화 우려로 완화적 기조' },
      { date: '2024.01', action: '3.50% 동결', impact: '부동산 시장 안정화 관망' },
      { date: '2023.11', action: '3.50% 동결', impact: '인플레이션 둔화 추세' }
    ]
  };

  const currentInfo = selectedInstitution === 'fed' ? fedInfo : bokInfo;

  const policyTools = [
    {
      name: '기준금리 조정',
      description: '시장 유동성과 차입비용을 직접 조절',
      mechanism: '금리 ↑ → 자금조달 비용 ↑ → 투자/소비 ↓ → 경기 진정',
      effectiveness: '즉각적이고 강력한 효과',
      examples: ['2008년: 0-0.25%로 제로금리', '2022년: 인플레이션 대응 급속 인상']
    },
    {
      name: '양적완화 (QE)',
      description: '장기 국채 매입을 통한 시장 유동성 공급',
      mechanism: '국채 매입 → 시장에 현금 공급 → 금리 하락 → 투자 활성화',
      effectiveness: '금리 정책 한계시 사용하는 비전통적 수단',
      examples: ['QE1,2,3 (2008-2014)', '코로나19 대응 무제한 QE (2020)']
    },
    {
      name: '포워드 가이던스',
      description: '향후 정책 방향에 대한 의사소통',
      mechanism: '미래 정책 신호 → 시장 기대 조정 → 장기금리 영향',
      effectiveness: '실제 정책 변경 없이도 시장에 영향',
      examples: ['Dot Plot (금리 전망 점도표)', '성명서 문구 변화']
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
              <Building2 className="w-8 h-8 text-yellow-600 dark:text-yellow-400" />
            </div>
            <div className="text-left">
              <div className="text-sm text-gray-500 mb-1">Foundation Program • Chapter 8</div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                FOMC와 한국은행 통화정책
              </h1>
            </div>
          </div>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
            미국 연준과 한국은행의 통화정책 결정 과정을 이해하고, 정책 변화가 글로벌 금융시장에 미치는 영향을 분석해보세요.
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
              <span>FOMC와 한국은행 통화정책회의 이해</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold mt-1">•</span>
              <span>통화정책 도구와 전달 메커니즘 학습</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold mt-1">•</span>
              <span>정책 분기 시 시장 영향 분석</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold mt-1">•</span>
              <span>통화정책 기반 투자 전략 수립</span>
            </li>
          </ul>
        </div>

        {/* Main Content */}
        <div className="space-y-12">
          {/* Section 1: Central Bank Comparison */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              1️⃣ 연준 vs 한국은행 비교
            </h2>
            
            <div className="flex gap-2 mb-6">
              <button
                onClick={() => setSelectedInstitution('fed')}
                className={`px-4 py-2 rounded-lg font-medium transition-all ${
                  selectedInstitution === 'fed'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                🇺🇸 미국 연준 (Fed)
              </button>
              <button
                onClick={() => setSelectedInstitution('bok')}
                className={`px-4 py-2 rounded-lg font-medium transition-all ${
                  selectedInstitution === 'bok'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                🇰🇷 한국은행 (BOK)
              </button>
            </div>

            <div className={`rounded-xl p-6 mb-8 ${
              selectedInstitution === 'fed' 
                ? 'bg-blue-50 dark:bg-blue-900/10' 
                : 'bg-green-50 dark:bg-green-900/10'
            }`}>
              <div className="flex items-center gap-3 mb-4">
                <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                  selectedInstitution === 'fed'
                    ? 'bg-blue-100 dark:bg-blue-900/30'
                    : 'bg-green-100 dark:bg-green-900/30'
                }`}>
                  <Globe className={`w-6 h-6 ${
                    selectedInstitution === 'fed'
                      ? 'text-blue-600 dark:text-blue-400'
                      : 'text-green-600 dark:text-green-400'
                  }`} />
                </div>
                <div>
                  <h3 className={`text-xl font-bold ${
                    selectedInstitution === 'fed'
                      ? 'text-blue-800 dark:text-blue-300'
                      : 'text-green-800 dark:text-green-300'
                  }`}>
                    {currentInfo.fullName}
                  </h3>
                  <p className={`text-sm ${
                    selectedInstitution === 'fed'
                      ? 'text-blue-600 dark:text-blue-400'
                      : 'text-green-600 dark:text-green-400'
                  }`}>
                    의장: {currentInfo.chairman}
                  </p>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-700 p-4 rounded-lg">
                    <h4 className="font-semibold mb-2">설립 연도</h4>
                    <p className="text-gray-600 dark:text-gray-400">{currentInfo.established}</p>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-700 p-4 rounded-lg">
                    <h4 className="font-semibold mb-2">정책 목표</h4>
                    <p className="text-gray-600 dark:text-gray-400">{currentInfo.mandate}</p>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-700 p-4 rounded-lg">
                    <h4 className="font-semibold mb-2">주요 금리</h4>
                    <p className="text-gray-600 dark:text-gray-400">{currentInfo.keyRate}</p>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-700 p-4 rounded-lg">
                    <h4 className="font-semibold mb-2">회의 주기</h4>
                    <p className="text-gray-600 dark:text-gray-400">{currentInfo.meetings}</p>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-700 p-4 rounded-lg">
                    <h4 className="font-semibold mb-2">시장 영향력</h4>
                    <p className="text-gray-600 dark:text-gray-400">{currentInfo.influence}</p>
                  </div>
                </div>
              </div>

              <div className="mt-6">
                <h4 className="font-semibold mb-3">최근 주요 정책</h4>
                <div className="space-y-2">
                  {currentInfo.recentActions.map((action, index) => (
                    <div key={index} className="bg-white dark:bg-gray-700 p-3 rounded flex justify-between items-start">
                      <div>
                        <span className="font-medium">{action.date}</span>
                        <span className="ml-3 text-gray-600 dark:text-gray-400">{action.action}</span>
                      </div>
                      <span className="text-sm text-gray-500 ml-4">{action.impact}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>

          {/* Section 2: Policy Tools */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              2️⃣ 통화정책 도구
            </h2>
            
            <div className="grid gap-6">
              {policyTools.map((tool, index) => (
                <div key={index} className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-6">
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
                    {index + 1}. {tool.name}
                  </h3>
                  
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    {tool.description}
                  </p>

                  <div className="grid md:grid-cols-2 gap-4 mb-4">
                    <div className="bg-blue-50 dark:bg-blue-900/10 p-4 rounded-lg">
                      <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">
                        작동 메커니즘
                      </h4>
                      <p className="text-sm text-blue-700 dark:text-blue-300">
                        {tool.mechanism}
                      </p>
                    </div>

                    <div className="bg-green-50 dark:bg-green-900/10 p-4 rounded-lg">
                      <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2">
                        효과성
                      </h4>
                      <p className="text-sm text-green-700 dark:text-green-300">
                        {tool.effectiveness}
                      </p>
                    </div>
                  </div>

                  <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                    <h4 className="font-semibold text-gray-800 dark:text-gray-300 mb-2">
                      역사적 사례
                    </h4>
                    <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                      {tool.examples.map((example, idx) => (
                        <li key={idx}>• {example}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Section 3: Policy Impact Simulator */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              3️⃣ 통화정책 영향 시뮬레이터
            </h2>
            
            <MonetaryPolicySimulator />
          </section>

          {/* Section 4: Reading Policy Signals */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              4️⃣ 정책 신호 읽기
            </h2>
            
            <div className="bg-yellow-50 dark:bg-yellow-900/10 p-6 rounded-xl mb-8">
              <h3 className="text-xl font-bold text-yellow-800 dark:text-yellow-300 mb-4">
                🔍 중앙은행 언어 해석법
              </h3>
              <p className="text-yellow-700 dark:text-yellow-300">
                중앙은행은 시장 충격을 피하기 위해 매우 신중한 언어를 사용합니다. 
                작은 문구 변화도 큰 정책 변화의 신호일 수 있어요.
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
              <div className="bg-red-50 dark:bg-red-900/10 p-6 rounded-xl">
                <h3 className="text-lg font-bold text-red-800 dark:text-red-300 mb-4">
                  🦅 매파적 (Hawkish) 신호
                </h3>
                
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-700 p-3 rounded">
                    <h4 className="font-semibold text-red-700 dark:text-red-400 text-sm">핵심 키워드</h4>
                    <p className="text-xs text-red-600 dark:text-red-300 mt-1">
                      "인플레이션 우려", "경기 과열", "긴축적", "추가 조치 필요"
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-700 p-3 rounded">
                    <h4 className="font-semibold text-red-700 dark:text-red-400 text-sm">시장 해석</h4>
                    <p className="text-xs text-red-600 dark:text-red-300 mt-1">
                      금리 인상 가능성 ↑ → 주식 부담 ↑ → 통화 강세
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-700 p-3 rounded">
                    <h4 className="font-semibold text-red-700 dark:text-red-400 text-sm">투자 전략</h4>
                    <p className="text-xs text-red-600 dark:text-red-300 mt-1">
                      금융주 매수, 성장주 비중 축소, 채권 수익률 상승 대비
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/10 p-6 rounded-xl">
                <h3 className="text-lg font-bold text-blue-800 dark:text-blue-300 mb-4">
                  🕊️ 비둘기파 (Dovish) 신호
                </h3>
                
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-700 p-3 rounded">
                    <h4 className="font-semibold text-blue-700 dark:text-blue-400 text-sm">핵심 키워드</h4>
                    <p className="text-xs text-blue-600 dark:text-blue-300 mt-1">
                      "경기 둔화", "완화적", "지원적", "신중한 접근"
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-700 p-3 rounded">
                    <h4 className="font-semibold text-blue-700 dark:text-blue-400 text-sm">시장 해석</h4>
                    <p className="text-xs text-blue-600 dark:text-blue-300 mt-1">
                      금리 인하/동결 가능성 ↑ → 주식 상승 → 통화 약세
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-700 p-3 rounded">
                    <h4 className="font-semibold text-blue-700 dark:text-blue-400 text-sm">투자 전략</h4>
                    <p className="text-xs text-blue-600 dark:text-blue-300 mt-1">
                      성장주 매수, 부동산 리츠 고려, 금융주 주의
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/10 p-6 rounded-xl mt-6">
              <h3 className="text-lg font-bold text-purple-800 dark:text-purple-300 mb-4">
                📅 중요 일정 & 지켜볼 포인트
              </h3>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold text-purple-700 dark:text-purple-400 mb-3">FOMC 일정 (2024)</h4>
                  <div className="space-y-2">
                    <div className="bg-white dark:bg-gray-700 p-2 rounded text-sm">
                      <span className="font-medium">3월 19-20일</span> - 금리 결정 + 점도표
                    </div>
                    <div className="bg-white dark:bg-gray-700 p-2 rounded text-sm">
                      <span className="font-medium">5월 1일</span> - 금리 결정
                    </div>
                    <div className="bg-white dark:bg-gray-700 p-2 rounded text-sm">
                      <span className="font-medium">6월 11-12일</span> - 금리 결정 + 점도표
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-semibold text-purple-700 dark:text-purple-400 mb-3">핵심 관전 포인트</h4>
                  <div className="space-y-2">
                    <div className="bg-white dark:bg-gray-700 p-2 rounded text-sm">
                      <span className="font-medium">Dot Plot:</span> 위원들의 금리 전망
                    </div>
                    <div className="bg-white dark:bg-gray-700 p-2 rounded text-sm">
                      <span className="font-medium">기자회견:</span> 의장 발언과 질의응답
                    </div>
                    <div className="bg-white dark:bg-gray-700 p-2 rounded text-sm">
                      <span className="font-medium">성명서:</span> 문구 변화와 정책 톤
                    </div>
                  </div>
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
              <span className="text-2xl">🎓</span>
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                다음 단계로 진행
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                통화정책을 이해했다면 이제 종합적인 거시경제 분석 실습을 해보세요.
              </p>
            </div>
          </div>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                📊 Chapter 9: 거시경제 분석 실습
              </h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">
                실제 경제지표 발표를 분석하고 투자 전략을 수립하는 종합적인 실습을 진행합니다.
              </p>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <Clock className="w-4 h-4" />
                  <span>예상 학습시간: 120분</span>
                </div>
                <Link
                  href="/modules/stock-analysis/chapters/macro-practice"
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
                Foundation Program의 전체 학습 경로를 확인하고 나만의 학습 계획을 세워보세요.
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
              <span>Foundation Program 진행률</span>
              <span>8/9 완료</span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div className="bg-gradient-to-r from-yellow-400 to-orange-500 h-2 rounded-full" style={{ width: '89%' }}></div>
            </div>
          </div>
        </div>
      </div>

      {/* Chapter Navigation */}
      <ChapterNavigation currentChapterId="fomc-analysis" programType="foundation" />
    </div>
  );
}