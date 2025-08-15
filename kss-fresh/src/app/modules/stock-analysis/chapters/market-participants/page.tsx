'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Users, TrendingUp, AlertCircle, ChevronRight, Activity, PieChart, BarChart3, Target } from 'lucide-react';
import ChapterNavigation from '../../components/ChapterNavigation';

function QuizSection() {
  const [answers, setAnswers] = useState<{ q1: string; q2: string }>({ q1: '', q2: '' });
  const [showResults, setShowResults] = useState(false);
  
  const correctAnswers = {
    q1: 'q1-2', // 개인투자자가 전체 거래량의 60~70% 차지
    q2: 'q2-3'  // 글로벌 경제 상황과 환율에 민감하게 반응
  };
  
  const handleAnswerChange = (question: string, value: string) => {
    if (!showResults) {
      setAnswers(prev => ({ ...prev, [question]: value }));
    }
  };
  
  const checkAnswers = () => {
    if (answers.q1 && answers.q2) {
      setShowResults(true);
    } else {
      alert('모든 문제에 답해주세요.');
    }
  };
  
  const resetQuiz = () => {
    setAnswers({ q1: '', q2: '' });
    setShowResults(false);
  };
  
  const getResultStyle = (question: 'q1' | 'q2', optionValue: string) => {
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
  
  const getResultIcon = (question: 'q1' | 'q2', optionValue: string) => {
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
          score === 2 ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300' 
          : score === 1 ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300'
          : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
        }`}>
          <p className="font-semibold">
            {score === 2 ? '🎉 완벽합니다!' : score === 1 ? '😊 잘하셨어요!' : '💪 다시 도전해보세요!'}
            {` ${score}/2 문제를 맞추셨습니다.`}
          </p>
        </div>
      )}
      
      <div className="space-y-6">
        <div>
          <h3 className="font-semibold mb-3">Q1. 한국 주식시장의 투자자별 거래 비중에 대한 설명으로 옳은 것은?</h3>
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
                기관투자자가 전체 거래량의 70% 이상을 차지한다{getResultIcon('q1', 'q1-1')}
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
                개인투자자가 전체 거래량의 60~70%를 차지한다{getResultIcon('q1', 'q1-2')}
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
                외국인투자자가 전체 거래량의 80%를 차지한다{getResultIcon('q1', 'q1-3')}
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q1" 
                value="q1-4"
                checked={answers.q1 === 'q1-4'}
                onChange={(e) => handleAnswerChange('q1', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q1', 'q1-4')}>
                모든 투자자가 균등하게 33%씩 거래한다{getResultIcon('q1', 'q1-4')}
              </span>
            </label>
          </div>
          {showResults && (
            <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <p className="text-sm text-gray-700 dark:text-gray-300">
                💡 해설: 한국 주식시장은 개인투자자의 거래 비중이 매우 높은 것이 특징입니다. 전체 거래량의 60~70%를 개인투자자가 차지합니다.
              </p>
            </div>
          )}
        </div>

        <div>
          <h3 className="font-semibold mb-3">Q2. 외국인 투자자의 특징으로 가장 적절한 것은?</h3>
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
                주로 소형주와 테마주에 집중 투자한다{getResultIcon('q2', 'q2-1')}
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
                단기 매매를 선호하며 변동성이 큰 편이다{getResultIcon('q2', 'q2-2')}
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
                글로벌 경제 상황과 환율에 민감하게 반응한다{getResultIcon('q2', 'q2-3')}
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q2" 
                value="q2-4"
                checked={answers.q2 === 'q2-4'}
                onChange={(e) => handleAnswerChange('q2', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q2', 'q2-4')}>
                국내 정치 상황에 큰 영향을 받는다{getResultIcon('q2', 'q2-4')}
              </span>
            </label>
          </div>
          {showResults && (
            <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <p className="text-sm text-gray-700 dark:text-gray-300">
                💡 해설: 외국인 투자자는 글로벌 자금 흐름에 따라 움직이며, 환율 변동과 글로벌 경제 상황에 매우 민감합니다. 주로 대형주 위주로 투자합니다.
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="mt-6 flex justify-center">
        <button 
          onClick={showResults ? resetQuiz : checkAnswers}
          className="px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors shadow-sm"
        >
          {showResults ? '다시 풀기' : '정답 확인하기'}
        </button>
      </div>
    </div>
  );
}

export default function MarketParticipantsChapter() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <Link 
            href="/modules/stock-analysis/stages/foundation"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Foundation Program으로 돌아가기</span>
          </Link>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Title Section */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            시장 참여자의 이해
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            주식시장을 움직이는 세 주체의 특성과 행동 패턴을 이해하고,
            이들의 매매 동향이 시장에 미치는 영향을 파악합니다.
          </p>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-blue-50 dark:bg-blue-900/10 rounded-xl p-8">
            <h2 className="text-2xl font-bold mb-4">왜 시장 참여자를 이해해야 할까?</h2>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              주식시장은 다양한 목적과 전략을 가진 투자자들이 모여 가격을 형성합니다. 
              각 투자 주체의 특성을 이해하면 시장의 흐름을 읽고 더 나은 투자 결정을 내릴 수 있습니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 mt-4">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Target className="w-5 h-5 text-blue-600" />
                실전 활용 예시
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                <li>• 외국인이 대규모 순매도 → 시장 조정 가능성 대비</li>
                <li>• 기관이 특정 섹터 집중 매수 → 업종 순환 신호 포착</li>
                <li>• 개인 매수세 급증 → 과열 구간 주의 신호</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 3대 투자 주체 Overview */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6">한국 주식시장의 3대 투자 주체</h2>
          
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-6">
            <div className="grid md:grid-cols-3 gap-6 text-center">
              <div>
                <div className="w-20 h-20 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center mx-auto mb-3">
                  <Users className="w-10 h-10 text-blue-600 dark:text-blue-400" />
                </div>
                <h3 className="font-bold text-lg mb-1">개인투자자</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">거래 비중 약 67%</p>
                <p className="text-xs text-gray-500 mt-1">"개미"</p>
              </div>
              
              <div>
                <div className="w-20 h-20 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center mx-auto mb-3">
                  <BarChart3 className="w-10 h-10 text-green-600 dark:text-green-400" />
                </div>
                <h3 className="font-bold text-lg mb-1">기관투자자</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">거래 비중 약 13%</p>
                <p className="text-xs text-gray-500 mt-1">"큰손"</p>
              </div>
              
              <div>
                <div className="w-20 h-20 bg-purple-100 dark:bg-purple-900/30 rounded-full flex items-center justify-center mx-auto mb-3">
                  <Activity className="w-10 h-10 text-purple-600 dark:text-purple-400" />
                </div>
                <h3 className="font-bold text-lg mb-1">외국인투자자</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">거래 비중 약 18%</p>
                <p className="text-xs text-gray-500 mt-1">"외인"</p>
              </div>
            </div>
          </div>

          <div className="bg-amber-50 dark:bg-amber-900/10 rounded-lg p-4">
            <p className="text-sm text-gray-700 dark:text-gray-300 flex items-start gap-2">
              <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
              <span>
                <strong>주의:</strong> 거래 비중과 시가총액 보유 비중은 다릅니다. 
                외국인은 거래 비중은 18%지만 시가총액의 28.7%를 보유(2024년 기준)하고 있어 영향력이 큽니다.
              </span>
            </p>
          </div>
        </section>

        {/* 개인투자자 Section */}
        <section className="mb-12">
          <div className="bg-blue-50 dark:bg-blue-900/10 rounded-xl p-8">
            <h3 className="text-2xl font-bold mb-6 flex items-center gap-3">
              <Users className="w-8 h-8 text-blue-600" />
              개인투자자 (개미)
            </h3>
            
            <div className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold mb-3">주요 특징</h4>
                  <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                    <li className="flex items-start gap-2">
                      <span className="text-blue-500 mt-1">•</span>
                      <span>높은 거래 빈도 - 하루에도 여러 번 매매</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-500 mt-1">•</span>
                      <span>단기 수익 추구 경향이 강함</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-500 mt-1">•</span>
                      <span>감정적 매매 - 공포와 탐욕에 좌우</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-500 mt-1">•</span>
                      <span>정보 비대칭에 취약</span>
                    </li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-semibold mb-3">투자 성향</h4>
                  <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                    <li className="flex items-start gap-2">
                      <span className="text-blue-500 mt-1">•</span>
                      <span>테마주, 소형주 선호</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-500 mt-1">•</span>
                      <span>뉴스와 루머에 민감하게 반응</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-500 mt-1">•</span>
                      <span>손절매가 어려워 장기 보유로 전환</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-500 mt-1">•</span>
                      <span>상승장 후반부에 대거 진입하는 경향</span>
                    </li>
                  </ul>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
                <h4 className="font-semibold mb-3">2025년 개인투자자 최신 트렌드</h4>
                <div className="grid md:grid-cols-2 gap-4 text-sm">
                  <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
                    <strong className="text-blue-600">통합 MTS 플랫폼</strong>
                    <p className="text-gray-600 dark:text-gray-400 mt-1">국내·해외주식·자산관리까지 하나의 앱에서 처리</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
                    <strong className="text-blue-600">미국 기술주 편중</strong>
                    <p className="text-gray-600 dark:text-gray-400 mt-1">해외투자 잔액의 90% 이상이 미국 시장에 집중</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
                    <strong className="text-blue-600">테마형 ETF 선호</strong>
                    <p className="text-gray-600 dark:text-gray-400 mt-1">친환경, 디지털 전환 등 신성장 산업 ETF 인기</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
                    <strong className="text-blue-600">주요 MTS 앱</strong>
                    <p className="text-gray-600 dark:text-gray-400 mt-1">키움 영웅문, NH 나무, 삼성 mPop이 상위권</p>
                  </div>
                </div>
                <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                  <p className="text-xs text-gray-600 dark:text-gray-300">
                    <strong>💡 투자 권고:</strong> 미국 기술주 중심의 편중된 포트폴리오를 
                    신흥국(중국 전기차, 인도 디지털금융) 및 다양한 섹터로 분산하는 것이 중요합니다.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 기관투자자 Section */}
        <section className="mb-12">
          <div className="bg-green-50 dark:bg-green-900/10 rounded-xl p-8">
            <h3 className="text-2xl font-bold mb-6 flex items-center gap-3">
              <BarChart3 className="w-8 h-8 text-green-600" />
              기관투자자
            </h3>
            
            <div className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold mb-3">구성</h4>
                  <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                    <li className="flex items-start gap-2">
                      <span className="text-green-500 mt-1">•</span>
                      <span><strong>자산운용사:</strong> 펀드 운용</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-500 mt-1">•</span>
                      <span><strong>보험사:</strong> 보험료 운용</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-500 mt-1">•</span>
                      <span><strong>연기금:</strong> 국민연금, 공무원연금 등</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-500 mt-1">•</span>
                      <span><strong>은행/증권사:</strong> 자기자본 투자</span>
                    </li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-semibold mb-3">투자 특징</h4>
                  <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                    <li className="flex items-start gap-2">
                      <span className="text-green-500 mt-1">•</span>
                      <span>중장기 투자 관점</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-500 mt-1">•</span>
                      <span>기업의 재무제표와 실적 분석 중심 (펀더멘털 분석)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-500 mt-1">•</span>
                      <span>리스크 관리 체계적</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-500 mt-1">•</span>
                      <span>대량 매매로 시장 영향력 큼</span>
                    </li>
                  </ul>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
                <h4 className="font-semibold mb-3">기관별 특성</h4>
                <div className="space-y-3">
                  <div className="border-l-4 border-green-500 pl-4">
                    <strong>국민연금</strong>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      국내 최대 기관투자자. 장기투자 원칙, 배당 중시, 주주권 행사로 기업 경영에 참여
                    </p>
                  </div>
                  <div className="border-l-4 border-green-500 pl-4">
                    <strong>자산운용사</strong>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      펀드 수익률 경쟁 치열. 분기/연간 성과 압박으로 단기 실적에 민감
                    </p>
                  </div>
                  <div className="border-l-4 border-green-500 pl-4">
                    <strong>보험사</strong>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      안정성 최우선. 고배당주 선호, 보험금 지급을 위한 자산-부채 매칭 전략
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 외국인투자자 Section */}
        <section className="mb-12">
          <div className="bg-purple-50 dark:bg-purple-900/10 rounded-xl p-8">
            <h3 className="text-2xl font-bold mb-6 flex items-center gap-3">
              <Activity className="w-8 h-8 text-purple-600" />
              외국인투자자
            </h3>
            
            <div className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold mb-3">주요 특징</h4>
                  <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                    <li className="flex items-start gap-2">
                      <span className="text-purple-500 mt-1">•</span>
                      <span>KOSPI 시가총액의 약 30% 보유</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-purple-500 mt-1">•</span>
                      <span>대형주 위주 투자 (삼성전자, SK하이닉스 등)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-purple-500 mt-1">•</span>
                      <span>글로벌 자금 흐름에 민감</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-purple-500 mt-1">•</span>
                      <span>환율 변동 리스크 헤지</span>
                    </li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-semibold mb-3">행동 패턴</h4>
                  <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                    <li className="flex items-start gap-2">
                      <span className="text-purple-500 mt-1">•</span>
                      <span>연초/연말 리밸런싱 매매</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-purple-500 mt-1">•</span>
                      <span>글로벌 위기 시 일괄 매도</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-purple-500 mt-1">•</span>
                      <span>한국 신용등급 변화에 민감</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-purple-500 mt-1">•</span>
                      <span>업종별 순환매 전략</span>
                    </li>
                  </ul>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
                <h4 className="font-semibold mb-3 flex items-center gap-2">
                  <PieChart className="w-5 h-5 text-purple-600" />
                  외국인 투자자 구성
                </h4>
                <div className="grid md:grid-cols-2 gap-4 text-sm">
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="font-medium">연기금/국부펀드</span>
                        <span className="font-medium">40%</span>
                      </div>
                      <p className="text-xs text-gray-500">국가가 운용하는 연금, 국가 자산</p>
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="font-medium">뮤추얼펀드 (공모펀드)</span>
                        <span className="font-medium">25%</span>
                      </div>
                      <p className="text-xs text-gray-500">일반인이 가입할 수 있는 펀드</p>
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="font-medium">헤지펀드</span>
                        <span className="font-medium">20%</span>
                      </div>
                      <p className="text-xs text-gray-500">고수익 추구하는 전문투자펀드</p>
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="font-medium">기타(ETF, 개인 등)</span>
                        <span className="font-medium">15%</span>
                      </div>
                    </div>
                  </div>
                  <div className="bg-purple-100 dark:bg-purple-900/20 rounded p-4">
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      <strong>주요 투자자 예시:</strong><br/>
                      • 노르웨이 국부펀드 (석유 수익 운용)<br/>
                      • 싱가포르 GIC (국가 자산 운용)<br/>
                      • 캐나다 연기금 (국민연금)
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 투자 주체별 상호작용 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6">투자 주체 간 상호작용</h2>
          
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <div className="space-y-4">
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-red-600">🔴 전형적인 하락장 패턴</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  외국인 대량 매도 → 기관 초기 방어 매수 → 개인 저가 매수 → 
                  추가 하락 시 기관도 매도 전환 → 개인 홀로 매수 (나홀로 떡상)
                </p>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-green-600">🟢 전형적인 상승장 패턴</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  외국인/기관 동반 매수 → 지수 상승 → 개인 추격 매수 → 
                  고점 부근 기관 매도 → 외국인 매도 전환 → 개인 홀로 매수 지속
                </p>
              </div>
            </div>
            
            <div className="mt-6 bg-amber-50 dark:bg-amber-900/10 rounded-lg p-4">
              <p className="text-sm flex items-start gap-2">
                <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                <span>
                  <strong>투자 교훈:</strong> 외국인과 기관은 정보력과 자금력이 우수해 "스마트 머니"라 불립니다.
                  이들의 동향을 참고하되, 맹목적으로 따라가지 말고 자신만의 투자 원칙을 세우세요.
                </span>
              </p>
            </div>
          </div>
        </section>

        {/* 실전 활용법 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6">실전 활용법</h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="font-semibold mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-blue-600" />
                매매 동향 확인 방법
              </h3>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                <li>• 증권사 HTS/MTS에서 "투자자별 매매동향" 확인</li>
                <li>• 한국거래소 홈페이지 일일 투자자별 거래실적</li>
                <li>• 개별 종목의 외국인/기관 보유비율 변화 추적</li>
                <li>• 프로그램 매매 동향 (차익거래, 비차익거래)</li>
              </ul>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="font-semibold mb-4 flex items-center gap-2">
                <Target className="w-5 h-5 text-green-600" />
                활용 전략
              </h3>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                <li>• 외국인 10일 연속 순매수 종목 주목</li>
                <li>• 기관 업종별 순매수 추세 확인</li>
                <li>• 개인 매수세만 강한 종목은 신중히 접근</li>
                <li>• 세 주체가 동시에 매수하는 종목 발굴</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 핵심 정리 */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-yellow-100 to-yellow-50 dark:from-yellow-900/20 dark:to-yellow-800/10 rounded-xl p-8">
            <h2 className="text-2xl font-bold mb-6">📌 핵심 정리</h2>
            
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <span className="text-2xl font-bold text-yellow-600">1</span>
                <div>
                  <h3 className="font-semibold mb-1">개인투자자는 시장의 유동성 공급자</h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    높은 거래 비중으로 시장에 활력을 제공하지만, 정보력과 자금력의 한계로 수익률은 낮은 편입니다.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <span className="text-2xl font-bold text-yellow-600">2</span>
                <div>
                  <h3 className="font-semibold mb-1">기관투자자는 시장의 안정화 역할</h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    중장기 관점의 투자로 시장 변동성을 완화하며, 펀더멘털에 기반한 가치 투자를 추구합니다.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <span className="text-2xl font-bold text-yellow-600">3</span>
                <div>
                  <h3 className="font-semibold mb-1">외국인투자자는 시장의 방향성 결정</h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    글로벌 자금 흐름을 반영하며, 한국 시장의 중장기 트렌드를 주도하는 핵심 세력입니다.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Quiz Section */}
        <section className="mb-12">
          <QuizSection />
        </section>

        {/* Next Steps */}
        <section className="mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-8">
            <div className="flex flex-col sm:flex-row items-center justify-between gap-6">
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
                  다음 챕터
                </p>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-3">
                  매매 시스템 실습
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  호가창 읽기, 주문 유형별 특징, 체결 원리 등 
                  실제 매매에 필요한 기초 지식을 실습합니다.
                </p>
              </div>
              
              {/* Vertical Divider - Hidden on mobile */}
              <div className="hidden sm:block w-px h-20 bg-gray-200 dark:bg-gray-700" />
              
              <Link
                href="/modules/stock-analysis/chapters/trading-system"
                className="inline-flex items-center gap-2 px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors whitespace-nowrap"
              >
                다음 챕터로
                <ChevronRight className="w-5 h-5" />
              </Link>
            </div>
          </div>
        </section>
      </div>

      {/* Chapter Navigation */}
      <ChapterNavigation currentChapterId="market-participants" programType="foundation" />
    </div>
  );
}