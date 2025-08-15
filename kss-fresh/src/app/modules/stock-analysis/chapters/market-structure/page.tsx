'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Globe, Building2, Users, TrendingUp, AlertCircle, ChevronRight } from 'lucide-react';
import ChapterNavigation from '../../components/ChapterNavigation';

function QuizSection() {
  const [answers, setAnswers] = useState<{ q1: string; q2: string }>({ q1: '', q2: '' });
  const [showResults, setShowResults] = useState(false);
  
  const correctAnswers = {
    q1: 'q1-3', // 전문투자자만 참여 가능
    q2: 'q2-2'  // KOSPI 시가총액의 약 30%를 보유하고 있다
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
          <h3 className="font-semibold mb-3">Q1. 다음 중 KOSPI 시장의 특징이 아닌 것은?</h3>
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
                대형 우량기업 중심{getResultIcon('q1', 'q1-1')}
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
                시가총액 약 2,000조원{getResultIcon('q1', 'q1-2')}
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
                전문투자자만 참여 가능{getResultIcon('q1', 'q1-3')}
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
                일일 가격제한폭 ±30%{getResultIcon('q1', 'q1-4')}
              </span>
            </label>
          </div>
          {showResults && (
            <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <p className="text-sm text-gray-700 dark:text-gray-300">
                💡 해설: KOSPI는 개인투자자도 자유롭게 참여할 수 있습니다. 전문투자자만 참여 가능한 것은 KONEX시장입니다.
              </p>
            </div>
          )}
        </div>

        <div>
          <h3 className="font-semibold mb-3">Q2. 외국인 투자자의 특징으로 옳은 것은?</h3>
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
                주로 소형주에 투자한다{getResultIcon('q2', 'q2-1')}
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
                KOSPI 시가총액의 약 30%를 보유하고 있다{getResultIcon('q2', 'q2-2')}
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
                환율 변동과 무관하다{getResultIcon('q2', 'q2-3')}
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
                단기 매매를 선호한다{getResultIcon('q2', 'q2-4')}
              </span>
            </label>
          </div>
          {showResults && (
            <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <p className="text-sm text-gray-700 dark:text-gray-300">
                💡 해설: 외국인 투자자는 대형주 위주로 투자하며, 환율 변동에 민감하고, 장기 투자를 선호합니다.
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

export default function MarketStructureChapter() {
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

      {/* Chapter Header */}
      <div className="bg-yellow-50 dark:bg-gray-800 py-8">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center">
              <Globe className="w-6 h-6 text-yellow-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Week 1 - Chapter 1</p>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                글로벌 금융시장의 구조
              </h1>
            </div>
          </div>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            한국 주식시장은 어떻게 작동하며, 세계 시장과 어떻게 연결되어 있을까?
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6">
            <h2 className="text-xl font-bold mb-4">이 챕터에서 배울 내용</h2>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-blue-500 mt-1">•</span>
                <span>주식시장의 탄생과 존재 이유</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500 mt-1">•</span>
                <span>한국증권거래소(KRX)의 구조와 역할</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500 mt-1">•</span>
                <span>글로벌 시장과의 연결고리</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500 mt-1">•</span>
                <span>시장 참여자들의 역할과 영향력</span>
              </li>
            </ul>
          </div>
        </section>

        {/* Section 1: Why Stock Market Exists */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6">1. 주식시장은 왜 존재하는가?</h2>
          
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <p className="mb-4">
              주식시장은 단순히 돈을 벌고 잃는 도박장이 아닙니다. 
              <strong>자본주의 경제의 핵심 인프라</strong>로서, 기업과 투자자를 연결하는 중요한 역할을 합니다.
            </p>

            <div className="grid md:grid-cols-2 gap-6 my-8">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <Building2 className="w-5 h-5 text-blue-500" />
                  기업의 입장
                </h3>
                <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                  <li>• 대규모 자금 조달 가능</li>
                  <li>• 은행 대출보다 유리한 조건</li>
                  <li>• 기업 가치의 객관적 평가</li>
                  <li>• 인수합병(M&A) 수단</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <Users className="w-5 h-5 text-green-500" />
                  투자자의 입장
                </h3>
                <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                  <li>• 기업 성장의 과실 공유</li>
                  <li>• 유동성 있는 자산 투자</li>
                  <li>• 분산투자로 위험 관리</li>
                  <li>• 인플레이션 헤지</li>
                </ul>
              </div>
            </div>

            <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6 my-8">
              <h4 className="font-semibold mb-3">💡 핵심 개념: IPO (Initial Public Offering)</h4>
              <p className="text-gray-700 dark:text-gray-300">
                비상장 기업이 처음으로 주식시장에 상장하는 것을 IPO라고 합니다. 
                최근 카카오뱅크(2021년), 크래프톤(2021년) 등의 대형 IPO가 있었고, 
                이를 통해 기업은 수조원의 자금을 조달했습니다.
              </p>
            </div>
          </div>
        </section>

        {/* Section 2: Structure of KRX */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6">2. 한국증권거래소(KRX)의 구조</h2>

          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
            <div className="p-6">
              <h3 className="text-xl font-semibold mb-4">시장 구분</h3>
              
              <div className="space-y-4">
                <div className="border-l-4 border-blue-500 pl-4">
                  <h4 className="font-semibold text-lg">KOSPI (코스피)</h4>
                  <p className="text-gray-600 dark:text-gray-400 mb-2">
                    대형 우량기업 중심의 주식시장
                  </p>
                  <ul className="text-sm space-y-1 text-gray-500">
                    <li>• 시가총액: 약 2,000조원</li>
                    <li>• 상장기업: 약 800개</li>
                    <li>• 대표기업: 삼성전자, SK하이닉스, 네이버, 카카오</li>
                  </ul>
                </div>

                <div className="border-l-4 border-green-500 pl-4">
                  <h4 className="font-semibold text-lg">KOSDAQ (코스닥)</h4>
                  <p className="text-gray-600 dark:text-gray-400 mb-2">
                    성장 가능성 높은 중소/벤처기업 시장
                  </p>
                  <ul className="text-sm space-y-1 text-gray-500">
                    <li>• 시가총액: 약 400조원</li>
                    <li>• 상장기업: 약 1,500개</li>
                    <li>• 대표기업: 셀트리온헬스케어, 에코프로비엠, 펄어비스</li>
                  </ul>
                </div>

                <div className="border-l-4 border-purple-500 pl-4">
                  <h4 className="font-semibold text-lg">KONEX (코넥스)</h4>
                  <p className="text-gray-600 dark:text-gray-400 mb-2">
                    초기 중소기업 전용시장 (전문투자자 중심)
                  </p>
                  <ul className="text-sm space-y-1 text-gray-500">
                    <li>• 개인투자자 참여 제한</li>
                    <li>• KOSDAQ 이전상장 준비 단계</li>
                  </ul>
                </div>

                <div className="border-l-4 border-orange-500 pl-4">
                  <h4 className="font-semibold text-lg">K-OTC (장외시장)</h4>
                  <p className="text-gray-600 dark:text-gray-400 mb-2">
                    비상장기업 주식거래 플랫폼
                  </p>
                  <ul className="text-sm space-y-1 text-gray-500">
                    <li>• 한국금융투자협회 운영</li>
                    <li>• 일일 1회 체결 (15:00)</li>
                    <li>• 가격제한폭 없음</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-8 grid md:grid-cols-2 gap-6">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
              <h4 className="font-semibold mb-3">거래 시간</h4>
              <ul className="space-y-2 text-sm">
                <li><strong>장전 시간외:</strong> 08:00 ~ 09:00</li>
                <li className="text-xs text-gray-600 dark:text-gray-400 ml-4">→ 전일 종가 기준 ±10% 단일가 거래</li>
                <li><strong>정규장:</strong> 09:00 ~ 15:30</li>
                <li className="text-xs text-gray-600 dark:text-gray-400 ml-4">→ 실시간 체결, 가격제한폭 ±30%</li>
                <li><strong>장후 시간외:</strong> 15:40 ~ 16:00</li>
                <li className="text-xs text-gray-600 dark:text-gray-400 ml-4">→ 당일 종가 기준 ±10% 단일가 거래</li>
                <li><strong>야간 선물:</strong> 18:00 ~ 익일 05:00</li>
              </ul>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
              <h4 className="font-semibold mb-3">가격제한폭</h4>
              <ul className="space-y-2 text-sm">
                <li><strong>일일 변동제한:</strong> ±30%</li>
                <li><strong>가격제한폭 적용:</strong> 전일 종가 기준</li>
                <li><strong>상한가/하한가:</strong> 추가 매매 가능</li>
                <li><strong>서킷브레이커:</strong> 급락 시 거래 일시정지</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Section 3: Global Market Connection */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6">3. 글로벌 시장과의 연결</h2>

          <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-8">
            <h3 className="text-xl font-semibold mb-4">24시간 돌아가는 글로벌 시장</h3>
            <p className="mb-4">
              주식시장은 지구 자전과 함께 24시간 돌아갑니다. 
              한국 시장이 문을 닫으면 유럽이 열리고, 이어서 미국 시장이 열립니다.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-medium mb-2">🌏 아시아 시장</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  한국, 일본, 중국, 홍콩<br/>
                  09:00 ~ 16:00 (KST)
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-medium mb-2">🌍 유럽 시장</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  런던, 프랑크푸르트, 파리<br/>
                  17:00 ~ 01:30 (KST)
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-medium mb-2">🌎 미국 시장</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  NYSE, NASDAQ<br/>
                  23:30 ~ 06:00 (KST)<br/>
                  <span className="text-xs">서머타임: 22:30 ~ 05:00</span>
                </p>
              </div>
            </div>
          </div>

          <div className="prose prose-lg dark:prose-invert max-w-none">
            <h3 className="text-xl font-semibold mb-4">한국 시장에 미치는 영향</h3>
            
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <TrendingUp className="w-5 h-5 text-blue-500 mt-1" />
                <div>
                  <h4 className="font-medium mb-1">미국 시장의 영향</h4>
                  <p className="text-gray-600 dark:text-gray-400">
                    미국의 3대 지수 움직임은 다음날 한국 시장 개장에 직접적인 영향을 미칩니다.
                  </p>
                  
                  <div className="mt-3 space-y-3">
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                      <h5 className="font-semibold text-blue-700 dark:text-blue-300">나스닥 (NASDAQ Composite)</h5>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        미국 나스닥 거래소에 상장된 모든 주식을 포함하는 지수. 약 3,700개 종목으로 구성되며, 
                        애플, 마이크로소프트, 구글, 메타, 테슬라 등 기술주 비중이 높아 '기술주 지수'로 불립니다.
                      </p>
                      <p className="text-xs text-gray-500 mt-2">
                        → 한국 IT 섹터 (삼성전자, SK하이닉스 등)에 큰 영향
                      </p>
                    </div>
                    
                    <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3">
                      <h5 className="font-semibold text-green-700 dark:text-green-300">다우지수 (Dow Jones Industrial Average)</h5>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        미국에서 가장 오래된 주가지수(1896년~). 미국을 대표하는 30개 대형 우량기업만 포함. 
                        보잉, 코카콜라, 골드만삭스, 맥도날드 등 전통 산업 대기업 중심입니다.
                      </p>
                      <p className="text-xs text-gray-500 mt-2">
                        → KOSPI 대형주, 특히 금융/산업재 섹터에 영향
                      </p>
                    </div>
                    
                    <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3">
                      <h5 className="font-semibold text-purple-700 dark:text-purple-300">S&P 500 (Standard & Poor's 500)</h5>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        미국 시장 시가총액의 약 80%를 차지하는 500개 대형주로 구성. 시가총액 가중평균 방식으로 
                        계산되어 미국 경제 전체를 가장 잘 대표하는 지수로 평가받습니다.
                      </p>
                      <p className="text-xs text-gray-500 mt-2">
                        → 한국 주식시장 전체 방향성에 가장 큰 영향
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <TrendingUp className="w-5 h-5 text-green-500 mt-1" />
                <div>
                  <h4 className="font-medium mb-1">중국 시장의 영향</h4>
                  <p className="text-gray-600 dark:text-gray-400">
                    중국은 한국의 최대 교역국으로, 상하이종합지수와 항셍지수의 변동은 
                    한국 수출 기업들의 주가에 큰 영향을 미칩니다.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <TrendingUp className="w-5 h-5 text-purple-500 mt-1" />
                <div>
                  <h4 className="font-medium mb-1">환율의 영향</h4>
                  <p className="text-gray-600 dark:text-gray-400">
                    원/달러 환율은 외국인 투자자의 수익률에 직접적인 영향을 미치며, 
                    수출입 기업의 실적에도 큰 변수로 작용합니다.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 4: Market Participants */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6">4. 시장 참여자의 이해</h2>

          <div className="grid gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold mb-4 text-blue-600 dark:text-blue-400">
                개인투자자 (개미)
              </h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium mb-2">특징</h4>
                  <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                    <li>• 전체 거래량의 60~70% 차지</li>
                    <li>• 단기 매매 성향이 강함</li>
                    <li>• 정보 비대칭에 취약</li>
                    <li>• 감정적 매매 경향</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-medium mb-2">최근 트렌드</h4>
                  <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                    <li>• MTS 활용 급증</li>
                    <li>• 해외주식 투자 확대</li>
                    <li>• 공모주 청약 열풍</li>
                    <li>• ETF 투자 증가</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold mb-4 text-green-600 dark:text-green-400">
                기관투자자
              </h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium mb-2">주요 기관</h4>
                  <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                    <li>• 자산운용사 (펀드)</li>
                    <li>• 보험사</li>
                    <li>• 연기금 (국민연금 등)</li>
                    <li>• 은행</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-medium mb-2">투자 특징</h4>
                  <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                    <li>• 장기 투자 관점</li>
                    <li>• 리스크 관리 중시</li>
                    <li>• 대량 매매로 시장 영향력</li>
                    <li>• 펀더멘털 분석 중심</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold mb-4 text-purple-600 dark:text-purple-400">
                외국인투자자
              </h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium mb-2">특징</h4>
                  <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                    <li>• KOSPI 시가총액의 30% 보유</li>
                    <li>• 대형주 위주 투자</li>
                    <li>• 글로벌 자금 흐름에 민감</li>
                    <li>• 환율 리스크 헤지</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-medium mb-2">영향력</h4>
                  <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                    <li>• 일일 매매동향 주목 대상</li>
                    <li>• 대규모 매도 시 지수 급락</li>
                    <li>• 업종/종목 선호도 변화</li>
                    <li>• 한국 시장 평가 지표</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-8 bg-amber-50 dark:bg-amber-900/10 rounded-lg p-6">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-1" />
              <div>
                <h4 className="font-semibold mb-2">💡 투자 인사이트</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  각 투자 주체의 매매 동향을 파악하는 것은 시장의 방향성을 예측하는 중요한 지표입니다. 
                  특히 외국인과 기관의 순매수/순매도 동향은 중장기 추세를 가늠하는 척도가 됩니다.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Key Takeaways */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-yellow-100 to-yellow-50 dark:from-yellow-900/20 dark:to-yellow-800/10 rounded-xl p-8">
            <h2 className="text-2xl font-bold mb-6">📌 핵심 정리</h2>
            
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <span className="text-2xl font-bold text-yellow-600">1</span>
                <div>
                  <h3 className="font-semibold mb-1">주식시장은 자본주의의 핵심 인프라</h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    기업의 자금조달과 투자자의 수익 창출을 연결하는 중요한 플랫폼입니다.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <span className="text-2xl font-bold text-yellow-600">2</span>
                <div>
                  <h3 className="font-semibold mb-1">한국 시장은 글로벌 시장과 긴밀히 연결</h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    미국, 중국 시장의 움직임과 환율 변동이 한국 시장에 직접적인 영향을 미칩니다.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <span className="text-2xl font-bold text-yellow-600">3</span>
                <div>
                  <h3 className="font-semibold mb-1">시장 참여자의 특성을 이해하라</h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    개인, 기관, 외국인 각각의 투자 패턴과 영향력을 파악하면 시장을 더 잘 이해할 수 있습니다.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 시간외거래 상세 설명 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6">5. 시간외거래의 이해</h2>
          
          <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4">왜 정규시간 외에도 거래를 할까?</h3>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-semibold mb-3 text-blue-600 dark:text-blue-400">
                  장전 시간외거래 (08:00~09:00)
                </h4>
                <ul className="text-sm space-y-2 text-gray-600 dark:text-gray-400">
                  <li>• <strong>목적:</strong> 전날 미국시장 반영</li>
                  <li>• <strong>특징:</strong> 단일가 거래 (10분 단위)</li>
                  <li>• <strong>제한:</strong> 전일 종가 ±10%</li>
                  <li>• <strong>주요 참여자:</strong> 기관, 외국인</li>
                </ul>
                <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                  <p className="text-xs">
                    💡 전날 밤 미국 증시가 급등/급락했다면 장전 시간외에서 가격이 크게 움직입니다
                  </p>
                </div>
              </div>
              
              <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-semibold mb-3 text-green-600 dark:text-green-400">
                  장후 시간외거래 (15:40~16:00)
                </h4>
                <ul className="text-sm space-y-2 text-gray-600 dark:text-gray-400">
                  <li>• <strong>목적:</strong> 장마감 후 뉴스/공시 반영</li>
                  <li>• <strong>특징:</strong> 단일가 거래 (10분 단위)</li>
                  <li>• <strong>제한:</strong> 당일 종가 ±10%</li>
                  <li>• <strong>활용:</strong> 실적발표, 공시 대응</li>
                </ul>
                <div className="mt-3 p-3 bg-green-50 dark:bg-green-900/20 rounded">
                  <p className="text-xs">
                    💡 15:30 이후 발표되는 기업 실적이나 중요 공시에 대응할 수 있습니다
                  </p>
                </div>
              </div>
            </div>
            
            <div className="mt-6 bg-amber-50 dark:bg-amber-900/10 rounded-lg p-4">
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-amber-600" />
                시간외거래 주의사항
              </h4>
              <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                <li>• 거래량이 적어 가격 변동성이 큼</li>
                <li>• 개인투자자 참여율이 낮아 불리할 수 있음</li>
                <li>• 호가 간격이 넓어 원하는 가격에 체결 어려움</li>
              </ul>
            </div>
          </div>
        </section>

        {/* K-OTC (장외거래) Section */}
        <section className="mb-12">
          <div className="bg-purple-50 dark:bg-purple-900/10 rounded-xl p-8">
            <h3 className="text-2xl font-bold mb-6 flex items-center gap-3">
              <Building2 className="w-8 h-8 text-purple-600" />
              K-OTC (한국장외주식시장)
            </h3>
            
            <div className="space-y-6">
              <div>
                <h4 className="font-semibold mb-3">K-OTC란?</h4>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  한국금융투자협회가 운영하는 비상장 기업 주식 거래 플랫폼입니다. 
                  코스피·코스닥에 상장되지 않은 중소·벤처기업의 주식을 거래할 수 있습니다.
                </p>
              </div>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold mb-3">주요 특징</h4>
                  <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                    <li className="flex items-start gap-2">
                      <span className="text-purple-500 mt-1">•</span>
                      <span>비상장 기업 주식 거래 가능</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-purple-500 mt-1">•</span>
                      <span>정규시장 대비 완화된 상장 요건</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-purple-500 mt-1">•</span>
                      <span>벤처·중소기업 중심</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-purple-500 mt-1">•</span>
                      <span>일일 가격제한폭 ±30%</span>
                    </li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-semibold mb-3">거래 정보</h4>
                  <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                    <li className="flex items-start gap-2">
                      <span className="text-purple-500 mt-1">•</span>
                      <span><strong>거래시간:</strong> 09:00 ~ 15:30</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-purple-500 mt-1">•</span>
                      <span><strong>거래방식:</strong> 호가 매매</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-purple-500 mt-1">•</span>
                      <span><strong>결제:</strong> T+2일</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-purple-500 mt-1">•</span>
                      <span><strong>최소거래단위:</strong> 1주</span>
                    </li>
                  </ul>
                </div>
              </div>
              
              <div className="bg-yellow-50 dark:bg-yellow-900/10 rounded-lg p-4">
                <p className="text-sm flex items-start gap-2">
                  <AlertCircle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                  <span>
                    <strong>투자 주의사항:</strong> K-OTC 시장은 정보 비대칭이 크고 유동성이 낮아 
                    투자 위험이 높습니다. 기업 정보를 충분히 분석하고, 소액으로 분산 투자하는 것이 중요합니다.
                  </span>
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* 미국 시장 시간외 거래 Section */}
        <section className="mb-12">
          <div className="bg-indigo-50 dark:bg-indigo-900/10 rounded-xl p-8">
            <h3 className="text-2xl font-bold mb-6 flex items-center gap-3">
              <Globe className="w-8 h-8 text-indigo-600" />
              미국 주식시장 시간외 거래
            </h3>
            
            <div className="space-y-6">
              <div>
                <h4 className="font-semibold mb-3">미국 시장 거래 시간 (한국 시간 기준)</h4>
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">프리마켓 (Pre-market)</span>
                      <span className="text-indigo-600 dark:text-indigo-400 font-mono">17:00 ~ 22:30</span>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      미국 동부시간 04:00 ~ 09:30 / 정규장 개장 전 거래
                    </p>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">정규장 (Regular Hours)</span>
                      <span className="text-indigo-600 dark:text-indigo-400 font-mono">22:30 ~ 05:00</span>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      미국 동부시간 09:30 ~ 16:00 / NYSE, NASDAQ 정규 거래
                    </p>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">애프터마켓 (After-hours)</span>
                      <span className="text-indigo-600 dark:text-indigo-400 font-mono">05:00 ~ 09:00</span>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      미국 동부시간 16:00 ~ 20:00 / 정규장 마감 후 거래
                    </p>
                  </div>
                </div>
              </div>
              
              <div>
                <h4 className="font-semibold mb-3">시간외 거래 특징</h4>
                <ul className="space-y-2 list-disc list-inside text-gray-600 dark:text-gray-400">
                  <li><strong>낮은 유동성:</strong> 정규장 대비 거래량이 크게 감소</li>
                  <li><strong>높은 변동성:</strong> 적은 거래량으로도 가격이 크게 움직임</li>
                  <li><strong>넓은 스프레드:</strong> 매수/매도 호가 차이가 크게 벌어짐</li>
                  <li><strong>제한된 주문:</strong> 일부 브로커는 지정가 주문만 허용</li>
                  <li><strong>실적 발표 반영:</strong> 장 마감 후 발표된 실적이 즉시 반영</li>
                </ul>
              </div>
              
              <div>
                <h4 className="font-semibold mb-3">한국 투자자를 위한 팁</h4>
                <div className="bg-indigo-100 dark:bg-indigo-900/20 rounded-lg p-4 space-y-2">
                  <p className="text-sm">
                    <strong>오후 5시 ~ 밤 10시 30분:</strong> 프리마켓에서 전일 뉴스와 아시아 시장 반응 확인
                  </p>
                  <p className="text-sm">
                    <strong>밤 10시 30분 ~ 새벽 5시:</strong> 정규장 거래로 최고의 유동성 확보
                  </p>
                  <p className="text-sm">
                    <strong>새벽 5시 ~ 오전 9시:</strong> 애프터마켓에서 당일 실적 발표 반응 확인
                  </p>
                </div>
              </div>
              
              <div className="mt-4 p-4 bg-yellow-100 dark:bg-yellow-900/20 rounded-lg">
                <p className="text-sm flex items-start gap-2">
                  <AlertCircle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                  <span>
                    <strong>주의사항:</strong> 시간외 거래는 유동성이 낮아 불리한 가격에 체결될 수 있습니다. 
                    특히 마켓 주문은 피하고, 반드시 지정가 주문을 사용하세요. 서머타임 적용 시 
                    모든 시간이 1시간 앞당겨집니다 (3월 둘째 주 ~ 11월 첫째 주).
                  </span>
                </p>
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
                  시장 참여자의 이해
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  개인, 기관, 외국인 투자자의 특성과 투자 패턴을 분석하고,
                  각 참여자가 시장에 미치는 영향을 알아봅니다.
                </p>
              </div>
              
              {/* Vertical Divider - Hidden on mobile */}
              <div className="hidden sm:block w-px h-20 bg-gray-200 dark:bg-gray-700" />
              
              <Link
                href="/modules/stock-analysis/chapters/market-participants"
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
      <ChapterNavigation currentChapterId="market-structure" programType="foundation" />
    </div>
  );
}