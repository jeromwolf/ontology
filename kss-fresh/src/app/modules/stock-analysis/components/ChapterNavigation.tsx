'use client';

import Link from 'next/link';
import { ArrowLeft, ArrowRight, Home } from 'lucide-react';

interface Chapter {
  id: string;
  title: string;
  week: string;
}

interface ChapterNavigationProps {
  currentChapterId: string;
  programType: 'baby-chick' | 'foundation' | 'advanced' | 'professional';
}

export default function ChapterNavigation({ currentChapterId, programType }: ChapterNavigationProps) {
  // Baby Chick Program 챕터 순서
  const babyChickChapters: Chapter[] = [
    { id: 'what-is-stock', title: '주식이란 무엇인가?', week: 'Week 1' },
    { id: 'open-account', title: '증권계좌 개설하기', week: 'Week 1' },
    { id: 'trading-app-basics', title: 'HTS/MTS 사용법', week: 'Week 1' },
    { id: 'understanding-candles', title: '캔들 차트 이해하기', week: 'Week 2' },
    { id: 'volume-basics', title: '거래량이 말해주는 것', week: 'Week 2' },
    { id: 'order-book', title: '호가창 완전정복', week: 'Week 2' },
    { id: 'basic-terms', title: '꼭 알아야 할 투자 용어', week: 'Week 3' },
    { id: 'reading-news', title: '투자 뉴스 읽는 법', week: 'Week 3' },
    { id: 'sectors-themes', title: '업종과 테마 이해하기', week: 'Week 3' },
    { id: 'stock-selection', title: '종목 선택의 기초', week: 'Week 4' },
    { id: 'small-start', title: '소액으로 시작하기', week: 'Week 4' },
    { id: 'trading-diary', title: '매매일지 작성하기', week: 'Week 4' }
  ];

  // Foundation Program 챕터 순서
  const foundationChapters: Chapter[] = [
    { id: 'chart-basics', title: '차트 분석의 핵심 원리', week: 'Week 1-2' },
    { id: 'technical-indicators', title: '주요 기술적 지표 마스터', week: 'Week 1-2' },
    { id: 'pattern-recognition', title: '차트 패턴 인식과 매매', week: 'Week 1-2' },
    { id: 'financial-statements', title: '재무제표 읽기의 정석', week: 'Week 3-4' },
    { id: 'valuation-basics', title: '기업가치 평가의 기초', week: 'Week 3-4' },
    { id: 'industry-analysis', title: '산업 분석과 기업 비교', week: 'Week 3-4' },
    { id: 'global-brokerage-accounts', title: '해외 증권사 계좌 개설', week: 'Week 5' },
    { id: 'global-sectors-understanding', title: '글로벌 섹터 이해', week: 'Week 5' },
    { id: 'gaap-vs-ifrs', title: 'GAAP vs IFRS 회계기준', week: 'Week 6' },
    { id: 'investment-strategies', title: '검증된 투자 전략 학습', week: 'Week 7' },
    { id: 'portfolio-basics', title: '포트폴리오 구성의 기본', week: 'Week 7' },
    { id: 'risk-control', title: '리스크 관리와 손절매', week: 'Week 8' },
    { id: 'market-timing', title: '시장 타이밍과 매매 시점', week: 'Week 9' },
    { id: 'real-trading', title: '실전 매매 시뮬레이션', week: 'Week 9' },
    { id: 'investment-plan', title: '나만의 투자 계획 수립', week: 'Week 10' }
  ];

  // Advanced Program 챕터 순서
  const advancedChapters: Chapter[] = [
    { id: 'advanced-technical-analysis', title: '고급 차트 패턴과 하모닉 트레이딩', week: 'Week 1-3' },
    { id: 'system-trading-basics', title: '시스템 트레이딩 입문', week: 'Week 1-3' },
    { id: 'automated-strategies', title: '자동매매 전략 구축', week: 'Week 1-3' },
    { id: 'quantitative-basics', title: '퀀트 투자의 이해', week: 'Week 4-6' },
    { id: 'financial-data-analysis', title: '금융 빅데이터 분석', week: 'Week 4-6' },
    { id: 'factor-models', title: '팩터 모델 구축', week: 'Week 4-6' },
    { id: 'derivatives-basics', title: '옵션 거래 전략', week: 'Week 7-9' },
    { id: 'advanced-options', title: '고급 옵션 전략', week: 'Week 7-9' },
    { id: 'hedging-strategies', title: '헤지 전략과 리스크 관리', week: 'Week 7-9' },
    { id: 'global-markets', title: '글로벌 시장 투자', week: 'Week 10-11' },
    { id: 'alternative-investments', title: '대안 투자 전략', week: 'Week 10-11' },
    { id: 'macro-trading', title: '매크로 트레이딩', week: 'Week 12' },
    { id: 'currency-hedging-strategies', title: '통화 헤지 전략', week: 'Week 13' },
    { id: 'global-macro-investing', title: '글로벌 매크로 투자', week: 'Week 14' },
    { id: 'international-diversification', title: '국제 분산투자', week: 'Week 15' }
  ];

  // 현재 프로그램에 따른 챕터 목록 선택
  const chapters = programType === 'baby-chick' ? babyChickChapters :
                   programType === 'foundation' ? foundationChapters :
                   programType === 'advanced' ? advancedChapters : [];
  
  // 현재 챕터 인덱스 찾기
  const currentIndex = chapters.findIndex(chapter => chapter.id === currentChapterId);
  
  if (currentIndex === -1) return null; // 챕터를 찾을 수 없는 경우
  
  const previousChapter = currentIndex > 0 ? chapters[currentIndex - 1] : null;
  const nextChapter = currentIndex < chapters.length - 1 ? chapters[currentIndex + 1] : null;
  const currentChapter = chapters[currentIndex];

  const getProgramName = (type: string) => {
    switch (type) {
      case 'baby-chick': return '🐣 Baby Chick Program';
      case 'foundation': return 'Foundation Program';
      case 'advanced': return 'Advanced Program';
      case 'professional': return 'Professional Program';
      default: return 'Program';
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Progress Indicator */}
        <div className="mb-6">
          <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400 mb-2">
            <span>{currentChapter.week}</span>
            <span>{currentIndex + 1} / {chapters.length}</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div 
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${((currentIndex + 1) / chapters.length) * 100}%` }}
            />
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            {getProgramName(programType)} 진행률: {Math.round(((currentIndex + 1) / chapters.length) * 100)}%
          </div>
        </div>

        {/* Navigation Buttons */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Previous Chapter */}
          <div className="md:col-span-1">
            {previousChapter ? (
              <Link
                href={`/modules/stock-analysis/chapters/${previousChapter.id}`}
                className="group flex items-center gap-3 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
              >
                <ArrowLeft className="w-5 h-5 text-gray-400 group-hover:text-blue-500 transition-colors" />
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-gray-500 dark:text-gray-400">이전 챕터</p>
                  <p className="font-medium text-gray-900 dark:text-white truncate group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                    {previousChapter.title}
                  </p>
                </div>
              </Link>
            ) : (
              <div className="flex items-center gap-3 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg opacity-50">
                <ArrowLeft className="w-5 h-5 text-gray-300" />
                <div className="flex-1">
                  <p className="text-xs text-gray-400">첫 번째 챕터입니다</p>
                </div>
              </div>
            )}
          </div>

          {/* Home Button */}
          <div className="md:col-span-1">
            <Link
              href={`/modules/stock-analysis/stages/${programType}`}
              className="group flex items-center justify-center gap-3 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors"
            >
              <Home className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              <span className="font-medium text-blue-600 dark:text-blue-400">
                {getProgramName(programType)}
              </span>
            </Link>
          </div>

          {/* Next Chapter */}
          <div className="md:col-span-1">
            {nextChapter ? (
              <Link
                href={`/modules/stock-analysis/chapters/${nextChapter.id}`}
                className="group flex items-center gap-3 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
              >
                <div className="flex-1 min-w-0 text-right">
                  <p className="text-xs text-gray-500 dark:text-gray-400">다음 챕터</p>
                  <p className="font-medium text-gray-900 dark:text-white truncate group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                    {nextChapter.title}
                  </p>
                </div>
                <ArrowRight className="w-5 h-5 text-gray-400 group-hover:text-blue-500 transition-colors" />
              </Link>
            ) : (
              <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-4">
                <div className="text-center">
                  <p className="text-xs text-green-600 dark:text-green-400 mb-1">🎉 완료!</p>
                  <p className="font-medium text-green-700 dark:text-green-300">
                    {getProgramName(programType)} 완료
                  </p>
                  <Link
                    href="/modules/stock-analysis/stages/advanced"
                    className="inline-flex items-center gap-2 mt-2 text-xs text-green-600 dark:text-green-400 hover:text-green-700 dark:hover:text-green-300 transition-colors"
                  >
                    Advanced Program 시작하기
                    <ArrowRight className="w-3 h-3" />
                  </Link>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Current Chapter Info */}
        <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-blue-900 dark:text-blue-100">
                현재 학습 중: {currentChapter.title}
              </h3>
              <p className="text-sm text-blue-700 dark:text-blue-300">
                {currentChapter.week} • {currentIndex + 1}번째 챕터
              </p>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                {Math.round(((currentIndex + 1) / chapters.length) * 100)}%
              </div>
              <div className="text-xs text-blue-500 dark:text-blue-400">
                완료율
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}