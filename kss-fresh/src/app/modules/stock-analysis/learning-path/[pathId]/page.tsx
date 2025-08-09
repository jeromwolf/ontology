'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { 
  ArrowLeft, 
  PlayCircle, 
  Clock, 
  CheckCircle,
  Lock,
  Trophy,
  BookOpen,
  Target
} from 'lucide-react';

interface Chapter {
  id: string;
  title: string;
  duration: string;
  completed?: boolean;
}

interface Simulator {
  id: string;
  title: string;
  description: string;
}

interface LearningPathData {
  id: string;
  title: string;
  description: string;
  duration: string;
  chapters: Chapter[];
  simulators: Simulator[];
  prerequisites?: string[];
  nextPath?: string;
}

export default function LearningPathPage() {
  const params = useParams();
  const pathId = params.pathId as string;
  const [progress, setProgress] = useState(0);
  const [completedChapters, setCompletedChapters] = useState<Set<string>>(new Set());

  // 학습 경로 데이터
  const learningPaths: { [key: string]: LearningPathData } = {
    'absolute-beginner': {
      id: 'absolute-beginner',
      title: '주식이 뭔가요?',
      description: '주식을 처음 들어보는 왕초보를 위한 3일 완성 코스입니다. 주식의 기본 개념부터 시작해서 왜 사람들이 투자하는지, 주식시장이 어떻게 돌아가는지 쉽게 배워봅시다.',
      duration: '3일 (하루 20분)',
      chapters: [
        { id: 'what-is-stock', title: '주식이 도대체 뭔가요?', duration: '20분' },
        { id: 'why-invest', title: '왜 사람들이 주식을 살까?', duration: '20분' },
        { id: 'stock-market-basics', title: '주식시장은 어떻게 돌아갈까?', duration: '20분' }
      ],
      simulators: [
        { id: 'stock-basics-simulator', title: '주식이 뭔지 알아보기', description: '기업과 주식의 관계를 시각적으로 이해해봅시다' },
        { id: 'simple-calculator', title: '수익률 계산해보기', description: '투자 수익이 어떻게 계산되는지 체험해봅시다' }
      ],
      nextPath: 'beginner'
    },
    'beginner': {
      id: 'beginner',
      title: '첫 주식 사보기',
      description: '이제 실제로 주식을 사볼 준비를 해봅시다. 증권 계좌를 만들고, 주문하는 방법을 배우고, 첫 주식을 고르는 기준까지 알아봅니다.',
      duration: '1주 (하루 30분)',
      prerequisites: ['주식의 기본 개념 이해'],
      chapters: [
        { id: 'how-to-start', title: '증권 계좌 만들기 A to Z', duration: '30분' },
        { id: 'order-types', title: '매수, 매도 주문하는 법', duration: '30분' },
        { id: 'first-stock-selection', title: '내 첫 주식 고르기', duration: '30분' }
      ],
      simulators: [
        { id: 'trading-practice', title: '가상으로 주식 사보기', description: '실제 돈을 쓰지 않고 연습해봅시다' },
        { id: 'simple-portfolio', title: '내 주식 관리하기', description: '포트폴리오 관리의 기초를 배워봅시다' }
      ],
      nextPath: 'chart-basics'
    },
    'chart-basics': {
      id: 'chart-basics',
      title: '차트 읽기 기초',
      description: '주식 차트를 보면 빨간색, 파란색 막대가 나옵니다. 이게 뭔지부터 시작해서 기본적인 차트 읽는 법을 배워봅시다.',
      duration: '2주 (하루 40분)',
      prerequisites: ['주식 거래 경험'],
      chapters: [
        { id: 'basic-chart-reading', title: '차트의 빨간색 파란색이 뭔가요?', duration: '40분' },
        { id: 'simple-indicators', title: '이동평균선과 거래량 보기', duration: '40분' },
        { id: 'trend-basics', title: '상승장 하락장 구분하기', duration: '40분' }
      ],
      simulators: [
        { id: 'chart-practice', title: '차트 보는 연습', description: '실제 차트를 보며 패턴을 익혀봅시다' },
        { id: 'pattern-game', title: '패턴 찾기 게임', description: '게임으로 재미있게 차트 패턴을 배워봅시다' }
      ],
      nextPath: 'smart-investor'
    },
    'smart-investor': {
      id: 'smart-investor',
      title: '똑똑한 투자자 되기',
      description: '이제 좀 더 체계적으로 투자해봅시다. 좋은 회사를 고르는 방법, 적정 주가를 판단하는 법, 매매 타이밍을 잡는 법을 배웁니다.',
      duration: '4주 (하루 1시간)',
      prerequisites: ['차트 읽기 능력'],
      chapters: [
        { id: 'company-analysis-basics', title: '좋은 회사 고르는 법', duration: '1시간' },
        { id: 'simple-valuation', title: '주가가 싼지 비싼지 알아보기', duration: '1시간' },
        { id: 'buy-sell-timing', title: '언제 사고 팔아야 할까?', duration: '1시간' }
      ],
      simulators: [
        { id: 'company-analyzer', title: '회사 분석 도구', description: '기업의 재무 상태를 쉽게 분석해봅시다' },
        { id: 'simple-trading-game', title: '모의 투자 게임', description: '실전처럼 투자 전략을 테스트해봅시다' }
      ],
      nextPath: 'technical-analysis'
    },
    'technical-analysis': {
      id: 'technical-analysis',
      title: '기술적 분석 배우기',
      description: '차트를 더 깊이 분석하는 방법을 배웁니다. RSI, MACD 같은 지표들을 활용하고, 차트 패턴으로 미래를 예측하는 방법을 익힙니다.',
      duration: '6주 (하루 1시간)',
      prerequisites: ['기본적 분석 능력', '차트 읽기 숙달'],
      chapters: [
        { id: 'technical-indicators', title: 'RSI, MACD 등 지표 활용하기', duration: '1시간' },
        { id: 'chart-patterns', title: '차트 패턴으로 예측하기', duration: '1시간' },
        { id: 'trading-strategies', title: '나만의 매매 전략 만들기', duration: '1시간' }
      ],
      simulators: [
        { id: 'chart-analyzer', title: 'AI 차트 분석기', description: 'AI가 도와주는 차트 분석을 체험해봅시다' },
        { id: 'backtesting-engine', title: '전략 테스트하기', description: '내 전략이 과거에 얼마나 수익을 냈을지 확인해봅시다' }
      ],
      nextPath: 'professional'
    },
    'professional': {
      id: 'professional',
      title: '전문 투자자 과정',
      description: '이제 전문가 수준의 투자를 배워봅시다. 재무제표를 깊이 분석하고, 포트폴리오를 체계적으로 관리하며, 리스크를 관리하는 방법을 익힙니다.',
      duration: '8주 (하루 2시간)',
      prerequisites: ['기술적 분석 숙달', '투자 경험 1년 이상'],
      chapters: [
        { id: 'financial-analysis', title: '재무제표 깊이 분석하기', duration: '2시간' },
        { id: 'portfolio-management', title: '포트폴리오 관리 전략', duration: '2시간' },
        { id: 'risk-management', title: '리스크 관리와 헤지 전략', duration: '2시간' }
      ],
      simulators: [
        { id: 'dcf-valuation-model', title: 'DCF 가치평가 모델', description: '기업의 진정한 가치를 계산해봅시다' },
        { id: 'portfolio-optimizer', title: '포트폴리오 최적화', description: '최적의 자산 배분을 찾아봅시다' },
        { id: 'ai-mentor', title: 'AI 투자 조언', description: 'AI가 제공하는 전문가급 투자 조언을 받아봅시다' }
      ]
    }
  };

  const currentPath = learningPaths[pathId];

  useEffect(() => {
    // 진행률 계산
    const saved = localStorage.getItem(`stock-path-${pathId}-completed`);
    if (saved) {
      setCompletedChapters(new Set(JSON.parse(saved)));
    }
  }, [pathId]);

  useEffect(() => {
    const completed = completedChapters.size;
    const total = currentPath?.chapters.length || 1;
    setProgress((completed / total) * 100);
  }, [completedChapters, currentPath]);

  const handleChapterComplete = (chapterId: string) => {
    const newCompleted = new Set(completedChapters);
    if (newCompleted.has(chapterId)) {
      newCompleted.delete(chapterId);
    } else {
      newCompleted.add(chapterId);
    }
    setCompletedChapters(newCompleted);
    localStorage.setItem(`stock-path-${pathId}-completed`, JSON.stringify(Array.from(newCompleted)));
  };

  if (!currentPath) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">학습 경로를 찾을 수 없습니다</h1>
          <Link href="/modules/stock-analysis" className="text-blue-600 hover:underline">
            돌아가기
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* 헤더 */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <Link 
              href="/modules/stock-analysis"
              className="flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>학습 경로 선택으로 돌아가기</span>
            </Link>
            
            <div className="flex items-center gap-4">
              <div className="text-right">
                <p className="text-sm text-gray-600 dark:text-gray-400">전체 진행률</p>
                <p className="text-lg font-bold text-gray-900 dark:text-white">{Math.round(progress)}%</p>
              </div>
              <div className="w-32 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-blue-500 to-purple-600 transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 메인 콘텐츠 */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* 경로 소개 */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 mb-8 shadow-lg">
          <div className="flex items-start justify-between mb-6">
            <div className="flex-1">
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
                {currentPath.title}
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400 mb-4">
                {currentPath.description}
              </p>
              <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                <div className="flex items-center gap-1">
                  <Clock className="w-4 h-4" />
                  <span>{currentPath.duration}</span>
                </div>
                <div className="flex items-center gap-1">
                  <BookOpen className="w-4 h-4" />
                  <span>{currentPath.chapters.length}개 챕터</span>
                </div>
                <div className="flex items-center gap-1">
                  <Target className="w-4 h-4" />
                  <span>{currentPath.simulators.length}개 실습</span>
                </div>
              </div>
            </div>
            
            {progress === 100 && (
              <div className="flex items-center gap-2 px-4 py-2 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded-lg">
                <Trophy className="w-5 h-5" />
                <span className="font-medium">완료!</span>
              </div>
            )}
          </div>

          {/* 선수 과목 */}
          {currentPath.prerequisites && (
            <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <p className="text-sm text-yellow-800 dark:text-yellow-300">
                <strong>선수 지식:</strong> {currentPath.prerequisites.join(', ')}
              </p>
            </div>
          )}
        </div>

        {/* 학습 내용 */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* 챕터 목록 */}
          <div>
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-blue-500" />
              학습 내용
            </h2>
            <div className="space-y-3">
              {currentPath.chapters.map((chapter, index) => {
                const isCompleted = completedChapters.has(chapter.id);
                const isLocked = index > 0 && !completedChapters.has(currentPath.chapters[index - 1].id);
                
                return (
                  <div
                    key={chapter.id}
                    className={`bg-white dark:bg-gray-800 rounded-lg p-4 border ${
                      isCompleted 
                        ? 'border-green-500 bg-green-50 dark:bg-green-900/10' 
                        : isLocked
                        ? 'border-gray-200 dark:border-gray-700 opacity-50'
                        : 'border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600'
                    } transition-all duration-200`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium ${
                          isCompleted
                            ? 'bg-green-500 text-white'
                            : isLocked
                            ? 'bg-gray-200 dark:bg-gray-700 text-gray-400'
                            : 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                        }`}>
                          {isCompleted ? <CheckCircle className="w-5 h-5" /> : isLocked ? <Lock className="w-5 h-5" /> : index + 1}
                        </div>
                        <div className="flex-1">
                          <h3 className="font-medium text-gray-900 dark:text-white">
                            {chapter.title}
                          </h3>
                          <p className="text-sm text-gray-500 dark:text-gray-400">
                            {chapter.duration}
                          </p>
                        </div>
                      </div>
                      
                      {!isLocked && (
                        <button
                          onClick={() => handleChapterComplete(chapter.id)}
                          className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                            isCompleted
                              ? 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-300 dark:hover:bg-gray-600'
                              : 'bg-blue-600 text-white hover:bg-blue-700'
                          }`}
                        >
                          {isCompleted ? '다시 학습' : '학습하기'}
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 실습 도구 */}
          <div>
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-purple-500" />
              실습 도구
            </h2>
            <div className="space-y-3">
              {currentPath.simulators.map((simulator) => (
                <div
                  key={simulator.id}
                  className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700 hover:border-purple-300 dark:hover:border-purple-600 transition-all duration-200"
                >
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 rounded-lg flex items-center justify-center">
                      <PlayCircle className="w-5 h-5" />
                    </div>
                    <div className="flex-1">
                      <h3 className="font-medium text-gray-900 dark:text-white mb-1">
                        {simulator.title}
                      </h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {simulator.description}
                      </p>
                    </div>
                    <Link
                      href={`/modules/stock-analysis/simulators/${simulator.id}`}
                      className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium"
                    >
                      실습하기
                    </Link>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* 다음 경로 추천 */}
        {currentPath.nextPath && progress === 100 && (
          <div className="mt-8 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-2xl p-8 text-center">
            <Trophy className="w-12 h-12 text-yellow-500 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              축하합니다! 🎉
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-6">
              {currentPath.title} 과정을 모두 완료했습니다!
            </p>
            <Link
              href={`/modules/stock-analysis/learning-path/${currentPath.nextPath}`}
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:shadow-lg transition-all duration-200 font-medium"
            >
              다음 단계로 진행하기
              <ArrowLeft className="w-5 h-5 rotate-180" />
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}