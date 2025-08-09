'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { 
  Sparkles, 
  GraduationCap, 
  Target, 
  TrendingUp, 
  Award,
  ChevronRight,
  Clock,
  CheckCircle,
  Lock,
  PlayCircle,
  BookOpen,
  Zap
} from 'lucide-react';

interface LearningPath {
  id: string;
  title: string;
  description: string;
  level: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  duration: string;
  chapters: string[];
  simulators: string[];
  icon: React.ReactNode;
  color: string;
  badge: string;
}

export function LearningPathGuide() {
  const [selectedPath, setSelectedPath] = useState<string>('beginner');
  const [userLevel, setUserLevel] = useState<string>('');

  const learningPaths: LearningPath[] = [
    {
      id: 'absolute-beginner',
      title: '주식이 뭔가요?',
      description: '주식을 처음 들어보는 왕초보를 위한 3일 완성 코스',
      level: 'beginner',
      duration: '3일 (하루 20분)',
      chapters: ['what-is-stock', 'why-invest', 'stock-market-basics'],
      simulators: ['stock-basics-simulator', 'simple-calculator'],
      icon: <Sparkles className="w-6 h-6" />,
      color: 'from-green-400 to-green-500',
      badge: '🌱 왕초보'
    },
    {
      id: 'beginner',
      title: '첫 주식 사보기',
      description: '계좌 만들고 실제로 주식 사는 법 배우기',
      level: 'beginner',
      duration: '1주 (하루 30분)',
      chapters: ['how-to-start', 'order-types', 'first-stock-selection'],
      simulators: ['trading-practice', 'simple-portfolio'],
      icon: <PlayCircle className="w-6 h-6" />,
      color: 'from-green-500 to-blue-500',
      badge: '🎯 입문자'
    },
    {
      id: 'chart-basics',
      title: '차트 읽기 기초',
      description: '빨간색 파란색부터 시작하는 차트 읽기',
      level: 'intermediate',
      duration: '2주 (하루 40분)',
      chapters: ['basic-chart-reading', 'simple-indicators', 'trend-basics'],
      simulators: ['chart-practice', 'pattern-game'],
      icon: <BookOpen className="w-6 h-6" />,
      color: 'from-blue-400 to-blue-600',
      badge: '📊 초급자'
    },
    {
      id: 'smart-investor',
      title: '똑똑한 투자자 되기',
      description: '기업 분석하고 좋은 주식 고르는 법',
      level: 'intermediate',
      duration: '4주 (하루 1시간)',
      chapters: ['company-analysis-basics', 'simple-valuation', 'buy-sell-timing'],
      simulators: ['company-analyzer', 'simple-trading-game'],
      icon: <GraduationCap className="w-6 h-6" />,
      color: 'from-blue-600 to-purple-500',
      badge: '🎓 중급자'
    },
    {
      id: 'technical-analysis',
      title: '기술적 분석 배우기',
      description: '차트 패턴과 지표로 매매 타이밍 잡기',
      level: 'advanced',
      duration: '6주 (하루 1시간)',
      chapters: ['technical-indicators', 'chart-patterns', 'trading-strategies'],
      simulators: ['chart-analyzer', 'backtesting-engine'],
      icon: <TrendingUp className="w-6 h-6" />,
      color: 'from-purple-500 to-pink-500',
      badge: '📈 상급자'
    },
    {
      id: 'professional',
      title: '전문 투자자 과정',
      description: '재무제표 분석부터 포트폴리오 관리까지',
      level: 'expert',
      duration: '8주 (하루 2시간)',
      chapters: ['financial-analysis', 'portfolio-management', 'risk-management'],
      simulators: ['dcf-valuation-model', 'portfolio-optimizer', 'ai-mentor'],
      icon: <Award className="w-6 h-6" />,
      color: 'from-pink-500 to-red-600',
      badge: '🚀 전문가'
    }
  ];

  const quickStartQuestions = [
    {
      question: "주식 투자 경험이 있으신가요?",
      options: [
        { text: "주식이 뭔지 모르겠어요", value: 'absolute-beginner' },
        { text: "들어는 봤는데 해본 적 없어요", value: 'beginner' },
        { text: "계좌는 있는데 잘 모르겠어요", value: 'chart-basics' },
        { text: "조금씩 사보고 있어요", value: 'smart-investor' },
        { text: "1년 이상 투자하고 있어요", value: 'technical-analysis' },
        { text: "전문적으로 투자해요", value: 'professional' }
      ]
    }
  ];

  const currentPath = learningPaths.find(p => p.id === selectedPath);

  return (
    <div className="space-y-8">
      {/* 빠른 시작 퀴즈 */}
      {!userLevel && (
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-2xl p-8 text-center">
          <h3 className="text-2xl font-bold mb-4">🎯 나에게 맞는 학습 경로 찾기</h3>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            간단한 질문으로 최적의 학습 경로를 추천해드려요
          </p>
          <div className="space-y-4 max-w-md mx-auto">
            {quickStartQuestions[0].options.map((option) => (
              <button
                key={option.value}
                onClick={() => {
                  setUserLevel(option.value);
                  setSelectedPath(option.value);
                }}
                className="w-full p-4 bg-white dark:bg-gray-800 rounded-lg hover:shadow-lg transition-all duration-200 text-left flex items-center justify-between group"
              >
                <span className="font-medium">{option.text}</span>
                <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-blue-500 group-hover:translate-x-1 transition-all" />
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 학습 경로 선택 */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        {learningPaths.map((path) => (
          <button
            key={path.id}
            onClick={() => setSelectedPath(path.id)}
            className={`relative p-6 rounded-xl transition-all duration-200 ${
              selectedPath === path.id
                ? 'ring-2 ring-blue-500 shadow-lg scale-105 bg-blue-50 dark:bg-blue-900/20'
                : 'bg-white dark:bg-gray-800 hover:shadow-md'
            } ${
              userLevel && userLevel !== path.id && path.level === 'expert'
                ? 'opacity-50'
                : ''
            }`}
          >
            <div>
              <div className="flex items-start justify-between mb-4">
                <div className={`p-3 rounded-lg bg-gradient-to-r ${path.color} text-white`}>
                  {path.icon}
                </div>
                <span className="text-xs px-2 py-1 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300">
                  {path.badge}
                </span>
              </div>
              
              <h3 className="font-bold text-lg mb-2 text-gray-900 dark:text-white">
                {path.title}
              </h3>
              
              <p className="text-sm mb-4 text-gray-600 dark:text-gray-400">
                {path.description}
              </p>
              
              <div className="flex items-center gap-1 text-xs text-gray-500">
                <Clock className="w-3 h-3" />
                <span>{path.duration}</span>
              </div>
            </div>
            
            {userLevel === path.id && (
              <div className="absolute -top-2 -right-2">
                <span className="flex h-6 w-6">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-6 w-6 bg-green-500 items-center justify-center">
                    <Zap className="w-3 h-3 text-white" />
                  </span>
                </span>
              </div>
            )}
          </button>
        ))}
      </div>

      {/* 선택된 경로 상세 정보 */}
      {currentPath && (
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              <div className={`p-4 rounded-xl bg-gradient-to-r ${currentPath.color} text-white`}>
                {currentPath.icon}
              </div>
              <div>
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                  {currentPath.title}
                </h2>
                <p className="text-gray-600 dark:text-gray-400">
                  {currentPath.duration} 완성 과정
                </p>
              </div>
            </div>
            <Link
              href={`/modules/stock-analysis/learning-path/${currentPath.id}`}
              className={`px-6 py-3 rounded-lg font-medium bg-gradient-to-r ${currentPath.color} text-white hover:shadow-lg transition-all duration-200 flex items-center gap-2`}
            >
              <PlayCircle className="w-5 h-5" />
              학습 시작하기
            </Link>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {/* 포함된 챕터 */}
            <div>
              <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-blue-500" />
                학습할 내용
              </h3>
              <div className="space-y-3">
                {currentPath.chapters.map((chapterId, index) => {
                  const chapterNames: { [key: string]: string } = {
                    // 왕초보 과정
                    'what-is-stock': '주식이 도대체 뭔가요?',
                    'why-invest': '왜 사람들이 주식을 살까?',
                    'stock-market-basics': '주식시장은 어떻게 돌아갈까?',
                    
                    // 입문자 과정
                    'how-to-start': '증권 계좌 만들기 A to Z',
                    'order-types': '매수, 매도 주문하는 법',
                    'first-stock-selection': '내 첫 주식 고르기',
                    
                    // 초급자 과정
                    'basic-chart-reading': '차트의 빨간색 파란색이 뭔가요?',
                    'simple-indicators': '이동평균선과 거래량 보기',
                    'trend-basics': '상승장 하락장 구분하기',
                    
                    // 중급자 과정
                    'company-analysis-basics': '좋은 회사 고르는 법',
                    'simple-valuation': '주가가 싼지 비싼지 알아보기',
                    'buy-sell-timing': '언제 사고 팔아야 할까?',
                    
                    // 상급자 과정
                    'technical-indicators': 'RSI, MACD 등 지표 활용하기',
                    'chart-patterns': '차트 패턴으로 예측하기',
                    'trading-strategies': '나만의 매매 전략 만들기',
                    
                    // 전문가 과정
                    'financial-analysis': '재무제표 깊이 분석하기',
                    'portfolio-management': '포트폴리오 관리 전략',
                    'risk-management': '리스크 관리와 헤지 전략'
                  };
                  
                  return (
                    <div key={chapterId} className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-full flex items-center justify-center text-sm font-medium">
                        {index + 1}
                      </div>
                      <span className="text-gray-700 dark:text-gray-300">
                        {chapterNames[chapterId] || chapterId}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* 사용할 시뮬레이터 */}
            <div>
              <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                <Target className="w-5 h-5 text-purple-500" />
                실습 도구
              </h3>
              <div className="space-y-3">
                {currentPath.simulators.map((simulatorId) => {
                  const simulatorNames: { [key: string]: string } = {
                    // 초보자용
                    'stock-basics-simulator': '주식이 뭔지 알아보기',
                    'simple-calculator': '수익률 계산해보기',
                    'trading-practice': '가상으로 주식 사보기',
                    'simple-portfolio': '내 주식 관리하기',
                    
                    // 초급자용
                    'chart-practice': '차트 보는 연습',
                    'pattern-game': '패턴 찾기 게임',
                    
                    // 중급자용
                    'company-analyzer': '회사 분석 도구',
                    'simple-trading-game': '모의 투자 게임',
                    
                    // 상급자용
                    'chart-analyzer': 'AI 차트 분석기',
                    'backtesting-engine': '전략 테스트하기',
                    
                    // 전문가용
                    'dcf-valuation-model': 'DCF 가치평가 모델',
                    'portfolio-optimizer': '포트폴리오 최적화',
                    'ai-mentor': 'AI 투자 조언',
                    
                    // 기타
                    'financial-calculator': '재무 분석 도구',
                    'earnings-forecast-model': '실적 예측 모델',
                    'dividend-growth-analyzer': '배당 분석기',
                    'factor-investing-lab': '팩터 투자 실험실',
                    'correlation-matrix-analyzer': '상관관계 분석',
                    'macro-economic-dashboard': '경제 지표 대시보드'
                  };
                  
                  return (
                    <div key={simulatorId} className="flex items-center gap-3">
                      <CheckCircle className="w-5 h-5 text-green-500" />
                      <span className="text-gray-700 dark:text-gray-300">
                        {simulatorNames[simulatorId] || simulatorId}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* 학습 후 도달 수준 */}
          <div className="mt-8 p-6 bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-700/30 dark:to-gray-800/30 rounded-xl">
            <h3 className="font-semibold text-lg mb-3 flex items-center gap-2">
              <Award className="w-5 h-5 text-yellow-500" />
              이 과정을 마치면
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              {currentPath.id === 'absolute-beginner' && (
                <>
                  <li>✅ 주식이 무엇인지 이해하게 됩니다</li>
                  <li>✅ 주식시장이 어떻게 돌아가는지 알게 됩니다</li>
                  <li>✅ 주식 투자를 시작할 준비가 됩니다</li>
                </>
              )}
              {currentPath.id === 'beginner' && (
                <>
                  <li>✅ 증권 계좌를 만들 수 있습니다</li>
                  <li>✅ 실제로 주식을 사고 팔 수 있습니다</li>
                  <li>✅ 기본적인 주문 방법을 익힙니다</li>
                </>
              )}
              {currentPath.id === 'chart-basics' && (
                <>
                  <li>✅ 차트를 보고 주가 흐름을 읽을 수 있습니다</li>
                  <li>✅ 이동평균선의 의미를 이해합니다</li>
                  <li>✅ 상승장과 하락장을 구분할 수 있습니다</li>
                </>
              )}
              {currentPath.id === 'smart-investor' && (
                <>
                  <li>✅ 좋은 회사를 고르는 기준이 생깁니다</li>
                  <li>✅ 주가가 적정한지 판단할 수 있습니다</li>
                  <li>✅ 매수와 매도 타이밍을 잡을 수 있습니다</li>
                </>
              )}
              {currentPath.id === 'technical-analysis' && (
                <>
                  <li>✅ 기술적 지표를 활용할 수 있습니다</li>
                  <li>✅ 차트 패턴을 읽고 예측할 수 있습니다</li>
                  <li>✅ 자신만의 매매 전략을 만들 수 있습니다</li>
                </>
              )}
              {currentPath.id === 'professional' && (
                <>
                  <li>✅ 재무제표를 깊이 분석할 수 있습니다</li>
                  <li>✅ 포트폴리오를 전문적으로 관리합니다</li>
                  <li>✅ 리스크를 체계적으로 관리할 수 있습니다</li>
                </>
              )}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}