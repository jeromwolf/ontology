'use client';

import Link from 'next/link';
import { ArrowLeft, BookOpen, Target, TrendingUp, Mouse, Smartphone, ChevronRight, Play, Clock, Award, Sparkles, BarChart, FileText, Users, AlertTriangle } from 'lucide-react';

export default function BabyChickStagePage() {
  const curriculum = [
    {
      week: 'Week 1',
      title: '주식 투자의 첫걸음',
      chapters: [
        {
          id: 'what-is-stock',
          title: '주식이란 무엇인가?',
          description: '기업의 일부를 소유한다는 의미, 주식이 오르고 내리는 이유, 배당금이란?',
          duration: '30분',
          type: 'theory',
          level: 'Beginner'
        },
        {
          id: 'open-account',
          title: '증권계좌 개설하기',
          description: '증권사 선택 기준, 비대면 계좌 개설 과정, 수수료 비교하기',
          duration: '45분',
          type: 'practice',
          level: 'Beginner'
        },
        {
          id: 'trading-app-basics',
          title: 'HTS/MTS 사용법',
          description: '주식 앱 설치하고 둘러보기, 관심종목 등록, 첫 주문 넣어보기',
          duration: '60분',
          type: 'hands-on',
          level: 'Beginner'
        }
      ]
    },
    {
      week: 'Week 2',
      title: '차트 읽기의 기초',
      chapters: [
        {
          id: 'understanding-candles',
          title: '캔들 차트 이해하기',
          description: '빨간색과 파란색의 의미, 몸통과 꼬리 읽기, 일봉/주봉/월봉의 차이',
          duration: '45분',
          type: 'visual',
          level: 'Beginner'
        },
        {
          id: 'volume-basics',
          title: '거래량이 말해주는 것',
          description: '거래량이 많다는 것의 의미, 가격과 거래량의 관계, 거래량 급증 신호',
          duration: '30분',
          type: 'analysis',
          level: 'Beginner'
        },
        {
          id: 'order-book',
          title: '호가창 완전정복',
          description: '매수/매도 호가 읽기, 잔량의 의미, 시장가와 지정가 주문',
          duration: '45분',
          type: 'interactive',
          level: 'Beginner'
        }
      ]
    },
    {
      week: 'Week 3',
      title: '기초 용어와 정보 수집',
      chapters: [
        {
          id: 'basic-terms',
          title: '꼭 알아야 할 투자 용어',
          description: '시가총액, PER, PBR, ROE, 배당수익률 등 핵심 지표 이해하기',
          duration: '60분',
          type: 'theory',
          level: 'Beginner'
        },
        {
          id: 'reading-news',
          title: '투자 뉴스 읽는 법',
          description: '어떤 뉴스가 주가에 영향을 미치나, 공시 읽기, 루머와 사실 구분하기',
          duration: '45분',
          type: 'analysis',
          level: 'Beginner'
        },
        {
          id: 'sectors-themes',
          title: '업종과 테마 이해하기',
          description: 'KOSPI 업종 분류, 테마주란?, 관련주 찾는 방법',
          duration: '45분',
          type: 'research',
          level: 'Beginner'
        }
      ]
    },
    {
      week: 'Week 4',
      title: '첫 투자 시작하기',
      chapters: [
        {
          id: 'stock-selection',
          title: '종목 선택의 기초',
          description: '대형주 vs 중소형주, 안전한 종목 찾기, 위험한 종목 피하기',
          duration: '60분',
          type: 'strategy',
          level: 'Beginner'
        },
        {
          id: 'small-start',
          title: '소액으로 시작하기',
          description: '얼마로 시작할까?, 분할 매수의 중요성, 첫 수익과 손실 대처법',
          duration: '45분',
          type: 'practice',
          level: 'Beginner'
        },
        {
          id: 'trading-diary',
          title: '매매일지 작성하기',
          description: '왜 샀는지 기록하기, 매도 이유 정리하기, 실수에서 배우기',
          duration: '30분',
          type: 'reflection',
          level: 'Beginner'
        }
      ]
    }
  ];

  const tools = [
    {
      name: '모의투자 시뮬레이터',
      description: '실제 시장 데이터로 연습하는 가상 투자 플랫폼',
      icon: Mouse,
      href: '/modules/stock-analysis/simulators/paper-trading',
      badge: '인기'
    },
    {
      name: '용어 사전',
      description: '초보자를 위한 주식 용어 완벽 정리',
      icon: FileText,
      href: '/modules/stock-analysis/tools/glossary',
      badge: '필수'
    },
    {
      name: '차트 연습장',
      description: '캔들 패턴 그려보고 익히기',
      icon: BarChart,
      href: '/modules/stock-analysis/simulators/chart-practice',
      badge: '추천'
    }
  ];

  const achievements = [
    {
      title: '시장 이해도',
      description: '주식시장의 작동 원리와 참여자 이해',
      icon: '🏛️'
    },
    {
      title: '심리 통제력',
      description: '투자 심리의 함정을 인식하고 극복',
      icon: '🧠'
    },
    {
      title: '거시 분석력',
      description: '경제지표가 시장에 미치는 영향 파악',
      icon: '📊'
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

      {/* Hero Section */}
      <div className="bg-yellow-50 dark:bg-gray-800 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-6 mb-6">
            <div className="w-20 h-20 bg-yellow-100 rounded-full flex items-center justify-center">
              <span className="text-4xl">🐥</span>
            </div>
            <div>
              <div className="flex items-center gap-3 mb-2">
                <span className="text-sm font-medium text-gray-500">Stage 1</span>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                  Baby Chick - 병아리 투자자
                </h1>
              </div>
              <p className="text-lg text-gray-600 dark:text-gray-400">
                시장의 작동 원리와 기본 메커니즘을 이해하는 단계
              </p>
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-4 mt-8">
            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <div className="flex items-center gap-3 mb-2">
                <Clock className="w-5 h-5 text-gray-500" />
                <span className="text-sm font-medium">학습 기간</span>
              </div>
              <p className="text-lg font-semibold">4주 과정</p>
            </div>
            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <div className="flex items-center gap-3 mb-2">
                <BookOpen className="w-5 h-5 text-gray-500" />
                <span className="text-sm font-medium">커리큘럼</span>
              </div>
              <p className="text-lg font-semibold">9개 챕터</p>
            </div>
            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <div className="flex items-center gap-3 mb-2">
                <Award className="w-5 h-5 text-gray-500" />
                <span className="text-sm font-medium">목표</span>
              </div>
              <p className="text-lg font-semibold">시장 기초 마스터</p>
            </div>
          </div>
        </div>
      </div>

      {/* Warning Banner */}
      <div className="bg-amber-50 border-l-4 border-amber-400 p-4 my-6 max-w-7xl mx-auto">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5" />
          <div>
            <h3 className="text-sm font-medium text-amber-800">중요 안내</h3>
            <p className="text-sm text-amber-700 mt-1">
              이 단계에서는 실제 매매보다는 시장의 작동 원리를 이해하는 것이 중요합니다. 
              충분한 학습 없이 실전 투자를 시작하면 큰 손실을 볼 수 있습니다.
            </p>
          </div>
        </div>
      </div>

      {/* Curriculum Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
          학습 커리큘럼
        </h2>

        <div className="space-y-8">
          {curriculum.map((week) => (
            <div key={week.week} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
              <div className="bg-gray-50 dark:bg-gray-700 px-6 py-4">
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-sm font-medium text-gray-500 dark:text-gray-400">{week.week}</span>
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white mt-1">
                      {week.title}
                    </h3>
                  </div>
                  <TrendingUp className="w-6 h-6 text-gray-400" />
                </div>
              </div>

              <div className="p-6 space-y-4">
                {week.chapters.map((chapter, index) => (
                  <div key={chapter.id} className="flex items-start gap-4">
                    <div className="w-8 h-8 bg-gray-100 dark:bg-gray-600 rounded-full flex items-center justify-center text-sm font-medium">
                      {index + 1}
                    </div>
                    <div className="flex-1">
                      <h4 className="font-semibold text-gray-900 dark:text-white mb-1">
                        {chapter.title}
                      </h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        {chapter.description}
                      </p>
                      <div className="flex items-center gap-4 text-xs">
                        <span className="flex items-center gap-1 text-gray-500">
                          <Clock className="w-3 h-3" />
                          {chapter.duration}
                        </span>
                        <span className={`px-2 py-1 rounded-full ${
                          chapter.type === 'theory' 
                            ? 'bg-blue-100 text-blue-700' 
                            : chapter.type === 'practice'
                            ? 'bg-green-100 text-green-700'
                            : 'bg-purple-100 text-purple-700'
                        }`}>
                          {chapter.type === 'theory' ? '이론' : chapter.type === 'practice' ? '실습' : '시뮬레이션'}
                        </span>
                      </div>
                    </div>
                    <Link
                      href={`/modules/stock-analysis/chapters/${chapter.id}`}
                      className="inline-flex items-center gap-2 px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-white rounded-lg font-medium transition-colors"
                    >
                      <Play className="w-4 h-4" />
                      시작하기
                    </Link>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Tools Section */}
      <div className="bg-gray-100 dark:bg-gray-800 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Stage 1 전용 도구
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            {tools.map((tool) => {
              const Icon = tool.icon;
              return (
                <Link
                  key={tool.name}
                  href={tool.href}
                  className="bg-white dark:bg-gray-700 rounded-xl p-6 hover:shadow-lg transition-shadow"
                >
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 bg-yellow-100 dark:bg-yellow-900/30 rounded-lg flex items-center justify-center">
                      <Icon className="w-6 h-6 text-yellow-600 dark:text-yellow-400" />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
                        {tool.name}
                      </h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {tool.description}
                      </p>
                    </div>
                    <ChevronRight className="w-5 h-5 text-gray-400" />
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      </div>

      {/* Achievement Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
          학습 목표 달성
        </h2>

        <div className="grid md:grid-cols-3 gap-6">
          {achievements.map((achievement) => (
            <div key={achievement.title} className="text-center">
              <div className="w-20 h-20 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-3xl">{achievement.icon}</span>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                {achievement.title}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {achievement.description}
              </p>
            </div>
          ))}
        </div>

        {/* Next Steps */}
        <div className="mt-16 text-center bg-gradient-to-r from-yellow-100 to-orange-100 dark:from-gray-800 dark:to-gray-700 rounded-2xl p-12">
          <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            다음 단계: Foundation Program
          </h3>
          <p className="text-lg text-gray-600 dark:text-gray-300 mb-8 max-w-2xl mx-auto">
            Baby Chick 과정을 마치면, 차트 분석과 기업 분석을 배우는 Foundation Program으로 진급할 수 있습니다.
          </p>
          <div className="flex items-center justify-center gap-4">
            <Link
              href="/modules/stock-analysis/stages/foundation"
              className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white rounded-xl font-semibold transition-all transform hover:scale-105"
            >
              Foundation Program 미리보기
              <ChevronRight className="w-5 h-5" />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}