'use client';

import Link from 'next/link';
import { ArrowLeft, BookOpen, Target, TrendingUp, BarChart3, AlertTriangle, ChevronRight, Play, Clock, Award } from 'lucide-react';

export default function BabyChickStagePage() {
  const curriculum = [
    {
      week: '1주차',
      title: '주식시장의 기본 이해',
      chapters: [
        {
          id: 'market-structure',
          title: '글로벌 금융시장의 구조',
          description: '한국 주식시장은 어떻게 작동하며, 세계 시장과 어떻게 연결되어 있을까?',
          duration: '45분',
          type: 'theory'
        },
        {
          id: 'market-participants',
          title: '시장 참여자의 이해',
          description: '개인, 기관, 외국인 투자자들의 행동 패턴과 시장에 미치는 영향',
          duration: '30분',
          type: 'theory'
        },
        {
          id: 'trading-system',
          title: '매매 시스템 실습',
          description: '호가창 읽기, 주문 유형, 체결 원리 등 실전 매매 기초',
          duration: '60분',
          type: 'practice'
        }
      ]
    },
    {
      week: '2주차',
      title: '투자 심리와 행동재무학',
      chapters: [
        {
          id: 'investor-psychology',
          title: '투자자 심리의 함정',
          description: '손실회피, 확증편향, 군중심리 등 투자 실패의 주요 원인',
          duration: '40분',
          type: 'theory'
        },
        {
          id: 'risk-management-basics',
          title: '리스크 관리 기초',
          description: '손절선 설정, 포지션 사이징, 분산투자의 기본 원칙',
          duration: '45분',
          type: 'theory'
        },
        {
          id: 'psychology-simulation',
          title: '투자 심리 시뮬레이션',
          description: '실제 시장 상황에서 심리적 함정을 체험하고 극복하는 훈련',
          duration: '90분',
          type: 'simulation'
        }
      ]
    },
    {
      week: '3-4주차',
      title: '거시경제 지표 분석',
      chapters: [
        {
          id: 'economic-indicators',
          title: '주요 경제지표의 이해',
          description: 'GDP, 금리, 환율, 인플레이션이 주식시장에 미치는 영향',
          duration: '60분',
          type: 'theory'
        },
        {
          id: 'fomc-analysis',
          title: 'FOMC와 한국은행 통화정책',
          description: '중앙은행 정책이 주식시장에 미치는 영향 분석',
          duration: '45분',
          type: 'theory'
        },
        {
          id: 'macro-practice',
          title: '거시경제 분석 실습',
          description: '실제 경제지표 발표를 분석하고 투자 전략 수립하기',
          duration: '120분',
          type: 'practice'
        }
      ]
    }
  ];

  const tools = [
    {
      name: '호가창 시뮬레이터',
      description: '실시간 호가 변화를 관찰하고 주문 체결 원리 학습',
      icon: BarChart3,
      href: '/modules/stock-analysis/simulators/order-book'
    },
    {
      name: '기초 재무제표 계산기',
      description: 'PER, PBR, ROE 등 기본 지표 계산 연습',
      icon: Target,
      href: '/modules/stock-analysis/tools/basic-calculator'
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
                      className="inline-flex items-center gap-1 text-sm font-medium text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300"
                    >
                      <Play className="w-4 h-4" />
                      시작
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

        <div className="mt-12 text-center">
          <Link
            href="/modules/stock-analysis/stages/young-eagle"
            className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
          >
            다음 단계로 진행
            <ChevronRight className="w-5 h-5" />
          </Link>
          <p className="text-sm text-gray-500 mt-3">
            모든 챕터를 완료하면 Young Eagle 단계로 진급할 수 있습니다
          </p>
        </div>
      </div>
    </div>
  );
}