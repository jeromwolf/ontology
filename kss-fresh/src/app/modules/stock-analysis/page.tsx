'use client';

import Link from 'next/link';
import { TrendingUp, Shield, ChevronRight, BarChart3, DollarSign, Activity, Target } from 'lucide-react';

export default function StockAnalysisModulePage() {
  // 3단계 성장 경로 (동물 캐릭터)
  const growthStages = [
    {
      id: 'baby-chick',
      stage: 1,
      title: 'Baby Chick',
      koreanTitle: '병아리 투자자',
      description: '시장의 작동 원리와 기본 메커니즘 이해',
      chapters: [
        { id: 'market-fundamentals', title: '글로벌 금융시장의 구조' },
        { id: 'investment-psychology', title: '투자 심리와 행동재무학' },
        { id: 'economic-indicators', title: '거시경제 지표 분석' }
      ],
      tools: ['호가창 시뮬레이터', '기초 재무제표 계산기'],
      color: 'bg-yellow-50'
    },
    {
      id: 'young-eagle',
      stage: 2,
      title: 'Young Eagle',
      koreanTitle: '독수리 훈련생',
      description: '기업 분석과 투자 전략 수립 역량 구축',
      chapters: [
        { id: 'financial-statements-deep', title: '재무제표 완전 분석' },
        { id: 'valuation-methods', title: '기업가치 평가 방법론' },
        { id: 'technical-analysis-foundation', title: '기술적 분석 기초' }
      ],
      tools: ['DCF 가치평가 모델', '차트 패턴 인식기', 'AI 종목 분석기'],
      color: 'bg-blue-50'
    },
    {
      id: 'lion-king',
      stage: 3,
      title: 'Lion King',
      koreanTitle: '시장의 사자',
      description: '포트폴리오 관리와 고급 투자 전략 마스터',
      chapters: [
        { id: 'modern-portfolio-theory', title: '현대 포트폴리오 이론' },
        { id: 'risk-management-advanced', title: '고급 리스크 관리' },
        { id: 'algo-trading-systems', title: '알고리즘 트레이딩' }
      ],
      tools: ['포트폴리오 최적화', 'AI 트레이딩 봇', '리스크 대시보드'],
      color: 'bg-red-50'
    }
  ];


  // 핵심 리얼 도구 (시뮬레이터 아닌 실전 도구)
  const realTools = [
    {
      id: 'real-time-analyzer',
      name: '실시간 종목 분석기',
      description: 'DART 공시 연동, AI 기반 투자포인트 추출',
      features: ['실시간 재무제표', '산업 비교 분석', 'AI 리포트 생성'],
      icon: Activity,
      status: 'live'
    },
    {
      id: 'smart-timing',
      name: '스마트 매매 타이밍',
      description: '기술적 지표와 수급 분석 기반 매매 신호',
      features: ['실시간 알림', '백테스팅 검증', '성과 추적'],
      icon: Target,
      status: 'live'
    },
    {
      id: 'portfolio-optimizer',
      name: '포트폴리오 최적화',
      description: '현대 포트폴리오 이론 기반 최적 자산배분',
      features: ['리스크 분석', '리밸런싱 제안', '세금 최적화'],
      icon: Shield,
      status: 'beta'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">

      {/* Hero Section - 전문가 톤 */}
      <div className="bg-slate-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="max-w-3xl">
            <h1 className="text-4xl font-bold mb-4">
              Stock Analysis
            </h1>
            <p className="text-lg text-gray-300 mb-6 leading-relaxed">
              기관투자자의 분석 방법론과 헤지펀드의 트레이딩 전략을 체계적으로 학습하고,
              실전에서 바로 활용 가능한 투자 도구를 제공합니다.
            </p>
            <div className="flex items-center gap-6 text-gray-400">
              <div className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                <span>실전 도구 14개</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Growth Path Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-3">
            단계별 성장 경로
          </h2>
          <p className="text-base text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            체계적인 커리큘럼을 통해 개인 투자자에서 전문 투자자로 성장합니다
          </p>
        </div>

        {/* Stage Cards */}
        <div className="space-y-4">
          {growthStages.map((stage) => (
            <div 
              key={stage.id}
              className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden hover:shadow-lg transition-shadow"
            >
              <div className="p-4">
                <div className="flex items-start gap-6">
                  {/* Stage Number */}
                  <div className={`w-16 h-16 rounded-lg bg-gradient-to-br ${stage.color} flex items-center justify-center shadow-sm`}>
                    <span className="text-2xl font-bold text-gray-700">{stage.stage}</span>
                  </div>

                  {/* Stage Content */}
                  <div className="flex-1">
                    <div className="flex items-center gap-4 mb-3">
                      <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
                        Stage {stage.stage}
                      </span>
                      <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
                        {stage.title}
                      </h3>
                      <span className="text-lg text-gray-600 dark:text-gray-400">
                        {stage.koreanTitle}
                      </span>
                    </div>

                    <p className="text-gray-600 dark:text-gray-400 mb-4">
                      {stage.description}
                    </p>

                    <div className="grid md:grid-cols-3 gap-6">
                      {/* Curriculum */}
                      <div>
                        <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                          핵심 커리큘럼
                        </h4>
                        <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                          {stage.chapters.map((chapter) => (
                            <li key={chapter.id} className="flex items-center gap-2">
                              <span className="w-1 h-1 bg-gray-400 rounded-full" />
                              {chapter.title}
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Tools */}
                      <div>
                        <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                          실전 도구
                        </h4>
                        <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                          {stage.tools.map((tool) => (
                            <li key={tool} className="flex items-center gap-2">
                              <span className="w-1 h-1 bg-gray-400 rounded-full" />
                              {tool}
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Start Learning */}
                      <div className="flex items-end justify-end">
                        <Link
                          href={`/modules/stock-analysis/stages/${stage.id}`}
                          className="inline-flex items-center gap-1 text-sm font-medium text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300"
                        >
                          학습 시작하기
                          <ChevronRight className="w-4 h-4" />
                        </Link>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

      </div>

      {/* Real Tools Section */}
      <div className="bg-gray-100 dark:bg-gray-800 py-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-3">
              실전 투자 도구
            </h2>
            <p className="text-base text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              교육용 시뮬레이터가 아닌, 실제 투자에 활용 가능한 전문가급 도구
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-6">
            {realTools.map((tool) => {
              const Icon = tool.icon;
              return (
                <div
                  key={tool.id}
                  className="bg-white dark:bg-gray-900 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden hover:shadow-xl transition-shadow"
                >
                  <div className="p-4">
                    <div className="flex items-center justify-between mb-4">
                      <div className="w-12 h-12 bg-red-100 dark:bg-red-900/30 rounded-lg flex items-center justify-center">
                        <Icon className="w-6 h-6 text-red-600 dark:text-red-400" />
                      </div>
                      <span className={`text-xs px-3 py-1 rounded-full font-medium ${
                        tool.status === 'live' 
                          ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                          : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                      }`}>
                        {tool.status === 'live' ? 'Live' : 'Beta'}
                      </span>
                    </div>

                    <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-1">
                      {tool.name}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                      {tool.description}
                    </p>

                    <div className="flex flex-wrap gap-2 mb-3">
                      {tool.features.map((feature) => (
                        <span key={feature} className="text-xs text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                          {feature}
                        </span>
                      ))}
                    </div>

                    <div className="pt-4 border-t border-gray-200 dark:border-gray-700 flex justify-end">
                      <Link
                        href={`/modules/stock-analysis/tools/${tool.id}`}
                        className="text-sm font-medium text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300"
                      >
                        사용하기 →
                      </Link>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-8 text-center">
            <Link
              href="/modules/stock-analysis/tools"
              className="inline-flex items-center gap-2 px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors"
            >
              <DollarSign className="w-5 h-5" />
              모든 투자 도구 보기
              <ChevronRight className="w-5 h-5" />
            </Link>
          </div>
        </div>
      </div>

    </div>
  );
}