'use client';

import Link from 'next/link';
import { TrendingUp, Shield, ChevronRight, BarChart3, DollarSign, Activity, Target, Database, LineChart, Brain, Clock, Award, BookOpen, Users } from 'lucide-react';

export default function StockAnalysisModulePage() {
  // Professional Investment Analysis Programs
  const programs = [
    {
      id: 'baby-chick',
      level: '🐣 Baby Chick',
      title: '주식 투자 첫걸음',
      koreanTitle: '완전 초보자 과정',
      description: '주식이 무엇인지부터 차근차근 배우는 4주 입문 과정',
      modules: [
        { id: 'what-is-stock', title: '주식이란 무엇인가?' },
        { id: 'chart-basics', title: '차트 읽기 기초' },
        { id: 'first-investment', title: '첫 투자 시작하기' }
      ],
      tools: ['모의투자 시뮬레이터', '용어 사전', '차트 연습장'],
      duration: '4주',
      participants: 3520,
      color: 'from-yellow-500 to-orange-500'
    },
    {
      id: 'foundation',
      level: 'Level 1',
      title: 'Foundation Program',
      koreanTitle: '투자 분석 기초',
      description: '차트 분석, 기업 분석, 투자 전략의 기초를 배우는 8주 과정',
      modules: [
        { id: 'technical-analysis', title: '기술적 분석 기초' },
        { id: 'fundamental-analysis', title: '기본적 분석 입문' },
        { id: 'risk-management', title: '리스크 관리 기초' }
      ],
      tools: ['Chart Analyzer', 'Financial Calculator', 'Portfolio Tracker'],
      duration: '8주',
      participants: 1250,
      color: 'from-blue-600 to-blue-700'
    },
    {
      id: 'advanced',
      level: 'Level 2',
      title: 'Advanced Program',
      koreanTitle: '고급 투자 분석',
      description: '퀀트 투자, 머신러닝, 알고리즘 트레이딩을 배우는 16주 전문 과정',
      modules: [
        { id: 'quantitative-analysis', title: '퀀트 투자 전략' },
        { id: 'machine-learning', title: '머신러닝 투자 모델' },
        { id: 'algorithmic-trading', title: '알고리즘 트레이딩' }
      ],
      tools: ['Quant Research Platform', 'AI Trading Lab', 'Backtesting Engine'],
      duration: '16주',
      participants: 820,
      color: 'from-purple-600 to-indigo-700'
    }
  ];

  // Professional Trading Tools
  const tradingTools = [
    {
      id: 'order-flow',
      name: 'Order Flow Analytics',
      description: 'Level 2 데이터 기반 기관투자자 주문흐름 분석',
      features: ['Dark Pool 거래 감지', '대량매매 분석', 'HFT 패턴 인식'],
      icon: Database,
      status: 'live',
      users: '2.3K'
    },
    {
      id: 'risk-dashboard',
      name: 'Risk Management Dashboard',
      description: 'VaR, Stress Testing, 포지션 리스크 실시간 모니터링',
      features: ['Portfolio VaR', 'Factor Analysis', 'Scenario Testing'],
      icon: Shield,
      status: 'live',
      users: '1.8K'
    },
    {
      id: 'algo-trading',
      name: 'Algorithmic Trading Platform',
      description: '백테스팅부터 실전 트레이딩까지 통합 플랫폼',
      features: ['Strategy Builder', 'Paper Trading', 'Live Execution'],
      icon: Brain,
      status: 'beta',
      users: '750'
    }
  ];

  // Key Metrics
  const metrics = [
    { label: '총 학습 시간', value: '240시간', icon: Clock },
    { label: '실습 프로젝트', value: '45개', icon: Target },
    { label: '전문 도구', value: '14개', icon: BarChart3 },
    { label: '수료생', value: '2,520명', icon: Users }
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      
      {/* Professional Hero Section */}
      <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="max-w-4xl">
            <div className="flex items-center gap-4 mb-6">
              <span className="px-4 py-2 bg-blue-500/20 text-blue-400 rounded-full text-sm font-medium">
                Professional Track
              </span>
              <span className="text-gray-400">Since 2024</span>
            </div>
            <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
              Professional Investment Analysis
            </h1>
            <p className="text-xl text-gray-300 mb-8 leading-relaxed">
              기관투자자의 분석 방법론과 헤지펀드의 트레이딩 전략을 체계적으로 학습하는 
              실무 중심의 전문가 양성 프로그램
            </p>
            
            {/* Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-12">
              {metrics.map((metric) => {
                const Icon = metric.icon;
                return (
                  <div key={metric.label} className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
                    <Icon className="w-6 h-6 text-gray-400 mb-2" />
                    <div className="text-2xl font-bold mb-1">{metric.value}</div>
                    <div className="text-sm text-gray-400">{metric.label}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Programs Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            Structured Learning Programs
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
            체계적인 커리큘럼을 통해 투자 분석 전문가로 성장하는 단계별 프로그램
          </p>
        </div>

        {/* Program Cards */}
        <div className="space-y-8">
          {programs.map((program, index) => (
            <div 
              key={program.id}
              className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg overflow-hidden hover:shadow-xl transition-all"
            >
              <div className="flex flex-col lg:flex-row">
                {/* Left Section - Program Info */}
                <div className="flex-1 p-8 lg:p-10">
                  <div className="flex items-center gap-4 mb-4">
                    <span className={`px-4 py-2 bg-gradient-to-r ${program.color} text-white rounded-lg text-sm font-medium`}>
                      {program.level}
                    </span>
                    <span className="text-gray-500 dark:text-gray-400">
                      {program.duration} · {program.participants.toLocaleString()}명 수료
                    </span>
                  </div>
                  
                  <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                    {program.title}
                  </h3>
                  <p className="text-lg text-gray-600 dark:text-gray-400 mb-6">
                    {program.description}
                  </p>

                  <div className="space-y-4">
                    <div>
                      <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                        Core Modules
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {program.modules.map((module) => (
                          <span 
                            key={module.id}
                            className="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg text-sm"
                          >
                            {module.title}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div>
                      <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                        Professional Tools
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {program.tools.map((tool) => (
                          <span 
                            key={tool}
                            className="px-3 py-1 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded-lg text-sm"
                          >
                            {tool}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Right Section - CTA */}
                <div className={`bg-gradient-to-br ${program.color} p-8 lg:p-10 flex items-center justify-center lg:w-80`}>
                  <div className="text-center">
                    <div className="w-20 h-20 bg-white/20 rounded-full flex items-center justify-center mx-auto mb-4">
                      <span className="text-3xl font-bold text-white">{index + 1}</span>
                    </div>
                    <Link
                      href={`/modules/stock-analysis/stages/${program.id}`}
                      className="inline-flex items-center gap-2 px-6 py-3 bg-white text-gray-900 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
                    >
                      Program Details
                      <ChevronRight className="w-5 h-5" />
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Professional Trading Tools */}
      <div className="bg-gray-900 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">
              Professional Trading Tools
            </h2>
            <p className="text-lg text-gray-400 max-w-3xl mx-auto">
              실제 기관투자자와 헤지펀드에서 사용하는 수준의 분석 도구를 직접 체험하고 활용하세요
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-8">
            {tradingTools.map((tool) => {
              const Icon = tool.icon;
              return (
                <div
                  key={tool.id}
                  className="bg-gray-800 rounded-xl p-8 hover:bg-gray-750 transition-all group"
                >
                  <div className="flex items-start justify-between mb-6">
                    <div className="w-14 h-14 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center">
                      <Icon className="w-8 h-8 text-white" />
                    </div>
                    <div className="flex items-center gap-3">
                      <span className={`text-xs px-3 py-1 rounded-full font-medium ${
                        tool.status === 'live' 
                          ? 'bg-green-500/20 text-green-400'
                          : 'bg-yellow-500/20 text-yellow-400'
                      }`}>
                        {tool.status === 'live' ? 'Live' : 'Beta'}
                      </span>
                      <span className="text-sm text-gray-400">{tool.users} users</span>
                    </div>
                  </div>

                  <h3 className="text-xl font-bold mb-3 group-hover:text-blue-400 transition-colors">
                    {tool.name}
                  </h3>
                  <p className="text-gray-400 mb-4">
                    {tool.description}
                  </p>

                  <div className="space-y-2">
                    {tool.features.map((feature) => (
                      <div key={feature} className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 bg-blue-400 rounded-full" />
                        <span className="text-sm text-gray-400">{feature}</span>
                      </div>
                    ))}
                  </div>

                  <Link
                    href={`/modules/stock-analysis/tools/${tool.id}`}
                    className="inline-flex items-center gap-2 mt-6 text-blue-400 hover:text-blue-300 transition-colors"
                  >
                    <span className="text-sm font-medium">Launch Tool</span>
                    <ChevronRight className="w-4 h-4" />
                  </Link>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-700 py-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white mb-4">
            투자 분석 전문가로의 여정을 시작하세요
          </h2>
          <p className="text-xl text-blue-100 mb-8">
            240시간의 체계적인 커리큘럼과 실무 도구로 전문 투자자의 역량을 갖추세요
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/modules/stock-analysis/stages/foundation"
              className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-white text-blue-700 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
            >
              Foundation Program 시작하기
              <ChevronRight className="w-5 h-5" />
            </Link>
            <Link
              href="/modules/stock-analysis/tools"
              className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-blue-700 text-white rounded-lg font-semibold hover:bg-blue-800 transition-colors border border-white/20"
            >
              <BarChart3 className="w-5 h-5" />
              전체 도구 둘러보기
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}