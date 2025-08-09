'use client';

import Link from 'next/link';
import { ArrowLeft, Sparkles, GraduationCap, Target, Award, PlayCircle, Clock, Users, ChevronRight, BookOpen, Calculator, BarChart3, PieChart, Activity, Brain, DollarSign, TrendingUp, AlertTriangle, Shield, Microscope, Newspaper } from 'lucide-react';

export default function StockAnalysisModulePage() {
  const learningTracks = [
    {
      id: 'beginner',
      title: '주식 투자 첫걸음',
      description: '주식이 뭔지도 모르는 완전 초보자를 위한 코스',
      duration: '1주일',
      level: 'beginner',
      icon: <Sparkles className="w-6 h-6" />,
      color: 'from-green-400 to-green-600',
      topics: ['주식이란?', '계좌 개설', '첫 거래'],
      students: 15420,
      rating: 4.8
    },
    {
      id: 'basic',
      title: '차트 읽기 기초',
      description: '빨간색 파란색부터 시작하는 차트 읽기',
      duration: '2주일',
      level: 'basic',
      icon: <BookOpen className="w-6 h-6" />,
      color: 'from-blue-400 to-blue-600',
      topics: ['캔들차트', '이동평균선', '거래량'],
      students: 12350,
      rating: 4.7
    },
    {
      id: 'intermediate',
      title: '똑똑한 투자자 되기',
      description: '기업 분석하고 좋은 주식 고르는 법',
      duration: '4주일',
      level: 'intermediate',
      icon: <GraduationCap className="w-6 h-6" />,
      color: 'from-purple-400 to-purple-600',
      topics: ['재무제표 기초', '가치평가', '매매타이밍'],
      students: 8920,
      rating: 4.9
    },
    {
      id: 'advanced',
      title: '기술적 분석 마스터',
      description: '차트 패턴과 지표로 매매 타이밍 잡기',
      duration: '6주일',
      level: 'advanced',
      icon: <Target className="w-6 h-6" />,
      color: 'from-orange-400 to-orange-600',
      topics: ['기술적 지표', '차트 패턴', '매매 전략'],
      students: 5230,
      rating: 4.8
    },
    {
      id: 'professional',
      title: '전문 투자자 과정',
      description: '포트폴리오 관리와 리스크 헤지 전략',
      duration: '8주일',
      level: 'professional',
      icon: <Award className="w-6 h-6" />,
      color: 'from-red-400 to-red-600',
      topics: ['포트폴리오 이론', '리스크 관리', 'AI 투자'],
      students: 2140,
      rating: 4.9
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <Link 
              href="/"
              className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>홈으로 돌아가기</span>
            </Link>
          </div>
        </div>
      </div>

      {/* Hero Section */}
      <div className="bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
            스마트 주식투자 배우기
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-400 mb-8 max-w-3xl mx-auto">
            주식이 처음이어도 괜찮아요! 기초부터 전문가 수준까지, 
            나만의 속도로 차근차근 배우는 체계적인 투자 교육
          </p>
          <div className="flex items-center justify-center gap-8 text-sm text-gray-600 dark:text-gray-400">
            <div className="flex items-center gap-2">
              <Users className="w-5 h-5" />
              <span>44,260명 수강중</span>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="w-5 h-5" />
              <span>평균 완주율 89%</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-yellow-500">★★★★★</span>
              <span>4.8/5.0</span>
            </div>
          </div>
        </div>
      </div>

      {/* Learning Tracks */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            나에게 맞는 학습 코스를 선택하세요
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            수준별로 준비된 5가지 학습 트랙 중 선택하세요
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {learningTracks.map((track) => (
            <Link
              key={track.id}
              href={`/modules/stock-analysis/learn/${track.id}`}
              className="group relative bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-xl hover:scale-105 transition-all duration-200 overflow-hidden"
            >
              {/* Level Badge */}
              <div className="absolute top-4 right-4 z-10">
                <span className={`text-xs px-3 py-1 rounded-full font-medium ${
                  track.level === 'beginner' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                  track.level === 'basic' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' :
                  track.level === 'intermediate' ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400' :
                  track.level === 'advanced' ? 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400' :
                  'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                }`}>
                  {track.level === 'beginner' ? '입문' : 
                   track.level === 'basic' ? '초급' :
                   track.level === 'intermediate' ? '중급' :
                   track.level === 'advanced' ? '상급' : '전문가'}
                </span>
              </div>

              {/* Card Content */}
              <div className="p-6">
                <div className={`w-12 h-12 rounded-lg bg-gradient-to-r ${track.color} flex items-center justify-center text-white mb-4`}>
                  {track.icon}
                </div>
                
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2 group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-red-500 group-hover:to-orange-500 transition-all">
                  {track.title}
                </h3>
                
                <p className="text-gray-600 dark:text-gray-400 mb-4 text-sm">
                  {track.description}
                </p>

                {/* Topics */}
                <div className="flex flex-wrap gap-2 mb-4">
                  {track.topics.map((topic, index) => (
                    <span key={index} className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded-full text-gray-600 dark:text-gray-400">
                      {topic}
                    </span>
                  ))}
                </div>

                {/* Stats */}
                <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-500 mb-4">
                  <div className="flex items-center gap-1">
                    <Clock className="w-4 h-4" />
                    <span>{track.duration}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Users className="w-4 h-4" />
                    <span>{track.students.toLocaleString()}명</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="text-yellow-500">★</span>
                    <span>{track.rating}</span>
                  </div>
                </div>

                {/* CTA Button */}
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    학습 시작하기
                  </span>
                  <PlayCircle className="w-5 h-5 text-gray-400 group-hover:text-red-500 transition-colors" />
                </div>
              </div>

              {/* Hover Effect */}
              <div className={`absolute inset-0 bg-gradient-to-r ${track.color} opacity-0 group-hover:opacity-10 transition-opacity duration-200`} />
            </Link>
          ))}
        </div>

        {/* Quick Start Guide */}
        <div className="mt-12 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-2xl p-8">
          <div className="max-w-3xl mx-auto text-center">
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              어떤 코스를 선택해야 할지 모르겠나요? 🤔
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-6">
              30초만에 당신에게 딱 맞는 학습 경로를 찾아드립니다!
            </p>
            <Link
              href="/modules/stock-analysis/quiz"
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-medium hover:shadow-lg hover:scale-105 transition-all duration-200"
            >
              <Sparkles className="w-5 h-5" />
              맞춤 학습 경로 찾기
              <ChevronRight className="w-5 h-5" />
            </Link>
          </div>
        </div>

        {/* All Simulators Gallery */}
        <div className="mt-12">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
              모든 시뮬레이터 체험하기 🛠️
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              11개의 전문가급 투자 도구를 자유롭게 체험해보세요
            </p>
          </div>

          {/* Featured Simulators */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            {[
              {
                id: 'financial-calculator',
                name: '재무제표 분석기',
                description: '70개 재무비율 자동계산',
                icon: Calculator,
                color: 'from-blue-500 to-cyan-500'
              },
              {
                id: 'chart-analyzer',
                name: 'AI 차트 분석기',
                description: '50가지 패턴 자동 인식',
                icon: BarChart3,
                color: 'from-green-500 to-emerald-500'
              },
              {
                id: 'portfolio-optimizer',
                name: '포트폴리오 최적화',
                description: '마코위츠 이론 기반',
                icon: PieChart,
                color: 'from-purple-500 to-violet-500'
              },
              {
                id: 'ai-mentor',
                name: 'AI 투자 멘토',
                description: 'GPT-4 기반 맞춤 조언',
                icon: Brain,
                color: 'from-pink-500 to-rose-500'
              }
            ].map((simulator) => {
              const IconComponent = simulator.icon;
              return (
                <Link
                  key={simulator.id}
                  href={`/modules/stock-analysis/simulators/${simulator.id}`}
                  className="group bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700 hover:shadow-lg hover:scale-105 transition-all duration-200"
                >
                  <div className={`w-12 h-12 rounded-lg bg-gradient-to-r ${simulator.color} flex items-center justify-center text-white mb-3 group-hover:scale-110 transition-transform`}>
                    <IconComponent className="w-6 h-6" />
                  </div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-1 group-hover:text-red-600 dark:group-hover:text-red-400 transition-colors">
                    {simulator.name}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {simulator.description}
                  </p>
                </Link>
              );
            })}
          </div>

          {/* All Simulators Categories */}
          <div className="grid md:grid-cols-3 gap-6">
            {/* Basic Analysis Tools */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <Calculator className="w-5 h-5 text-blue-500" />
                기본 분석 도구
              </h3>
              <div className="space-y-3">
                {[
                  { id: 'financial-calculator', name: '재무제표 분석기' },
                  { id: 'dcf-valuation-model', name: 'DCF 가치평가 모델' },
                  { id: 'earnings-forecast-model', name: 'AI 실적 예측 모델' },
                  { id: 'dividend-growth-analyzer', name: '배당성장 분석기' }
                ].map((tool) => (
                  <Link
                    key={tool.id}
                    href={`/modules/stock-analysis/simulators/${tool.id}`}
                    className="block text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                  >
                    • {tool.name}
                  </Link>
                ))}
              </div>
            </div>

            {/* Technical Analysis Tools */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-green-500" />
                기술적 분석 도구
              </h3>
              <div className="space-y-3">
                {[
                  { id: 'chart-analyzer', name: 'AI 차트 분석기' },
                  { id: 'backtesting-engine', name: '백테스팅 엔진' },
                  { id: 'market-sentiment-gauge', name: '시장 심리 측정기' },
                  { id: 'sector-rotation-tracker', name: '섹터 로테이션 추적기' }
                ].map((tool) => (
                  <Link
                    key={tool.id}
                    href={`/modules/stock-analysis/simulators/${tool.id}`}
                    className="block text-sm text-gray-600 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400 transition-colors"
                  >
                    • {tool.name}
                  </Link>
                ))}
              </div>
            </div>

            {/* Portfolio & Risk Management */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <PieChart className="w-5 h-5 text-purple-500" />
                포트폴리오 & 리스크
              </h3>
              <div className="space-y-3">
                {[
                  { id: 'portfolio-optimizer', name: '포트폴리오 최적화기' },
                  { id: 'risk-management-dashboard', name: '리스크 관리 대시보드' },
                  { id: 'correlation-matrix-analyzer', name: '상관관계 분석기' },
                  { id: 'etf-overlap-analyzer', name: 'ETF 중복도 분석기' }
                ].map((tool) => (
                  <Link
                    key={tool.id}
                    href={`/modules/stock-analysis/simulators/${tool.id}`}
                    className="block text-sm text-gray-600 dark:text-gray-400 hover:text-purple-600 dark:hover:text-purple-400 transition-colors"
                  >
                    • {tool.name}
                  </Link>
                ))}
              </div>
            </div>
          </div>

          {/* Advanced Tools Section */}
          <div className="mt-8 bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-700 rounded-xl p-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
              <Brain className="w-6 h-6 text-red-500" />
              실전 투자에 꼭 필요한 고급 도구
            </h3>
            <div className="grid md:grid-cols-3 lg:grid-cols-7 gap-3">
              {[
                { id: 'real-time-dashboard', name: '실시간 시장 데이터', icon: Activity, color: 'text-blue-500' },
                { id: 'risk-management-dashboard', name: '리스크 관리', icon: Shield, color: 'text-red-500' },
                { id: 'backtesting-engine', name: '백테스팅', icon: BarChart3, color: 'text-green-500' },
                { id: 'dcf-valuation-model', name: 'DCF 가치평가', icon: DollarSign, color: 'text-purple-500' },
                { id: 'options-strategy-analyzer', name: '옵션 전략', icon: TrendingUp, color: 'text-orange-500' },
                { id: 'factor-investing-lab', name: '팩터 투자', icon: Microscope, color: 'text-cyan-500' },
                { id: 'news-impact-analyzer', name: 'AI 뉴스 분석', icon: Newspaper, color: 'text-pink-500' }
              ].map((tool) => {
                const IconComponent = tool.icon;
                return (
                  <Link
                    key={tool.id}
                    href={`/modules/stock-analysis/simulators/${tool.id}`}
                    className="group bg-white dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-600 hover:shadow-md hover:scale-105 transition-all duration-200"
                  >
                    <div className="flex flex-col items-center gap-2 text-center">
                      <div className="w-10 h-10 bg-gray-100 dark:bg-gray-700 rounded-lg flex items-center justify-center group-hover:bg-red-50 dark:group-hover:bg-red-900/20 transition-colors">
                        <IconComponent className={`w-5 h-5 ${tool.color} group-hover:scale-110 transition-transform`} />
                      </div>
                      <h4 className="font-medium text-gray-900 dark:text-white text-xs group-hover:text-red-600 dark:group-hover:text-red-400 transition-colors">
                        {tool.name}
                      </h4>
                    </div>
                  </Link>
                );
              })}
            </div>
          </div>

          {/* View All Button */}
          <div className="mt-8 text-center">
            <Link
              href="/modules/stock-analysis/simulators"
              className="inline-flex items-center gap-2 px-8 py-3 bg-gradient-to-r from-red-600 to-orange-600 text-white rounded-lg font-medium hover:shadow-lg hover:scale-105 transition-all duration-200"
            >
              <PlayCircle className="w-5 h-5" />
              모든 시뮬레이터 보기 (11개)
              <ChevronRight className="w-5 h-5" />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}