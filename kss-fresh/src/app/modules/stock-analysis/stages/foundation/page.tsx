'use client';

import Link from 'next/link';
import { ArrowLeft, BookOpen, Target, TrendingUp, BarChart3, AlertTriangle, ChevronRight, Play, Clock, Award, Shield, Database, LineChart } from 'lucide-react';

export default function FoundationStagePage() {
  const curriculum = [
    {
      week: 'Week 1-2',
      title: '기술적 분석의 기초',
      chapters: [
        {
          id: 'chart-basics',
          title: '차트 분석의 핵심 원리',
          description: '추세선, 지지와 저항, 거래량 분석, 이동평균선 활용법',
          duration: '90분',
          type: 'theory',
          level: 'Foundation'
        },
        {
          id: 'technical-indicators',
          title: '주요 기술적 지표 마스터',
          description: 'RSI, MACD, 볼린저밴드, 스토캐스틱 지표의 실전 활용',
          duration: '120분',
          type: 'practice',
          level: 'Foundation'
        },
        {
          id: 'pattern-recognition',
          title: '차트 패턴 인식과 매매',
          description: '헤드앤숄더, 삼각패턴, 깃발패턴 등 주요 패턴 실습',
          duration: '90분',
          type: 'analysis',
          level: 'Foundation'
        }
      ]
    },
    {
      week: 'Week 3-4',
      title: '기본적 분석 입문',
      chapters: [
        {
          id: 'financial-statements',
          title: '재무제표 읽기의 정석',
          description: '손익계산서, 재무상태표, 현금흐름표 분석 방법',
          duration: '120분',
          type: 'theory',
          level: 'Foundation'
        },
        {
          id: 'valuation-basics',
          title: '기업가치 평가의 기초',
          description: 'PER, PBR, EV/EBITDA 등 밸류에이션 지표 이해와 활용',
          duration: '90분',
          type: 'analysis',
          level: 'Foundation'
        },
        {
          id: 'industry-analysis',
          title: '산업 분석과 기업 비교',
          description: '섹터별 특성 이해, 경쟁사 분석, 성장성 평가',
          duration: '120분',
          type: 'practice',
          level: 'Foundation'
        }
      ]
    },
    {
      week: 'Week 5-6',
      title: '투자 전략과 포트폴리오',
      chapters: [
        {
          id: 'investment-strategies',
          title: '검증된 투자 전략 학습',
          description: '가치투자, 성장투자, 모멘텀 투자 전략의 이해와 적용',
          duration: '90분',
          type: 'theory',
          level: 'Foundation'
        },
        {
          id: 'portfolio-basics',
          title: '포트폴리오 구성의 기본',
          description: '분산투자의 원칙, 자산배분 전략, 리밸런싱 방법',
          duration: '120분',
          type: 'practice',
          level: 'Foundation'
        },
        {
          id: 'risk-control',
          title: '리스크 관리와 손절매',
          description: '포지션 사이징, 손절매 원칙, 리스크/리워드 비율',
          duration: '90분',
          type: 'analysis',
          level: 'Foundation'
        }
      ]
    },
    {
      week: 'Week 7-8',
      title: '실전 투자와 종합 연습',
      chapters: [
        {
          id: 'market-timing',
          title: '시장 타이밍과 매매 시점',
          description: '시장 사이클 이해, 매수/매도 신호 포착, 투자 심리 극복',
          duration: '120분',
          type: 'practice',
          level: 'Foundation'
        },
        {
          id: 'real-trading',
          title: '실전 매매 시뮬레이션',
          description: '모의투자로 배운 내용 종합 실습, 매매일지 작성과 분석',
          duration: '180분',
          type: 'simulation',
          level: 'Foundation'
        },
        {
          id: 'investment-plan',
          title: '나만의 투자 계획 수립',
          description: '투자 목표 설정, 전략 선택, 실행 계획 작성',
          duration: '90분',
          type: 'project',
          level: 'Foundation'
        }
      ]
    }
  ];

  const tools = [
    {
      name: '차트 분석 연습장',
      description: '실시간 차트에서 기술적 분석 도구를 직접 그려보고 연습',
      icon: LineChart,
      href: '/modules/stock-analysis/simulators/chart-analysis',
      badge: '필수'
    },
    {
      name: '재무제표 분석기',
      description: '실제 기업의 재무제표를 쉽게 분석하고 비교하는 도구',
      icon: Database,
      href: '/modules/stock-analysis/tools/financial-analyzer',
      badge: '추천'
    },
    {
      name: '포트폴리오 시뮬레이터',
      description: '가상 포트폴리오 구성하고 수익률 추적하기',
      icon: Shield,
      href: '/modules/stock-analysis/simulators/portfolio-builder',
      badge: '인기'
    }
  ];

  const competencies = [
    {
      title: '차트 분석 능력',
      description: '기술적 지표와 차트 패턴을 활용한 매매 시점 판단',
      icon: LineChart,
      metrics: ['추세 판단', '지표 활용', '패턴 인식']
    },
    {
      title: '기업 분석 역량',
      description: '재무제표 분석과 기업가치 평가 능력',
      icon: Database,
      metrics: ['재무제표 읽기', '밸류에이션', '산업 분석']
    },
    {
      title: '포트폴리오 관리',
      description: '분산투자와 리스크 관리를 통한 안정적 수익 추구',
      icon: Shield,
      metrics: ['자산 배분', '리스크 관리', '리밸런싱']
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
            <span>Investment Analysis로 돌아가기</span>
          </Link>
        </div>
      </div>

      {/* Hero Section */}
      <div className="bg-gradient-to-br from-blue-600 to-indigo-700 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-8">
            <div className="flex-1">
              <div className="flex items-center gap-4 mb-4">
                <span className="px-3 py-1 bg-orange-500/20 text-orange-300 rounded-full text-sm font-medium">
                  Stage 2: Foundation
                </span>
                <span className="text-gray-300">8주 체계적 과정</span>
              </div>
              <h1 className="text-4xl md:text-5xl font-bold mb-4">
                Foundation Program - 기초 투자자
              </h1>
              <p className="text-xl text-gray-100 max-w-3xl">
                차트 분석부터 기업 분석까지, 성공적인 투자를 위한 필수 지식과 실전 기술을 체계적으로 학습하는 프로그램
              </p>
            </div>
            <div className="hidden lg:block">
              <div className="w-32 h-32 bg-gradient-to-br from-orange-400 to-orange-500 rounded-2xl flex items-center justify-center">
                <TrendingUp className="w-16 h-16 text-white" />
              </div>
            </div>
          </div>

          {/* Key Metrics */}
          <div className="grid md:grid-cols-4 gap-6 mt-12">
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6">
              <Clock className="w-8 h-8 text-orange-300 mb-3" />
              <div className="text-3xl font-bold mb-1">8주</div>
              <div className="text-gray-300">학습 기간</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6">
              <BookOpen className="w-8 h-8 text-green-400 mb-3" />
              <div className="text-3xl font-bold mb-1">12개</div>
              <div className="text-gray-300">핵심 챕터</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6">
              <Target className="w-8 h-8 text-yellow-400 mb-3" />
              <div className="text-3xl font-bold mb-1">3개</div>
              <div className="text-gray-300">실습 도구</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6">
              <Award className="w-8 h-8 text-purple-400 mb-3" />
              <div className="text-3xl font-bold mb-1">실전</div>
              <div className="text-gray-300">투자 준비</div>
            </div>
          </div>
        </div>
      </div>

      {/* Prerequisites Notice */}
      <div className="bg-orange-50 dark:bg-orange-900/20 border-l-4 border-orange-500 p-6 my-8 max-w-7xl mx-auto">
        <div className="flex items-start gap-4">
          <AlertTriangle className="w-6 h-6 text-orange-600 dark:text-orange-400 mt-1" />
          <div>
            <h3 className="text-lg font-semibold text-orange-900 dark:text-orange-100 mb-2">
              수강 전 필수 사항
            </h3>
            <ul className="text-orange-800 dark:text-orange-200 space-y-1">
              <li>• Baby Chick 과정 수료 또는 기본 투자 용어 이해</li>
              <li>• 증권계좌 보유 및 HTS/MTS 기본 사용 가능</li>
              <li>• 주 10-15시간 학습 시간 확보 가능</li>
              <li>• 실습을 위한 소액 투자금 준비 (선택사항)</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Curriculum Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            학습 커리큘럼
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            차트 분석, 기업 분석, 투자 전략까지 성공적인 투자에 필요한 모든 기초를 체계적으로 학습합니다.
          </p>
        </div>

        <div className="space-y-8">
          {curriculum.map((week) => (
            <div key={week.week} className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
              <div className="bg-gradient-to-r from-gray-900 to-gray-800 text-white px-8 py-6">
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-sm font-medium text-gray-300">{week.week}</span>
                    <h3 className="text-2xl font-bold mt-1">{week.title}</h3>
                  </div>
                  <TrendingUp className="w-8 h-8 text-gray-400" />
                </div>
              </div>

              <div className="p-8 space-y-6">
                {week.chapters.map((chapter, index) => (
                  <div key={chapter.id} className="flex items-start gap-6 pb-6 border-b border-gray-200 dark:border-gray-700 last:border-0 last:pb-0">
                    <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center text-white font-bold">
                      {index + 1}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h4 className="text-xl font-semibold text-gray-900 dark:text-white">
                          {chapter.title}
                        </h4>
                        <span className="px-3 py-1 bg-gray-100 dark:bg-gray-700 rounded-full text-xs font-medium text-gray-600 dark:text-gray-300">
                          {chapter.level}
                        </span>
                      </div>
                      <p className="text-gray-600 dark:text-gray-400 mb-3">
                        {chapter.description}
                      </p>
                      <div className="flex items-center gap-6 text-sm">
                        <span className="flex items-center gap-2 text-gray-500">
                          <Clock className="w-4 h-4" />
                          {chapter.duration}
                        </span>
                        <span className={`px-3 py-1 rounded-full font-medium ${
                          chapter.type === 'theory' 
                            ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' 
                            : chapter.type === 'practice'
                            ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                            : chapter.type === 'analysis'
                            ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400'
                            : chapter.type === 'quantitative'
                            ? 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400'
                            : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                        }`}>
                          {chapter.type === 'theory' ? 'Theory' : 
                           chapter.type === 'practice' ? 'Practice' : 
                           chapter.type === 'analysis' ? 'Analysis' :
                           chapter.type === 'quantitative' ? 'Quantitative' : 'Simulation'}
                        </span>
                      </div>
                    </div>
                    <Link
                      href={`/modules/stock-analysis/chapters/${chapter.id}`}
                      className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
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
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Foundation 전용 실습 도구</h2>
            <p className="text-xl text-gray-400">배운 내용을 직접 실습하고 연습할 수 있는 전용 도구들</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {tools.map((tool) => {
              const Icon = tool.icon;
              return (
                <Link
                  key={tool.name}
                  href={tool.href}
                  className="bg-gray-800 rounded-xl p-8 hover:bg-gray-700 transition-all hover:scale-105 group"
                >
                  <div className="flex items-start justify-between mb-6">
                    <div className="w-14 h-14 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center">
                      <Icon className="w-8 h-8 text-white" />
                    </div>
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                      tool.badge === '필수' ? 'bg-red-500/20 text-red-400' :
                      tool.badge === '추천' ? 'bg-blue-500/20 text-blue-400' :
                      'bg-green-500/20 text-green-400'
                    }`}>
                      {tool.badge}
                    </span>
                  </div>
                  <h3 className="text-xl font-semibold mb-3 group-hover:text-blue-400 transition-colors">
                    {tool.name}
                  </h3>
                  <p className="text-gray-400">
                    {tool.description}
                  </p>
                  <div className="mt-6 flex items-center gap-2 text-orange-400">
                    <span className="text-sm font-medium">도구 실행</span>
                    <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      </div>

      {/* Core Competencies */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            학습 목표 달성
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-400">
            Foundation Program을 통해 습득하게 될 3대 핵심 역량
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {competencies.map((comp) => {
            const Icon = comp.icon;
            return (
              <div key={comp.title} className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center mb-6">
                  <Icon className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-3">
                  {comp.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                  {comp.description}
                </p>
                <div className="space-y-2">
                  {comp.metrics.map((metric) => (
                    <div key={metric} className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">{metric}</span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        {/* Next Steps */}
        <div className="mt-16 text-center bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-gray-800 dark:to-gray-700 rounded-2xl p-12">
          <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            다음 단계: Advanced Program
          </h3>
          <p className="text-lg text-gray-600 dark:text-gray-300 mb-8 max-w-2xl mx-auto">
            Foundation Program을 마치면, 고급 투자 전략과 퀀트 분석을 배우는 Advanced Program으로 진급할 수 있습니다.
          </p>
          <div className="flex items-center justify-center gap-4">
            <Link
              href="/modules/stock-analysis/stages/advanced"
              className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl font-semibold transition-all transform hover:scale-105"
            >
              Advanced Program 미리보기
              <ChevronRight className="w-5 h-5" />
            </Link>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-4">
            * Foundation Program 전 과정을 80% 이상 수료 시 진급 가능
          </p>
        </div>
      </div>
    </div>
  );
}