'use client';

import Link from 'next/link';
import { ArrowLeft, BookOpen, Target, TrendingUp, Cpu, Brain, ChevronRight, Play, Clock, Award, Zap, BarChart3, Database, Server, LineChart, Activity } from 'lucide-react';

export default function AdvancedStagePage() {
  const curriculum = [
    {
      week: 'Week 1-3',
      title: '고급 기술적 분석과 시스템 트레이딩',
      chapters: [
        {
          id: 'advanced-technical-analysis',
          title: '고급 차트 패턴과 하모닉 트레이딩',
          description: '엘리어트 파동, 하모닉 패턴, 피보나치 고급 활용, 멀티 타임프레임 분석',
          duration: '120분',
          type: 'technical',
          level: 'Advanced'
        },
        {
          id: 'system-trading-basics',
          title: '시스템 트레이딩 입문',
          description: '트레이딩 시스템 설계, 백테스팅 기초, 성과 평가 지표, 오버피팅 방지',
          duration: '150분',
          type: 'system',
          level: 'Advanced'
        },
        {
          id: 'automated-strategies',
          title: '자동매매 전략 구축',
          description: 'Python으로 구현하는 자동매매, API 연동, 실시간 데이터 처리',
          duration: '180분',
          type: 'automation',
          level: 'Advanced'
        }
      ]
    },
    {
      week: 'Week 4-6',
      title: '퀀트 투자 기초와 데이터 분석',
      chapters: [
        {
          id: 'quantitative-basics',
          title: '퀀트 투자의 이해',
          description: '팩터 투자 기초, 백테스팅 심화, 샤프 비율과 최대 낙폭 분석',
          duration: '120분',
          type: 'quantitative',
          level: 'Advanced'
        },
        {
          id: 'financial-data-analysis',
          title: '금융 빅데이터 분석',
          description: 'Pandas를 활용한 시계열 분석, 상관관계 분석, 포트폴리오 리스크 계산',
          duration: '150분',
          type: 'data-analysis',
          level: 'Advanced'
        },
        {
          id: 'factor-models',
          title: '팩터 모델 구축',
          description: '가치, 성장, 모멘텀 팩터 구현, 멀티팩터 포트폴리오 구성',
          duration: '180분',
          type: 'modeling',
          level: 'Advanced'
        }
      ]
    },
    {
      week: 'Week 7-9',
      title: '파생상품과 헤지 전략',
      chapters: [
        {
          id: 'derivatives-basics',
          title: '옵션 거래 전략',
          description: '옵션 가격 결정 모델, 그릭스 이해, 기본 옵션 전략 (콜/풋 스프레드)',
          duration: '150분',
          type: 'derivatives',
          level: 'Advanced'
        },
        {
          id: 'advanced-options',
          title: '고급 옵션 전략',
          description: '변동성 거래, 아이언 콘도르, 캘린더 스프레드, 델타 중립 전략',
          duration: '180분',
          type: 'options',
          level: 'Advanced'
        },
        {
          id: 'hedging-strategies',
          title: '헤지 전략과 리스크 관리',
          description: '포트폴리오 헤지, 통화 헤지, VIX 활용, 테일 리스크 관리',
          duration: '150분',
          type: 'hedging',
          level: 'Advanced'
        }
      ]
    },
    {
      week: 'Week 10-12',
      title: '글로벌 투자와 대안 투자',
      chapters: [
        {
          id: 'global-markets',
          title: '글로벌 시장 투자',
          description: '해외 주식 투자, ADR/GDR, 환율 영향 분석, 국가별 투자 전략',
          duration: '120분',
          type: 'global',
          level: 'Advanced'
        },
        {
          id: 'alternative-investments',
          title: '대안 투자 전략',
          description: 'REITs, 원자재, 암호화폐, P2P 투자 등 대안 자산 투자',
          duration: '150분',
          type: 'alternative',
          level: 'Advanced'
        },
        {
          id: 'macro-trading',
          title: '매크로 트레이딩',
          description: '경제 지표 활용, 중앙은행 정책 분석, 통화/금리/원자재 연계 전략',
          duration: '180분',
          type: 'macro',
          level: 'Advanced'
        }
      ]
    }
  ];

  const tools = [
    {
      name: '시스템 트레이딩 백테스터',
      description: 'Python 기반 전략 백테스팅, 최적화, 성과 분석 플랫폼',
      icon: Cpu,
      href: '/modules/stock-analysis/simulators/system-backtester',
      badge: '필수'
    },
    {
      name: '퀀트 분석 도구',
      description: '팩터 분석, 포트폴리오 최적화, 리스크 측정 통합 환경',
      icon: Database,
      href: '/modules/stock-analysis/simulators/quant-analyzer',
      badge: '고급'
    },
    {
      name: '옵션 시뮬레이터',
      description: '옵션 가격 계산, 그릭스 분석, 전략 시뮬레이션',
      icon: Activity,
      href: '/modules/stock-analysis/simulators/options-simulator',
      badge: '파생'
    },
    {
      name: '글로벌 투자 플랫폼',
      description: '해외 시장 분석, 환율 계산, 국가별 투자 전략 도구',
      icon: BarChart3,
      href: '/modules/stock-analysis/tools/global-platform',
      badge: '글로벌'
    }
  ];

  const competencies = [
    {
      title: '시스템 트레이딩',
      description: '자동매매 시스템 설계와 백테스팅을 통한 전략 검증',
      icon: Server,
      metrics: ['전략 코딩', '백테스팅', '최적화', '실전 운용']
    },
    {
      title: '퀀트 분석 능력',
      description: '데이터 기반의 체계적인 투자 전략 수립',
      icon: Database,
      metrics: ['팩터 분석', '통계 모델링', '포트폴리오 이론', '리스크 측정']
    },
    {
      title: '파생상품 활용',
      description: '옵션과 선물을 활용한 고급 투자 전략',
      icon: Activity,
      metrics: ['옵션 전략', '헤지 기법', '변동성 거래', '구조화 상품']
    },
    {
      title: '글로벌 투자',
      description: '전 세계 시장에서 기회를 포착하는 능력',
      icon: LineChart,
      metrics: ['해외 시장 분석', '환율 헤지', '국가별 전략', '대안 투자']
    }
  ];

  const prerequisites = [
    'Foundation Program 수료 또는 동등한 투자 지식',
    '프로그래밍 기초: Python 또는 Excel VBA 활용 가능',
    '투자 경험: 최소 1년 이상의 실전 투자 경험',
    '학습 시간: 주 15-20시간 학습 가능'
  ];

  const careerPaths = [
    {
      role: '시스템 트레이더',
      description: '자동매매 시스템을 운용하는 전문 트레이더',
      skills: ['시스템 설계', '백테스팅', '리스크 관리'],
      salary: '1억 ~ 3억+'
    },
    {
      role: '퀀트 애널리스트',
      description: '증권사/자산운용사의 퀀트 전략 개발자',
      skills: ['팩터 분석', '포트폴리오 최적화', '데이터 분석'],
      salary: '8천만 ~ 2억+'
    },
    {
      role: '헤지펀드 매니저',
      description: '고급 전략을 운용하는 펀드 매니저',
      skills: ['매크로 분석', '파생상품', '글로벌 투자'],
      salary: '1.5억 ~ 10억+'
    },
    {
      role: '투자 자문가',
      description: '고액 자산가를 위한 투자 컨설턴트',
      skills: ['자산 배분', '세무 전략', '대안 투자'],
      salary: '1억 ~ 5억+'
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

      {/* Hero Section - Advanced Design */}
      <div className="bg-gradient-to-br from-purple-900 via-indigo-900 to-blue-900 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-8">
            <div className="flex-1">
              <div className="flex items-center gap-4 mb-4">
                <span className="px-3 py-1 bg-purple-500/20 text-purple-400 rounded-full text-sm font-medium">
                  Stage 3: Advanced
                </span>
                <span className="text-gray-300">12주 전문 과정</span>
              </div>
              <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-white to-purple-200 bg-clip-text text-transparent">
                Advanced Program - 고급 투자자
              </h1>
              <p className="text-xl text-gray-300 max-w-3xl">
                시스템 트레이딩, 퀀트 투자, 파생상품, 글로벌 투자까지 전문 투자자로 도약하는 고급 과정
              </p>
            </div>
            <div className="hidden lg:block">
              <div className="w-40 h-40 relative">
                <div className="absolute inset-0 bg-gradient-to-br from-purple-500 to-blue-600 rounded-2xl animate-pulse"></div>
                <div className="relative w-full h-full bg-gradient-to-br from-purple-600 to-blue-700 rounded-2xl flex items-center justify-center">
                  <Brain className="w-20 h-20 text-white" />
                </div>
              </div>
            </div>
          </div>

          {/* Key Metrics */}
          <div className="grid md:grid-cols-5 gap-6 mt-16">
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20">
              <Clock className="w-8 h-8 text-purple-400 mb-3" />
              <div className="text-3xl font-bold mb-1">12주</div>
              <div className="text-gray-300">학습 기간</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20">
              <BookOpen className="w-8 h-8 text-blue-400 mb-3" />
              <div className="text-3xl font-bold mb-1">12개</div>
              <div className="text-gray-300">전문 모듈</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20">
              <Target className="w-8 h-8 text-green-400 mb-3" />
              <div className="text-3xl font-bold mb-1">30+</div>
              <div className="text-gray-300">실전 프로젝트</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20">
              <Zap className="w-8 h-8 text-yellow-400 mb-3" />
              <div className="text-3xl font-bold mb-1">GPU</div>
              <div className="text-gray-300">가속 컴퓨팅</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20">
              <Award className="w-8 h-8 text-orange-400 mb-3" />
              <div className="text-3xl font-bold mb-1">인증서</div>
              <div className="text-gray-300">수료 인증</div>
            </div>
          </div>
        </div>
      </div>

      {/* Prerequisites Alert */}
      <div className="bg-purple-50 dark:bg-purple-900/20 border-l-4 border-purple-500 p-6 my-8 max-w-7xl mx-auto">
        <div className="flex items-start gap-4">
          <Zap className="w-6 h-6 text-purple-600 dark:text-purple-400 mt-1" />
          <div>
            <h3 className="text-lg font-semibold text-purple-900 dark:text-purple-100 mb-2">
              필수 선수 지식 & 기술 요구사항
            </h3>
            <ul className="text-purple-800 dark:text-purple-200 space-y-1">
              {prerequisites.map((prereq, idx) => (
                <li key={idx}>• {prereq}</li>
              ))}
            </ul>
            <div className="mt-4 p-4 bg-purple-100 dark:bg-purple-800/30 rounded-lg">
              <p className="text-sm font-medium text-purple-900 dark:text-purple-100">
                ⚡ 기술 스택: Python, SQL, Git, Docker, Cloud Computing (AWS/GCP)
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Curriculum Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            Advanced Curriculum
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            업계 최고 수준의 퀀트 투자 및 AI 트레이딩 전문 교육 과정
          </p>
        </div>

        <div className="space-y-8">
          {curriculum.map((week, weekIdx) => (
            <div key={week.week} className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden border border-gray-200 dark:border-gray-700">
              <div className="bg-gradient-to-r from-purple-900 via-indigo-900 to-blue-900 text-white px-8 py-6">
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-sm font-medium text-gray-300">{week.week}</span>
                    <h3 className="text-2xl font-bold mt-1">{week.title}</h3>
                  </div>
                  <div className="flex items-center gap-3">
                    {weekIdx === 0 && <Database className="w-8 h-8" />}
                    {weekIdx === 1 && <Brain className="w-8 h-8" />}
                    {weekIdx === 2 && <Server className="w-8 h-8" />}
                    {weekIdx === 3 && <LineChart className="w-8 h-8" />}
                  </div>
                </div>
              </div>

              <div className="p-8 space-y-6">
                {week.chapters.map((chapter, index) => (
                  <div key={chapter.id} className="flex items-start gap-6 pb-6 border-b border-gray-200 dark:border-gray-700 last:border-0 last:pb-0">
                    <div className="w-14 h-14 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center text-white font-bold text-lg">
                      {weekIdx * 3 + index + 1}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h4 className="text-xl font-semibold text-gray-900 dark:text-white">
                          {chapter.title}
                        </h4>
                        <span className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 rounded-full text-xs font-medium text-purple-700 dark:text-purple-300">
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
                          chapter.type === 'quantitative' 
                            ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' 
                            : chapter.type === 'modeling'
                            ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                            : chapter.type === 'strategy'
                            ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400'
                            : chapter.type === 'engineering'
                            ? 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400'
                            : chapter.type === 'ai-modeling'
                            ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400'
                            : chapter.type === 'ai-advanced'
                            ? 'bg-pink-100 text-pink-700 dark:bg-pink-900/30 dark:text-pink-400'
                            : chapter.type === 'infrastructure'
                            ? 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
                            : chapter.type === 'execution'
                            ? 'bg-teal-100 text-teal-700 dark:bg-teal-900/30 dark:text-teal-400'
                            : chapter.type === 'hft'
                            ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                            : chapter.type === 'risk'
                            ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                            : chapter.type === 'optimization'
                            ? 'bg-cyan-100 text-cyan-700 dark:bg-cyan-900/30 dark:text-cyan-400'
                            : 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400'
                        }`}>
                          {chapter.type.charAt(0).toUpperCase() + chapter.type.slice(1).replace('-', ' ')}
                        </span>
                      </div>
                    </div>
                    <Link
                      href={`/modules/stock-analysis/chapters/${chapter.id}`}
                      className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white rounded-lg font-medium transition-all transform hover:scale-105"
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

      {/* Professional Tools Section */}
      <div className="bg-gradient-to-br from-gray-900 to-purple-900 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Enterprise-Grade Tools</h2>
            <p className="text-xl text-gray-300">실제 퀀트 펀드에서 사용하는 수준의 전문 도구</p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {tools.map((tool) => {
              const Icon = tool.icon;
              return (
                <Link
                  key={tool.name}
                  href={tool.href}
                  className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 hover:bg-gray-700/50 transition-all hover:scale-105 group"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center">
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                    <span className="px-2 py-1 bg-purple-500/20 text-purple-400 rounded-full text-xs font-medium">
                      {tool.badge}
                    </span>
                  </div>
                  <h3 className="text-lg font-semibold mb-2 group-hover:text-purple-400 transition-colors">
                    {tool.name}
                  </h3>
                  <p className="text-gray-400 text-sm">
                    {tool.description}
                  </p>
                  <div className="mt-4 flex items-center gap-2 text-purple-400">
                    <span className="text-sm font-medium">Launch</span>
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
            Advanced Competencies
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-400">
            Advanced Program을 통해 마스터하게 될 전문 역량
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {competencies.map((comp) => {
            const Icon = comp.icon;
            return (
              <div key={comp.title} className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
                <div className="w-14 h-14 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center mb-4">
                  <Icon className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                  {comp.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-4 text-sm">
                  {comp.description}
                </p>
                <div className="space-y-2">
                  {comp.metrics.map((metric) => (
                    <div key={metric} className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-purple-500 rounded-full" />
                      <span className="text-xs text-gray-600 dark:text-gray-400">{metric}</span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Career Paths Section */}
      <div className="bg-gray-100 dark:bg-gray-800 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
              Career Opportunities
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-400">
              Advanced Program 수료 후 진출 가능한 커리어 패스
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {careerPaths.map((career) => (
              <div key={career.role} className="bg-white dark:bg-gray-700 rounded-xl p-6 shadow-lg">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                  {career.role}
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-4 text-sm">
                  {career.description}
                </p>
                <div className="mb-4">
                  <span className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                    {career.salary}
                  </span>
                  <span className="text-sm text-gray-500 dark:text-gray-400 ml-2">연봉</span>
                </div>
                <div className="space-y-1">
                  {career.skills.map((skill) => (
                    <div key={skill} className="text-xs bg-gray-100 dark:bg-gray-600 text-gray-700 dark:text-gray-300 px-2 py-1 rounded inline-block mr-2 mb-1">
                      {skill}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Next Steps */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="bg-gradient-to-r from-red-600 to-purple-600 rounded-2xl p-12 text-white text-center">
          <h3 className="text-3xl font-bold mb-4">
            전문 투자자로의 도약, 준비되셨나요?
          </h3>
          <p className="text-xl mb-8 max-w-3xl mx-auto">
            Foundation Program을 마치셨다면, 이제 시스템 트레이딩과 퀀트 투자의 세계로 들어올 준비가 되었습니다.
          </p>
          <div className="flex items-center justify-center gap-4 flex-wrap">
            <Link
              href="/modules/stock-analysis/stages/professional"
              className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-xl font-semibold transition-all transform hover:scale-105"
            >
              Professional Program 보기
              <ChevronRight className="w-5 h-5" />
            </Link>
          </div>
          <p className="text-sm text-red-200 mt-6">
            * Advanced 전 과정을 80% 이상 수료 시 Professional 과정 진급 가능
          </p>
        </div>
      </div>
    </div>
  );
}