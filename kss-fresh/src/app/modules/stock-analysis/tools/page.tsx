'use client';

import Link from 'next/link';
import { ArrowLeft, Activity, Target, Shield, BarChart3, TrendingUp, LineChart, Calculator, Brain, Zap, DollarSign, AlertCircle, ChevronRight, Users } from 'lucide-react';

export default function StockAnalysisToolsPage() {
  const tools = [
    {
      category: '종목 분석 도구',
      items: [
        {
          id: 'real-time-analyzer',
          name: '실시간 종목 분석기',
          description: '한국 주식시장 전종목을 실시간으로 분석합니다. DART 공시 데이터와 재무제표를 AI가 분석해 핵심 투자 포인트를 추출합니다.',
          features: ['실시간 재무제표 업데이트', '산업 평균 대비 분석', 'AI 기반 투자 의견'],
          icon: Activity,
          status: 'live',
          difficulty: 'intermediate',
          techStack: ['Python', 'TensorFlow', 'FastAPI', 'Redis'],
          dataSource: 'DART API, KRX, Yahoo Finance'
        },
        {
          id: 'fundamental-screener',
          name: '펀더멘털 스크리너',
          description: '벤저민 그레이엄의 가치투자 기준으로 저평가 우량주를 발굴합니다. 커스텀 필터링과 백테스팅을 지원합니다.',
          features: ['40+ 재무지표 필터링', '커스텀 스크리닝 룰', '과거 성과 백테스팅'],
          icon: Shield,
          status: 'live',
          difficulty: 'advanced',
          techStack: ['React', 'PostgreSQL', 'Python'],
          dataSource: 'FnGuide, QuantiWise'
        },
        {
          id: 'investor-flow-tracker',
          name: '투자자별 매매동향 분석기',
          description: '개인, 기관, 외국인의 매매 패턴을 실시간으로 추적합니다. 연속 순매수, 업종별 자금흐름, 삼박자 종목을 한눈에 파악할 수 있습니다.',
          features: ['외국인 연속 순매수 추적', '기관 업종별 자금흐름', '삼박자 종목 스캐너', '매매주체 편중도 경고'],
          icon: Users,
          status: 'live',
          difficulty: 'intermediate',
          techStack: ['React', 'WebSocket', 'Python', 'ClickHouse'],
          dataSource: 'KRX 투자자별 매매동향, 실시간 체결 데이터'
        }
      ]
    },
    {
      category: '가치평가 도구',
      items: [
        {
          id: 'dcf-calculator',
          name: 'DCF 가치평가 계산기',
          description: '현금흐름할인법(DCF)으로 기업의 내재가치를 계산합니다. 애스워스 다모다란 교수의 방법론을 구현했습니다.',
          features: ['자동 재무데이터 입력', '민감도 분석', '몬테카를로 시뮬레이션'],
          icon: Calculator,
          status: 'live',
          difficulty: 'expert',
          techStack: ['React', 'D3.js', 'Python', 'NumPy'],
          dataSource: 'DART, Bloomberg Terminal API'
        },
        {
          id: 'relative-valuation',
          name: '상대가치 평가 도구',
          description: 'PER, PBR, EV/EBITDA 등 멀티플을 활용한 동종업계 비교 분석. 섹터별 적정 밸류에이션을 제시합니다.',
          features: ['실시간 멀티플 계산', '업종별 히트맵', '역사적 밸류에이션 추이'],
          icon: BarChart3,
          status: 'beta',
          difficulty: 'intermediate',
          techStack: ['Vue.js', 'ECharts', 'Django'],
          dataSource: 'KRX, FnGuide'
        }
      ]
    },
    {
      category: '차트 분석 도구',
      items: [
        {
          id: 'advanced-charting',
          name: '고급 차트 분석 플랫폼',
          description: 'TradingView를 능가하는 전문가급 차트 도구. 150개 이상의 기술적 지표와 커스텀 인디케이터를 지원합니다.',
          features: ['150+ 기술적 지표', '멀티 타임프레임', 'Pine Script 호환'],
          icon: LineChart,
          status: 'development',
          difficulty: 'advanced',
          techStack: ['TradingView Charting Library', 'WebSocket', 'Redis'],
          dataSource: 'Real-time market data feed'
        },
        {
          id: 'pattern-ai',
          name: 'AI 차트 패턴 인식기',
          description: '딥러닝으로 차트 패턴을 자동 인식합니다. 과거 10년 데이터를 학습한 AI가 높은 확률의 패턴만 알려드립니다.',
          features: ['20+ 패턴 자동 인식', '성공률 통계 제공', '실시간 알림'],
          icon: Brain,
          status: 'beta',
          difficulty: 'intermediate',
          techStack: ['PyTorch', 'OpenCV', 'Kafka'],
          dataSource: 'Historical price data (10 years)'
        }
      ]
    },
    {
      category: '포트폴리오 관리',
      items: [
        {
          id: 'portfolio-optimizer',
          name: '포트폴리오 최적화 엔진',
          description: '마코위츠 현대포트폴리오이론과 블랙-리터만 모델을 활용한 최적 자산배분. 노벨경제학상 수상 이론을 구현했습니다.',
          features: ['효율적 투자선 계산', '리스크 패리티', '다기간 최적화'],
          icon: Target,
          status: 'live',
          difficulty: 'expert',
          techStack: ['Python', 'CVXPY', 'React', 'Plotly'],
          dataSource: 'Global asset prices, Economic indicators'
        },
        {
          id: 'risk-analytics',
          name: '리스크 분석 대시보드',
          description: '포트폴리오의 리스크를 실시간으로 모니터링합니다. VaR, CVaR, 최대낙폭, 샤프지수 등을 시각화합니다.',
          features: ['실시간 리스크 지표', '스트레스 테스트', '시나리오 분석'],
          icon: AlertCircle,
          status: 'beta',
          difficulty: 'expert',
          techStack: ['R', 'Shiny', 'QuantLib', 'MongoDB'],
          dataSource: 'Real-time portfolio data'
        }
      ]
    },
    {
      category: 'AI 기반 도구',
      items: [
        {
          id: 'news-sentiment',
          name: '뉴스 감성분석 AI',
          description: 'BERT 기반 자연어처리로 뉴스와 공시의 긍/부정을 분석합니다. 시장 심리 변화를 실시간으로 포착합니다.',
          features: ['실시간 뉴스 분석', '종목별 감성 점수', '이벤트 임팩트 예측'],
          icon: Zap,
          status: 'live',
          difficulty: 'intermediate',
          techStack: ['BERT', 'Elasticsearch', 'Apache Spark'],
          dataSource: 'News API, Social media, DART'
        },
        {
          id: 'earnings-predictor',
          name: '실적 예측 AI',
          description: '과거 실적 패턴과 산업 동향을 학습한 AI가 분기 실적을 예측합니다. 70% 이상의 정확도를 자랑합니다.',
          features: ['분기 실적 예측', '컨센서스 대비 분석', '서프라이즈 확률'],
          icon: TrendingUp,
          status: 'development',
          difficulty: 'advanced',
          techStack: ['XGBoost', 'LSTM', 'Apache Airflow'],
          dataSource: 'Historical earnings, Analyst estimates'
        }
      ]
    }
  ];

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'text-green-600 bg-green-50';
      case 'intermediate': return 'text-blue-600 bg-blue-50';
      case 'advanced': return 'text-orange-600 bg-orange-50';
      case 'expert': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getDifficultyLabel = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return '초급';
      case 'intermediate': return '중급';
      case 'advanced': return '고급';
      case 'expert': return '전문가';
      default: return '';
    }
  };

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
      <div className="bg-slate-900 text-white py-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold mb-4">
            실전 투자 도구
          </h1>
          <p className="text-lg text-gray-300 max-w-3xl">
            전문 투자자들이 사용하는 분석 도구를 제공합니다. 
            실시간 데이터와 AI 기술을 활용해 더 나은 투자 결정을 내리세요.
          </p>
        </div>
      </div>

      {/* Tools Grid */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {tools.map((category) => (
          <div key={category.category} className="mb-12">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              {category.category}
            </h2>
            
            <div className="grid md:grid-cols-2 gap-6">
              {category.items.map((tool) => {
                const Icon = tool.icon;
                
                return (
                  <div
                    key={tool.id}
                    className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden hover:shadow-lg transition-shadow"
                  >
                    <div className="p-6">
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex items-center gap-4">
                          <div className="w-12 h-12 bg-gray-100 dark:bg-gray-700 rounded-lg flex items-center justify-center">
                            <Icon className="w-6 h-6 text-gray-700 dark:text-gray-300" />
                          </div>
                          <div>
                            <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                              {tool.name}
                            </h3>
                            <div className="flex items-center gap-3 mt-1">
                              <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                                tool.status === 'live' 
                                  ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                                  : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                              }`}>
                                {tool.status === 'live' ? 'Live' : 'Beta'}
                              </span>
                              <span className={`text-xs px-2 py-1 rounded-full font-medium ${getDifficultyColor(tool.difficulty)}`}>
                                {getDifficultyLabel(tool.difficulty)}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>

                      <p className="text-gray-600 dark:text-gray-400 mb-4">
                        {tool.description}
                      </p>

                      <div className="space-y-3 mb-4">
                        <div className="flex flex-wrap gap-2">
                          {tool.features.map((feature) => (
                            <span key={feature} className="text-xs text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-700 px-3 py-1 rounded-full">
                              {feature}
                            </span>
                          ))}
                        </div>
                        
                        {tool.techStack && (
                          <div className="text-xs">
                            <span className="font-medium text-gray-500">기술 스택:</span>
                            <span className="ml-2 text-gray-600 dark:text-gray-400">
                              {tool.techStack.join(', ')}
                            </span>
                          </div>
                        )}
                        
                        {tool.dataSource && (
                          <div className="text-xs">
                            <span className="font-medium text-gray-500">데이터 소스:</span>
                            <span className="ml-2 text-gray-600 dark:text-gray-400">
                              {tool.dataSource}
                            </span>
                          </div>
                        )}
                      </div>

                      <div className="flex justify-end">
                        <Link
                          href={`/modules/stock-analysis/tools/${tool.id}`}
                          className="inline-flex items-center gap-1 text-sm font-medium text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300"
                        >
                          도구 사용하기
                          <ChevronRight className="w-4 h-4" />
                        </Link>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Bottom Notice */}
      <div className="bg-gray-100 dark:bg-gray-800 py-8">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <DollarSign className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            더 많은 도구가 준비 중입니다
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            사용자 피드백을 바탕으로 새로운 투자 도구를 지속적으로 추가하고 있습니다.
          </p>
        </div>
      </div>
    </div>
  );
}