'use client';

import Link from 'next/link';
import { 
  ArrowLeft, Search, Filter, ChevronRight, Users, Shield,
  // 일반용 아이콘
  Calculator, LineChart, PieChart, TrendingUp, DollarSign, BarChart3, 
  FileText, Target, Wallet, Globe,
  // 전문가용 아이콘
  Activity, Brain, Database, Zap, AlertTriangle, Settings,
  Eye, GitBranch, Cpu, Binary
} from 'lucide-react';
import { useState } from 'react';

interface Simulator {
  id: string;
  name: string;
  description: string;
  icon: any;
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  category: 'basic' | 'analysis' | 'portfolio' | 'trading' | 'risk' | 'ai';
  users: number;
  isNew?: boolean;
  isPremium?: boolean;
}

export default function SimulatorsOverviewPage() {
  const [viewMode, setViewMode] = useState<'all' | 'beginner' | 'professional'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  // 일반용 시뮬레이터 (초급-중급)
  const beginnerSimulators: Simulator[] = [
    // 기초 도구
    {
      id: 'financial-calculator',
      name: '투자 수익률 계산기',
      description: 'PER, PBR, ROE 등 기본 투자지표 계산',
      icon: Calculator,
      difficulty: 'beginner',
      category: 'basic',
      users: 5200
    },
    {
      id: 'simple-chart-viewer',
      name: '차트 기초 학습기',
      description: '캔들차트, 이동평균선 등 차트 읽기 연습',
      icon: LineChart,
      difficulty: 'beginner',
      category: 'basic',
      users: 4800
    },
    {
      id: 'portfolio-tracker',
      name: '포트폴리오 관리',
      description: '보유 종목 관리 및 수익률 추적',
      icon: PieChart,
      difficulty: 'beginner',
      category: 'portfolio',
      users: 3900
    },
    {
      id: 'stock-screener-basic',
      name: '종목 스크리너',
      description: '조건에 맞는 종목 찾기 (기초)',
      icon: Search,
      difficulty: 'beginner',
      category: 'analysis',
      users: 3500
    },
    // 중급 도구
    {
      id: 'technical-indicator-lab',
      name: '기술적 지표 실습',
      description: 'MACD, RSI, 볼린저밴드 등 지표 학습',
      icon: TrendingUp,
      difficulty: 'intermediate',
      category: 'analysis',
      users: 2800
    },
    {
      id: 'dcf-calculator',
      name: 'DCF 가치평가',
      description: '현금흐름할인법으로 적정주가 계산',
      icon: DollarSign,
      difficulty: 'intermediate',
      category: 'analysis',
      users: 2400
    },
    {
      id: 'sector-comparison',
      name: '섹터별 비교 분석',
      description: '업종별 성과 비교 및 순환 분석',
      icon: BarChart3,
      difficulty: 'intermediate',
      category: 'analysis',
      users: 2100
    },
    {
      id: 'earnings-calendar',
      name: '실적 발표 캘린더',
      description: '기업 실적 일정 및 컨센서스 확인',
      icon: FileText,
      difficulty: 'intermediate',
      category: 'basic',
      users: 1900
    },
    {
      id: 'dividend-tracker',
      name: '배당 투자 도구',
      description: '배당수익률 계산 및 배당주 분석',
      icon: Wallet,
      difficulty: 'intermediate',
      category: 'portfolio',
      users: 1700,
      isNew: true
    },
    {
      id: 'etf-comparator',
      name: 'ETF 비교 분석',
      description: '국내외 ETF 수수료, 수익률 비교',
      icon: Globe,
      difficulty: 'intermediate',
      category: 'portfolio',
      users: 1500
    }
  ];

  // 전문가용 시뮬레이터 (고급-전문가)
  const professionalSimulators: Simulator[] = [
    // 고급 분석
    {
      id: 'ai-chart-analyzer',
      name: 'AI 차트 패턴 분석기',
      description: '딥러닝 기반 차트 패턴 자동 인식',
      icon: Brain,
      difficulty: 'advanced',
      category: 'ai',
      users: 3800,
      isPremium: true
    },
    {
      id: 'portfolio-optimizer',
      name: '포트폴리오 최적화',
      description: '현대 포트폴리오 이론 기반 자산배분',
      icon: Cpu,
      difficulty: 'advanced',
      category: 'portfolio',
      users: 2100,
      isPremium: true
    },
    {
      id: 'risk-management-dashboard',
      name: '리스크 관리 대시보드',
      description: 'VaR, 스트레스 테스트, 팩터 분석',
      icon: Shield,
      difficulty: 'expert',
      category: 'risk',
      users: 1800,
      isPremium: true
    },
    {
      id: 'options-strategy-analyzer',
      name: '옵션 전략 분석기',
      description: '그릭스 계산 및 옵션 전략 시뮬레이션',
      icon: GitBranch,
      difficulty: 'expert',
      category: 'trading',
      users: 1100,
      isPremium: true
    },
    // 트레이딩 도구
    {
      id: 'algorithmic-trading-lab',
      name: '알고리즘 트레이딩',
      description: '백테스팅 및 자동매매 전략 개발',
      icon: Binary,
      difficulty: 'expert',
      category: 'trading',
      users: 750,
      isPremium: true
    },
    {
      id: 'order-flow-analytics',
      name: '주문흐름 분석',
      description: 'Level 2 데이터 기반 기관 동향 파악',
      icon: Database,
      difficulty: 'expert',
      category: 'trading',
      users: 650,
      isPremium: true
    },
    {
      id: 'market-microstructure-lab',
      name: '시장 미시구조 연구',
      description: '틱 데이터 분석 및 HFT 패턴 인식',
      icon: Activity,
      difficulty: 'expert',
      category: 'trading',
      users: 420,
      isPremium: true
    },
    // AI & 퀀트
    {
      id: 'news-sentiment-analyzer',
      name: 'NLP 뉴스 분석기',
      description: '뉴스 감성분석 및 시장영향도 측정',
      icon: Eye,
      difficulty: 'advanced',
      category: 'ai',
      users: 3200
    },
    {
      id: 'factor-investing-lab',
      name: '팩터 투자 연구소',
      description: '멀티팩터 모델 구축 및 백테스팅',
      icon: Zap,
      difficulty: 'expert',
      category: 'ai',
      users: 890,
      isPremium: true
    },
    {
      id: 'monte-carlo-simulator',
      name: '몬테카를로 시뮬레이션',
      description: '확률적 시나리오 분석 및 리스크 평가',
      icon: AlertTriangle,
      difficulty: 'advanced',
      category: 'risk',
      users: 1200,
      isNew: true
    }
  ];

  const allSimulators = [...beginnerSimulators, ...professionalSimulators];

  // 카테고리 정의
  const categories = [
    { id: 'all', name: '전체', count: allSimulators.length },
    { id: 'basic', name: '기초', count: allSimulators.filter(s => s.category === 'basic').length },
    { id: 'analysis', name: '분석', count: allSimulators.filter(s => s.category === 'analysis').length },
    { id: 'portfolio', name: '포트폴리오', count: allSimulators.filter(s => s.category === 'portfolio').length },
    { id: 'trading', name: '트레이딩', count: allSimulators.filter(s => s.category === 'trading').length },
    { id: 'risk', name: '리스크', count: allSimulators.filter(s => s.category === 'risk').length },
    { id: 'ai', name: 'AI/퀀트', count: allSimulators.filter(s => s.category === 'ai').length }
  ];

  // 필터링
  const getFilteredSimulators = () => {
    let simulators = allSimulators;
    
    // 뷰 모드 필터
    if (viewMode === 'beginner') {
      simulators = beginnerSimulators;
    } else if (viewMode === 'professional') {
      simulators = professionalSimulators;
    }
    
    // 카테고리 필터
    if (selectedCategory !== 'all') {
      simulators = simulators.filter(s => s.category === selectedCategory);
    }
    
    // 검색 필터
    if (searchQuery) {
      simulators = simulators.filter(s => 
        s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        s.description.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }
    
    return simulators;
  };

  const filteredSimulators = getFilteredSimulators();

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'intermediate': return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
      case 'advanced': return 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400';
      case 'expert': return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const getDifficultyText = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return '초급';
      case 'intermediate': return '중급';
      case 'advanced': return '고급';
      case 'expert': return '전문가';
      default: return difficulty;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <Link 
              href="/modules/stock-analysis"
              className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>Stock Analysis로 돌아가기</span>
            </Link>
          </div>
        </div>
      </div>

      {/* Hero */}
      <div className="bg-gradient-to-br from-blue-600 to-purple-700 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl md:text-5xl font-bold mb-4">
              투자 분석 시뮬레이터
            </h1>
            <p className="text-xl text-blue-100 max-w-3xl mx-auto mb-8">
              초보자부터 전문가까지, 수준별 맞춤 투자 도구를 만나보세요
            </p>
            
            {/* View Mode Selector */}
            <div className="inline-flex items-center bg-white/10 backdrop-blur-sm rounded-lg p-1">
              <button
                onClick={() => setViewMode('all')}
                className={`px-6 py-2 rounded-md font-medium transition-all ${
                  viewMode === 'all'
                    ? 'bg-white text-blue-600'
                    : 'text-white hover:bg-white/10'
                }`}
              >
                전체 보기
              </button>
              <button
                onClick={() => setViewMode('beginner')}
                className={`px-6 py-2 rounded-md font-medium transition-all ${
                  viewMode === 'beginner'
                    ? 'bg-white text-blue-600'
                    : 'text-white hover:bg-white/10'
                }`}
              >
                <Users className="w-4 h-4 inline mr-2" />
                일반용
              </button>
              <button
                onClick={() => setViewMode('professional')}
                className={`px-6 py-2 rounded-md font-medium transition-all ${
                  viewMode === 'professional'
                    ? 'bg-white text-blue-600'
                    : 'text-white hover:bg-white/10'
                }`}
              >
                <Shield className="w-4 h-4 inline mr-2" />
                전문가용
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
          {/* Search */}
          <div className="mb-6">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="시뮬레이터 검색..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-12 pr-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          {/* Categories */}
          <div className="flex flex-wrap gap-2">
            {categories.map((cat) => (
              <button
                key={cat.id}
                onClick={() => setSelectedCategory(cat.id)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  selectedCategory === cat.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                {cat.name} ({cat.count})
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Simulators Grid */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-16">
        <div className="mb-6 flex items-center justify-between">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            {filteredSimulators.length}개 시뮬레이터
          </h2>
          <div className="text-sm text-gray-500">
            <Filter className="w-4 h-4 inline mr-1" />
            {viewMode === 'beginner' ? '일반용' : viewMode === 'professional' ? '전문가용' : '전체'}
          </div>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredSimulators.map((simulator) => {
            const Icon = simulator.icon;
            const linkPath = simulator.id.includes('ai-chart-analyzer') || 
                           simulator.id.includes('portfolio-optimizer') || 
                           simulator.id.includes('news-sentiment-analyzer') ||
                           simulator.id.includes('risk-management-dashboard') ||
                           simulator.id.includes('options-strategy-analyzer')
              ? `/modules/stock-analysis/tools/${simulator.id}`
              : `/modules/stock-analysis/simulators/${simulator.id}`;
            
            return (
              <Link
                key={simulator.id}
                href={linkPath}
                className="group bg-white dark:bg-gray-800 rounded-xl shadow-sm hover:shadow-lg transition-all p-6"
              >
                {/* Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center text-white">
                    <Icon className="w-6 h-6" />
                  </div>
                  <div className="flex items-center gap-2">
                    {simulator.isNew && (
                      <span className="text-xs px-2 py-1 bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 rounded-full font-medium">
                        NEW
                      </span>
                    )}
                    {simulator.isPremium && (
                      <span className="text-xs px-2 py-1 bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400 rounded-full font-medium">
                        PRO
                      </span>
                    )}
                  </div>
                </div>

                {/* Content */}
                <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                  {simulator.name}
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 line-clamp-2">
                  {simulator.description}
                </p>

                {/* Footer */}
                <div className="flex items-center justify-between">
                  <span className={`text-xs px-3 py-1 rounded-full font-medium ${getDifficultyColor(simulator.difficulty)}`}>
                    {getDifficultyText(simulator.difficulty)}
                  </span>
                  <div className="flex items-center gap-3 text-sm text-gray-500">
                    <div className="flex items-center gap-1">
                      <Users className="w-4 h-4" />
                      <span>{simulator.users.toLocaleString()}</span>
                    </div>
                    <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </div>
                </div>
              </Link>
            );
          })}
        </div>

        {/* Empty State */}
        {filteredSimulators.length === 0 && (
          <div className="text-center py-16">
            <div className="w-20 h-20 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
              <Search className="w-10 h-10 text-gray-400" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
              검색 결과가 없습니다
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              다른 검색어나 필터를 시도해보세요
            </p>
          </div>
        )}
      </div>

      {/* Info Section */}
      {viewMode !== 'all' && (
        <div className="bg-gray-100 dark:bg-gray-800 py-12">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            {viewMode === 'beginner' ? (
              <>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                  일반 투자자를 위한 도구
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                  투자를 시작하는 분들을 위한 쉽고 직관적인 도구들입니다.
                  기본적인 투자 개념부터 중급 수준의 분석까지 단계별로 학습할 수 있습니다.
                </p>
              </>
            ) : (
              <>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                  전문 투자자를 위한 도구
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                  기관투자자와 전문 트레이더를 위한 고급 분석 도구들입니다.
                  퀀트 투자, 알고리즘 트레이딩, 리스크 관리 등 전문적인 기능을 제공합니다.
                </p>
              </>
            )}
            
            <Link
              href={viewMode === 'beginner' ? '/modules/stock-analysis/stages/foundation' : '/modules/stock-analysis/stages/advanced'}
              className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-colors"
            >
              {viewMode === 'beginner' ? 'Foundation Program 시작하기' : 'Advanced Program 시작하기'}
              <ChevronRight className="w-5 h-5" />
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}