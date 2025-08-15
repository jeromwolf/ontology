'use client';

import Link from 'next/link';
import { ArrowLeft, Search, Filter, ExternalLink, ChevronRight, Database, Shield, Brain, LineChart, BarChart3, Activity, Target, TrendingUp, Calculator, DollarSign, Eye, AlertTriangle, Newspaper, Settings, Zap, Globe, Lock, CheckCircle, Users, Clock } from 'lucide-react';
import { useState } from 'react';

interface Tool {
  id: string;
  name: string;
  description: string;
  category: 'analytics' | 'risk' | 'trading' | 'research' | 'portfolio';
  features: string[];
  icon: any;
  status: 'live' | 'beta' | 'coming-soon';
  users?: string;
  pricing: 'free' | 'premium' | 'enterprise';
  badge?: string;
  requirements?: string[];
}

export default function ToolsOverviewPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedStatus, setSelectedStatus] = useState<string>('all');

  // 전체 도구 목록
  const allTools: Tool[] = [
    // Analytics Tools
    {
      id: 'order-flow-analytics',
      name: 'Order Flow Analytics',
      description: 'Level 2 데이터 기반 실시간 주문흐름 분석',
      category: 'analytics',
      features: ['Dark Pool Detection', 'Block Trade Analysis', 'HFT Pattern Recognition', 'Liquidity Heatmap'],
      icon: Database,
      status: 'live',
      users: '2.3K',
      pricing: 'premium',
      badge: 'Institutional Grade',
      requirements: ['Level 2 Data Access', 'Premium Account']
    },
    {
      id: 'market-microstructure-lab',
      name: 'Market Microstructure Lab',
      description: '틱 데이터 분석 및 시장 미시구조 연구',
      category: 'analytics',
      features: ['Tick Data Analysis', 'Spread Decomposition', 'Market Impact Cost', 'Order Book Dynamics'],
      icon: Activity,
      status: 'live',
      users: '1.5K',
      pricing: 'enterprise',
      badge: 'Advanced'
    },
    {
      id: 'ai-chart-analyzer',
      name: 'AI Chart Pattern Analyzer',
      description: '딥러닝 기반 차트 패턴 자동 인식',
      category: 'analytics',
      features: ['Pattern Recognition', 'Trend Prediction', 'Support/Resistance', 'Multi-Timeframe Analysis'],
      icon: LineChart,
      status: 'live',
      users: '3.8K',
      pricing: 'premium',
      badge: 'AI Powered'
    },
    
    // Risk Management Tools
    {
      id: 'risk-dashboard',
      name: 'Risk Management Dashboard',
      description: 'VaR, Stress Testing, 포트폴리오 리스크 모니터링',
      category: 'risk',
      features: ['Portfolio VaR', 'Factor Analysis', 'Scenario Testing', 'Real-time Alerts'],
      icon: Shield,
      status: 'live',
      users: '1.8K',
      pricing: 'premium',
      badge: 'Professional'
    },
    {
      id: 'risk-metrics-calculator',
      name: 'Risk Metrics Calculator',
      description: 'Sharpe, Sortino, Maximum Drawdown 등 리스크 지표 산출',
      category: 'risk',
      features: ['20+ Risk Metrics', 'Historical Analysis', 'Peer Comparison', 'Custom Benchmarks'],
      icon: Calculator,
      status: 'live',
      users: '4.2K',
      pricing: 'free',
      badge: 'Essential'
    },
    {
      id: 'stress-testing-suite',
      name: 'Stress Testing Suite',
      description: '시나리오 기반 포트폴리오 스트레스 테스트',
      category: 'risk',
      features: ['Historical Scenarios', 'Custom Stress Tests', 'Monte Carlo Simulation', 'Regulatory Compliance'],
      icon: AlertTriangle,
      status: 'beta',
      users: '650',
      pricing: 'enterprise',
      badge: 'Regulatory Compliant'
    },
    
    // Trading Tools
    {
      id: 'algo-trading-platform',
      name: 'Algorithmic Trading Platform',
      description: '백테스팅부터 실전 거래까지 통합 플랫폼',
      category: 'trading',
      features: ['Strategy Builder', 'Backtesting Engine', 'Paper Trading', 'Live Execution'],
      icon: Brain,
      status: 'beta',
      users: '750',
      pricing: 'enterprise',
      badge: 'Quant Platform'
    },
    {
      id: 'smart-order-router',
      name: 'Smart Order Router',
      description: '최적 체결을 위한 지능형 주문 라우팅',
      category: 'trading',
      features: ['Best Execution', 'Multi-Venue Routing', 'Slippage Minimization', 'Transaction Cost Analysis'],
      icon: Zap,
      status: 'live',
      users: '920',
      pricing: 'enterprise',
      badge: 'HFT Ready'
    },
    {
      id: 'options-strategy-analyzer',
      name: 'Options Strategy Analyzer',
      description: '옵션 전략 분석 및 Greeks 계산',
      category: 'trading',
      features: ['Greeks Calculator', 'Strategy Builder', 'Volatility Surface', 'P&L Simulation'],
      icon: TrendingUp,
      status: 'live',
      users: '1.1K',
      pricing: 'premium',
      badge: 'Derivatives'
    },
    
    // Research Tools
    {
      id: 'financial-statement-analyzer',
      name: 'Financial Statement Analyzer',
      description: 'AI 기반 재무제표 자동 분석 및 이상 징후 탐지',
      category: 'research',
      features: ['Automated Analysis', 'Peer Comparison', 'Fraud Detection', 'Trend Analysis'],
      icon: Eye,
      status: 'live',
      users: '2.7K',
      pricing: 'premium',
      badge: 'AI Enhanced'
    },
    {
      id: 'dcf-valuation-model',
      name: 'DCF Valuation Model',
      description: '현금흐름할인법 기반 기업가치 평가',
      category: 'research',
      features: ['Sensitivity Analysis', 'Scenario Modeling', 'Comparable Analysis', 'Report Generation'],
      icon: DollarSign,
      status: 'live',
      users: '1.9K',
      pricing: 'free',
      badge: 'Fundamental'
    },
    {
      id: 'news-sentiment-analyzer',
      name: 'News Sentiment Analyzer',
      description: 'NLP 기반 뉴스 감성 분석 및 시장 영향도 측정',
      category: 'research',
      features: ['Real-time Analysis', 'Multi-language Support', 'Source Credibility', 'Impact Prediction'],
      icon: Newspaper,
      status: 'live',
      users: '3.2K',
      pricing: 'premium',
      badge: 'NLP Powered'
    },
    
    // Portfolio Management Tools
    {
      id: 'portfolio-optimizer',
      name: 'Portfolio Optimizer',
      description: '현대 포트폴리오 이론 기반 자산배분 최적화',
      category: 'portfolio',
      features: ['Efficient Frontier', 'Black-Litterman', 'Risk Parity', 'Rebalancing Alerts'],
      icon: BarChart3,
      status: 'live',
      users: '2.1K',
      pricing: 'premium',
      badge: 'MPT Based'
    },
    {
      id: 'factor-investing-lab',
      name: 'Factor Investing Lab',
      description: '팩터 기반 투자 전략 연구 및 백테스팅',
      category: 'portfolio',
      features: ['Factor Analysis', 'Multi-factor Models', 'Factor Timing', 'Custom Factors'],
      icon: Target,
      status: 'live',
      users: '890',
      pricing: 'enterprise',
      badge: 'Quantitative'
    },
    {
      id: 'tax-optimization-engine',
      name: 'Tax Optimization Engine',
      description: '세금 효율적인 포트폴리오 관리',
      category: 'portfolio',
      features: ['Tax Loss Harvesting', 'Holding Period Optimization', 'After-tax Returns', 'Regulatory Compliance'],
      icon: Settings,
      status: 'coming-soon',
      pricing: 'enterprise',
      badge: 'Coming Q2 2025'
    }
  ];

  // 카테고리 정의
  const categories = [
    { id: 'all', name: '전체', count: allTools.length },
    { id: 'analytics', name: 'Analytics', count: allTools.filter(t => t.category === 'analytics').length },
    { id: 'risk', name: 'Risk Management', count: allTools.filter(t => t.category === 'risk').length },
    { id: 'trading', name: 'Trading', count: allTools.filter(t => t.category === 'trading').length },
    { id: 'research', name: 'Research', count: allTools.filter(t => t.category === 'research').length },
    { id: 'portfolio', name: 'Portfolio', count: allTools.filter(t => t.category === 'portfolio').length }
  ];

  // 필터링 로직
  const filteredTools = allTools.filter(tool => {
    const matchesSearch = tool.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         tool.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         tool.features.some(f => f.toLowerCase().includes(searchQuery.toLowerCase()));
    const matchesCategory = selectedCategory === 'all' || tool.category === selectedCategory;
    const matchesStatus = selectedStatus === 'all' || tool.status === selectedStatus;
    
    return matchesSearch && matchesCategory && matchesStatus;
  });

  const getPricingBadgeColor = (pricing: string) => {
    switch (pricing) {
      case 'free': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case 'premium': return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
      case 'enterprise': return 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400';
      default: return 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300';
    }
  };

  const getStatusBadgeColor = (status: string) => {
    switch (status) {
      case 'live': return 'bg-green-500/20 text-green-400';
      case 'beta': return 'bg-yellow-500/20 text-yellow-400';
      case 'coming-soon': return 'bg-gray-500/20 text-gray-400';
      default: return 'bg-gray-500/20 text-gray-400';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'analytics': return Database;
      case 'risk': return Shield;
      case 'trading': return Brain;
      case 'research': return Eye;
      case 'portfolio': return BarChart3;
      default: return Activity;
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
              <span>Investment Analysis로 돌아가기</span>
            </Link>
          </div>
        </div>
      </div>

      {/* Hero Section */}
      <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl md:text-5xl font-bold mb-4">
              Professional Trading Tools
            </h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto mb-8">
              기관투자자급 분석 도구부터 퀀트 트레이딩 플랫폼까지, 
              전문 투자를 위한 모든 도구를 한 곳에서
            </p>
            
            {/* Stats */}
            <div className="grid grid-cols-3 gap-8 max-w-2xl mx-auto">
              <div>
                <div className="text-3xl font-bold mb-1">{allTools.length}</div>
                <div className="text-gray-400">전체 도구</div>
              </div>
              <div>
                <div className="text-3xl font-bold mb-1">
                  {allTools.filter(t => t.status === 'live').length}
                </div>
                <div className="text-gray-400">Live 도구</div>
              </div>
              <div>
                <div className="text-3xl font-bold mb-1">12.5K+</div>
                <div className="text-gray-400">활성 사용자</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Filters and Search */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
          {/* Search Bar */}
          <div className="relative mb-6">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="도구 이름, 기능으로 검색..."
              className="w-full pl-12 pr-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          {/* Filters */}
          <div className="flex flex-col md:flex-row gap-6">
            {/* Category Filter */}
            <div className="flex-1">
              <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">카테고리</h3>
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

            {/* Status Filter */}
            <div className="md:w-48">
              <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">상태</h3>
              <select
                className="w-full px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={selectedStatus}
                onChange={(e) => setSelectedStatus(e.target.value)}
              >
                <option value="all">전체</option>
                <option value="live">Live</option>
                <option value="beta">Beta</option>
                <option value="coming-soon">Coming Soon</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Tools Grid */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-16">
        <div className="mb-6 flex items-center justify-between">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            {filteredTools.length}개 도구
          </h2>
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <Filter className="w-4 h-4" />
            <span>필터 적용됨</span>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {filteredTools.map((tool) => {
            const Icon = tool.icon;
            const CategoryIcon = getCategoryIcon(tool.category);
            
            return (
              <div
                key={tool.id}
                className="bg-white dark:bg-gray-800 rounded-xl shadow-sm hover:shadow-lg transition-all p-6"
              >
                {/* Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center">
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-1">
                        {tool.name}
                      </h3>
                      <p className="text-gray-600 dark:text-gray-400">
                        {tool.description}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Badges */}
                <div className="flex flex-wrap gap-2 mb-4">
                  <span className={`text-xs px-3 py-1 rounded-full font-medium ${getStatusBadgeColor(tool.status)}`}>
                    {tool.status === 'live' ? 'Live' : tool.status === 'beta' ? 'Beta' : 'Coming Soon'}
                  </span>
                  <span className={`text-xs px-3 py-1 rounded-full font-medium ${getPricingBadgeColor(tool.pricing)}`}>
                    {tool.pricing === 'free' ? 'Free' : tool.pricing === 'premium' ? 'Premium' : 'Enterprise'}
                  </span>
                  {tool.badge && (
                    <span className="text-xs px-3 py-1 rounded-full font-medium bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300">
                      {tool.badge}
                    </span>
                  )}
                  {tool.users && (
                    <span className="text-xs px-3 py-1 rounded-full font-medium bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300">
                      <Users className="w-3 h-3 inline mr-1" />
                      {tool.users} users
                    </span>
                  )}
                </div>

                {/* Features */}
                <div className="mb-4">
                  <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                    주요 기능
                  </h4>
                  <div className="grid grid-cols-2 gap-2">
                    {tool.features.slice(0, 4).map((feature) => (
                      <div key={feature} className="flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Requirements */}
                {tool.requirements && (
                  <div className="mb-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                    <div className="flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 text-yellow-600 dark:text-yellow-400 mt-0.5" />
                      <div>
                        <p className="text-xs font-medium text-yellow-800 dark:text-yellow-200">요구사항</p>
                        <p className="text-xs text-yellow-700 dark:text-yellow-300">
                          {tool.requirements.join(', ')}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* CTA */}
                <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
                  <div className="flex items-center gap-2">
                    <CategoryIcon className="w-4 h-4 text-gray-400" />
                    <span className="text-sm text-gray-500 dark:text-gray-400">
                      {tool.category.charAt(0).toUpperCase() + tool.category.slice(1)}
                    </span>
                  </div>
                  {tool.status === 'live' || tool.status === 'beta' ? (
                    <Link
                      href={`/modules/stock-analysis/tools/${tool.id}`}
                      className="inline-flex items-center gap-2 text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 font-medium"
                    >
                      <span className="text-sm">도구 시작하기</span>
                      <ChevronRight className="w-4 h-4" />
                    </Link>
                  ) : (
                    <div className="flex items-center gap-2 text-gray-400">
                      <Lock className="w-4 h-4" />
                      <span className="text-sm">준비중</span>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Empty State */}
        {filteredTools.length === 0 && (
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

      {/* CTA Section */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-700 py-12">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-2xl font-bold text-white mb-4">
            더 많은 도구가 필요하신가요?
          </h2>
          <p className="text-lg text-blue-100 mb-6">
            지속적으로 새로운 도구를 추가하고 있습니다. 필요한 도구를 제안해주세요.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/modules/stock-analysis/request-tool"
              className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-white text-blue-700 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
            >
              <ExternalLink className="w-5 h-5" />
              도구 제안하기
            </Link>
            <Link
              href="/modules/stock-analysis/api-docs"
              className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-blue-700 text-white rounded-lg font-semibold hover:bg-blue-800 transition-colors border border-white/20"
            >
              <Globe className="w-5 h-5" />
              API 문서 보기
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}