'use client';

import Link from 'next/link';
import { ArrowLeft, Search, Sparkles, Calculator, BarChart3, PieChart, Activity, Brain, DollarSign, TrendingUp, AlertTriangle, Settings, Eye, Shield, Newspaper, Database, LineChart } from 'lucide-react';
import { useState } from 'react';
import { stockAnalysisModule } from '../metadata';

export default function StockAnalysisSimulatorsPage() {
  const [searchQuery, setSearchQuery] = useState('');
  
  // 시뮬레이터 카테고리 정의
  const categories = [
    {
      id: 'basic-analysis',
      name: '핵심 분석 도구',
      icon: Calculator,
      color: 'from-blue-500 to-cyan-500',
      simulators: ['financial-calculator', 'chart-analyzer', 'portfolio-optimizer', 'backtesting-engine']
    },
    {
      id: 'advanced-analysis',
      name: '고급 분석 도구',
      icon: BarChart3,
      color: 'from-purple-500 to-violet-500',
      simulators: ['real-time-dashboard', 'risk-management-dashboard', 'factor-investing-lab', 'options-strategy-analyzer']
    },
    {
      id: 'ai-tools',
      name: 'AI 도구',
      icon: Brain,
      color: 'from-green-500 to-emerald-500',
      simulators: ['ai-mentor', 'news-impact-analyzer', 'dcf-valuation-model']
    }
  ];

  // 모든 시뮬레이터가 구현됨
  const implementedSimulators = stockAnalysisModule.simulators.map(s => s.id);

  // 검색 필터링
  const filteredSimulators = stockAnalysisModule.simulators.filter(sim =>
    sim.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    sim.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getSimulatorsByCategory = (categorySimulators: string[]) => {
    return categorySimulators
      .map(id => stockAnalysisModule.simulators.find(sim => sim.id === id))
      .filter(Boolean);
  };

  const getSimulatorIcon = (simulatorId: string) => {
    const iconMap: { [key: string]: any } = {
      'financial-calculator': Calculator,
      'chart-analyzer': LineChart,
      'portfolio-optimizer': PieChart,
      'backtesting-engine': Activity,
      'ai-mentor': Brain,
      'dcf-valuation-model': DollarSign,
      'options-strategy-analyzer': TrendingUp,
      'risk-management-dashboard': Shield,
      'factor-investing-lab': BarChart3,
      'earnings-forecast-model': Eye,
      'market-sentiment-gauge': AlertTriangle,
      'real-time-dashboard': Activity,
      'news-impact-analyzer': Newspaper,
      'news-ontology': Database,
      'cache-dashboard': Settings
    };
    return iconMap[simulatorId] || Calculator;
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
              <span>주식 분석 모듈로 돌아가기</span>
            </Link>
          </div>
        </div>
      </div>

      {/* Hero Section */}
      <div className="bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
            모든 시뮬레이터 보기
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-400 mb-8 max-w-3xl mx-auto">
            전문가급 {stockAnalysisModule.simulators.length}개 투자 도구로 실전과 같은 연습을 해보세요
          </p>

          {/* Search Bar */}
          <div className="max-w-2xl mx-auto">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="시뮬레이터 검색..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-3 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent transition-all"
              />
            </div>
          </div>

          {/* Stats */}
          <div className="mt-8 flex items-center justify-center gap-8 text-sm text-gray-600 dark:text-gray-400">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-green-500" />
              <span>
                <span className="font-bold text-green-600">{stockAnalysisModule.simulators.length}개</span> 시뮬레이터
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-500" />
              <span>실시간 데이터 연동</span>
            </div>
            <div className="flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-500" />
              <span>AI 기반 분석</span>
            </div>
          </div>
        </div>
      </div>

      {/* Simulators by Category */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {searchQuery ? (
          // Search Results
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              검색 결과 ({filteredSimulators.length}개)
            </h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredSimulators.map((simulator) => {
                const IconComponent = getSimulatorIcon(simulator.id);
                const isImplemented = implementedSimulators.includes(simulator.id);
                
                return (
                  <Link
                    key={simulator.id}
                    href={`/modules/stock-analysis/simulators/${simulator.id}`}
                    className={`group relative bg-white dark:bg-gray-800 rounded-xl p-6 border ${
                      isImplemented
                        ? 'border-gray-200 dark:border-gray-700 hover:shadow-lg hover:scale-105'
                        : 'border-gray-300 dark:border-gray-600 opacity-60 cursor-not-allowed'
                    } transition-all duration-200`}
                  >
                    {!isImplemented && (
                      <div className="absolute top-2 right-2">
                        <span className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 rounded-full">
                          준비 중
                        </span>
                      </div>
                    )}
                    
                    <div className="flex items-start gap-4">
                      <div className={`w-12 h-12 rounded-lg bg-gradient-to-r ${
                        isImplemented ? 'from-red-500 to-orange-500' : 'from-gray-400 to-gray-500'
                      } flex items-center justify-center text-white flex-shrink-0`}>
                        <IconComponent className="w-6 h-6" />
                      </div>
                      
                      <div className="flex-1">
                        <h3 className="font-semibold text-gray-900 dark:text-white mb-2 group-hover:text-red-600 dark:group-hover:text-red-400 transition-colors">
                          {simulator.name}
                        </h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
                          {simulator.description}
                        </p>
                      </div>
                    </div>
                  </Link>
                );
              })}
            </div>
          </div>
        ) : (
          // Categories View
          <div className="space-y-12">
            {categories.map((category) => {
              const categorySimulators = getSimulatorsByCategory(category.simulators);
              const IconComponent = category.icon;
              
              return (
                <div key={category.id}>
                  <div className="flex items-center gap-3 mb-6">
                    <div className={`w-10 h-10 rounded-lg bg-gradient-to-r ${category.color} flex items-center justify-center text-white`}>
                      <IconComponent className="w-6 h-6" />
                    </div>
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                      {category.name}
                    </h2>
                    <span className="text-sm text-gray-500 dark:text-gray-400">
                      ({categorySimulators.length}개)
                    </span>
                  </div>
                  
                  <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {categorySimulators.map((simulator) => {
                      if (!simulator) return null;
                      const SimIcon = getSimulatorIcon(simulator.id);
                      const isImplemented = implementedSimulators.includes(simulator.id);
                      
                      return (
                        <Link
                          key={simulator.id}
                          href={`/modules/stock-analysis/simulators/${simulator.id}`}
                          className={`group relative bg-white dark:bg-gray-800 rounded-xl p-6 border ${
                            isImplemented
                              ? 'border-gray-200 dark:border-gray-700 hover:shadow-lg hover:scale-105'
                              : 'border-gray-300 dark:border-gray-600 opacity-60 cursor-not-allowed'
                          } transition-all duration-200`}
                        >
                          {!isImplemented && (
                            <div className="absolute top-2 right-2">
                              <span className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 rounded-full">
                                준비 중
                              </span>
                            </div>
                          )}
                          
                          <div className="flex items-start gap-4">
                            <div className={`w-12 h-12 rounded-lg bg-gradient-to-r ${
                              isImplemented ? category.color : 'from-gray-400 to-gray-500'
                            } flex items-center justify-center text-white flex-shrink-0`}>
                              <SimIcon className="w-6 h-6" />
                            </div>
                            
                            <div className="flex-1">
                              <h3 className="font-semibold text-gray-900 dark:text-white mb-2 group-hover:text-red-600 dark:group-hover:text-red-400 transition-colors">
                                {simulator.name}
                              </h3>
                              <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
                                {simulator.description}
                              </p>
                              
                              {isImplemented && (
                                <div className="mt-3 flex items-center gap-2 text-xs text-green-600 dark:text-green-400">
                                  <Activity className="w-3 h-3" />
                                  <span>사용 가능</span>
                                </div>
                              )}
                            </div>
                          </div>
                        </Link>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* CTA Section */}
        <div className="mt-16 bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-2xl p-8">
          <div className="max-w-3xl mx-auto text-center">
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              시뮬레이터를 활용한 실전 투자 학습 🎯
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-6">
              모든 시뮬레이터는 실제 시장 데이터와 전문가급 분석 기법을 기반으로 만들어졌습니다.
              이론을 실습으로 체득하며 투자 실력을 키워보세요.
            </p>
            <div className="flex items-center justify-center gap-4">
              <Link
                href="/modules/stock-analysis"
                className="inline-flex items-center gap-2 px-6 py-3 bg-white dark:bg-gray-800 text-gray-900 dark:text-white border border-gray-300 dark:border-gray-600 rounded-lg font-medium hover:shadow-md transition-all duration-200"
              >
                <ArrowLeft className="w-5 h-5" />
                학습 과정 보기
              </Link>
              <Link
                href="/modules/stock-analysis/learn/beginner"
                className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-red-600 to-orange-600 text-white rounded-lg font-medium hover:shadow-lg hover:scale-105 transition-all duration-200"
              >
                <Sparkles className="w-5 h-5" />
                투자 학습 시작하기
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}