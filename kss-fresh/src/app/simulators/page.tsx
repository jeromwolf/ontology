'use client'

import React, { useState, useMemo } from 'react';
import Link from 'next/link';
import { Search, Filter, Sparkles, ChevronRight } from 'lucide-react';
import { getAllSimulators, getCategories, getSimulatorStats } from '@/lib/simulatorRegistry';

export default function SimulatorsPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('All');

  const simulators = getAllSimulators();
  const categories = getCategories();
  const stats = getSimulatorStats();

  // Filter simulators based on search and category
  const filteredSimulators = useMemo(() => {
    let filtered = simulators;

    // Category filter
    if (selectedCategory !== 'All') {
      filtered = filtered.filter(sim => sim.category === selectedCategory);
    }

    // Search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(sim =>
        sim.name.toLowerCase().includes(query) ||
        sim.description.toLowerCase().includes(query) ||
        sim.moduleName.toLowerCase().includes(query)
      );
    }

    return filtered;
  }, [simulators, searchQuery, selectedCategory]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50 dark:from-gray-900 dark:via-gray-900 dark:to-purple-900">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        {/* Background decoration */}
        <div className="absolute inset-0 bg-grid-slate-900/[0.04] dark:bg-grid-slate-400/[0.05]" />
        <div className="absolute inset-0 bg-gradient-to-t from-white/80 dark:from-gray-900/80" />

        <div className="relative max-w-7xl mx-auto px-6 py-20">
          {/* Header */}
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-purple-100 dark:bg-purple-900/30 rounded-full mb-6">
              <Sparkles className="w-4 h-4 text-purple-600 dark:text-purple-400" />
              <span className="text-sm font-medium text-purple-900 dark:text-purple-300">
                {stats.total}+ Interactive Simulators
              </span>
            </div>

            <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-purple-600 via-blue-600 to-indigo-600 bg-clip-text text-transparent">
              Simulator Gallery
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto mb-8">
              모든 모듈의 인터랙티브 시뮬레이터를 한곳에서 탐색하고 체험하세요.
              <br />
              실전 중심의 학습 도구로 복잡한 개념을 직접 경험할 수 있습니다.
            </p>

            {/* Stats Dashboard */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto mb-12">
              <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg border border-gray-100 dark:border-gray-700">
                <div className="text-3xl font-bold text-purple-600 dark:text-purple-400 mb-1">
                  {stats.total}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">시뮬레이터</div>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg border border-gray-100 dark:border-gray-700">
                <div className="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-1">
                  {stats.modules}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">모듈</div>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg border border-gray-100 dark:border-gray-700">
                <div className="text-3xl font-bold text-indigo-600 dark:text-indigo-400 mb-1">
                  {stats.categories}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">카테고리</div>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg border border-gray-100 dark:border-gray-700">
                <div className="text-3xl font-bold text-pink-600 dark:text-pink-400 mb-1">
                  100%
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">무료 체험</div>
              </div>
            </div>
          </div>

          {/* Search Bar */}
          <div className="max-w-2xl mx-auto mb-8">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="시뮬레이터 이름, 설명, 모듈로 검색..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-12 pr-4 py-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-2xl text-gray-900 dark:text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 dark:focus:ring-purple-600 focus:border-transparent shadow-lg"
              />
            </div>
          </div>

          {/* Category Filter */}
          <div className="flex items-center justify-center gap-3 flex-wrap mb-8">
            <Filter className="w-5 h-5 text-gray-500 dark:text-gray-400" />
            <button
              onClick={() => setSelectedCategory('All')}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
                selectedCategory === 'All'
                  ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/30'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700 hover:border-purple-500 dark:hover:border-purple-600'
              }`}
            >
              All ({simulators.length})
            </button>
            {categories.map((category) => (
              <button
                key={category}
                onClick={() => setSelectedCategory(category)}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
                  selectedCategory === category
                    ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/30'
                    : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700 hover:border-purple-500 dark:hover:border-purple-600'
                }`}
              >
                {category} ({stats.byCategory.find(c => c.name === category)?.count || 0})
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Simulators Grid */}
      <section className="max-w-7xl mx-auto px-6 pb-20">
        {/* Results count */}
        <div className="mb-6 text-center">
          <p className="text-gray-600 dark:text-gray-400">
            {filteredSimulators.length === simulators.length
              ? `전체 ${filteredSimulators.length}개의 시뮬레이터`
              : `${filteredSimulators.length}개의 시뮬레이터 검색됨`}
          </p>
        </div>

        {/* Empty state */}
        {filteredSimulators.length === 0 && (
          <div className="text-center py-20">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gray-100 dark:bg-gray-800 rounded-full mb-4">
              <Search className="w-8 h-8 text-gray-400" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
              검색 결과가 없습니다
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-6">
              다른 검색어나 카테고리를 시도해보세요
            </p>
            <button
              onClick={() => {
                setSearchQuery('');
                setSelectedCategory('All');
              }}
              className="px-6 py-3 bg-purple-600 text-white rounded-xl hover:bg-purple-700 transition-colors"
            >
              전체 보기
            </button>
          </div>
        )}

        {/* Simulator Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredSimulators.map((simulator) => (
            <Link
              key={`${simulator.moduleId}-${simulator.id}`}
              href={simulator.url}
              className="group"
            >
              <div className="h-full bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 overflow-hidden hover:shadow-2xl hover:border-purple-500 dark:hover:border-purple-600 transition-all duration-300 hover:-translate-y-1">
                {/* Module Badge */}
                <div className={`h-2 bg-gradient-to-r ${simulator.gradient}`} />

                <div className="p-6">
                  {/* Module Tag */}
                  <div className="inline-flex items-center gap-2 mb-4">
                    <span className={`text-xs font-semibold px-3 py-1 rounded-full bg-gradient-to-r ${simulator.gradient} text-white`}>
                      {simulator.moduleName}
                    </span>
                  </div>

                  {/* Simulator Name */}
                  <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-3 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors line-clamp-2">
                    {simulator.name}
                  </h3>

                  {/* Description */}
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 line-clamp-2">
                    {simulator.description}
                  </p>

                  {/* Launch Button */}
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500 dark:text-gray-500">
                      {simulator.category}
                    </span>
                    <div className="flex items-center gap-1 text-purple-600 dark:text-purple-400 text-sm font-medium">
                      <span>실행하기</span>
                      <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                    </div>
                  </div>
                </div>

                {/* Hover Gradient Effect */}
                <div className={`h-1 bg-gradient-to-r ${simulator.gradient} opacity-0 group-hover:opacity-100 transition-opacity`} />
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="max-w-4xl mx-auto px-6 pb-20">
        <div className="bg-gradient-to-r from-purple-600 via-blue-600 to-indigo-600 rounded-3xl p-12 text-center shadow-2xl">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            더 많은 시뮬레이터가 추가되고 있습니다
          </h2>
          <p className="text-xl text-purple-100 mb-8">
            새로운 모듈과 시뮬레이터가 매주 업데이트됩니다
          </p>
          <Link
            href="/"
            className="inline-flex items-center gap-2 px-8 py-4 bg-white text-purple-600 rounded-xl font-semibold hover:bg-gray-100 transition-colors shadow-xl"
          >
            모듈 탐색하기
            <ChevronRight className="w-5 h-5" />
          </Link>
        </div>
      </section>
    </div>
  );
}
