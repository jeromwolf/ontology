'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { moduleCategories } from '@/data/modules';

export default function ModuleCatalog() {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const getDifficultyColor = (status: string) => {
    switch (status) {
      case '학습 가능': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      case '개발중': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400';
      case '준비중': return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400';
      default: return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
    }
  };

  const filteredCategories = selectedCategory
    ? moduleCategories.filter(cat => cat.id === selectedCategory)
    : moduleCategories;

  const totalModules = moduleCategories.reduce((sum, cat) => sum + cat.modules.length, 0);

  return (
    <section id="learning-modules" className="py-20 px-6 bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Learning Modules
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto mb-2">
            {totalModules}개의 전문 모듈로 AI부터 엔지니어링까지 완전 정복
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-500">
            카테고리를 선택하거나 전체 모듈을 탐색하세요
          </p>
        </div>

        {/* Category Filter */}
        <div className="flex flex-wrap gap-3 justify-center mb-12">
          <button
            onClick={() => setSelectedCategory(null)}
            className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
              selectedCategory === null
                ? 'bg-blue-600 text-white shadow-lg scale-105'
                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 border border-gray-200 dark:border-gray-700'
            }`}
          >
            전체 ({totalModules})
          </button>
          {moduleCategories.map((category) => (
            <button
              key={category.id}
              onClick={() => setSelectedCategory(category.id)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-all flex items-center gap-2 ${
                selectedCategory === category.id
                  ? 'bg-blue-600 text-white shadow-lg scale-105'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 border border-gray-200 dark:border-gray-700'
              }`}
            >
              <span>{category.icon}</span>
              <span>{category.title}</span>
              <span className="text-xs opacity-70">({category.modules.length})</span>
            </button>
          ))}
        </div>

        {/* Module Grid by Category */}
        <div className="space-y-12">
          {filteredCategories.map((category) => (
            <div key={category.id} className="space-y-6">
              {/* Category Header */}
              <div className="flex items-center gap-6 pb-6 mb-2">
                <div className="w-16 h-16 bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-700 rounded-2xl flex items-center justify-center shadow-md">
                  <span className="text-4xl">{category.icon}</span>
                </div>
                <div className="flex-grow">
                  <h3 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                    {category.title}
                  </h3>
                  <p className="text-base text-gray-600 dark:text-gray-400">
                    {category.description}
                  </p>
                </div>
                <div className="bg-blue-50 dark:bg-blue-900/30 px-6 py-3 rounded-xl border border-blue-200 dark:border-blue-800">
                  <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {category.modules.length}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400 font-medium">
                    모듈
                  </div>
                </div>
              </div>

              {/* Modules in Category */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {category.modules.map((module) => (
                  <Link key={module.id} href={module.href} className="block group">
                    <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-md border-2 border-gray-100 dark:border-gray-700 hover:border-blue-400 dark:hover:border-blue-500 hover:shadow-2xl transition-all duration-300 cursor-pointer h-full flex flex-col group-hover:-translate-y-1">
                      {/* Icon with gradient background */}
                      <div className={`w-16 h-16 bg-gradient-to-r ${module.gradient} rounded-2xl flex items-center justify-center mb-6 shadow-lg group-hover:scale-110 transition-transform duration-300`}>
                        <span className="text-white text-2xl">{module.icon}</span>
                      </div>

                      {/* Title */}
                      <h4 className="text-xl font-bold text-gray-900 dark:text-white mb-4 leading-tight group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                        {module.title}
                      </h4>

                      {/* Description */}
                      <p className="text-gray-600 dark:text-gray-300 text-base leading-relaxed mb-6 flex-grow">
                        {module.description}
                      </p>

                      {/* Footer with duration and status */}
                      <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
                        <div className="flex items-center gap-2">
                          <svg className="w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
                            {module.duration}
                          </span>
                        </div>
                        <span className={`px-3 py-1.5 rounded-full text-xs font-semibold ${getDifficultyColor(module.status)}`}>
                          {module.status}
                        </span>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Summary Stats */}
        {selectedCategory === null && (
          <div className="mt-16 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-2xl p-8 border border-blue-200 dark:border-blue-800">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
              <div>
                <div className="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-2">
                  {totalModules}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  전체 모듈
                </div>
              </div>
              <div>
                <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-2">
                  {moduleCategories.reduce((sum, cat) =>
                    sum + cat.modules.filter(m => m.status === '학습 가능').length, 0
                  )}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  학습 가능
                </div>
              </div>
              <div>
                <div className="text-3xl font-bold text-yellow-600 dark:text-yellow-400 mb-2">
                  {moduleCategories.reduce((sum, cat) =>
                    sum + cat.modules.filter(m => m.status === '개발중').length, 0
                  )}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  개발중
                </div>
              </div>
              <div>
                <div className="text-3xl font-bold text-purple-600 dark:text-purple-400 mb-2">
                  {moduleCategories.length}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  카테고리
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
