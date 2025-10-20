'use client';

import { useState } from 'react';
import Link from 'next/link';
import { BookOpen, Zap } from 'lucide-react';
import { moduleMetadata } from './metadata';
import ChapterContent from './components/ChapterContent';

type TabType = 'chapters' | 'simulators';

export default function ModulePage() {
  const [activeTab, setActiveTab] = useState<TabType>('chapters');
  const [selectedChapterId, setSelectedChapterId] = useState<string | null>(null);

  const tabs = [
    { id: 'chapters' as TabType, label: '📖 학습', icon: BookOpen, count: moduleMetadata.chapters.length },
    { id: 'simulators' as TabType, label: '🎮 시뮬레이터', icon: Zap, count: moduleMetadata.simulators.length },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link href="/" className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 mb-4 inline-block">
            ← 홈으로 돌아가기
          </Link>
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8">
            <div className="flex items-center gap-4 mb-6">
              <div className={`w-16 h-16 bg-gradient-to-r ${moduleMetadata.gradient} rounded-xl flex items-center justify-center`}>
                <span className="text-3xl">{moduleMetadata.icon}</span>
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">{moduleMetadata.title}</h1>
                <p className="text-gray-600 dark:text-gray-300 mt-2">{moduleMetadata.description}</p>
              </div>
            </div>
            <div className="flex gap-4 text-sm flex-wrap">
              <span className="px-3 py-1 bg-gray-100 dark:bg-gray-700 rounded-full text-gray-800 dark:text-gray-200">
                {moduleMetadata.category}
              </span>
              <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 rounded-full">
                {moduleMetadata.difficulty}
              </span>
              <span className="px-3 py-1 bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 rounded-full">
                {moduleMetadata.estimatedHours}시간
              </span>
              <span className="px-3 py-1 bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300 rounded-full">
                {moduleMetadata.chapters.length}개 챕터
              </span>
              <span className="px-3 py-1 bg-orange-100 dark:bg-orange-900 text-orange-700 dark:text-orange-300 rounded-full">
                {moduleMetadata.simulators.length}개 시뮬레이터
              </span>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
          <div className="flex border-b border-gray-200 dark:border-gray-700">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveTab(tab.id);
                  setSelectedChapterId(null);
                }}
                className={`flex-1 px-6 py-4 font-semibold text-center transition-all relative ${
                  activeTab === tab.id
                    ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20'
                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700/50'
                }`}
              >
                <span className="flex items-center justify-center gap-2">
                  <tab.icon className="w-5 h-5" />
                  <span>{tab.label}</span>
                  <span className={`ml-2 px-2 py-0.5 rounded-full text-xs ${
                    activeTab === tab.id
                      ? 'bg-blue-600 dark:bg-blue-500 text-white'
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                  }`}>
                    {tab.count}
                  </span>
                </span>
                {activeTab === tab.id && (
                  <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 to-purple-600" />
                )}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="p-6">
            {activeTab === 'chapters' && (
              <div>
                {!selectedChapterId ? (
                  <>
                    <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">학습 챕터</h2>
                    <div className="grid gap-4">
                      {moduleMetadata.chapters.map((chapter, index) => (
                        <button
                          key={chapter.id}
                          onClick={() => setSelectedChapterId(chapter.id)}
                          className="bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 rounded-lg shadow p-6 text-left transition-all border-2 border-transparent hover:border-blue-500"
                        >
                          <div className="flex items-start gap-4">
                            <div className="flex-shrink-0 w-10 h-10 bg-gradient-to-r from-red-500 to-orange-500 rounded-lg flex items-center justify-center text-white font-bold">
                              {index + 1}
                            </div>
                            <div className="flex-1">
                              <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">{chapter.title}</h3>
                              <p className="text-gray-600 dark:text-gray-300 mb-2">{chapter.description}</p>
                              <p className="text-sm text-gray-500 dark:text-gray-400">{chapter.estimatedMinutes}분</p>
                            </div>
                          </div>
                        </button>
                      ))}
                    </div>
                  </>
                ) : (
                  <div>
                    <button
                      onClick={() => setSelectedChapterId(null)}
                      className="mb-4 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 flex items-center gap-2"
                    >
                      ← 챕터 목록으로 돌아가기
                    </button>
                    <ChapterContent chapterId={selectedChapterId} />
                  </div>
                )}
              </div>
            )}

            {activeTab === 'simulators' && (
              <div>
                <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">인터랙티브 시뮬레이터</h2>
                <div className="grid md:grid-cols-2 gap-6">
                  {moduleMetadata.simulators.map((simulator) => (
                    <Link
                      key={simulator.id}
                      href={`/modules/cyber-security/simulators/${simulator.id}`}
                      className="bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 rounded-lg shadow p-6 transition-all border-2 border-transparent hover:border-purple-500 group"
                    >
                      <div className="flex items-start gap-4">
                        <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center text-2xl group-hover:scale-110 transition-transform">
                          🔒
                        </div>
                        <div className="flex-1">
                          <h3 className="text-lg font-semibold mb-2 text-gray-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400">
                            {simulator.title}
                          </h3>
                          <p className="text-gray-600 dark:text-gray-300 text-sm">{simulator.description}</p>
                        </div>
                      </div>
                    </Link>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
