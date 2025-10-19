'use client'

import { useState } from 'react'
import Link from 'next/link'
import dynamic from 'next/dynamic'
import { Play, Clock, Target, BookOpen, Zap, CheckCircle2, FileText } from 'lucide-react'
import { llmModule } from './metadata'
import ModuleRelatedPapers from '@/components/papers/ModuleRelatedPapers'

// Dynamic import for LLM Simulators
const LLMSimulators = dynamic(
  () => import('@/components/llm-simulators/LLMSimulators'),
  { loading: () => <div>시뮬레이터 로딩 중...</div> }
)

type TabType = 'chapters' | 'simulators' | 'papers'

export default function LLMMainPage() {
  const [completedChapters, setCompletedChapters] = useState<string[]>([])
  const [activeTab, setActiveTab] = useState<TabType>('chapters')

  const progress = (completedChapters.length / llmModule.chapters.length) * 100

  const tabs = [
    { id: 'chapters' as TabType, label: '📖 학습', icon: BookOpen, count: llmModule.chapters.length },
    { id: 'simulators' as TabType, label: '🎮 시뮬레이터', icon: Zap, count: 5 },
    { id: 'papers' as TabType, label: '📄 관련 논문', icon: FileText, count: null }
  ]

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <section className="text-center py-12">
        <div className="w-20 h-20 mx-auto rounded-3xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-4xl mb-6">
          {llmModule.icon}
        </div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          {llmModule.nameKo}
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-2xl mx-auto">
          {llmModule.description}
        </p>

        {/* Progress */}
        <div className="max-w-md mx-auto mb-8">
          <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
            <span>학습 진도</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
            <div
              className="bg-gradient-to-r from-indigo-500 to-purple-600 h-3 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </div>

        <Link
          href={`/modules/llm/${llmModule.chapters[0].id}`}
          className="inline-flex items-center gap-2 bg-gradient-to-r from-indigo-500 to-purple-600 text-white px-8 py-4 rounded-xl font-semibold hover:shadow-lg transition-all duration-200 hover:-translate-y-1"
        >
          <Play size={20} />
          학습 시작하기
        </Link>
      </section>

      {/* 학습 목표 */}
      <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Target className="text-indigo-500" size={24} />
          학습 목표
        </h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h3 className="font-semibold text-gray-800 dark:text-gray-200">핵심 개념 이해</h3>
            <ul className="space-y-2 text-gray-600 dark:text-gray-400">
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                Transformer 아키텍처 완전 이해
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                Attention 메커니즘 동작 원리
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                사전훈련과 파인튜닝 과정
              </li>
            </ul>
          </div>
          <div className="space-y-4">
            <h3 className="font-semibold text-gray-800 dark:text-gray-200">실전 활용</h3>
            <ul className="space-y-2 text-gray-600 dark:text-gray-400">
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                프롬프트 엔지니어링 기법
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                모델 선택과 최적화
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                실무 프로젝트 구현
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Tab Navigation */}
      <section>
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
          {/* Tab Headers */}
          <div className="flex border-b border-gray-200 dark:border-gray-700">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 px-6 py-4 font-semibold transition-all duration-200 relative ${
                  activeTab === tab.id
                    ? 'text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-900/20'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-50 dark:hover:bg-gray-700/50'
                }`}
              >
                <div className="flex items-center justify-center gap-2">
                  <span className="text-lg">{tab.label}</span>
                  {tab.count !== null && (
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      activeTab === tab.id
                        ? 'bg-indigo-600 text-white'
                        : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                    }`}>
                      {tab.count}
                    </span>
                  )}
                </div>
                {activeTab === tab.id && (
                  <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-indigo-500 to-purple-600"></div>
                )}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="p-8">
            {/* 챕터 목록 */}
            {activeTab === 'chapters' && (
              <div className="space-y-4">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-3">
                    <BookOpen className="text-indigo-500" size={24} />
                    챕터 목록
                  </h2>
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    {llmModule.chapters.length}개 챕터
                  </span>
                </div>
                <div className="grid gap-4">
                  {llmModule.chapters.map((chapter, index) => {
                    const isCompleted = completedChapters.includes(chapter.id)
                    const isLocked = index > 0 && !completedChapters.includes(llmModule.chapters[index - 1].id)

                    return (
                      <Link
                        key={chapter.id}
                        href={isLocked ? '#' : `/modules/llm/${chapter.id}`}
                        className={`block p-6 rounded-xl border transition-all duration-200 ${
                          isLocked
                            ? 'bg-gray-50 dark:bg-gray-800/50 border-gray-200 dark:border-gray-700 cursor-not-allowed opacity-60'
                            : isCompleted
                            ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-700 hover:shadow-md'
                            : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:shadow-md hover:border-indigo-300 dark:hover:border-indigo-600'
                        }`}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-3 mb-2">
                              <span className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                                isCompleted
                                  ? 'bg-green-500 text-white'
                                  : isLocked
                                  ? 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400'
                                  : 'bg-indigo-100 dark:bg-indigo-900 text-indigo-600 dark:text-indigo-400'
                              }`}>
                                {isCompleted ? <CheckCircle2 size={16} /> : index + 1}
                              </span>
                              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                                {chapter.title}
                              </h3>
                            </div>
                            <p className="text-gray-600 dark:text-gray-400 mb-3">
                              {chapter.description}
                            </p>
                            <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                              <div className="flex items-center gap-1">
                                <Clock size={14} />
                                <span>{chapter.estimatedMinutes}분</span>
                              </div>
                              <div className="flex items-center gap-2">
                                {chapter.keywords.slice(0, 3).map((keyword, i) => (
                                  <span key={i} className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">
                                    {keyword}
                                  </span>
                                ))}
                              </div>
                            </div>
                          </div>
                          {!isLocked && (
                            <div className="text-indigo-500">
                              <Play size={20} />
                            </div>
                          )}
                        </div>
                      </Link>
                    )
                  })}
                </div>
              </div>
            )}

            {/* 시뮬레이터 */}
            {activeTab === 'simulators' && (
              <div>
                <LLMSimulators />
              </div>
            )}

            {/* 관련 논문 */}
            {activeTab === 'papers' && (
              <div>
                <ModuleRelatedPapers
                  moduleId="llm"
                  maxPapers={20}
                  showStats={true}
                />
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  )
}
