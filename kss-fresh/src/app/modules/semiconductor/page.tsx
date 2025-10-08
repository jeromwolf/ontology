'use client'

import { useState } from 'react'
import Link from 'next/link'
import { moduleInfo } from './metadata'
import { beginnerCurriculum } from '@/data/semiconductor/beginnerCurriculum'
import { intermediateCurriculum } from '@/data/semiconductor/intermediateCurriculum'
import { advancedCurriculum } from '@/data/semiconductor/advancedCurriculum'

type Level = 'beginner' | 'intermediate' | 'advanced' | 'simulators'

export default function SemiconductorPage() {
  const [activeTab, setActiveTab] = useState<Level>('beginner')

  const tabs = [
    { id: 'beginner' as Level, label: '초급', icon: '🌱', color: 'blue' },
    { id: 'intermediate' as Level, label: '중급', icon: '🚀', color: 'purple' },
    { id: 'advanced' as Level, label: '고급', icon: '⚡', color: 'indigo' },
    { id: 'simulators' as Level, label: '시뮬레이터', icon: '🔬', color: 'pink' }
  ]

  const getCurriculum = () => {
    switch (activeTab) {
      case 'beginner':
        return beginnerCurriculum
      case 'intermediate':
        return intermediateCurriculum
      case 'advanced':
        return advancedCurriculum
      default:
        return null
    }
  }

  const curriculum = getCurriculum()

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-4 mb-6">
            <span className="text-6xl">{moduleInfo.icon}</span>
            <div>
              <h1 className="text-5xl font-bold mb-2">{moduleInfo.title}</h1>
              <p className="text-xl text-blue-100">{moduleInfo.description}</p>
            </div>
          </div>

          {/* Stats */}
          <div className="flex gap-6 mt-8 text-sm">
            <div className="flex items-center gap-2">
              <span className="text-blue-200">⏱️</span>
              <span>{moduleInfo.duration}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-blue-200">👥</span>
              <span>{moduleInfo.students}명 수강</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-blue-200">⭐</span>
              <span>{moduleInfo.rating}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-blue-200">📚</span>
              <span>8개 챕터</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-blue-200">🔬</span>
              <span>6개 시뮬레이터</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 sticky top-16 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex gap-4 overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-6 py-4 font-medium text-sm whitespace-nowrap transition-all border-b-2 ${
                  activeTab === tab.id
                    ? `border-${tab.color}-500 text-${tab.color}-600 dark:text-${tab.color}-400`
                    : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Curriculum Content */}
        {activeTab !== 'simulators' && curriculum && (
          <div>
            {/* Curriculum Header */}
            <div className="mb-8">
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                {curriculum.title}
              </h2>
              <p className="text-lg text-gray-600 dark:text-gray-400 mb-4">
                {curriculum.description}
              </p>
              <div className="flex items-center gap-4">
                <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full text-sm font-medium">
                  총 학습 시간: {curriculum.duration}
                </span>
                <span className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-full text-sm font-medium">
                  {curriculum.modules.length}개 모듈
                </span>
              </div>
            </div>

            {/* Curriculum Modules */}
            <div className="space-y-4">
              {curriculum.modules.map((module, index) => (
                <div
                  key={module.id}
                  className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                        {module.title}
                      </h3>
                      <span className="text-sm text-gray-500 dark:text-gray-400">
                        ⏱️ {module.duration}
                      </span>
                    </div>
                    <div className="ml-4">
                      {module.completed ? (
                        <span className="inline-flex items-center px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded-full text-sm font-medium">
                          ✓ 완료
                        </span>
                      ) : (
                        <span className="inline-flex items-center px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full text-sm font-medium">
                          학습 대기
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Topics */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {module.topics.map((topic, topicIndex) => (
                      <div
                        key={topicIndex}
                        className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400"
                      >
                        <span className="text-blue-500">▸</span>
                        {topic}
                      </div>
                    ))}
                  </div>

                  {/* Start Button */}
                  <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                    <Link
                      href={`/modules/semiconductor/${(module as any).chapterId || 'basics'}`}
                      className="inline-block px-4 py-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg hover:shadow-lg transition-all font-medium text-sm"
                    >
                      학습 시작 →
                    </Link>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Simulators */}
        {activeTab === 'simulators' && (
          <div>
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
              🔬 인터랙티브 시뮬레이터
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {moduleInfo.simulators.map((simulator) => (
                <Link
                  key={simulator.id}
                  href={`/modules/semiconductor/simulators/${simulator.id}`}
                  className="block p-6 bg-gradient-to-br from-blue-500 to-indigo-600 text-white rounded-xl shadow-lg hover:shadow-2xl transition-all hover:-translate-y-1"
                >
                  <h3 className="text-xl font-bold mb-2">{simulator.title}</h3>
                  <p className="text-blue-100 text-sm mb-4">{simulator.description}</p>
                  <div className="inline-flex items-center gap-2 text-sm font-medium">
                    체험하기 →
                  </div>
                </Link>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
