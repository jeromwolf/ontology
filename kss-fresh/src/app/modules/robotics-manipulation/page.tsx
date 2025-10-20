'use client'

import Navigation from '@/components/Navigation'
import Link from 'next/link'
import { ArrowRight, Play, BookOpen, Wrench } from 'lucide-react'
import { moduleMetadata } from './metadata'

export default function RoboticsManipulationPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Navigation />

      <main className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        <div className="bg-gradient-to-r from-orange-600 to-red-600 rounded-2xl p-8 md:p-12 text-white mb-8">
          <div className="max-w-4xl">
            <div className="inline-block px-4 py-1 bg-white/20 rounded-full text-sm font-medium mb-4">
              🦾 Advanced Level • {moduleMetadata.duration}
            </div>
            <h1 className="text-4xl md:text-5xl font-bold mb-4">
              {moduleMetadata.title}
            </h1>
            <p className="text-xl text-white/90 mb-6">
              {moduleMetadata.description}
            </p>
            <div className="flex flex-wrap gap-4">
              <Link
                href="#chapters"
                className="inline-flex items-center gap-2 px-6 py-3 bg-white text-orange-600 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
              >
                <BookOpen className="w-5 h-5" />
                학습 시작하기
              </Link>
              <Link
                href="#simulators"
                className="inline-flex items-center gap-2 px-6 py-3 bg-orange-700 text-white rounded-lg font-semibold hover:bg-orange-800 transition-colors"
              >
                <Play className="w-5 h-5" />
                시뮬레이터 체험
              </Link>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <div className="text-3xl font-bold text-orange-600 mb-2">{moduleMetadata.chapters.length}</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Chapters</div>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <div className="text-3xl font-bold text-orange-600 mb-2">{moduleMetadata.simulators.length}</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Simulators</div>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <div className="text-3xl font-bold text-orange-600 mb-2">{moduleMetadata.duration}</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Duration</div>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <div className="text-3xl font-bold text-orange-600 mb-2">{moduleMetadata.difficulty}</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Level</div>
          </div>
        </div>

        {/* Chapters */}
        <section id="chapters" className="mb-12">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-6">
            📚 학습 커리큘럼
          </h2>
          <div className="grid gap-4">
            {moduleMetadata.chapters.map((chapter) => (
              <Link
                key={chapter.id}
                href={`/modules/robotics-manipulation/chapter-${chapter.id}`}
                className="block bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 hover:border-orange-500 dark:hover:border-orange-500 transition-colors group"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 bg-orange-100 dark:bg-orange-900/30 rounded-lg flex items-center justify-center text-orange-600 dark:text-orange-400 font-bold text-lg">
                      {chapter.id}
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-gray-900 dark:text-white group-hover:text-orange-600 dark:group-hover:text-orange-400 transition-colors">
                        {chapter.title}
                      </h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {chapter.description}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
                    <span>{chapter.duration}</span>
                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  {chapter.learningObjectives.slice(0, 3).map((objective, idx) => (
                    <span
                      key={idx}
                      className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded"
                    >
                      {objective.length > 40 ? objective.slice(0, 40) + '...' : objective}
                    </span>
                  ))}
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* Simulators */}
        <section id="simulators" className="mb-12">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-6">
            🎮 인터랙티브 시뮬레이터
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {moduleMetadata.simulators.map((simulator) => (
              <Link
                key={simulator.id}
                href={`/modules/robotics-manipulation/simulators/${simulator.id}`}
                className="block bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 hover:border-orange-500 dark:hover:border-orange-500 transition-all hover:shadow-lg group"
              >
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-12 h-12 bg-gradient-to-br from-orange-500 to-red-500 rounded-lg flex items-center justify-center">
                    <Wrench className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-bold text-gray-900 dark:text-white group-hover:text-orange-600 dark:group-hover:text-orange-400 transition-colors">
                      {simulator.title}
                    </h3>
                    <span className={`text-xs px-2 py-0.5 rounded ${
                      simulator.difficulty === 'beginner' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                      simulator.difficulty === 'intermediate' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' :
                      'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                    }`}>
                      {simulator.difficulty}
                    </span>
                  </div>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {simulator.description}
                </p>
              </Link>
            ))}
          </div>
        </section>

        {/* Prerequisites & Outcomes */}
        <div className="grid md:grid-cols-2 gap-8 mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              📋 선수 지식
            </h3>
            <ul className="space-y-2">
              {moduleMetadata.prerequisites.map((prereq, idx) => (
                <li key={idx} className="flex items-start gap-2 text-gray-600 dark:text-gray-400">
                  <span className="text-orange-500 mt-1">•</span>
                  {prereq}
                </li>
              ))}
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              🎯 학습 성과
            </h3>
            <ul className="space-y-2">
              {moduleMetadata.outcomes.map((outcome, idx) => (
                <li key={idx} className="flex items-start gap-2 text-gray-600 dark:text-gray-400">
                  <span className="text-orange-500 mt-1">✓</span>
                  {outcome}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Tools */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
            🛠️ 사용 도구 및 프레임워크
          </h3>
          <div className="flex flex-wrap gap-2">
            {moduleMetadata.tools.map((tool, idx) => (
              <span
                key={idx}
                className="px-3 py-1 bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400 rounded-lg text-sm font-medium"
              >
                {tool}
              </span>
            ))}
          </div>
        </div>
      </main>
    </div>
  )
}
