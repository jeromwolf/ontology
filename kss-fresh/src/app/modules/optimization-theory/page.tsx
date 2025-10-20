'use client'

import { useState } from 'react'
import Link from 'next/link'
import { Play, Clock, Target, BookOpen, CheckCircle2, TrendingUp, Minimize, Maximize } from 'lucide-react'
import { moduleMetadata } from './metadata'

export default function OptimizationTheoryMainPage() {
  const [completedChapters, setCompletedChapters] = useState<string[]>([])

  const progress = (completedChapters.length / moduleMetadata.chapters.length) * 100

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="text-center py-16">
        <div className="w-20 h-20 mx-auto rounded-3xl bg-gradient-to-br from-emerald-600 to-teal-700 flex items-center justify-center text-white text-4xl mb-6">
          {moduleMetadata.icon}
        </div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          {moduleMetadata.title}
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-2xl mx-auto">
          {moduleMetadata.description}
        </p>

        {/* Progress */}
        <div className="max-w-md mx-auto mb-8">
          <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
            <span>학습 진도</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
            <div
              className="bg-gradient-to-r from-emerald-600 to-teal-700 h-3 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </div>

        <Link
          href={`/modules/optimization-theory/${moduleMetadata.chapters[0].id}`}
          className="inline-flex items-center gap-2 bg-gradient-to-r from-emerald-600 to-teal-700 text-white px-8 py-4 rounded-xl font-semibold hover:shadow-lg transition-all duration-200 hover:-translate-y-1"
        >
          <Play size={20} />
          학습 시작하기
        </Link>
      </section>

      {/* 학습 목표 */}
      <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Target className="text-emerald-600" size={24} />
          학습 목표
        </h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-3">
              <Minimize className="text-emerald-600" size={20} />
              <h3 className="font-semibold text-gray-800 dark:text-gray-200">최적화 기초</h3>
            </div>
            <ul className="space-y-2 text-gray-600 dark:text-gray-400">
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                선형/비선형 최적화 이해
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                볼록 최적화 이론
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                제약 조건 처리 (KKT)
              </li>
            </ul>
          </div>
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="text-teal-600" size={20} />
              <h3 className="font-semibold text-gray-800 dark:text-gray-200">최적화 알고리즘</h3>
            </div>
            <ul className="space-y-2 text-gray-600 dark:text-gray-400">
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                Gradient Descent 계열
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                유전 알고리즘 & PSO
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                메타휴리스틱 기법
              </li>
            </ul>
          </div>
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-3">
              <Maximize className="text-emerald-500" size={20} />
              <h3 className="font-semibold text-gray-800 dark:text-gray-200">실전 응용</h3>
            </div>
            <ul className="space-y-2 text-gray-600 dark:text-gray-400">
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                하이퍼파라미터 튜닝
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                다목적 최적화 (Pareto)
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                실무 최적화 문제 해결
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 챕터 목록 */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <BookOpen className="text-emerald-600" size={24} />
          챕터 목록
        </h2>
        <div className="grid gap-4">
          {moduleMetadata.chapters.map((chapter, index) => {
            const isCompleted = completedChapters.includes(chapter.id)
            const isLocked = index > 0 && !completedChapters.includes(moduleMetadata.chapters[index - 1].id)

            return (
              <Link
                key={chapter.id}
                href={isLocked ? '#' : `/modules/optimization-theory/${chapter.id}`}
                className={`block p-6 rounded-xl border transition-all duration-200 ${
                  isLocked
                    ? 'bg-gray-50 dark:bg-gray-800/50 border-gray-200 dark:border-gray-700 cursor-not-allowed opacity-60'
                    : isCompleted
                    ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-700 hover:shadow-md'
                    : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:shadow-md hover:border-emerald-300 dark:hover:border-emerald-600'
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
                          : 'bg-emerald-100 dark:bg-emerald-900 text-emerald-600 dark:text-emerald-400'
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
                    </div>
                  </div>
                  {!isLocked && (
                    <div className="text-emerald-600">
                      <Play size={20} />
                    </div>
                  )}
                </div>
              </Link>
            )
          })}
        </div>
      </section>

      {/* 시뮬레이터 미리보기 */}
      <section className="bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-emerald-900/10 dark:to-teal-900/10 rounded-2xl p-8 border border-emerald-200 dark:border-emerald-800">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          인터랙티브 시뮬레이터
        </h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          {moduleMetadata.simulators.length}개의 실습 도구로 최적화 이론을 직접 체험하세요
        </p>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {moduleMetadata.simulators.map((simulator) => (
            <Link
              key={simulator.id}
              href={`/modules/optimization-theory/simulators/${simulator.id}`}
              className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-lg hover:border-emerald-300 dark:hover:border-emerald-600 transition-all duration-200 hover:-translate-y-1 block"
            >
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                {simulator.title}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {simulator.description}
              </p>
            </Link>
          ))}
        </div>
      </section>
    </div>
  )
}
