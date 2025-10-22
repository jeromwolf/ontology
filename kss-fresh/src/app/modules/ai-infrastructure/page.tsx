'use client'

import { useState } from 'react'
import Link from 'next/link'
import { Play, Clock, Target, BookOpen, CheckCircle2, Cpu, Database, Zap } from 'lucide-react'
import { moduleMetadata } from './metadata'

export default function AIInfrastructureMainPage() {
  const [completedChapters, setCompletedChapters] = useState<string[]>([])

  const progress = (completedChapters.length / moduleMetadata.chapters.length) * 100

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="text-center py-16">
        <div className="w-20 h-20 mx-auto rounded-3xl bg-gradient-to-br from-slate-600 to-gray-700 flex items-center justify-center text-white text-4xl mb-6">
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
              className="bg-gradient-to-r from-slate-600 to-gray-700 h-3 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </div>

        <Link
          href={`/modules/ai-infrastructure/${moduleMetadata.chapters[0].id}`}
          className="inline-flex items-center gap-2 bg-gradient-to-r from-slate-600 to-gray-700 text-white px-8 py-4 rounded-xl font-semibold hover:shadow-lg transition-all duration-200 hover:-translate-y-1"
        >
          <Play size={20} />
          학습 시작하기
        </Link>
      </section>

      {/* 학습 목표 */}
      <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Target className="text-slate-600" size={24} />
          학습 목표
        </h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-3">
              <Cpu className="text-slate-600" size={20} />
              <h3 className="font-semibold text-gray-800 dark:text-gray-200">AI 인프라 설계</h3>
            </div>
            <ul className="space-y-2 text-gray-600 dark:text-gray-400">
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                GPU 클러스터 아키텍처 설계
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                컨테이너 오케스트레이션 (K8s)
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                분산 학습 환경 구축
              </li>
            </ul>
          </div>
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-3">
              <Database className="text-gray-600" size={20} />
              <h3 className="font-semibold text-gray-800 dark:text-gray-200">MLOps 파이프라인</h3>
            </div>
            <ul className="space-y-2 text-gray-600 dark:text-gray-400">
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                모델 서빙 최적화
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                Feature Store 설계
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                실험 추적 및 관리
              </li>
            </ul>
          </div>
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-3">
              <Zap className="text-slate-700" size={20} />
              <h3 className="font-semibold text-gray-800 dark:text-gray-200">프로덕션 운영</h3>
            </div>
            <ul className="space-y-2 text-gray-600 dark:text-gray-400">
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                모니터링 & Observability
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                ML CI/CD 파이프라인
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="text-green-500 mt-1" />
                비용 최적화 전략
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 챕터 목록 */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <BookOpen className="text-slate-600" size={24} />
          챕터 목록
        </h2>
        <div className="grid gap-4">
          {moduleMetadata.chapters.map((chapter, index) => {
            const isCompleted = completedChapters.includes(chapter.id)
            const isLocked = index > 0 && !completedChapters.includes(moduleMetadata.chapters[index - 1].id)

            return (
              <Link
                key={chapter.id}
                href={isLocked ? '#' : `/modules/ai-infrastructure/${chapter.id}`}
                className={`block p-6 rounded-xl border transition-all duration-200 ${
                  isLocked
                    ? 'bg-gray-50 dark:bg-gray-800/50 border-gray-200 dark:border-gray-700 cursor-not-allowed opacity-60'
                    : isCompleted
                    ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-700 hover:shadow-md'
                    : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:shadow-md hover:border-slate-300 dark:hover:border-slate-600'
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
                          : 'bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-400'
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
                    <div className="text-slate-600">
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
      <section className="bg-gradient-to-br from-slate-50 to-gray-50 dark:from-slate-900/10 dark:to-gray-900/10 rounded-2xl p-8 border border-slate-200 dark:border-slate-800">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          인터랙티브 시뮬레이터
        </h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          {moduleMetadata.simulators.length}개의 실습 도구로 AI Infrastructure & MLOps를 직접 체험하세요
        </p>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {moduleMetadata.simulators.map((simulator) => (
            <Link
              key={simulator.id}
              href={`/modules/ai-infrastructure/simulators/${simulator.id}`}
              className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-lg hover:border-slate-300 dark:hover:border-slate-600 transition-all duration-200 hover:-translate-y-1 block"
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
