'use client'

import { useState } from 'react'
import Link from 'next/link'
import { Play, Clock, Target, BookOpen, CheckCircle2, Database, Layers, Zap, Users, Star, TrendingUp } from 'lucide-react'
import { moduleMetadata } from './metadata'

export default function DataEngineeringMainPage() {
  const [completedChapters, setCompletedChapters] = useState<string[]>([])

  const progress = (completedChapters.length / moduleMetadata.chapters.length) * 100

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="text-center py-16">
        <div className="w-20 h-20 mx-auto rounded-3xl bg-gradient-to-br from-indigo-600 to-blue-700 flex items-center justify-center text-white text-4xl mb-6">
          {moduleMetadata.icon}
        </div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          {moduleMetadata.title}
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-2xl mx-auto">
          {moduleMetadata.description}
        </p>

        {/* Stats */}
        <div className="flex justify-center gap-8 mb-8 text-sm text-gray-600 dark:text-gray-400">
          <div className="flex items-center gap-2">
            <Users size={18} />
            <span>{moduleMetadata.students.toLocaleString()}명 수강</span>
          </div>
          <div className="flex items-center gap-2">
            <Star size={18} className="text-yellow-500" />
            <span>{moduleMetadata.rating} / 5.0</span>
          </div>
          <div className="flex items-center gap-2">
            <Clock size={18} />
            <span>{moduleMetadata.estimatedHours}시간</span>
          </div>
        </div>

        {/* Progress */}
        <div className="max-w-md mx-auto mb-8">
          <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
            <span>학습 진도</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
            <div
              className="bg-gradient-to-r from-indigo-600 to-blue-700 h-3 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </div>

        <Link
          href={`/modules/data-engineering/${moduleMetadata.chapters[0].id}`}
          className="inline-flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-blue-700 text-white px-8 py-4 rounded-xl font-semibold hover:shadow-lg transition-all duration-200 hover:-translate-y-1"
        >
          <Play size={20} />
          학습 시작하기
        </Link>
      </section>

      {/* Prerequisites & Skills */}
      <section className="grid md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <CheckCircle2 className="text-indigo-600" size={20} />
            사전 요구사항
          </h3>
          <ul className="space-y-2">
            {moduleMetadata.prerequisites.map((prereq, index) => (
              <li key={index} className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                <div className="w-1.5 h-1.5 rounded-full bg-indigo-500"></div>
                {prereq}
              </li>
            ))}
          </ul>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <TrendingUp className="text-blue-600" size={20} />
            학습할 핵심 기술
          </h3>
          <div className="flex flex-wrap gap-2">
            {moduleMetadata.skills.slice(0, 6).map((skill, index) => (
              <span
                key={index}
                className="px-3 py-1 bg-indigo-50 dark:bg-indigo-900/20 text-indigo-700 dark:text-indigo-300 rounded-full text-sm"
              >
                {skill}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* Learning Path */}
      <section className="bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-indigo-900/10 dark:to-blue-900/10 rounded-2xl p-8 border border-indigo-200 dark:border-indigo-800">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Target className="text-indigo-600" size={24} />
          학습 로드맵
        </h2>
        <div className="grid md:grid-cols-4 gap-4">
          {moduleMetadata.learningPath.map((path, index) => (
            <div key={index} className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-600 to-blue-700 text-white flex items-center justify-center text-sm font-bold">
                  {index + 1}
                </div>
                <h3 className="font-bold text-gray-900 dark:text-white">{path.stage}</h3>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">{path.description}</p>
              <p className="text-xs text-gray-500 dark:text-gray-500">{path.chapters.length}개 챕터</p>
            </div>
          ))}
        </div>
      </section>

      {/* Chapters */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <BookOpen className="text-indigo-600" size={24} />
          전체 챕터 ({moduleMetadata.chapters.length}개)
        </h2>
        <div className="grid gap-4">
          {moduleMetadata.chapters.map((chapter, index) => {
            const isCompleted = completedChapters.includes(chapter.id)
            const isLocked = index > 0 && !completedChapters.includes(moduleMetadata.chapters[index - 1].id)

            return (
              <Link
                key={chapter.id}
                href={isLocked ? '#' : `/modules/data-engineering/${chapter.id}`}
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
                    <p className="text-gray-600 dark:text-gray-400 mb-3 ml-11">
                      {chapter.description}
                    </p>
                    <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400 ml-11">
                      <div className="flex items-center gap-1">
                        <Clock size={14} />
                        <span>{chapter.estimatedMinutes}분</span>
                      </div>
                    </div>
                  </div>
                  {!isLocked && (
                    <div className="text-indigo-600">
                      <Play size={20} />
                    </div>
                  )}
                </div>
              </Link>
            )
          })}
        </div>
      </section>

      {/* Simulators */}
      <section className="bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-indigo-900/10 dark:to-blue-900/10 rounded-2xl p-8 border border-indigo-200 dark:border-indigo-800">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-3">
          <Zap className="text-indigo-600" size={24} />
          인터랙티브 시뮬레이터 ({moduleMetadata.simulators.length}개)
        </h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          실무 도구를 직접 체험하면서 데이터 엔지니어링 기술을 마스터하세요
        </p>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {moduleMetadata.simulators.map((simulator) => (
            <Link
              key={simulator.id}
              href={`/modules/data-engineering/simulators/${simulator.id}`}
              className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-lg hover:border-indigo-300 dark:hover:border-indigo-600 transition-all duration-200 hover:-translate-y-1 block"
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

      {/* Tools */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Database className="text-indigo-600" size={24} />
          실전 도구 ({moduleMetadata.tools.length}개)
        </h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {moduleMetadata.tools.map((tool) => (
            <div
              key={tool.id}
              className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 hover:shadow-md transition-all"
            >
              <div className="text-3xl mb-3">{tool.icon}</div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                {tool.title}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {tool.description}
              </p>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
