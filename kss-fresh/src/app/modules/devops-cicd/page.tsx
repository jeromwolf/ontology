'use client'

import { useState } from 'react'
import { Clock, Users, BookOpen, Play, CheckCircle, ArrowRight, ChevronRight, Star, Target, Lightbulb } from 'lucide-react'
import { devopsMetadata } from './metadata'
import Link from 'next/link'

export default function DevOpsPage() {
  const [completedChapters, setCompletedChapters] = useState<string[]>([])
  
  const stats = [
    { label: '총 학습 시간', value: devopsMetadata.duration },
    { label: '챕터', value: `${devopsMetadata.chapters.length}개` },
    { label: '시뮬레이터', value: `${devopsMetadata.simulators.length}개` },
    { label: '난이도', value: '중급' },
  ]

  const toggleChapterComplete = (chapterId: string) => {
    setCompletedChapters(prev => 
      prev.includes(chapterId)
        ? prev.filter(id => id !== chapterId)
        : [...prev, chapterId]
    )
  }

  const progress = (completedChapters.length / devopsMetadata.chapters.length) * 100

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className={`bg-gradient-to-r ${devopsMetadata.moduleColor} text-white`}>
        <div className="container mx-auto px-6 py-12">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-12 h-12 bg-white/20 rounded-xl flex items-center justify-center">
                  <div className="w-6 h-6 bg-white rounded-sm"></div>
                </div>
                <div>
                  <div className="text-white/80 text-sm font-medium">{devopsMetadata.category}</div>
                  <h1 className="text-3xl font-bold">{devopsMetadata.title}</h1>
                </div>
              </div>
              <p className="text-xl text-white/90 mb-8 max-w-2xl leading-relaxed">
                {devopsMetadata.description}
              </p>
            </div>
            
            {/* Stats */}
            <div className="hidden lg:block">
              <div className="grid grid-cols-2 gap-6">
                {stats.map((stat, index) => (
                  <div key={index} className="text-center">
                    <div className="text-2xl font-bold text-white">{stat.value}</div>
                    <div className="text-white/70 text-sm">{stat.label}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          {/* Mobile Stats */}
          <div className="lg:hidden grid grid-cols-4 gap-4 mt-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="text-lg font-bold text-white">{stat.value}</div>
                <div className="text-white/70 text-xs">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-12">
            {/* Progress Section */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">학습 진행률</h2>
                <span className="text-2xl font-bold text-gray-600 dark:text-gray-300">
                  {Math.round(progress)}%
                </span>
              </div>
              
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4 mb-4">
                <div 
                  className={`bg-gradient-to-r ${devopsMetadata.moduleColor} h-4 rounded-full transition-all duration-500 ease-out`}
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
              
              <div className="text-gray-600 dark:text-gray-400">
                {completedChapters.length}개 챕터 완료 / 총 {devopsMetadata.chapters.length}개 챕터
              </div>
            </div>

            {/* Prerequisites */}
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-2xl p-8 border border-blue-200 dark:border-blue-800">
              <div className="flex items-center gap-3 mb-4">
                <Target className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">선수 지식</h2>
              </div>
              <div className="grid gap-3">
                {devopsMetadata.prerequisites.map((req, index) => (
                  <div key={index} className="flex items-center gap-2 text-gray-700 dark:text-gray-300">
                    <CheckCircle className="w-4 h-4 text-blue-500" />
                    <span>{req}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Chapters */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 border border-gray-200 dark:border-gray-700">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-8">챕터</h2>
              
              <div className="space-y-4">
                {devopsMetadata.chapters.map((chapter, index) => {
                  const isCompleted = completedChapters.includes(chapter.id)
                  const isLocked = index > 0 && !completedChapters.includes(devopsMetadata.chapters[index - 1].id)
                  
                  return (
                    <div
                      key={chapter.id}
                      className={`border rounded-xl p-6 transition-all ${
                        isCompleted
                          ? 'border-green-300 dark:border-green-700 bg-green-50 dark:bg-green-900/20'
                          : isLocked
                          ? 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 opacity-60'
                          : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 hover:border-gray-300 dark:hover:border-gray-600'
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-3">
                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold ${
                              isCompleted
                                ? 'bg-green-500 text-white'
                                : isLocked
                                ? 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400'
                                : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
                            }`}>
                              {index + 1}
                            </div>
                            <div>
                              <h3 className={`text-lg font-semibold ${
                                isLocked ? 'text-gray-400 dark:text-gray-600' : 'text-gray-900 dark:text-white'
                              }`}>
                                {chapter.title}
                              </h3>
                              <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                                <div className="flex items-center gap-1">
                                  <Clock className="w-4 h-4" />
                                  <span>{chapter.duration}</span>
                                </div>
                              </div>
                            </div>
                          </div>
                          
                          <p className={`mb-4 leading-relaxed ${
                            isLocked ? 'text-gray-400 dark:text-gray-600' : 'text-gray-600 dark:text-gray-300'
                          }`}>
                            {chapter.description}
                          </p>

                          {/* Learning Objectives */}
                          <div className="space-y-2">
                            <div className="flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300">
                              <Lightbulb className="w-4 h-4" />
                              <span>학습 목표</span>
                            </div>
                            <div className="grid gap-2 ml-6">
                              {chapter.learningObjectives.map((objective, objIndex) => (
                                <div key={objIndex} className={`flex items-start gap-2 text-sm ${
                                  isLocked ? 'text-gray-400 dark:text-gray-600' : 'text-gray-600 dark:text-gray-400'
                                }`}>
                                  <div className="w-1.5 h-1.5 bg-gray-400 rounded-full mt-2 flex-shrink-0"></div>
                                  <span>{objective}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>

                        <div className="flex flex-col gap-2 ml-4">
                          {!isLocked && (
                            <>
                              <Link
                                href={`/modules/devops-cicd/${chapter.id}`}
                                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                                  isCompleted
                                    ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 hover:bg-green-200 dark:hover:bg-green-900/50'
                                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                                }`}
                              >
                                {isCompleted ? '복습하기' : '학습하기'}
                              </Link>
                              <button
                                onClick={() => toggleChapterComplete(chapter.id)}
                                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                                  isCompleted
                                    ? 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600'
                                    : 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 hover:bg-blue-200 dark:hover:bg-blue-900/50'
                                }`}
                              >
                                {isCompleted ? '미완료로 표시' : '완료로 표시'}
                              </button>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-8">
            {/* Learning Outcomes */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">학습 성과</h3>
              <div className="space-y-3">
                {devopsMetadata.outcomes.map((outcome, index) => (
                  <div key={index} className="flex items-start gap-3">
                    <Star className="w-5 h-5 text-yellow-500 mt-0.5 flex-shrink-0" />
                    <span className="text-gray-600 dark:text-gray-300 text-sm leading-relaxed">{outcome}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Tools & Technologies */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">사용 도구</h3>
              <div className="grid grid-cols-2 gap-3">
                {devopsMetadata.tools.map((tool, index) => (
                  <div key={index} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 text-center">
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{tool}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Simulators */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">인터랙티브 시뮬레이터</h3>
              <div className="space-y-4">
                {devopsMetadata.simulators.map((simulator, index) => (
                  <div key={simulator.id} className="border border-gray-200 dark:border-gray-600 rounded-lg p-4 hover:border-gray-300 dark:hover:border-gray-500 transition-colors cursor-pointer">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-gray-900 dark:text-white">{simulator.title}</h4>
                      <div className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400">
                        <Clock className="w-3 h-3" />
                        <span>{simulator.estimatedTime}</span>
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">{simulator.description}</p>
                    <div className="flex items-center justify-between">
                      <span className={`px-2 py-1 text-xs font-medium rounded ${
                        simulator.difficulty === 'advanced' 
                          ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
                          : 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400'
                      }`}>
                        {simulator.difficulty === 'advanced' ? '고급' : '중급'}
                      </span>
                      <div className="flex items-center gap-1 text-blue-600 dark:text-blue-400 text-sm font-medium">
                        <Play className="w-4 h-4" />
                        <span>실습하기</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-2xl p-6 border border-blue-200 dark:border-blue-800">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">바로가기</h3>
              <div className="space-y-3">
                <Link
                  href="/modules/devops-cicd/devops-culture"
                  className="flex items-center justify-between p-3 bg-white dark:bg-gray-800 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                >
                  <span className="font-medium text-gray-900 dark:text-white">첫 챕터 시작하기</span>
                  <ArrowRight className="w-4 h-4 text-gray-400" />
                </Link>
                <button className="w-full flex items-center justify-between p-3 bg-white dark:bg-gray-800 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
                  <span className="font-medium text-gray-900 dark:text-white">진행 상황 초기화</span>
                  <ChevronRight className="w-4 h-4 text-gray-400" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}