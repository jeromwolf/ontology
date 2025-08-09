'use client'

import { useState, useEffect } from 'react'
import { ArrowLeft, ArrowRight, CheckCircle, Clock, BookOpen, Target } from 'lucide-react'
import { devopsMetadata } from '../metadata'
import ChapterContent from '../ChapterContent'
import Link from 'next/link'

interface PageProps {
  params: {
    chapterId: string
  }
}

export default function ChapterPage({ params }: PageProps) {
  const { chapterId } = params
  const [isCompleted, setIsCompleted] = useState(false)
  
  // Find current chapter
  const currentChapter = devopsMetadata.chapters.find(ch => ch.id === chapterId)
  const currentIndex = devopsMetadata.chapters.findIndex(ch => ch.id === chapterId)
  
  // Navigation
  const prevChapter = currentIndex > 0 ? devopsMetadata.chapters[currentIndex - 1] : null
  const nextChapter = currentIndex < devopsMetadata.chapters.length - 1 ? devopsMetadata.chapters[currentIndex + 1] : null
  
  // Load completion status
  useEffect(() => {
    const completed = localStorage.getItem(`devops-${chapterId}-completed`) === 'true'
    setIsCompleted(completed)
  }, [chapterId])
  
  // Toggle completion
  const toggleCompleted = () => {
    const newStatus = !isCompleted
    setIsCompleted(newStatus)
    localStorage.setItem(`devops-${chapterId}-completed`, newStatus.toString())
  }

  if (!currentChapter) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            챕터를 찾을 수 없습니다
          </h1>
          <Link
            href="/modules/devops-cicd"
            className="text-blue-600 dark:text-blue-400 hover:underline"
          >
            모듈 홈으로 돌아가기
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className={`bg-gradient-to-r ${devopsMetadata.moduleColor} text-white`}>
        <div className="container mx-auto px-6 py-8">
          <div className="flex items-center justify-between mb-6">
            <Link
              href="/modules/devops-cicd"
              className="flex items-center gap-2 text-white/80 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>DevOps & CI/CD 홈으로</span>
            </Link>
            
            <div className="flex items-center gap-4">
              <span className="text-white/80 text-sm">
                {currentIndex + 1} / {devopsMetadata.chapters.length}
              </span>
              <button
                onClick={toggleCompleted}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isCompleted
                    ? 'bg-green-500 hover:bg-green-600 text-white'
                    : 'bg-white/20 hover:bg-white/30 text-white'
                }`}
              >
                <CheckCircle className="w-4 h-4" />
                {isCompleted ? '완료됨' : '완료로 표시'}
              </button>
            </div>
          </div>
          
          <div className="flex items-center gap-4 mb-4">
            <div className="w-12 h-12 bg-white/20 rounded-xl flex items-center justify-center text-xl font-bold">
              {currentIndex + 1}
            </div>
            <div>
              <h1 className="text-3xl font-bold mb-2">{currentChapter.title}</h1>
              <div className="flex items-center gap-6 text-white/80">
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4" />
                  <span>{currentChapter.duration}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Target className="w-4 h-4" />
                  <span>{currentChapter.learningObjectives.length}개 학습 목표</span>
                </div>
              </div>
            </div>
          </div>
          
          <p className="text-xl text-white/90 leading-relaxed">
            {currentChapter.description}
          </p>
        </div>
      </div>

      <div className="container mx-auto px-6 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-12">
          {/* Main Content */}
          <div className="lg:col-span-3">
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700">
              <div className="p-8">
                <ChapterContent chapterId={chapterId} />
              </div>
            </div>

            {/* Navigation */}
            <div className="flex items-center justify-between mt-12">
              {prevChapter ? (
                <Link
                  href={`/modules/devops-cicd/${prevChapter.id}`}
                  className="flex items-center gap-3 px-6 py-4 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 transition-colors group"
                >
                  <ArrowLeft className="w-5 h-5 text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300" />
                  <div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">이전 챕터</div>
                    <div className="font-medium text-gray-900 dark:text-white">{prevChapter.title}</div>
                  </div>
                </Link>
              ) : (
                <div></div>
              )}

              {nextChapter ? (
                <Link
                  href={`/modules/devops-cicd/${nextChapter.id}`}
                  className="flex items-center gap-3 px-6 py-4 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 transition-colors group"
                >
                  <div className="text-right">
                    <div className="text-sm text-gray-500 dark:text-gray-400">다음 챕터</div>
                    <div className="font-medium text-gray-900 dark:text-white">{nextChapter.title}</div>
                  </div>
                  <ArrowRight className="w-5 h-5 text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300" />
                </Link>
              ) : (
                <Link
                  href="/modules/devops-cicd"
                  className="flex items-center gap-3 px-6 py-4 bg-green-500 hover:bg-green-600 text-white rounded-xl transition-colors"
                >
                  <CheckCircle className="w-5 h-5" />
                  <span className="font-medium">모듈 완료</span>
                </Link>
              )}
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-8">
            {/* Learning Objectives */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <Target className="w-5 h-5 text-blue-500" />
                학습 목표
              </h3>
              <div className="space-y-3">
                {currentChapter.learningObjectives.map((objective, index) => (
                  <div key={index} className="flex items-start gap-3">
                    <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                    <span className="text-gray-600 dark:text-gray-300 text-sm leading-relaxed">
                      {objective}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Chapter Progress */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">진행 상황</h3>
              <div className="space-y-3">
                {devopsMetadata.chapters.map((chapter, index) => {
                  const completed = localStorage.getItem(`devops-${chapter.id}-completed`) === 'true'
                  const current = chapter.id === chapterId
                  
                  return (
                    <Link
                      key={chapter.id}
                      href={`/modules/devops-cicd/${chapter.id}`}
                      className={`flex items-center gap-3 p-3 rounded-lg transition-colors ${
                        current
                          ? 'bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800'
                          : completed
                          ? 'bg-green-50 dark:bg-green-900/20 hover:bg-green-100 dark:hover:bg-green-900/30'
                          : 'hover:bg-gray-50 dark:hover:bg-gray-700'
                      }`}
                    >
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                        current
                          ? 'bg-blue-500 text-white'
                          : completed
                          ? 'bg-green-500 text-white'
                          : 'bg-gray-200 dark:bg-gray-600 text-gray-600 dark:text-gray-300'
                      }`}>
                        {completed ? <CheckCircle className="w-4 h-4" /> : index + 1}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className={`font-medium text-sm truncate ${
                          current
                            ? 'text-blue-900 dark:text-blue-300'
                            : completed
                            ? 'text-green-900 dark:text-green-300'
                            : 'text-gray-900 dark:text-white'
                        }`}>
                          {chapter.title}
                        </div>
                        <div className={`text-xs ${
                          current
                            ? 'text-blue-600 dark:text-blue-400'
                            : completed
                            ? 'text-green-600 dark:text-green-400'
                            : 'text-gray-500 dark:text-gray-400'
                        }`}>
                          {chapter.duration}
                        </div>
                      </div>
                    </Link>
                  )
                })}
              </div>
            </div>

            {/* Module Info */}
            <div className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-700 rounded-2xl p-6 border border-gray-200 dark:border-gray-600">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-8 h-8 bg-gray-500 rounded-lg flex items-center justify-center">
                  <BookOpen className="w-4 h-4 text-white" />
                </div>
                <div>
                  <div className="font-medium text-gray-900 dark:text-white">{devopsMetadata.title}</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">{devopsMetadata.category}</div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4 text-center">
                <div>
                  <div className="text-xl font-bold text-gray-900 dark:text-white">{devopsMetadata.chapters.length}</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">챕터</div>
                </div>
                <div>
                  <div className="text-xl font-bold text-gray-900 dark:text-white">{devopsMetadata.simulators.length}</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">시뮬레이터</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}