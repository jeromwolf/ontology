'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ArrowLeft, ChevronLeft, ChevronRight, Clock, Target, CheckCircle } from 'lucide-react'
import { smartFactoryModule } from '../metadata'
import ChapterContent from '../components/ChapterContent'

interface PageProps {
  params: {
    chapterId: string
  }
}

export default function SmartFactoryChapterPage({ params }: PageProps) {
  const [progress, setProgress] = useState<Record<string, boolean>>({})
  const chapterId = params.chapterId
  const chapter = smartFactoryModule.chapters.find(c => c.id === chapterId)

  useEffect(() => {
    const saved = localStorage.getItem('smart-factory-progress')
    if (saved) {
      setProgress(JSON.parse(saved))
    }
  }, [])

  const markAsCompleted = () => {
    const newProgress = { ...progress, [chapterId]: true }
    setProgress(newProgress)
    localStorage.setItem('smart-factory-progress', JSON.stringify(newProgress))
  }

  const markAsIncomplete = () => {
    const newProgress = { ...progress, [chapterId]: false }
    setProgress(newProgress)
    localStorage.setItem('smart-factory-progress', JSON.stringify(newProgress))
  }

  if (!chapter) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">챕터를 찾을 수 없습니다</h1>
          <Link 
            href="/modules/smart-factory"
            className="text-slate-600 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300"
          >
            Smart Factory 모듈로 돌아가기
          </Link>
        </div>
      </div>
    )
  }

  const currentIndex = smartFactoryModule.chapters.findIndex(c => c.id === chapterId)
  const prevChapter = currentIndex > 0 ? smartFactoryModule.chapters[currentIndex - 1] : null
  const nextChapter = currentIndex < smartFactoryModule.chapters.length - 1 ? smartFactoryModule.chapters[currentIndex + 1] : null

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Compact Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14">
            <Link 
              href="/modules/smart-factory"
              className="flex items-center gap-2 text-slate-600 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 text-sm"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Smart Factory 모듈로 돌아가기</span>
            </Link>
            {progress[chapterId] ? (
              <button
                onClick={markAsIncomplete}
                className="flex items-center gap-2 px-3 py-1.5 bg-slate-600 text-white rounded-md hover:bg-slate-700 text-sm"
              >
                <CheckCircle className="w-4 h-4" />
                완료됨
              </button>
            ) : (
              <button
                onClick={markAsCompleted}
                className="flex items-center gap-2 px-3 py-1.5 bg-slate-600 text-white rounded-md hover:bg-slate-700 text-sm"
              >
                <Target className="w-4 h-4" />
                완료 표시
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Full Width Content */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          {/* Chapter Header */}
          <div className="mb-6 pb-6 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-slate-100 dark:bg-slate-700 rounded-lg flex items-center justify-center border border-slate-200 dark:border-slate-600">
                <span className="text-slate-600 dark:text-slate-400 font-bold">{currentIndex + 1}</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">{chapter.title}</h1>
                <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400 mt-1">
                  <div className="flex items-center gap-1">
                    <Clock className="w-4 h-4" />
                    <span>{chapter.estimatedMinutes}분</span>
                  </div>
                </div>
              </div>
            </div>
            <p className="text-gray-600 dark:text-gray-400">{chapter.description}</p>
          </div>
          
          {/* Chapter Content */}
          <ChapterContent chapterId={chapterId} />
        </div>

        {/* Compact Navigation */}
        <div className="flex items-center justify-between mt-6">
          {prevChapter ? (
            <Link
              href={`/modules/smart-factory/${prevChapter.id}`}
              className="flex items-center gap-2 px-4 py-2 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-md hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors text-sm"
            >
              <ChevronLeft className="w-4 h-4" />
              <span>이전: {prevChapter.title.length > 20 ? prevChapter.title.substring(0, 20) + '...' : prevChapter.title}</span>
            </Link>
          ) : (
            <div></div>
          )}

          {nextChapter ? (
            <Link
              href={`/modules/smart-factory/${nextChapter.id}`}
              className="flex items-center gap-2 px-4 py-2 bg-slate-600 text-white rounded-md hover:bg-slate-700 transition-colors text-sm"
            >
              <span>다음: {nextChapter.title.length > 20 ? nextChapter.title.substring(0, 20) + '...' : nextChapter.title}</span>
              <ChevronRight className="w-4 h-4" />
            </Link>
          ) : (
            <Link
              href="/modules/smart-factory"
              className="flex items-center gap-2 px-4 py-2 bg-slate-600 text-white rounded-md hover:bg-slate-700 transition-colors text-sm"
            >
              <span>모듈 완료</span>
              <CheckCircle className="w-4 h-4" />
            </Link>
          )}
        </div>
      </div>
    </div>
  )
}