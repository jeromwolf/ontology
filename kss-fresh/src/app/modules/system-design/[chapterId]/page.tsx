'use client'

import { useParams } from 'next/navigation'
import Link from 'next/link'
import { ArrowLeft, Clock, BookOpen, Target, ChevronRight } from 'lucide-react'
import { metadata } from '../metadata'
import ChapterContent from '../components/ChapterContent'

export default function SystemDesignChapter() {
  const params = useParams()
  const chapterId = params.chapterId as string
  
  const chapter = metadata.chapters.find(ch => ch.id === chapterId)
  const chapterIndex = metadata.chapters.findIndex(ch => ch.id === chapterId)
  const nextChapter = metadata.chapters[chapterIndex + 1]
  const prevChapter = metadata.chapters[chapterIndex - 1]
  
  if (!chapter) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            챕터를 찾을 수 없습니다
          </h1>
          <Link
            href="/modules/system-design"
            className="text-purple-600 dark:text-purple-400 hover:underline"
          >
            모듈 홈으로 돌아가기
          </Link>
        </div>
      </div>
    )
  }
  
  return (
    <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-7xl">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/system-design"
          className="inline-flex items-center text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 mb-4"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          System Design 모듈로 돌아가기
        </Link>
        
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
          <div className="flex items-center gap-4 mb-4">
            <span className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-full text-sm font-semibold">
              Chapter {chapter.number}
            </span>
            <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
              <span className="flex items-center gap-1">
                <Clock className="w-4 h-4" />
                {chapter.duration}
              </span>
            </div>
          </div>
          
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            {chapter.title}
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300 mb-6">
            {chapter.description}
          </p>
          
          <div className="bg-purple-50 dark:bg-purple-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
              <Target className="w-5 h-5 text-purple-600 dark:text-purple-400" />
              학습 목표
            </h3>
            <ul className="space-y-2">
              {chapter.objectives.map((objective, idx) => (
                <li key={idx} className="flex items-start gap-2 text-gray-700 dark:text-gray-300">
                  <div className="w-1.5 h-1.5 bg-purple-500 rounded-full mt-2 flex-shrink-0" />
                  {objective}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
      
      {/* Content */}
      <ChapterContent chapterId={chapterId} />
      
      {/* Navigation */}
      <div className="mt-12 flex justify-between">
        {prevChapter ? (
          <Link
            href={`/modules/system-design/${prevChapter.id}`}
            className="flex items-center gap-2 px-6 py-3 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
          >
            <ChevronRight className="w-5 h-5 rotate-180" />
            <div className="text-left">
              <div className="text-sm text-gray-600 dark:text-gray-400">이전 챕터</div>
              <div className="font-semibold text-gray-900 dark:text-white">
                {prevChapter.title}
              </div>
            </div>
          </Link>
        ) : (
          <div />
        )}
        
        {nextChapter ? (
          <Link
            href={`/modules/system-design/${nextChapter.id}`}
            className="flex items-center gap-2 px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
          >
            <div className="text-right">
              <div className="text-sm text-purple-200">다음 챕터</div>
              <div className="font-semibold">
                {nextChapter.title}
              </div>
            </div>
            <ChevronRight className="w-5 h-5" />
          </Link>
        ) : (
          <Link
            href="/modules/system-design"
            className="flex items-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
          >
            <div className="text-right">
              <div className="text-sm text-green-200">모듈 완료</div>
              <div className="font-semibold">
                System Design 홈으로
              </div>
            </div>
            <ChevronRight className="w-5 h-5" />
          </Link>
        )}
      </div>
    </div>
  )
}