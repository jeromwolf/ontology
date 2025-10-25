'use client'

import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { ChevronLeft, ChevronRight, Clock, BookOpen, CheckCircle } from 'lucide-react'
import { vibeCodingMetadata } from '../metadata'
import ChapterContent from '../components/ChapterContent'

export default function VibeCodingChapterPage() {
  const params = useParams()
  const router = useRouter()
  const chapterId = params.chapterId as string

  const currentChapterIndex = vibeCodingMetadata.chapters.findIndex(ch => ch.id === chapterId)
  const currentChapter = vibeCodingMetadata.chapters[currentChapterIndex]
  const nextChapter = vibeCodingMetadata.chapters[currentChapterIndex + 1]
  const prevChapter = vibeCodingMetadata.chapters[currentChapterIndex - 1]

  if (!currentChapter) {
    router.push('/modules/vibe-coding')
    return null
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Chapter Header */}
      <div className="mb-8">
        <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-4">
          <Link href="/modules/vibe-coding" className="hover:text-purple-600 dark:hover:text-purple-400">
            Vibe Coding 모듈
          </Link>
          <span>/</span>
          <span>Chapter {currentChapterIndex + 1}</span>
        </div>

        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          {currentChapter.title}
        </h1>
        <p className="text-lg text-gray-600 dark:text-gray-400 mb-4">
          {currentChapter.description}
        </p>

        <div className="flex items-center gap-6 text-sm text-gray-500 dark:text-gray-400">
          <div className="flex items-center gap-1">
            <Clock size={16} />
            <span>{currentChapter.duration}</span>
          </div>
          <div className="flex items-center gap-1">
            <BookOpen size={16} />
            <span>{currentChapter.topics.length}개 핵심 개념</span>
          </div>
          <div className="px-3 py-1 rounded-full text-xs font-medium bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300">
            {currentChapter.level === 'beginner' ? '초급' : currentChapter.level === 'intermediate' ? '중급' : '고급'}
          </div>
        </div>
      </div>

      {/* Learning Objectives */}
      <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6 mb-8">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <CheckCircle className="text-purple-600 dark:text-purple-400" size={20} />
          학습 목표
        </h2>
        <ul className="space-y-2">
          {currentChapter.topics.map((topic, index) => (
            <li key={index} className="flex items-start gap-2">
              <span className="text-purple-600 dark:text-purple-400 mt-1">•</span>
              <span className="text-gray-700 dark:text-gray-300">{topic}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Chapter Content */}
      <ChapterContent chapterId={chapterId} />

      {/* Navigation */}
      <div className="flex justify-between items-center mt-12 pt-8 border-t border-gray-200 dark:border-gray-700">
        {prevChapter ? (
          <Link
            href={`/modules/vibe-coding/${prevChapter.id}`}
            className="flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-purple-600 dark:hover:text-purple-400 transition-colors"
          >
            <ChevronLeft size={20} />
            <div className="text-left">
              <div className="text-sm">이전 챕터</div>
              <div className="font-medium">{prevChapter.title}</div>
            </div>
          </Link>
        ) : (
          <div />
        )}

        {nextChapter ? (
          <Link
            href={`/modules/vibe-coding/${nextChapter.id}`}
            className="flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-purple-600 dark:hover:text-purple-400 transition-colors"
          >
            <div className="text-right">
              <div className="text-sm">다음 챕터</div>
              <div className="font-medium">{nextChapter.title}</div>
            </div>
            <ChevronRight size={20} />
          </Link>
        ) : (
          <Link
            href="/modules/vibe-coding"
            className="flex items-center gap-2 text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-500 transition-colors"
          >
            <div className="text-right">
              <div className="text-sm">학습 완료!</div>
              <div className="font-medium">모듈 홈으로</div>
            </div>
            <CheckCircle size={20} />
          </Link>
        )}
      </div>
    </div>
  )
}
