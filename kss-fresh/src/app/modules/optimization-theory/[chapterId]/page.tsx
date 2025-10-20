'use client'

import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { ChevronLeft, ChevronRight, Clock, BookOpen, CheckCircle } from 'lucide-react'
import { moduleMetadata } from '../metadata'
import ChapterContent from '../components/ChapterContent'

export default function OptimizationTheoryChapterPage() {
  const params = useParams()
  const router = useRouter()
  const chapterId = params.chapterId as string

  const currentChapterIndex = moduleMetadata.chapters.findIndex(ch => ch.id === chapterId)
  const currentChapter = moduleMetadata.chapters[currentChapterIndex]
  const nextChapter = moduleMetadata.chapters[currentChapterIndex + 1]
  const prevChapter = moduleMetadata.chapters[currentChapterIndex - 1]

  if (!currentChapter) {
    router.push('/modules/optimization-theory')
    return null
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Chapter Header */}
      <div className="mb-8">
        <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-4">
          <Link href="/modules/optimization-theory" className="hover:text-emerald-600 dark:hover:text-emerald-400">
            Optimization Theory
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
            <span>{currentChapter.estimatedMinutes}분</span>
          </div>
        </div>
      </div>

      {/* Chapter Content */}
      <ChapterContent chapterId={chapterId} />

      {/* Navigation */}
      <div className="flex justify-between items-center mt-12 pt-8 border-t border-gray-200 dark:border-gray-700">
        {prevChapter ? (
          <Link
            href={`/modules/optimization-theory/${prevChapter.id}`}
            className="flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors"
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
            href={`/modules/optimization-theory/${nextChapter.id}`}
            className="flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors"
          >
            <div className="text-right">
              <div className="text-sm">다음 챕터</div>
              <div className="font-medium">{nextChapter.title}</div>
            </div>
            <ChevronRight size={20} />
          </Link>
        ) : (
          <Link
            href="/modules/optimization-theory"
            className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-500 transition-colors"
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
