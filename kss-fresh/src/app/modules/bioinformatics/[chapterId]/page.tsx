'use client'

import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { bioinformaticsMetadata } from '../metadata'
import ChapterContent from '../components/ChapterContent'
import { ChevronLeft, ChevronRight, Home, BookOpen } from 'lucide-react'

export default function BioinformaticsChapterPage() {
  const params = useParams()
  const router = useRouter()
  const chapterId = params.chapterId as string

  // Redirect simulators to proper route
  if (chapterId === 'simulators') {
    router.push('/modules/bioinformatics')
    return null
  }

  const currentChapterIndex = bioinformaticsMetadata.chapters.findIndex(
    ch => ch.id === chapterId
  )
  const currentChapter = bioinformaticsMetadata.chapters[currentChapterIndex]
  const prevChapter = bioinformaticsMetadata.chapters[currentChapterIndex - 1]
  const nextChapter = bioinformaticsMetadata.chapters[currentChapterIndex + 1]

  if (!currentChapter) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-600 dark:text-gray-400">챕터를 찾을 수 없습니다.</p>
        <Link href="/modules/bioinformatics" className="text-emerald-600 hover:underline mt-4 inline-block">
          모듈 홈으로 돌아가기
        </Link>
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto">
      {/* Navigation Bar */}
      <nav className="bg-white dark:bg-gray-800 rounded-xl p-4 mb-6 shadow-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/modules/bioinformatics"
              className="p-2 hover:bg-emerald-100 dark:hover:bg-emerald-900/50 rounded-lg transition-colors"
            >
              <Home className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
            </Link>
            <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
              <BookOpen className="w-4 h-4" />
              <span>Chapter {currentChapterIndex + 1} / {bioinformaticsMetadata.chapters.length}</span>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {prevChapter && (
              <button
                onClick={() => router.push(`/modules/bioinformatics/${prevChapter.id}`)}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              >
                <ChevronLeft className="w-5 h-5" />
              </button>
            )}
            {nextChapter && (
              <button
                onClick={() => router.push(`/modules/bioinformatics/${nextChapter.id}`)}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              >
                <ChevronRight className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>
      </nav>

      {/* Chapter Header */}
      <header className="bg-gradient-to-r from-emerald-600 to-teal-600 rounded-2xl p-8 mb-8 text-white shadow-xl">
        <h1 className="text-3xl font-bold mb-3">{currentChapter.title}</h1>
        <p className="text-emerald-100 mb-6">{currentChapter.description}</p>
        
        <div className="flex flex-wrap gap-3">
          <span className="px-3 py-1 bg-white/20 rounded-full text-sm">
            {currentChapter.duration}
          </span>
          <span className="px-3 py-1 bg-white/20 rounded-full text-sm">
            {currentChapter.objectives.length}개 학습 목표
          </span>
        </div>
      </header>

      {/* Learning Objectives */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 mb-8 shadow-lg">
        <h2 className="text-xl font-bold text-gray-800 dark:text-gray-200 mb-4">
          🎯 학습 목표
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {currentChapter.objectives.map((objective, idx) => (
            <div key={idx} className="flex items-start gap-3">
              <div className="mt-1 w-6 h-6 bg-emerald-100 dark:bg-emerald-900/50 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-xs font-bold text-emerald-600 dark:text-emerald-400">
                  {idx + 1}
                </span>
              </div>
              <p className="text-gray-700 dark:text-gray-300">{objective}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Chapter Content */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <ChapterContent chapterId={chapterId} />
      </section>

      {/* Chapter Navigation */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4">
        {prevChapter && (
          <Link
            href={`/modules/bioinformatics/${prevChapter.id}`}
            className="group bg-white dark:bg-gray-800 rounded-xl p-4 shadow-lg hover:shadow-xl transition-all border-2 border-transparent hover:border-emerald-500"
          >
            <div className="flex items-center gap-3">
              <ChevronLeft className="w-5 h-5 text-gray-400 group-hover:text-emerald-500 transition-colors" />
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">이전 챕터</p>
                <p className="font-semibold text-gray-800 dark:text-gray-200">
                  {prevChapter.title}
                </p>
              </div>
            </div>
          </Link>
        )}
        
        {nextChapter && (
          <Link
            href={`/modules/bioinformatics/${nextChapter.id}`}
            className="group bg-white dark:bg-gray-800 rounded-xl p-4 shadow-lg hover:shadow-xl transition-all border-2 border-transparent hover:border-emerald-500 md:ml-auto"
          >
            <div className="flex items-center gap-3 justify-end">
              <div className="text-right">
                <p className="text-xs text-gray-500 dark:text-gray-400">다음 챕터</p>
                <p className="font-semibold text-gray-800 dark:text-gray-200">
                  {nextChapter.title}
                </p>
              </div>
              <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-emerald-500 transition-colors" />
            </div>
          </Link>
        )}
      </div>
    </div>
  )
}