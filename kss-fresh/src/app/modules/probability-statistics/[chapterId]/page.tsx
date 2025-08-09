'use client'

import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { ChevronLeft, ChevronRight, BookOpen } from 'lucide-react'
import { probabilityStatisticsModule } from '../metadata'
import Navigation from '@/components/Navigation'
import ChapterContent from '../components/ChapterContent'

export default function ProbabilityStatisticsChapterPage() {
  const params = useParams()
  const router = useRouter()
  const chapterId = params.chapterId as string
  
  const chapterIndex = probabilityStatisticsModule.chapters.findIndex(ch => ch.id === chapterId)
  const currentChapter = probabilityStatisticsModule.chapters[chapterIndex]
  const prevChapter = chapterIndex > 0 ? probabilityStatisticsModule.chapters[chapterIndex - 1] : null
  const nextChapter = chapterIndex < probabilityStatisticsModule.chapters.length - 1 
    ? probabilityStatisticsModule.chapters[chapterIndex + 1] : null

  if (!currentChapter) {
    router.push('/modules/probability-statistics')
    return null
  }

  return (
    <div className="min-h-screen">
      <Navigation />
      
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Breadcrumb */}
        <nav className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-8">
          <Link href="/" className="hover:text-purple-600 dark:hover:text-purple-400">
            홈
          </Link>
          <ChevronRight className="w-4 h-4" />
          <Link href="/modules/probability-statistics" className="hover:text-purple-600 dark:hover:text-purple-400">
            {probabilityStatisticsModule.name}
          </Link>
          <ChevronRight className="w-4 h-4" />
          <span className="text-gray-900 dark:text-white">{currentChapter.title}</span>
        </nav>

        {/* Chapter Header */}
        <div className="mb-12">
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 rounded-xl bg-gradient-to-br from-purple-100 to-pink-100 dark:from-purple-900/30 dark:to-pink-900/30">
              <BookOpen className="w-6 h-6 text-purple-600 dark:text-purple-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                Chapter {chapterIndex + 1}: {currentChapter.title}
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400 mt-2">
                {currentChapter.description}
              </p>
            </div>
          </div>
        </div>

        {/* Chapter Content */}
        <ChapterContent chapterId={chapterId} />

        {/* Navigation */}
        <div className="flex justify-between items-center mt-12 pt-8 border-t border-gray-200 dark:border-gray-700">
          {prevChapter ? (
            <Link
              href={`/modules/probability-statistics/${prevChapter.id}`}
              className="flex items-center gap-2 text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300"
            >
              <ChevronLeft className="w-5 h-5" />
              <span>{prevChapter.title}</span>
            </Link>
          ) : (
            <div />
          )}
          
          {nextChapter ? (
            <Link
              href={`/modules/probability-statistics/${nextChapter.id}`}
              className="flex items-center gap-2 text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300"
            >
              <span>{nextChapter.title}</span>
              <ChevronRight className="w-5 h-5" />
            </Link>
          ) : (
            <Link
              href="/modules/probability-statistics"
              className="flex items-center gap-2 text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300"
            >
              <span>모듈 홈으로</span>
              <ChevronRight className="w-5 h-5" />
            </Link>
          )}
        </div>
      </main>
    </div>
  )
}