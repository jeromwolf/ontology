'use client'

import React from 'react'
import Link from 'next/link'
import { ChevronLeft, ChevronRight } from 'lucide-react'

interface ChapterNavigationProps {
  currentChapter: number
  totalChapters: number
  moduleSlug: string
}

const chapterTitles: { [key: number]: string } = {
  1: '로봇 공학 기초',
  2: '순기구학 (Forward Kinematics)',
  3: '역기구학 (Inverse Kinematics)',
  4: '경로 계획 (Path Planning)',
  5: '궤적 생성 (Trajectory Generation)',
  6: '그리핑과 조작 (Grasping & Manipulation)',
  7: 'ROS2 프로그래밍',
  8: '협동 로봇 (Collaborative Robots)'
}

export default function ChapterNavigation({
  currentChapter,
  totalChapters,
  moduleSlug
}: ChapterNavigationProps) {
  const hasPrevious = currentChapter > 1
  const hasNext = currentChapter < totalChapters

  const getPreviousChapterSlug = () => {
    if (!hasPrevious) return null
    return `chapter-${currentChapter - 1}`
  }

  const getNextChapterSlug = () => {
    if (!hasNext) return null
    return `chapter-${currentChapter + 1}`
  }

  return (
    <div className="mt-12 pt-8 border-t border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between gap-4">
        {/* Previous Button */}
        {hasPrevious ? (
          <Link
            href={`/modules/${moduleSlug}/${getPreviousChapterSlug()}`}
            className="group flex items-center gap-3 px-6 py-4 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-orange-500 dark:hover:border-orange-500 transition-all hover:shadow-lg flex-1"
          >
            <div className="flex items-center justify-center w-10 h-10 rounded-full bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400 group-hover:bg-orange-200 dark:group-hover:bg-orange-900/50 transition-colors">
              <ChevronLeft className="w-5 h-5" />
            </div>
            <div className="flex-1 text-left">
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">이전 챕터</p>
              <p className="font-semibold text-gray-900 dark:text-white group-hover:text-orange-600 dark:group-hover:text-orange-400 transition-colors">
                Chapter {currentChapter - 1}: {chapterTitles[currentChapter - 1]}
              </p>
            </div>
          </Link>
        ) : (
          <div className="flex-1" />
        )}

        {/* Module Home Button */}
        <Link
          href={`/modules/${moduleSlug}`}
          className="px-6 py-4 bg-gray-100 dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-orange-500 dark:hover:border-orange-500 transition-all hover:shadow-lg"
        >
          <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 hover:text-orange-600 dark:hover:text-orange-400 transition-colors text-center">
            📚 모듈 홈
          </p>
        </Link>

        {/* Next Button */}
        {hasNext ? (
          <Link
            href={`/modules/${moduleSlug}/${getNextChapterSlug()}`}
            className="group flex items-center gap-3 px-6 py-4 bg-gradient-to-r from-orange-500 to-red-500 rounded-xl hover:from-orange-600 hover:to-red-600 transition-all hover:shadow-lg flex-1"
          >
            <div className="flex-1 text-right">
              <p className="text-xs text-white/80 mb-1">다음 챕터</p>
              <p className="font-semibold text-white">
                Chapter {currentChapter + 1}: {chapterTitles[currentChapter + 1]}
              </p>
            </div>
            <div className="flex items-center justify-center w-10 h-10 rounded-full bg-white/20 text-white group-hover:bg-white/30 transition-colors">
              <ChevronRight className="w-5 h-5" />
            </div>
          </Link>
        ) : (
          <div className="flex-1" />
        )}
      </div>

      {/* Progress Bar */}
      <div className="mt-6">
        <div className="flex items-center justify-between mb-2">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            학습 진행률
          </p>
          <p className="text-sm font-semibold text-orange-600 dark:text-orange-400">
            {currentChapter} / {totalChapters} 챕터
          </p>
        </div>
        <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-orange-500 to-red-500 rounded-full transition-all duration-500"
            style={{ width: `${(currentChapter / totalChapters) * 100}%` }}
          />
        </div>
      </div>
    </div>
  )
}
