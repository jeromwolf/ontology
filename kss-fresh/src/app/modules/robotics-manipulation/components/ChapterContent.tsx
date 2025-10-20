'use client'

import dynamic from 'next/dynamic'
import { Suspense } from 'react'

// Dynamic imports for all chapters
const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false })
const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false })
const Chapter3 = dynamic(() => import('./chapters/Chapter3'), { ssr: false })
const Chapter4 = dynamic(() => import('./chapters/Chapter4'), { ssr: false })
const Chapter5 = dynamic(() => import('./chapters/Chapter5'), { ssr: false })
const Chapter6 = dynamic(() => import('./chapters/Chapter6'), { ssr: false })
const Chapter7 = dynamic(() => import('./chapters/Chapter7'), { ssr: false })
const Chapter8 = dynamic(() => import('./chapters/Chapter8'), { ssr: false })

interface ChapterContentProps {
  chapterId: string
}

export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const getChapterComponent = () => {
    switch (chapterId) {
      case 'chapter-1':
        return <Chapter1 />
      case 'chapter-2':
        return <Chapter2 />
      case 'chapter-3':
        return <Chapter3 />
      case 'chapter-4':
        return <Chapter4 />
      case 'chapter-5':
        return <Chapter5 />
      case 'chapter-6':
        return <Chapter6 />
      case 'chapter-7':
        return <Chapter7 />
      case 'chapter-8':
        return <Chapter8 />
      default:
        return (
          <div className="bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              챕터를 찾을 수 없습니다
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              요청하신 챕터가 존재하지 않습니다.
            </p>
          </div>
        )
    }
  }

  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-orange-600 mx-auto mb-4"></div>
            <p className="text-gray-600 dark:text-gray-400">로딩 중...</p>
          </div>
        </div>
      }
    >
      {getChapterComponent()}
    </Suspense>
  )
}
