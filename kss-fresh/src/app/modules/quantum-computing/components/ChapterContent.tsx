'use client'

import dynamic from 'next/dynamic'

// Dynamic imports with SSR disabled for better performance
const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false })
const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false })
const Chapter3 = dynamic(() => import('./chapters/Chapter3'), { ssr: false })
const Chapter4 = dynamic(() => import('./chapters/Chapter4'), { ssr: false })
const Chapter5 = dynamic(() => import('./chapters/Chapter5'), { ssr: false })
const Chapter6 = dynamic(() => import('./chapters/Chapter6'), { ssr: false })
const Chapter7 = dynamic(() => import('./chapters/Chapter7'), { ssr: false })
const Chapter8 = dynamic(() => import('./chapters/Chapter8'), { ssr: false })

interface ChapterContentProps {
  chapterId: number
}

export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const getChapterComponent = () => {
    switch (chapterId) {
      case 1:
        return <Chapter1 />
      case 2:
        return <Chapter2 />
      case 3:
        return <Chapter3 />
      case 4:
        return <Chapter4 />
      case 5:
        return <Chapter5 />
      case 6:
        return <Chapter6 />
      case 7:
        return <Chapter7 />
      case 8:
        return <Chapter8 />
      default:
        return (
          <div className="p-8 text-center">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              챕터 {chapterId} 콘텐츠 준비 중
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              이 챕터의 콘텐츠가 곧 추가될 예정입니다.
            </p>
          </div>
        )
    }
  }

  return getChapterComponent()
}