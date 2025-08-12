'use client'

import dynamic from 'next/dynamic'

// Dynamic imports for better performance
const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false })
const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false })
const Chapter3 = dynamic(() => import('./chapters/Chapter3'), { ssr: false })
const Chapter4 = dynamic(() => import('./chapters/Chapter4'), { ssr: false })
const Chapter5 = dynamic(() => import('./chapters/Chapter5'), { ssr: false })
const Chapter6 = dynamic(() => import('./chapters/Chapter6'), { ssr: false })
const Chapter7 = dynamic(() => import('./chapters/Chapter7'), { ssr: false })
const Chapter8 = dynamic(() => import('./chapters/Chapter8'), { ssr: false })
const Chapter9 = dynamic(() => import('./chapters/Chapter9'), { ssr: false })

export default function ChapterContent({ chapterId }: { chapterId: number }) {
  const content = getChapterComponent(chapterId)
  return <div className="prose prose-lg dark:prose-invert max-w-none">{content}</div>
}

function getChapterComponent(chapterId: number) {
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
    case 9:
      return <Chapter9 />
    default:
      return (
        <div className="text-center py-12">
          <div className="text-6xl mb-4">🤖</div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            챕터를 준비 중입니다
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            AI 자동화 기술의 최신 내용을 업데이트하고 있습니다.
          </p>
        </div>
      )
  }
}