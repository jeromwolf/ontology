'use client';

import dynamic from 'next/dynamic';

// 챕터 컴포넌트들을 동적으로 import
const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false })
const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false })
const Chapter3 = dynamic(() => import('./chapters/Chapter3'), { ssr: false })
const Chapter4 = dynamic(() => import('./chapters/Chapter4'), { ssr: false })
const Chapter5 = dynamic(() => import('./chapters/Chapter5'), { ssr: false })
const Chapter6 = dynamic(() => import('./chapters/Chapter6'), { ssr: false })
const Chapter7 = dynamic(() => import('./chapters/Chapter7'), { ssr: false })
const Chapter8 = dynamic(() => import('./chapters/Chapter8'), { ssr: false })

export default function ChapterContent({ chapterId }: { chapterId: number }) {
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
        return <div className="text-center text-gray-500 dark:text-gray-400">챕터 콘텐츠를 준비 중입니다.</div>
    }
  }

  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      {getChapterComponent()}
    </div>
  )
}