'use client'

import dynamic from 'next/dynamic'

// 동적 임포트로 각 챕터 컴포넌트 로드
const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false })
const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false })
// const Chapter3 = dynamic(() => import('./chapters/Chapter3'), { ssr: false })
// const Chapter4 = dynamic(() => import('./chapters/Chapter4'), { ssr: false })
// const Chapter5 = dynamic(() => import('./chapters/Chapter5'), { ssr: false })
// const Chapter6 = dynamic(() => import('./chapters/Chapter6'), { ssr: false })
// const Chapter7 = dynamic(() => import('./chapters/Chapter7'), { ssr: false })
// const Chapter8 = dynamic(() => import('./chapters/Chapter8'), { ssr: false })
// const Chapter9 = dynamic(() => import('./chapters/Chapter9'), { ssr: false })
// const Chapter10 = dynamic(() => import('./chapters/Chapter10'), { ssr: false })
// const Chapter11 = dynamic(() => import('./chapters/Chapter11'), { ssr: false })
// const Chapter12 = dynamic(() => import('./chapters/Chapter12'), { ssr: false })

interface ChapterContentProps {
  chapterId: string
}

export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const getChapterComponent = () => {
    switch (chapterId) {
      case 'data-engineering-foundations':
        return <Chapter1 />
      case 'exploratory-data-analysis':
        return <Chapter2 />
      case 'data-architecture-patterns':
      case 'batch-processing':
      case 'stream-processing':
      case 'data-modeling-warehousing':
      case 'data-quality-governance':
      case 'cloud-data-platforms':
      case 'data-orchestration':
      case 'performance-optimization':
      case 'mlops-data-engineering':
      case 'real-world-projects':
        return <div className="p-8 text-center text-gray-500">이 챕터는 준비 중입니다.</div>
      default:
        return <div className="p-8 text-center text-gray-500">챕터를 찾을 수 없습니다.</div>
    }
  }

  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      {getChapterComponent()}
    </div>
  )
}