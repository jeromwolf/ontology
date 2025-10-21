'use client'

import dynamic from 'next/dynamic'

// 동적 임포트로 각 챕터 컴포넌트 로드
const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false })
const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false })
const Chapter3 = dynamic(() => import('./chapters/Chapter3'), { ssr: false })
const Chapter4 = dynamic(() => import('./chapters/Chapter4'), { ssr: false })
const Chapter5 = dynamic(() => import('./chapters/Chapter5'), { ssr: false })
const Chapter6 = dynamic(() => import('./chapters/Chapter6'), { ssr: false })
const Chapter7 = dynamic(() => import('./chapters/Chapter7'), { ssr: false })
const Chapter8 = dynamic(() => import('./chapters/Chapter8'), { ssr: false })
const Chapter9 = dynamic(() => import('./chapters/Chapter9'), { ssr: false })
const Chapter10 = dynamic(() => import('./chapters/Chapter10'), { ssr: false })

interface ChapterContentProps {
  chapterId: string
}

export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const getChapterComponent = () => {
    switch (chapterId) {
      case 'hpc-fundamentals':
        return <Chapter1 />
      case 'parallel-computing':
        return <Chapter2 />
      case 'cuda-programming':
        return <Chapter3 />
      case 'gpu-architecture':
        return <Chapter4 />
      case 'cluster-computing':
        return <Chapter5 />
      case 'performance-optimization':
        return <Chapter6 />
      case 'distributed-systems':
        return <Chapter7 />
      case 'scientific-computing':
        return <Chapter8 />
      case 'cloud-hpc':
        return <Chapter9 />
      case 'ai-acceleration':
        return <Chapter10 />
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
