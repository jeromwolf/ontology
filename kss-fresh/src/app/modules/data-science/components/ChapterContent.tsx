'use client';

import dynamic from 'next/dynamic';

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
const Chapter11 = dynamic(() => import('./chapters/Chapter11'), { ssr: false })
const Chapter12 = dynamic(() => import('./chapters/Chapter12'), { ssr: false })

interface ChapterContentProps {
  chapterId: string
  onComplete?: () => void
}

export default function ChapterContent({ chapterId, onComplete }: ChapterContentProps) {
  const getChapterComponent = () => {
    switch (chapterId) {
      case 'data-science-intro':
        return <Chapter1 onComplete={onComplete} />
      case 'statistical-thinking':
        return <Chapter2 onComplete={onComplete} />
      case 'eda-visualization':
        return <Chapter3 onComplete={onComplete} />
      case 'supervised-learning':
        return <Chapter4 onComplete={onComplete} />
      case 'unsupervised-learning':
        return <Chapter5 onComplete={onComplete} />
      case 'deep-learning-basics':
        return <Chapter6 onComplete={onComplete} />
      case 'time-series-analysis':
        return <Chapter7 onComplete={onComplete} />
      case 'nlp-fundamentals':
        return <Chapter8 onComplete={onComplete} />
      case 'ab-testing':
        return <Chapter9 onComplete={onComplete} />
      case 'model-deployment':
        return <Chapter10 onComplete={onComplete} />
      case 'business-analytics':
        return <Chapter11 onComplete={onComplete} />
      case 'case-studies':
        return <Chapter12 onComplete={onComplete} />
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