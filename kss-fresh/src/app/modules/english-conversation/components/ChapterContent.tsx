'use client';

import dynamic from 'next/dynamic';

// Dynamic imports for better performance
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
  const renderContent = () => {
    switch (chapterId) {
      case 'conversation-basics':
        return <Chapter1 />
      case 'daily-situations':
        return <Chapter2 />
      case 'business-english':
        return <Chapter3 />
      case 'travel-english':
        return <Chapter4 />
      case 'pronunciation-intonation':
        return <Chapter5 />
      case 'listening-comprehension':
        return <Chapter6 />
      case 'cultural-context':
        return <Chapter7 />
      case 'advanced-conversation':
        return <Chapter8 />
      default:
        return <div>챕터 콘텐츠를 불러올 수 없습니다.</div>
    }
  }

  return <div className="prose prose-lg dark:prose-invert max-w-none">{renderContent()}</div>
}