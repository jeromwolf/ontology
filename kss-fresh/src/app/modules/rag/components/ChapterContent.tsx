'use client';

import { ReactNode } from 'react';
import dynamic from 'next/dynamic';

// 동적 임포트로 각 챕터 컴포넌트 로드 (성능 최적화)
const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false })
const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false })
const Chapter3 = dynamic(() => import('./chapters/Chapter3'), { ssr: false })
const Chapter4 = dynamic(() => import('./chapters/Chapter4'), { ssr: false })
const Chapter5 = dynamic(() => import('./chapters/Chapter5'), { ssr: false })
const Chapter6 = dynamic(() => import('./chapters/Chapter6'), { ssr: false })

interface ChapterContentProps {
  chapterId: string
}

// 챕터별 콘텐츠를 렌더링하는 라우터 컴포넌트
export default function ChapterContent({ chapterId }: ChapterContentProps) {
  // 챕터별 콘텐츠 매핑
  const renderContent = (): ReactNode => {
    switch (chapterId) {
      case '01-what-is-rag':
        return <Chapter1 />
      case '02-document-processing':
        return <Chapter2 />
      case '03-embeddings':
        return <Chapter3 />
      case '04-vector-search':
        return <Chapter4 />
      case '05-answer-generation':
        return <Chapter5 />
      case '06-advanced-rag':
        return <Chapter6 />
      default:
        return <ComingSoonContent />
    }
  }

  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      {renderContent()}
    </div>
  )
}

// Coming Soon 컴포넌트

// Coming Soon
function ComingSoonContent() {
  return (
    <div className="text-center py-16">
      <div className="text-6xl mb-4">🚧</div>
      <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
        콘텐츠 준비 중
      </h2>
      <p className="text-gray-600 dark:text-gray-400">
        이 챕터의 콘텐츠는 곧 업데이트될 예정입니다.
      </p>
    </div>
  )
}