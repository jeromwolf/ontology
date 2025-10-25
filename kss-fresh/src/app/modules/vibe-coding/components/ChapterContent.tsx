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
const Chapter7 = dynamic(() => import('./chapters/Chapter7'), { ssr: false })
const Chapter8 = dynamic(() => import('./chapters/Chapter8'), { ssr: false })
const Chapter9 = dynamic(() => import('./chapters/Chapter9'), { ssr: false })
const Chapter10 = dynamic(() => import('./chapters/Chapter10'), { ssr: false })
const Chapter11 = dynamic(() => import('./chapters/Chapter11'), { ssr: false })
const Chapter12 = dynamic(() => import('./chapters/Chapter12'), { ssr: false })

interface ChapterContentProps {
  chapterId: string
}

// 챕터별 콘텐츠를 렌더링하는 라우터 컴포넌트
export default function ChapterContent({ chapterId }: ChapterContentProps) {
  // 챕터별 콘텐츠 매핑
  const renderContent = (): ReactNode => {
    switch (chapterId) {
      case 'ai-coding-revolution':
        return <Chapter1 />
      case 'cursor-mastery':
        return <Chapter2 />
      case 'github-copilot':
        return <Chapter3 />
      case 'claude-code-engineering':
        return <Chapter4 />
      case 'prompt-engineering':
        return <Chapter5 />
      case 'ai-test-generation':
        return <Chapter6 />
      case 'ai-code-review':
        return <Chapter7 />
      case 'ai-refactoring':
        return <Chapter8 />
      case 'ai-documentation':
        return <Chapter9 />
      case 'ai-workflow-automation':
        return <Chapter10 />
      case 'ai-security-practices':
        return <Chapter11 />
      case 'real-world-projects':
        return <Chapter12 />
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
