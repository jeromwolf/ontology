'use client';

import { ReactNode } from 'react';
import dynamic from 'next/dynamic';

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
  const renderContent = (): ReactNode => {
    switch (chapterId) {
      case 'cloud-fundamentals':
        return <Chapter1 />
      case 'aws-essentials':
        return <Chapter2 />
      case 'azure-fundamentals':
        return <Chapter3 />
      case 'gcp-overview':
        return <Chapter4 />
      case 'cloud-architecture':
        return <Chapter5 />
      case 'serverless':
        return <Chapter6 />
      case 'containerization':
        return <Chapter7 />
      case 'cloud-security':
        return <Chapter8 />
      case 'cost-optimization':
        return <Chapter9 />
      case 'multi-cloud':
        return <Chapter10 />
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

function ComingSoonContent() {
  return (
    <div className="text-center py-16">
      <div className="text-6xl mb-4">🚧</div>
      <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
        Content Coming Soon
      </h2>
      <p className="text-gray-600 dark:text-gray-400">
        This chapter will be updated soon.
      </p>
    </div>
  )
}
