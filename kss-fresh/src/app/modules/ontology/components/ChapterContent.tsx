'use client';

import { ReactNode } from 'react';
import dynamic from 'next/dynamic';

// Lazy load simulators
const SparqlPlayground = dynamic(() => 
  import('@/components/sparql-playground/SparqlPlayground').then(mod => ({ default: mod.SparqlPlayground })), 
  { 
    ssr: false,
    loading: () => <div className="h-96 flex items-center justify-center">SPARQL Playground 로딩 중...</div>
  }
)

// Lazy load chapters
const IntroContent = dynamic(() => import('./chapters/IntroContent'), { ssr: false })
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
const Chapter13 = dynamic(() => import('./chapters/Chapter13'), { ssr: false })
const Chapter14 = dynamic(() => import('./chapters/Chapter14'), { ssr: false })
const Chapter15 = dynamic(() => import('./chapters/Chapter15'), { ssr: false })
const Chapter16 = dynamic(() => import('./chapters/Chapter16'), { ssr: false })
const ComingSoon = dynamic(() => import('./chapters/ComingSoon'), { ssr: false })

const InferenceEngine = dynamic(() => 
  import('@/components/rdf-editor/components/InferenceEngine').then(mod => ({ default: mod.InferenceEngine })), 
  { 
    ssr: false,
    loading: () => <div className="h-32 flex items-center justify-center">추론 엔진 로딩 중...</div>
  }
)


interface ChapterContentProps {
  chapterId: string
}

export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const renderContent = (): ReactNode => {
    switch (chapterId) {
      case 'intro':
        return <IntroContent />
      case 'chapter01':
        return <Chapter1 />
      case 'chapter02':
        return <Chapter2 />
      case 'chapter03':
        return <Chapter3 />
      case 'chapter04':
        return <Chapter4 />
      case 'chapter05':
        return <Chapter5 />
      case 'chapter06':
        return <Chapter6 />
      case 'chapter07':
        return <Chapter7 />
      case 'chapter08':
        return <Chapter8 />
      case 'chapter09':
        return <Chapter9 />
      case 'chapter10':
        return <Chapter10 />
      case 'chapter11':
        return <Chapter11 />
      case 'chapter12':
        return <Chapter12 />
      case 'chapter13':
        return <Chapter13 />
      case 'chapter14':
        return <Chapter14 />
      case 'chapter15':
        return <Chapter15 />
      case 'chapter16':
        return <Chapter16 />
      default:
        return <ComingSoon />
    }
  }

  return (
    <div className="prose prose-lg dark:prose-invert max-w-none 
      prose-headings:text-gray-900 dark:prose-headings:text-white
      prose-p:text-gray-700 dark:prose-p:text-gray-300
      prose-strong:text-gray-900 dark:prose-strong:text-white
      prose-a:text-indigo-600 dark:prose-a:text-indigo-400
      prose-code:text-indigo-600 dark:prose-code:text-indigo-400
      prose-pre:bg-gray-50 dark:prose-pre:bg-gray-900
      prose-h1:text-4xl prose-h1:font-bold prose-h1:mb-8
      prose-h2:text-3xl prose-h2:font-bold prose-h2:mt-12 prose-h2:mb-6
      prose-h3:text-xl prose-h3:font-semibold prose-h3:mt-8 prose-h3:mb-4
      prose-p:leading-relaxed prose-p:mb-6
      prose-li:my-2">
      {renderContent()}
    </div>
  )
}
