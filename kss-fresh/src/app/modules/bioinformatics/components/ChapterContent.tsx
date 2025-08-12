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
const Chapter10 = dynamic(() => import('./chapters/Chapter10'), { ssr: false })

interface ChapterContentProps {
  chapterId: string
}

export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const renderContent = () => {
    switch (chapterId) {
      case 'biology-fundamentals':
        return <Chapter1 />
      case 'cell-genetics':
        return <Chapter2 />
      case 'genomics-sequencing':
        return <Chapter3 />
      case 'sequence-alignment':
        return <Chapter4 />
      case 'proteomics-structure':
        return <Chapter5 />
      case 'drug-discovery':
        return <Chapter6 />
      case 'omics-integration':
        return <Chapter7 />
      case 'ml-genomics':
        return <Chapter8 />
      case 'single-cell':
        return <Chapter9 />
      case 'clinical-applications':
        return <Chapter10 />
      default:
        return <div>챕터 콘텐츠를 불러올 수 없습니다.</div>
    }
  }

  return <div className="prose prose-lg dark:prose-invert max-w-none">{renderContent()}</div>
}