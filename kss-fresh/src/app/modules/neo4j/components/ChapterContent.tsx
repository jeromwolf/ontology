'use client'

import dynamic from 'next/dynamic'

// Dynamic imports for chapters with SSR disabled
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
  const renderChapterContent = () => {
    switch (chapterId) {
      case '01-introduction':
        return <Chapter1 />
      case '02-cypher-basics':
        return <Chapter2 />
      case '03-data-modeling':
        return <Chapter3 />
      case '04-advanced-cypher':
        return <Chapter4 />
      case '05-graph-algorithms':
        return <Chapter5 />
      case '06-integration':
        return <Chapter6 />
      case '07-performance':
        return <Chapter7 />
      case '08-real-world':
        return <Chapter8 />
      default:
        return <div>Content not found</div>
    }
  }

  return (
    <div className="prose prose-lg max-w-none dark:prose-invert">
      {renderChapterContent()}
    </div>
  )
}