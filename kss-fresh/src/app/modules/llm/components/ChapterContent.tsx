'use client';

import dynamic from 'next/dynamic';

// Dynamic imports for chapters
const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false })
const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false })
const Chapter3 = dynamic(() => import('./chapters/Chapter3'), { ssr: false })
const Chapter4 = dynamic(() => import('./chapters/Chapter4'), { ssr: false })
const Chapter5Applications1 = dynamic(() => import('./chapters/Chapter5Applications1'), { ssr: false })
const Chapter5Applications2 = dynamic(() => import('./chapters/Chapter5Applications2'), { ssr: false })
const Chapter5Applications3 = dynamic(() => import('./chapters/Chapter5Applications3'), { ssr: false })
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
      case '02-architecture':
        return <Chapter2 />
      case '03-training':
        return <Chapter3 />
      case '04-prompt-engineering':
        return <Chapter4 />
      case '05-applications-1':
        return <Chapter5Applications1 />
      case '05-applications-2':
        return <Chapter5Applications2 />
      case '05-applications-3':
        return <Chapter5Applications3 />
      case '06-advanced':
        return <Chapter6 />
      case '07-huggingface':
        return <Chapter7 />
      case '08-ai-services':
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