'use client'

import dynamic from 'next/dynamic'

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
      case 'optimization-fundamentals':
        return <Chapter1 />
      case 'linear-optimization':
        return <Chapter2 />
      case 'nonlinear-optimization':
        return <Chapter3 />
      case 'constrained-optimization':
        return <Chapter4 />
      case 'convex-optimization':
        return <Chapter5 />
      case 'ai-optimization':
        return <Chapter6 />
      case 'metaheuristics':
        return <Chapter7 />
      case 'multi-objective':
        return <Chapter8 />
      case 'dynamic-programming':
        return <Chapter9 />
      case 'optimization-applications':
        return <Chapter10 />
      default:
        return (
          <div className="min-h-screen bg-gradient-to-br from-emerald-900 via-gray-900 to-teal-900 text-white flex items-center justify-center">
            <div className="text-center">
              <h1 className="text-4xl font-bold mb-4">Chapter not found</h1>
              <p className="text-emerald-300">The requested chapter does not exist.</p>
            </div>
          </div>
        )
    }
  }

  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      {getChapterComponent()}
    </div>
  )
}
