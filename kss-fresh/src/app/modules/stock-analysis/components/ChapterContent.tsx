'use client'

import { ReactNode } from 'react'
import dynamic from 'next/dynamic'

// Dynamic imports for all chapters with SSR disabled
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
const Chapter17 = dynamic(() => import('./chapters/Chapter17'), { ssr: false })
const Chapter18 = dynamic(() => import('./chapters/Chapter18'), { ssr: false })

interface ChapterContentProps {
  chapterId: string
}

// Main chapter content router component
export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const renderContent = (): ReactNode => {
    switch (chapterId) {
      // 왕초보 과정
      case 'what-is-stock':
        return <Chapter6 />
      case 'why-invest':
        return <Chapter7 />
      case 'stock-market-basics':
        return <Chapter8 />
      
      // 입문자 과정
      case 'how-to-start':
        return <Chapter9 />
      case 'order-types':
        return <Chapter10 />
      case 'first-stock-selection':
        return <Chapter11 />
      
      // 초급자 과정
      case 'basic-chart-reading':
        return <Chapter12 />
      case 'simple-indicators':
        return <Chapter13 />
      case 'trend-basics':
        return <Chapter14 />
      
      // 중급자 과정
      case 'company-analysis-basics':
        return <Chapter15 />
      case 'simple-valuation':
        return <Chapter16 />
      case 'buy-sell-timing':
        return <Chapter17 />
      
      // 기존 고급 콘텐츠
      case 'foundation':
        return <Chapter1 />
      case 'fundamental-analysis':
        return <Chapter2 />
      case 'technical-analysis':
        return <Chapter3 />
      case 'portfolio-management':
        return <Chapter4 />
      case 'ai-quant-investing':
        return <Chapter5 />
      
      // Coming Soon
      case 'coming-soon':
      default:
        return <Chapter18 />
    }
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      {renderContent()}
    </div>
  )
}