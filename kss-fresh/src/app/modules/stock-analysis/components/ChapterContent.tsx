'use client';

import { ReactNode } from 'react';
import dynamic from 'next/dynamic';

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
const Chapter19 = dynamic(() => import('./chapters/Chapter19'), { ssr: false })
const Chapter20 = dynamic(() => import('./chapters/Chapter20'), { ssr: false })
const Chapter21 = dynamic(() => import('./chapters/Chapter21'), { ssr: false })
const Chapter22 = dynamic(() => import('./chapters/Chapter22'), { ssr: false })
const Chapter23 = dynamic(() => import('./chapters/Chapter23'), { ssr: false })
const Chapter24 = dynamic(() => import('./chapters/Chapter24'), { ssr: false })
const Chapter25 = dynamic(() => import('./chapters/Chapter25'), { ssr: false })
const Chapter26 = dynamic(() => import('./chapters/Chapter26'), { ssr: false })
const Chapter27 = dynamic(() => import('./chapters/Chapter27'), { ssr: false })
const Chapter28 = dynamic(() => import('./chapters/Chapter28'), { ssr: false })
const Chapter29 = dynamic(() => import('./chapters/Chapter29'), { ssr: false })
const Chapter30 = dynamic(() => import('./chapters/Chapter30'), { ssr: false })
const Chapter31 = dynamic(() => import('./chapters/Chapter31'), { ssr: false })
const Chapter32 = dynamic(() => import('./chapters/Chapter32'), { ssr: false })
const Chapter33 = dynamic(() => import('./chapters/Chapter33'), { ssr: false })
const Chapter34 = dynamic(() => import('./chapters/Chapter34'), { ssr: false })
const Chapter35 = dynamic(() => import('./chapters/Chapter35'), { ssr: false })
const Chapter36 = dynamic(() => import('./chapters/Chapter36'), { ssr: false })
const Chapter37 = dynamic(() => import('./chapters/Chapter37'), { ssr: false })
const Chapter38 = dynamic(() => import('./chapters/Chapter38'), { ssr: false })
const Chapter39 = dynamic(() => import('./chapters/Chapter39'), { ssr: false })

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
      
      // Baby Chick 과정 (완전 초보자)
      case 'what-is-stock':
        return <Chapter6 />
      case 'open-account':
        return <Chapter7 />
      case 'trading-app-basics':
        return <Chapter8 />
      case 'understanding-candles':
        return <Chapter9 />
      case 'volume-basics':
        return <Chapter10 />
      case 'order-book':
        return <Chapter11 />
      case 'basic-terms':
        return <Chapter12 />
      case 'reading-news':
        return <Chapter13 />
      case 'sectors-themes':
        return <Chapter14 />
      case 'stock-selection':
        return <Chapter15 />
      case 'small-start':
        return <Chapter16 />
      case 'trading-diary':
        return <Chapter17 />
      
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
      
      // Foundation Program (기초 과정)
      case 'chart-basics':
        return <Chapter1 />
      case 'technical-indicators':
        return <Chapter2 />
      case 'pattern-recognition':
        return <Chapter3 />
      case 'financial-statements':
        return <Chapter4 />
      case 'valuation-basics':
        return <Chapter5 />
      case 'industry-analysis':
        return <Chapter19 />
      case 'investment-strategies':
        return <Chapter1 />
      case 'portfolio-basics':
        return <Chapter4 />
      case 'risk-control':
        return <Chapter5 />
      case 'market-timing':
        return <Chapter3 />
      case 'real-trading':
        return <Chapter20 />
      case 'investment-plan':
        return <Chapter21 />
      
      // 기존 고급 콘텐츠 (나중에 재배치 예정)
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
      
      // Advanced Program (고급 과정)
      case 'advanced-technical-analysis':
        return <Chapter22 />
      case 'system-trading-basics':
        return <Chapter23 />
      case 'automated-strategies':
        return <Chapter24 />
      case 'quantitative-basics':
        return <Chapter25 />
      case 'financial-data-analysis':
        return <Chapter26 />
      case 'factor-models':
        return <Chapter27 />
      case 'derivatives-basics':
        return <Chapter28 />
      case 'advanced-options':
        return <Chapter29 />
      case 'hedging-strategies':
        return <Chapter30 />
      case 'global-markets':
        return <Chapter31 />
      case 'alternative-investments':
        return <Chapter32 />
      case 'macro-trading':
        return <Chapter33 />
      
      // Foundation Program - Global Investment Chapters
      case 'global-brokerage-accounts':
        return <Chapter34 />
      case 'global-sectors-understanding':
        return <Chapter35 />
      case 'gaap-vs-ifrs':
        return <Chapter36 />
      
      // Advanced Program - Global Investment Chapters
      case 'currency-hedging-strategies':
        return <Chapter37 />
      case 'global-macro-investing':
        return <Chapter38 />
      case 'international-diversification':
        return <Chapter39 />
      
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