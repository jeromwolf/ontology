'use client';

import { ReactNode } from 'react';
import dynamic from 'next/dynamic';

// Dynamic imports for all chapters with SSR disabled
const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false }); const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false })
const Chapter3 = dynamic(() => import('./chapters/Chapter3'), { ssr: false }); const Chapter4 = dynamic(() => import('./chapters/Chapter4'), { ssr: false })
const Chapter5 = dynamic(() => import('./chapters/Chapter5'), { ssr: false }); const Chapter6 = dynamic(() => import('./chapters/Chapter6'), { ssr: false })
const Chapter7 = dynamic(() => import('./chapters/Chapter7'), { ssr: false }); const Chapter8 = dynamic(() => import('./chapters/Chapter8'), { ssr: false })

interface ChapterContentProps {
  chapterId: string
}

// Main chapter content router component
export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const renderContent = (): ReactNode => {
    switch (chapterId) {
      case 'creative-ai-intro':
        return <Chapter1 />
      case 'image-generation':
        return <Chapter2 />
      case 'stable-diffusion':
        return <Chapter3 />
      case 'style-transfer':
        return <Chapter4 />
      case 'ai-music':
        return <Chapter5 />
      case 'video-generation':
        return <Chapter6 />
      case 'creative-workflow':
        return <Chapter7 />
      case 'commercial-use':
        return <Chapter8 />
      default:
        return <Chapter1 />
    }
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      {renderContent()}
    </div>
  )
}
