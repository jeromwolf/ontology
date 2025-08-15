'use client';

import React from 'react';
import dynamic from 'next/dynamic';

// Dynamic imports for better code splitting
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
  const getChapterComponent = () => {
    switch (chapterId) {
      case 'fundamentals':
        return <Chapter1 />
      case 'scaling':
        return <Chapter2 />
      case 'caching':
        return <Chapter3 />
      case 'database':
        return <Chapter4 />
      case 'messaging':
        return <Chapter5 />
      case 'microservices':
        return <Chapter6 />
      case 'monitoring':
        return <Chapter7 />
      case 'case-studies':
        return <Chapter8 />
      default:
        return (
          <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
            <p className="text-gray-600 dark:text-gray-400">
              챕터 콘텐츠를 불러올 수 없습니다.
            </p>
          </div>
        )
    }
  }
  
  return <>{getChapterComponent()}</>
}