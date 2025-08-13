'use client'

import React from 'react'
import dynamic from 'next/dynamic'

// 동적 임포트로 각 챕터 컴포넌트 로드
const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false })
const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false })
const Chapter3 = dynamic(() => import('./chapters/Chapter3'), { ssr: false })
const Chapter4 = dynamic(() => import('./chapters/Chapter4'), { ssr: false })
const Chapter5 = dynamic(() => import('./chapters/Chapter5'), { ssr: false })
const Chapter6 = dynamic(() => import('./chapters/Chapter6'), { ssr: false })
const Chapter7 = dynamic(() => import('./chapters/Chapter7'), { ssr: false })
const Chapter8 = dynamic(() => import('./chapters/Chapter8'), { ssr: false })
const Chapter9 = dynamic(() => import('./chapters/Chapter9'), { ssr: false })

interface ChapterContentProps {
  chapterId: number
}

export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const renderChapterContent = () => {
    switch (chapterId) {
      case 1:
        return <Chapter1 />
      case 2:
        return <Chapter2 />
      case 3:
        return <Chapter3 />
      case 4:
        return <Chapter4 />
      case 5:
        return <Chapter5 />
      case 6:
        return <Chapter6 />
      case 7:
        return <Chapter7 />
      case 8:
        return <Chapter8 />
      case 9:
        return <Chapter9 />
      default:
        return <div>챕터를 찾을 수 없습니다.</div>
    }
  }

  return (
    <div>
      {renderChapterContent()}
    </div>
  )
}