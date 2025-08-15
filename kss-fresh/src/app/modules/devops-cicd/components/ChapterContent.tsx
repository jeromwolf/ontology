'use client';

import React from 'react';
import dynamic from 'next/dynamic';

// Dynamic imports for all chapters
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

const ChapterContent: React.FC<ChapterContentProps> = ({ chapterId }) => {
  const getChapterComponent = () => {
    switch (chapterId) {
      case 'devops-culture':
        return <Chapter1 />
      case 'docker-fundamentals':
        return <Chapter2 />
      case 'docker-advanced':
        return <Chapter3 />
      case 'kubernetes-basics':
        return <Chapter4 />
      case 'kubernetes-advanced':
        return <Chapter5 />
      case 'cicd-pipelines':
        return <Chapter6 />
      case 'gitops-deployment':
        return <Chapter7 />
      case 'monitoring-security':
        return <Chapter8 />
      default:
        return (
          <div className="prose prose-lg max-w-none dark:prose-invert">
            <h1>챕터를 찾을 수 없습니다</h1>
            <p>요청하신 챕터가 존재하지 않습니다.</p>
          </div>
        )
    }
  }

  return getChapterComponent()
}

export default ChapterContent