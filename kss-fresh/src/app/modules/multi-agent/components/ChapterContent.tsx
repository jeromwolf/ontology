'use client';

import React from 'react';
import dynamic from 'next/dynamic';

// 동적 임포트로 각 챕터 컴포넌트 로드
const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false });
const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false });
const Chapter3 = dynamic(() => import('./chapters/Chapter3'), { ssr: false });
const Chapter4 = dynamic(() => import('./chapters/Chapter4'), { ssr: false });
const Chapter5 = dynamic(() => import('./chapters/Chapter5'), { ssr: false });
const Chapter6 = dynamic(() => import('./chapters/Chapter6'), { ssr: false });
const Chapter7 = dynamic(() => import('./chapters/Chapter7'), { ssr: false });
const Chapter8 = dynamic(() => import('./chapters/Chapter8'), { ssr: false });

interface ChapterContentProps {
  chapterId: string;
}

export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const getChapterComponent = () => {
    switch(chapterId) {
      case 'intro-multi-agent':
        return <Chapter1 />;
      case 'a2a-communication':
        return <Chapter2 />;
      case 'crewai-framework':
        return <Chapter3 />;
      case 'autogen-systems':
        return <Chapter4 />;
      case 'consensus-algorithms':
        return <Chapter5 />;
      case 'orchestration-patterns':
        return <Chapter6 />;
      case 'langgraph-systems':
        return <Chapter7 />;
      case 'swarm-patterns':
        return <Chapter8 />;
      default:
        return (
          <div className="text-center py-12">
            <p className="text-gray-500 dark:text-gray-400">챕터 콘텐츠를 불러올 수 없습니다.</p>
          </div>
        );
    }
  };

  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      {getChapterComponent()}
    </div>
  );
}