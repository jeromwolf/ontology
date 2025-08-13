'use client';

import React from 'react';
import dynamic from 'next/dynamic';

// Dynamic imports for all chapters
const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false });
const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false });
const Chapter3 = dynamic(() => import('./chapters/Chapter3'), { ssr: false });
const Chapter4 = dynamic(() => import('./chapters/Chapter4'), { ssr: false });
const Chapter5 = dynamic(() => import('./chapters/Chapter5'), { ssr: false });
const Chapter6 = dynamic(() => import('./chapters/Chapter6'), { ssr: false });

interface ChapterContentProps {
  chapterId: string;
}

export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const renderContent = () => {
    switch(chapterId) {
      case '1':
        return <Chapter1 />;
      case '2':
        return <Chapter2 />;
      case '3':
        return <Chapter3 />;
      case '4':
        return <Chapter4 />;
      case '5':
        return <Chapter5 />;
      case '6':
        return <Chapter6 />;
      default:
        return <div>챕터 콘텐츠를 찾을 수 없습니다.</div>;
    }
  };

  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      {renderContent()}
    </div>
  );
}