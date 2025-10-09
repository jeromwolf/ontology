'use client';

import dynamic from 'next/dynamic';

const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false });
const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false });
const Chapter3 = dynamic(() => import('./chapters/Chapter3'), { ssr: false });
const Chapter4 = dynamic(() => import('./chapters/Chapter4'), { ssr: false });
const Chapter5 = dynamic(() => import('./chapters/Chapter5'), { ssr: false });
const Chapter6 = dynamic(() => import('./chapters/Chapter6'), { ssr: false });
const Chapter7 = dynamic(() => import('./chapters/Chapter7'), { ssr: false });
const Chapter8 = dynamic(() => import('./chapters/Chapter8'), { ssr: false });
const Chapter9 = dynamic(() => import('./chapters/Chapter9'), { ssr: false });
const Chapter10 = dynamic(() => import('./chapters/Chapter10'), { ssr: false });

interface ChapterContentProps {
  chapterId: string;
}

export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const getChapterComponent = () => {
    switch (chapterId) {
      case 'python-basics':
        return <Chapter1 />;
      case 'data-types-collections':
        return <Chapter2 />;
      case 'functions-modules':
        return <Chapter3 />;
      case 'file-io':
        return <Chapter4 />;
      case 'oop-basics':
        return <Chapter5 />;
      case 'exception-handling':
        return <Chapter6 />;
      case 'standard-library':
        return <Chapter7 />;
      case 'decorators-generators':
        return <Chapter8 />;
      case 'async-programming':
        return <Chapter9 />;
      case 'best-practices':
        return <Chapter10 />;
      default:
        return (
          <div className="text-center py-12">
            <p className="text-gray-600 dark:text-gray-400">Chapter not found</p>
          </div>
        );
    }
  };

  return <>{getChapterComponent()}</>;
}
