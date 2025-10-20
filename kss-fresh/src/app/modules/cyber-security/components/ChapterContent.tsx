import React from 'react';
import dynamic from 'next/dynamic';

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
    switch (chapterId) {
      case 'introduction':
        return <Chapter1 />;
      case 'network-security':
        return <Chapter2 />;
      case 'application-security':
        return <Chapter3 />;
      case 'cloud-security':
        return <Chapter4 />;
      case 'zero-trust':
        return <Chapter5 />;
      case 'penetration-testing':
        return <Chapter6 />;
      case 'incident-response':
        return <Chapter7 />;
      case 'security-operations':
        return <Chapter8 />;
      default:
        return (
          <div className="p-8 text-center">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              챕터를 찾을 수 없습니다
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              요청하신 챕터가 존재하지 않습니다.
            </p>
          </div>
        );
    }
  };

  return <div className="chapter-content">{getChapterComponent()}</div>;
}
