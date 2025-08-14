'use client';

import dynamic from 'next/dynamic';
import Link from 'next/link';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { aiSecurityMetadata } from '../metadata';

// Dynamic imports for all chapters
const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false });
const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false });
const Chapter3 = dynamic(() => import('./chapters/Chapter3'), { ssr: false });
const Chapter4 = dynamic(() => import('./chapters/Chapter4'), { ssr: false });
const Chapter5 = dynamic(() => import('./chapters/Chapter5'), { ssr: false });
const Chapter6 = dynamic(() => import('./chapters/Chapter6'), { ssr: false });
const Chapter7 = dynamic(() => import('./chapters/Chapter7'), { ssr: false });
const Chapter8 = dynamic(() => import('./chapters/Chapter8'), { ssr: false });

interface Props {
  chapterId: string;
}

export default function ChapterContent({ chapterId }: Props) {
  const chapterIndex = aiSecurityMetadata.chapters.findIndex(ch => ch.id === chapterId);
  const chapter = aiSecurityMetadata.chapters[chapterIndex];
  const prevChapter = chapterIndex > 0 ? aiSecurityMetadata.chapters[chapterIndex - 1] : null;
  const nextChapter = chapterIndex < aiSecurityMetadata.chapters.length - 1 ? aiSecurityMetadata.chapters[chapterIndex + 1] : null;

  const getChapterComponent = () => {
    switch (chapterId) {
      case 'fundamentals':
        return <Chapter1 />;
      case 'adversarial-attacks':
        return <Chapter2 />;
      case 'model-security':
        return <Chapter3 />;
      case 'privacy-preserving':
        return <Chapter4 />;
      case 'robustness':
        return <Chapter5 />;
      case 'security-testing':
        return <Chapter6 />;
      case 'deployment-security':
        return <Chapter7 />;
      case 'case-studies':
        return <Chapter8 />;
      default:
        return <div>Chapter not found</div>;
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <Link
          href="/modules/ai-security"
          className="inline-flex items-center text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 mb-4"
        >
          <ChevronLeft className="w-4 h-4 mr-1" />
          목차로 돌아가기
        </Link>
        
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          {chapter.title}
        </h1>
        <p className="text-gray-600 dark:text-gray-300">
          {chapter.description}
        </p>
      </div>

      {getChapterComponent()}

      <div className="mt-12 flex justify-between">
        {prevChapter && (
          <Link
            href={`/modules/ai-security/${prevChapter.id}`}
            className="inline-flex items-center px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700"
          >
            <ChevronLeft className="w-4 h-4 mr-1" />
            이전: {prevChapter.title}
          </Link>
        )}
        
        {nextChapter && (
          <Link
            href={`/modules/ai-security/${nextChapter.id}`}
            className="inline-flex items-center px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 ml-auto"
          >
            다음: {nextChapter.title}
            <ChevronRight className="w-4 h-4 ml-1" />
          </Link>
        )}
      </div>
    </div>
  );
}