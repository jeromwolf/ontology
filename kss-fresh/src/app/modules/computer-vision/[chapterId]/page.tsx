'use client';

import { useParams } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, BookOpen, Clock, Target, CheckCircle } from 'lucide-react';
import ChapterContent from '../components/ChapterContent';
import { moduleMetadata } from '../metadata';

export default function ChapterPage() {
  const params = useParams();
  const chapterId = params.chapterId as string;
  
  const chapter = moduleMetadata.chapters.find(ch => ch.id === chapterId);
  const chapterIndex = moduleMetadata.chapters.findIndex(ch => ch.id === chapterId);
  
  if (!chapter) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500 dark:text-gray-400 mb-4">챕터를 찾을 수 없습니다.</p>
        <Link
          href="/modules/computer-vision"
          className="text-teal-600 dark:text-teal-400 hover:underline"
        >
          모듈 홈으로 돌아가기
        </Link>
      </div>
    );
  }

  const nextChapter = moduleMetadata.chapters[chapterIndex + 1];
  const prevChapter = moduleMetadata.chapters[chapterIndex - 1];

  return (
    <div className="max-w-5xl mx-auto">
      {/* Chapter Header */}
      <div className="mb-8">
        <div className="flex items-center gap-4 mb-4">
          <div className="flex-shrink-0 w-12 h-12 bg-teal-100 dark:bg-teal-900/30 rounded-lg flex items-center justify-center text-teal-600 dark:text-teal-400 font-bold text-lg">
            {chapterIndex + 1}
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
              {chapter.title}
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              {chapter.description}
            </p>
          </div>
        </div>

        {/* Learning Objectives */}
        <div className="bg-teal-50 dark:bg-teal-900/20 border border-teal-200 dark:border-teal-800 rounded-lg p-6 mb-6">
          <h2 className="flex items-center gap-2 text-lg font-semibold text-teal-900 dark:text-teal-100 mb-3">
            <Target className="w-5 h-5" />
            학습 목표
          </h2>
          <ul className="space-y-2">
            {chapter.topics.map((topic, index) => (
              <li key={index} className="flex items-start gap-2 text-teal-800 dark:text-teal-200">
                <span className="text-teal-600 dark:text-teal-400 mt-0.5">•</span>
                <span>{topic}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Estimated Time */}
        <div className="flex items-center gap-6 text-sm text-gray-600 dark:text-gray-400 mb-6">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4" />
            <span>예상 학습 시간: 30-45분</span>
          </div>
          <div className="flex items-center gap-2">
            <BookOpen className="w-4 h-4" />
            <span>난이도: {chapterIndex < 3 ? '초급' : chapterIndex < 6 ? '중급' : '고급'}</span>
          </div>
        </div>
      </div>

      {/* Chapter Content */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <ChapterContent chapterId={chapterId} />
      </div>

      {/* Navigation */}
      <div className="flex items-center justify-between mt-8 pt-8 border-t border-gray-200 dark:border-gray-700">
        {prevChapter ? (
          <Link
            href={`/modules/computer-vision/${prevChapter.id}`}
            className="flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-teal-600 dark:hover:text-teal-400 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            <div className="text-left">
              <div className="text-sm">이전 챕터</div>
              <div className="font-medium">{prevChapter.title}</div>
            </div>
          </Link>
        ) : (
          <div />
        )}

        {nextChapter ? (
          <Link
            href={`/modules/computer-vision/${nextChapter.id}`}
            className="flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-teal-600 dark:hover:text-teal-400 transition-colors text-right"
          >
            <div>
              <div className="text-sm">다음 챕터</div>
              <div className="font-medium">{nextChapter.title}</div>
            </div>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        ) : (
          <Link
            href="/modules/computer-vision"
            className="flex items-center gap-2 text-teal-600 dark:text-teal-400 hover:text-teal-700 dark:hover:text-teal-300 transition-colors font-medium"
          >
            <CheckCircle className="w-4 h-4" />
            학습 완료
          </Link>
        )}
      </div>
    </div>
  );
}