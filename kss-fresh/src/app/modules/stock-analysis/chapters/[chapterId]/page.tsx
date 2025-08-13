'use client';

import React from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import ChapterContent from '../../components/ChapterContent';

export default function ChapterPage() {
  const params = useParams();
  const chapterId = params.chapterId as string;

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-4xl mx-auto px-4 py-8">
        <Link 
          href="/modules/stock-analysis"
          className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors mb-8"
        >
          <ArrowLeft className="w-5 h-5" />
          <span>학습 경로로 돌아가기</span>
        </Link>
        
        <ChapterContent chapterId={chapterId} />
      </div>
    </div>
  );
}