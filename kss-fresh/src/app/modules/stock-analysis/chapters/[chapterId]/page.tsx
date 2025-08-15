'use client';

import React from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import ChapterContent from '../../components/ChapterContent';
import ChapterNavigation from '../../components/ChapterNavigation';

export default function ChapterPage() {
  const params = useParams();
  const chapterId = params.chapterId as string;

  // Baby Chick 챕터들
  const babyChickChapterIds = [
    'what-is-stock', 'open-account', 'trading-app-basics',
    'understanding-candles', 'volume-basics', 'order-book',
    'basic-terms', 'reading-news', 'sectors-themes',
    'stock-selection', 'small-start', 'trading-diary'
  ];

  // Foundation 챕터들
  const foundationChapterIds = [
    'chart-basics', 'technical-indicators', 'pattern-recognition',
    'financial-statements', 'valuation-basics', 'industry-analysis',
    'investment-strategies', 'portfolio-basics', 'risk-control',
    'market-timing', 'real-trading', 'investment-plan'
  ];

  // Advanced 챕터들
  const advancedChapterIds = [
    'advanced-technical-analysis', 'system-trading-basics', 'automated-strategies',
    'quantitative-basics', 'financial-data-analysis', 'factor-models',
    'derivatives-basics', 'advanced-options', 'hedging-strategies',
    'global-markets', 'alternative-investments', 'macro-trading'
  ];

  // 현재 챕터가 어느 프로그램에 속하는지 판단
  const getProgramType = () => {
    if (babyChickChapterIds.includes(chapterId)) return 'baby-chick';
    if (foundationChapterIds.includes(chapterId)) return 'foundation';
    if (advancedChapterIds.includes(chapterId)) return 'advanced';
    return 'foundation'; // 기본값
  };

  const programType = getProgramType();
  
  // 프로그램별 돌아가기 링크
  const getBackLink = () => {
    switch (programType) {
      case 'baby-chick':
        return '/modules/stock-analysis/stages/baby-chick';
      case 'foundation':
        return '/modules/stock-analysis/stages/foundation';
      case 'advanced':
        return '/modules/stock-analysis/stages/advanced';
      default:
        return '/modules/stock-analysis';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-4xl mx-auto px-4 py-8">
        <Link 
          href={getBackLink()}
          className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors mb-8"
        >
          <ArrowLeft className="w-5 h-5" />
          <span>프로그램으로 돌아가기</span>
        </Link>
        
        <ChapterContent chapterId={chapterId} />
      </div>
      
      {/* Chapter Navigation */}
      <ChapterNavigation 
        currentChapterId={chapterId} 
        programType={programType as 'baby-chick' | 'foundation' | 'advanced' | 'professional'} 
      />
    </div>
  );
}