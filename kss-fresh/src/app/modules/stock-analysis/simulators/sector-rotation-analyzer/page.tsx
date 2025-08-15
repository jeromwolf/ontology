'use client';

import SectorRotationAnalyzer from '@/app/modules/stock-analysis/components/simulators/SectorRotationAnalyzer';

export default function SectorRotationAnalyzerPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">글로벌 섹터 로테이션 분석기</h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          경제 사이클에 따른 섹터별 성과를 분석하고 최적의 섹터 로테이션 전략을 수립하세요
        </p>
        <SectorRotationAnalyzer />
      </div>
    </div>
  );
}