'use client';

import ETFComparator from '@/app/modules/stock-analysis/components/simulators/ETFComparator';

export default function ETFComparatorPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">ETF 비교 분석기</h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          다양한 ETF의 성과, 리스크, 비용을 종합적으로 비교하고 최적의 포트폴리오를 구성하세요
        </p>
        <ETFComparator />
      </div>
    </div>
  );
}