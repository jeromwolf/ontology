'use client';

import PairTradingAnalyzer from '@/app/modules/stock-analysis/components/simulators/PairTradingAnalyzer';

export default function PairTradingAnalyzerPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">페어 트레이딩 분석기</h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          상관관계가 높은 주식 쌍을 찾아 통계적 차익거래 기회를 포착하고 백테스트하세요
        </p>
        <PairTradingAnalyzer />
      </div>
    </div>
  );
}