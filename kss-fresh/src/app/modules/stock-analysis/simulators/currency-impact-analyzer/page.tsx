'use client';

import CurrencyImpactAnalyzer from '@/app/modules/stock-analysis/components/simulators/CurrencyImpactAnalyzer';

export default function CurrencyImpactAnalyzerPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">환율 영향 분석기</h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          해외 주식 투자 시 환율 변동이 수익률에 미치는 영향을 분석하고, 환헤지 전략을 시뮬레이션하세요
        </p>
        <CurrencyImpactAnalyzer />
      </div>
    </div>
  );
}