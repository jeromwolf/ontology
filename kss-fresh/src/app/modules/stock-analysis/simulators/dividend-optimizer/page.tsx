'use client';

import DividendOptimizer from '@/app/modules/stock-analysis/components/simulators/DividendOptimizer';

export default function DividendOptimizerPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">배당 수익률 최적화</h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          다양한 배당 전략으로 안정적인 현금흐름을 창출하고 장기적인 배당 성장을 추구하세요
        </p>
        <DividendOptimizer />
      </div>
    </div>
  );
}