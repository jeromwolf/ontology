'use client';

import TradingCostCalculator from '@/app/modules/stock-analysis/components/simulators/TradingCostCalculator';

export default function TradingCostCalculatorPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">거래 비용 상세 모델링</h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          슬리피지, 시장충격, 세금, 수수료 등 모든 거래 비용을 정확히 계산하고 최적화 전략을 수립하세요
        </p>
        <TradingCostCalculator />
      </div>
    </div>
  );
}