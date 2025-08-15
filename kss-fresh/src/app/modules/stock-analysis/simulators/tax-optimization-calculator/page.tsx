'use client';

import TaxOptimizationCalculator from '@/app/modules/stock-analysis/components/simulators/TaxOptimizationCalculator';

export default function TaxOptimizationCalculatorPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">세금 최적화 계산기</h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          미국과 한국 주식 투자의 세금을 계산하고 절세 전략을 수립하세요
        </p>
        <TaxOptimizationCalculator />
      </div>
    </div>
  );
}