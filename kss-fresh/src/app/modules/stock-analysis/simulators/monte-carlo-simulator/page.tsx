'use client';

import MonteCarloSimulator from '@/app/modules/stock-analysis/components/simulators/MonteCarloSimulator';

export default function MonteCarloSimulatorPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">몬테카를로 시뮬레이션</h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          10,000회 이상의 시뮬레이션으로 포트폴리오의 미래 가치를 확률적으로 분석하고 리스크를 측정하세요
        </p>
        <MonteCarloSimulator />
      </div>
    </div>
  );
}