'use client';

import OptionsStrategySimulator from '@/app/modules/stock-analysis/components/simulators/OptionsStrategySimulator';

export default function OptionsStrategySimulatorPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">옵션 전략 시뮬레이터</h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          다양한 옵션 전략의 손익 구조를 시각화하고 Greeks를 분석하여 최적의 옵션 전략을 수립하세요
        </p>
        <OptionsStrategySimulator />
      </div>
    </div>
  );
}