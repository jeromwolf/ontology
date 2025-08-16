'use client';

import StressTestScenarios from '@/app/modules/stock-analysis/components/simulators/StressTestScenarios';

export default function StressTestScenariosPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">스트레스 테스트 시나리오</h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          역사적 금융위기 시나리오로 포트폴리오의 취약점을 분석하고 헤징 전략을 수립하세요
        </p>
        <StressTestScenarios />
      </div>
    </div>
  );
}