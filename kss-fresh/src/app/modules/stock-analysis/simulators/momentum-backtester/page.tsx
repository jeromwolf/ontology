'use client';

import MomentumBacktester from '@/app/modules/stock-analysis/components/simulators/MomentumBacktester';

export default function MomentumBacktesterPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">모멘텀 전략 백테스터</h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          다양한 모멘텀 지표를 활용하여 추세 추종 전략을 백테스트하고 최적화하세요
        </p>
        <MomentumBacktester />
      </div>
    </div>
  );
}