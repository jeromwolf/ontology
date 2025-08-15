'use client';

import GlobalMacroDashboard from '@/app/modules/stock-analysis/components/simulators/GlobalMacroDashboard';

export default function GlobalMacroDashboardPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">글로벌 매크로 대시보드</h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          전 세계 경제 지표와 시장 동향을 한눈에 파악하고 매크로 투자 전략을 수립하세요
        </p>
        <GlobalMacroDashboard />
      </div>
    </div>
  );
}