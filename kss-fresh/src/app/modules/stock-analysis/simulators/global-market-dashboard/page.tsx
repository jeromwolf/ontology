'use client';

import GlobalMarketDashboard from '@/app/modules/stock-analysis/components/simulators/GlobalMarketDashboard';

export default function GlobalMarketDashboardPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">글로벌 실시간 대시보드</h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          전 세계 주요 시장의 실시간 현황을 한눈에 확인하고, 환율 변동과 거래 시간을 모니터링하세요
        </p>
        <GlobalMarketDashboard />
      </div>
    </div>
  );
}