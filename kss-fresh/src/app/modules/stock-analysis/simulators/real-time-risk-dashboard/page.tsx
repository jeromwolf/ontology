'use client';

import RealTimeRiskDashboard from '@/app/modules/stock-analysis/components/simulators/RealTimeRiskDashboard';

export default function RealTimeRiskDashboardPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">실시간 리스크 대시보드</h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          포트폴리오의 리스크를 실시간으로 모니터링하고 한도 관리를 통해 안정적인 투자를 유지하세요
        </p>
        <RealTimeRiskDashboard />
      </div>
    </div>
  );
}