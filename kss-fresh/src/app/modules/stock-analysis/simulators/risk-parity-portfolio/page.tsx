'use client';

import RiskParityPortfolio from '@/app/modules/stock-analysis/components/simulators/RiskParityPortfolio';

export default function RiskParityPortfolioPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">리스크 패리티 포트폴리오</h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          각 자산의 리스크 기여도를 균등하게 배분하여 안정적이고 효율적인 포트폴리오를 구성하세요
        </p>
        <RiskParityPortfolio />
      </div>
    </div>
  );
}