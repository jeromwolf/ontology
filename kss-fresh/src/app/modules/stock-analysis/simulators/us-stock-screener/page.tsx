'use client';

import USStockScreener from '@/app/modules/stock-analysis/components/simulators/USStockScreener';

export default function USStockScreenerPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">미국 주식 스크리너</h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          100개 이상의 조건으로 미국 전체 상장 종목을 필터링하고 투자 기회를 발굴하세요
        </p>
        <USStockScreener />
      </div>
    </div>
  );
}