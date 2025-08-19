import dynamic from 'next/dynamic';

const ProTradingChartClient = dynamic(
  () => import('./ProTradingChartClient'),
  { 
    ssr: false,
    loading: () => (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 flex items-center justify-center">
        <div className="text-center text-white">
          <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Professional Trading Chart 로딩 중...</p>
          <p className="text-xs text-gray-500 mt-2">실시간 데이터 연결 준비 중...</p>
        </div>
      </div>
    )
  }
);

export default function ProTradingChartPage() {
  return <ProTradingChartClient />;
}