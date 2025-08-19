import dynamic from 'next/dynamic';

const KRStockChartClient = dynamic(
  () => import('./KRStockChartClient'),
  { 
    ssr: false,
    loading: () => (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 flex items-center justify-center">
        <div className="text-center text-white">
          <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-400">한국 주식 차트 로딩 중...</p>
        </div>
      </div>
    )
  }
);

export default function KRStockChartPage() {
  return <KRStockChartClient />;
}