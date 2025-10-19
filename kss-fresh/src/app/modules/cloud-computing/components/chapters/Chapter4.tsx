'use client';

// Chapter 4: Google Cloud Platform
export default function Chapter4() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">Google Cloud Platform (GCP) 소개</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          Google의 클라우드 플랫폼으로, 데이터 분석과 머신러닝 분야에서 강력한 서비스를 제공합니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">GCP 핵심 서비스</h2>
        <div className="grid gap-4">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-red-800 dark:text-red-200 mb-2">Compute Engine</h3>
            <p className="text-gray-700 dark:text-gray-300">가상 머신 인스턴스</p>
          </div>
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">BigQuery</h3>
            <p className="text-gray-700 dark:text-gray-300">페타바이트급 데이터 분석</p>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-green-800 dark:text-green-200 mb-2">Cloud Run</h3>
            <p className="text-gray-700 dark:text-gray-300">컨테이너 기반 서버리스</p>
          </div>
        </div>
      </section>

      <section className="bg-gradient-to-r from-red-50 to-yellow-50 dark:from-red-900/20 dark:to-yellow-900/20 rounded-xl p-6 mt-8">
        <h2 className="text-xl font-bold mb-4 text-red-800 dark:text-red-200">📚 핵심 정리</h2>
        <ul className="space-y-2">
          <li className="flex items-start gap-2">
            <span className="text-red-600 dark:text-red-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">Compute Engine, BigQuery, Cloud Run</span>
          </li>
        </ul>
      </section>
    </div>
  )
}
