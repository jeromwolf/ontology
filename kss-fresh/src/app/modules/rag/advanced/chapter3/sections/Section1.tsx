import { Globe } from 'lucide-react'

export default function Section1() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-indigo-100 dark:bg-indigo-900/20 flex items-center justify-center">
          <Globe className="text-indigo-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">3.1 엔터프라이즈급 분산 RAG의 도전과제</h2>
          <p className="text-gray-600 dark:text-gray-400">단일 서버의 한계를 넘어서는 대규모 시스템</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-xl border border-indigo-200 dark:border-indigo-700">
          <h3 className="font-bold text-indigo-800 dark:text-indigo-200 mb-4">단일 노드 RAG의 물리적 한계</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>실제 Netflix의 콘텐츠 추천 시스템은 다음과 같은 규모를 처리합니다:</strong>
            </p>
            <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
              <li><strong>2억+ 사용자</strong>: 실시간 개인화 추천 요구</li>
              <li><strong>수십억 개의 콘텐츠 메타데이터</strong>: 영화, 시리즈, 자막, 리뷰</li>
              <li><strong>초당 10만+ 쿼리</strong>: 피크 시간대 동시 접속</li>
              <li><strong>99.99% 가용성 요구</strong>: 연간 52분 이하 다운타임</li>
              <li><strong>&lt; 100ms 응답 시간</strong>: 사용자 경험을 위한 엄격한 레이턴시 요구</li>
            </ul>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-red-600 dark:text-red-400 mb-2">❌ 단일 노드 한계</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• RAM 용량 한계 (최대 수TB)</li>
                <li>• CPU/GPU 처리 능력 제한</li>
                <li>• 네트워크 대역폭 병목</li>
                <li>• 장애 시 전체 시스템 마비</li>
                <li>• 수직 확장의 비용 급증</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-green-600 dark:text-green-400 mb-2">✅ 분산 시스템 장점</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 무제한 수평 확장 가능</li>
                <li>• 부하 분산으로 성능 향상</li>
                <li>• 부분 장애 허용 (Fault Tolerance)</li>
                <li>• 지역별 데이터 로컬리티</li>
                <li>• 비용 효율적 확장</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">실제 사례: Uber의 분산 검색 시스템</h3>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border mb-4">
            <h4 className="font-medium text-gray-900 dark:text-white mb-3">Uber의 도전과제와 해결책</h4>
            <div className="grid md:grid-cols-3 gap-4 text-center">
              <div className="bg-blue-100 dark:bg-blue-900/30 p-3 rounded">
                <p className="text-2xl font-bold text-blue-600">40억+</p>
                <p className="text-xs text-blue-700 dark:text-blue-300">일일 검색 쿼리</p>
              </div>
              <div className="bg-green-100 dark:bg-green-900/30 p-3 rounded">
                <p className="text-2xl font-bold text-green-600">&lt;50ms</p>
                <p className="text-xs text-green-700 dark:text-green-300">P99 레이턴시</p>
              </div>
              <div className="bg-purple-100 dark:bg-purple-900/30 p-3 rounded">
                <p className="text-2xl font-bold text-purple-600">99.95%</p>
                <p className="text-xs text-purple-700 dark:text-purple-300">가용성 SLA</p>
              </div>
            </div>
          </div>

          <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded">
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong>아키텍처 핵심:</strong> Uber는 도시별로 분산된 검색 클러스터를 운영하며,
              각 클러스터는 해당 지역의 드라이버, 음식점, 경로 데이터를 처리합니다.
              글로벌 라우터가 사용자 위치에 따라 적절한 클러스터로 쿼리를 전달합니다.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
