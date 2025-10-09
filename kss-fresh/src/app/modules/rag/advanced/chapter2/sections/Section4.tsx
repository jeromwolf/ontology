import { Zap } from 'lucide-react'

export default function Section4() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-orange-100 dark:bg-orange-900/20 flex items-center justify-center">
          <Zap className="text-orange-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.4 Multi-Agent 성능 최적화</h2>
          <p className="text-gray-600 dark:text-gray-400">대규모 멀티 에이전트 시스템의 효율성 극대화</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl border border-orange-200 dark:border-orange-700">
          <h3 className="font-bold text-orange-800 dark:text-orange-200 mb-4">핵심 최적화 전략</h3>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-blue-600 dark:text-blue-400 mb-3">🚀 병렬 처리 최적화</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 에이전트별 독립적 리소스 할당</li>
                <li>• 비동기 메시지 처리</li>
                <li>• 로드 밸런싱 구현</li>
                <li>• 작업 큐 관리</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-green-600 dark:text-green-400 mb-3">💾 메모리 관리</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 컨텍스트 윈도우 최적화</li>
                <li>• 메시지 히스토리 압축</li>
                <li>• 에이전트별 메모리 풀</li>
                <li>• 가비지 컬렉션 튜닝</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-purple-600 dark:text-purple-400 mb-3">⚡ 통신 최적화</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 메시지 압축</li>
                <li>• 배치 처리</li>
                <li>• 우선순위 큐</li>
                <li>• 연결 풀링</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">성능 벤치마크 결과</h3>

          <div className="overflow-x-auto mb-4">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-blue-300 dark:border-blue-600">
                  <th className="text-left py-2 text-blue-800 dark:text-blue-200">시스템 구성</th>
                  <th className="text-left py-2 text-blue-800 dark:text-blue-200">응답 시간</th>
                  <th className="text-left py-2 text-blue-800 dark:text-blue-200">정확도</th>
                  <th className="text-left py-2 text-blue-800 dark:text-blue-200">처리량 (QPS)</th>
                </tr>
              </thead>
              <tbody className="text-blue-700 dark:text-blue-300">
                <tr className="border-b border-blue-200 dark:border-blue-700">
                  <td className="py-2">Single RAG Agent</td>
                  <td className="py-2">2.1초</td>
                  <td className="py-2">78%</td>
                  <td className="py-2">45</td>
                </tr>
                <tr className="border-b border-blue-200 dark:border-blue-700">
                  <td className="py-2">3-Agent System</td>
                  <td className="py-2">3.8초</td>
                  <td className="py-2">89%</td>
                  <td className="py-2">32</td>
                </tr>
                <tr className="border-b border-blue-200 dark:border-blue-700">
                  <td className="py-2">5-Agent System</td>
                  <td className="py-2">4.2초</td>
                  <td className="py-2">94%</td>
                  <td className="py-2">28</td>
                </tr>
                <tr>
                  <td className="py-2">Optimized 5-Agent</td>
                  <td className="py-2">2.9초</td>
                  <td className="py-2">93%</td>
                  <td className="py-2">38</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="p-3 bg-blue-100 dark:bg-blue-900/40 rounded">
            <p className="text-xs text-blue-800 dark:text-blue-200">
              💡 <strong>핵심 인사이트:</strong> 적절한 최적화를 통해 Multi-Agent 시스템은
              높은 정확도를 유지하면서도 단일 에이전트 대비 합리적인 성능을 달성할 수 있습니다.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
