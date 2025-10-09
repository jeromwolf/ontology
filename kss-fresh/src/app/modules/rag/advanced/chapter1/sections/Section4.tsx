import { Zap } from 'lucide-react'

export default function Section4() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-orange-100 dark:bg-orange-900/20 flex items-center justify-center">
          <Zap className="text-orange-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.4 GraphRAG 성능 최적화</h2>
          <p className="text-gray-600 dark:text-gray-400">대규모 지식 그래프를 위한 확장성 전략</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl border border-orange-200 dark:border-orange-700">
          <h3 className="font-bold text-orange-800 dark:text-orange-200 mb-4">핵심 최적화 전략</h3>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-3">🚀 인덱싱 최적화</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <li><strong>병렬 처리</strong><br/>문서별 엔티티 추출 병렬화</li>
                <li><strong>배치 처리</strong><br/>LLM API 호출 최적화</li>
                <li><strong>캐싱 전략</strong><br/>추출 결과 Redis 캐싱</li>
                <li><strong>점진적 업데이트</strong><br/>새 문서만 처리하여 그래프 확장</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-3">⚡ 쿼리 최적화</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <li><strong>커뮤니티 인덱싱</strong><br/>벡터 검색을 위한 커뮤니티 임베딩</li>
                <li><strong>계층적 검색</strong><br/>상위 레벨부터 점진적 탐색</li>
                <li><strong>결과 캐싱</strong><br/>유사 질문에 대한 답변 재사용</li>
                <li><strong>컨텍스트 압축</strong><br/>긴 커뮤니티 요약 압축</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-xl border border-yellow-200 dark:border-yellow-700">
          <h3 className="font-bold text-yellow-800 dark:text-yellow-200 mb-4">💰 비용 최적화</h3>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border mb-4">
            <h4 className="font-medium text-gray-900 dark:text-white mb-3">LLM API 사용량 관리</h4>

            <div className="grid grid-cols-3 gap-4 text-center mb-4">
              <div className="bg-blue-100 dark:bg-blue-900/30 p-3 rounded">
                <p className="text-lg font-bold text-blue-600">$0.03</p>
                <p className="text-xs text-blue-700 dark:text-blue-300">문서당 평균 비용</p>
              </div>
              <div className="bg-green-100 dark:bg-green-900/30 p-3 rounded">
                <p className="text-lg font-bold text-green-600">70%</p>
                <p className="text-xs text-green-700 dark:text-green-300">캐싱으로 절약</p>
              </div>
              <div className="bg-purple-100 dark:bg-purple-900/30 p-3 rounded">
                <p className="text-lg font-bold text-purple-600">5:1</p>
                <p className="text-xs text-purple-700 dark:text-purple-300">배치 처리 효율</p>
              </div>
            </div>

            <div className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <p><strong>전략 1:</strong> 엔티티 추출을 위해 더 저렴한 모델(GPT-3.5) 사용</p>
              <p><strong>전략 2:</strong> 커뮤니티 요약만 고급 모델(GPT-4) 사용</p>
              <p><strong>전략 3:</strong> 반복적 추출 결과 캐싱으로 중복 호출 방지</p>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">📊 실제 성능 벤치마크</h3>

          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-blue-300 dark:border-blue-600">
                  <th className="text-left py-2 text-blue-800 dark:text-blue-200">지표</th>
                  <th className="text-left py-2 text-blue-800 dark:text-blue-200">기존 RAG</th>
                  <th className="text-left py-2 text-blue-800 dark:text-blue-200">GraphRAG</th>
                  <th className="text-left py-2 text-blue-800 dark:text-blue-200">개선율</th>
                </tr>
              </thead>
              <tbody className="text-blue-700 dark:text-blue-300">
                <tr className="border-b border-blue-200 dark:border-blue-700">
                  <td className="py-2">복잡 질문 정확도</td>
                  <td className="py-2">64%</td>
                  <td className="py-2">87%</td>
                  <td className="py-2 text-green-600 font-bold">+36%</td>
                </tr>
                <tr className="border-b border-blue-200 dark:border-blue-700">
                  <td className="py-2">답변 포괄성</td>
                  <td className="py-2">2.1/5</td>
                  <td className="py-2">4.3/5</td>
                  <td className="py-2 text-green-600 font-bold">+105%</td>
                </tr>
                <tr className="border-b border-blue-200 dark:border-blue-700">
                  <td className="py-2">응답 시간</td>
                  <td className="py-2">1.2초</td>
                  <td className="py-2">2.8초</td>
                  <td className="py-2 text-red-600">+133%</td>
                </tr>
                <tr>
                  <td className="py-2">인덱싱 시간</td>
                  <td className="py-2">5분</td>
                  <td className="py-2">45분</td>
                  <td className="py-2 text-red-600">+800%</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="mt-4 p-3 bg-blue-100 dark:bg-blue-900/40 rounded">
            <p className="text-xs text-blue-800 dark:text-blue-200">
              💡 <strong>성능 트레이드오프:</strong> GraphRAG는 더 높은 품질의 답변을 제공하지만,
              초기 인덱싱과 쿼리 처리 시간이 증가합니다.
              복잡한 분석이 필요한 도메인에서 특히 유용합니다.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
