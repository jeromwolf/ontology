import { Users } from 'lucide-react'

export default function Section1() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
          <Users className="text-purple-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.1 Multi-Agent RAG의 필요성</h2>
          <p className="text-gray-600 dark:text-gray-400">복잡한 질문을 전문 에이전트들이 협력하여 해결</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
          <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">단일 Agent vs Multi-Agent 접근법</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>복잡한 질문은 종종 여러 도메인의 전문 지식을 필요로 합니다.</strong>
              단일 RAG 시스템으로는 한계가 있는 상황에서, Multi-Agent RAG는 각 도메인에 특화된
              에이전트들이 협력하여 더 정확하고 포괄적인 답변을 제공할 수 있습니다.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              <strong>핵심 아키텍처 요소:</strong>
            </p>
            <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
              <li><strong>Orchestrator Agent</strong>: 질문을 분석하고 적절한 전문 에이전트들에게 배분</li>
              <li><strong>Specialist Agents</strong>: 각 도메인별 전문 지식을 보유한 RAG 시스템</li>
              <li><strong>Synthesis Agent</strong>: 여러 에이전트의 결과를 통합하여 최종 답변 생성</li>
              <li><strong>Quality Validator</strong>: 답변의 일관성과 품질을 검증</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border">
            <h4 className="font-medium text-gray-900 dark:text-white mb-3">실제 활용 사례: 의료 진단 시스템</h4>
            <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded">
              <p className="text-sm text-gray-700 dark:text-gray-300">
                <strong>질문:</strong> "40세 남성 환자가 가슴 통증과 호흡곤란을 호소합니다.
                최근 장거리 비행 후 발생했으며, 가족력상 심장병이 있습니다."
              </p>
            </div>
            <div className="mt-3 grid md:grid-cols-3 gap-3 text-xs">
              <div className="bg-blue-50 dark:bg-blue-900/30 p-2 rounded">
                <strong>심장내과 Agent</strong><br/>
                심근경색, 협심증 가능성 분석
              </div>
              <div className="bg-green-50 dark:bg-green-900/30 p-2 rounded">
                <strong>호흡기내과 Agent</strong><br/>
                폐색전증, 기흉 가능성 검토
              </div>
              <div className="bg-orange-50 dark:bg-orange-900/30 p-2 rounded">
                <strong>응급의학 Agent</strong><br/>
                중증도 판단 및 응급 처치 가이드
              </div>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">Multi-Agent RAG의 장점</h3>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-green-600 dark:text-green-400 mb-2">✅ 성능 향상</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 도메인별 최적화된 검색 성능</li>
                <li>• 전문 지식의 깊이 증가</li>
                <li>• 교차 검증을 통한 정확도 향상</li>
                <li>• 편향 감소 (다양한 관점)</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-blue-600 dark:text-blue-400 mb-2">⚡ 확장성</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 새로운 도메인 에이전트 쉽게 추가</li>
                <li>• 병렬 처리로 응답 속도 최적화</li>
                <li>• 모듈화된 유지보수</li>
                <li>• 독립적 에이전트 업데이트 가능</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
