import { Network } from 'lucide-react'

export default function Section1() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
          <Network className="text-blue-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.1 Microsoft GraphRAG의 혁신</h2>
          <p className="text-gray-600 dark:text-gray-400">전통적 RAG의 한계를 극복한 그래프 기반 검색</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">기존 RAG vs GraphRAG 비교</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>Microsoft Research에서 2024년 발표한 GraphRAG는 기존 RAG의 근본적 한계를 해결합니다.</strong>
              전통적 벡터 검색은 지역적 유사성에만 의존하지만, GraphRAG는 글로벌 지식 구조를 파악하여
              복잡한 질문에 대해 더 포괄적인 답변을 제공할 수 있습니다.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              <strong>핵심 혁신:</strong>
            </p>
            <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
              <li><strong>Community Detection</strong>: 문서 내 엔티티들을 의미론적 클러스터로 그룹화</li>
              <li><strong>Hierarchical Summarization</strong>: 각 커뮤니티의 계층적 요약 생성</li>
              <li><strong>Global Query Processing</strong>: 전체 지식 그래프를 활용한 추론</li>
              <li><strong>Multi-perspective Reasoning</strong>: 다양한 관점에서의 종합적 분석</li>
            </ul>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-red-600 dark:text-red-400 mb-2">❌ 기존 RAG 한계</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 지역적 검색에만 의존</li>
                <li>• 문서 간 연결성 무시</li>
                <li>• 복잡한 질문에 대한 불완전한 답변</li>
                <li>• 전체적 맥락 파악 어려움</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-green-600 dark:text-green-400 mb-2">✅ GraphRAG 장점</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 글로벌 지식 구조 활용</li>
                <li>• 엔티티 관계 기반 추론</li>
                <li>• 포괄적이고 다각적 답변</li>
                <li>• 계층적 정보 구조화</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
          <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">실제 성능 비교 연구</h3>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border mb-4">
            <h4 className="font-medium text-gray-900 dark:text-white mb-3">Microsoft 연구 결과 (2024)</h4>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div className="bg-green-100 dark:bg-green-900/30 p-3 rounded">
                <p className="text-2xl font-bold text-green-600">41%</p>
                <p className="text-xs text-green-700 dark:text-green-300">답변 포괄성 향상</p>
              </div>
              <div className="bg-blue-100 dark:bg-blue-900/30 p-3 rounded">
                <p className="text-2xl font-bold text-blue-600">32%</p>
                <p className="text-xs text-blue-700 dark:text-blue-300">다각적 관점 증가</p>
              </div>
              <div className="bg-purple-100 dark:bg-purple-900/30 p-3 rounded">
                <p className="text-2xl font-bold text-purple-600">67%</p>
                <p className="text-xs text-purple-700 dark:text-purple-300">복잡 질문 해결률</p>
              </div>
            </div>
          </div>

          <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg">
            <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-2">💡 테스트 도메인</p>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              팟캐스트 전사본, 뉴스 기사, 연구 논문 등 다양한 텍스트 도메인에서
              "이 주제에 대한 주요 관점들은 무엇인가?", "핵심 이해관계자들 간의 관계는?"
              등의 복잡한 질문에서 GraphRAG가 일관되게 우수한 성능을 보임
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
