import { GitBranch } from 'lucide-react'

export default function Section2() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
          <GitBranch className="text-purple-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.2 GraphRAG 아키텍처 분석</h2>
          <p className="text-gray-600 dark:text-gray-400">인덱싱부터 쿼리 처리까지의 완전한 파이프라인</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
          <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">GraphRAG 파이프라인 구조</h3>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border">
            <pre className="text-xs overflow-x-auto">
{`┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Documents │ →  │ Entity Extraction│ →  │ Relationship    │
│   (PDF, Text)   │    │ (LLM-based NER)  │    │ Identification  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ↓                       ↓                       ↓
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Knowledge Graph │ ←  │ Community       │ ←  │ Graph           │
│ Construction    │    │ Detection       │    │ Construction    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ↓
┌─────────────────┐    ┌─────────────────┐
│ Hierarchical    │ →  │ Community       │
│ Clustering      │    │ Summarization   │
└─────────────────┘    └─────────────────┘`}
            </pre>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-xl">
            <h4 className="font-bold text-gray-900 dark:text-white mb-4">📝 인덱싱 단계 (Offline)</h4>
            <ol className="space-y-3 text-sm text-gray-700 dark:text-gray-300">
              <li><strong>1. 엔티티 추출</strong><br/>LLM을 사용한 named entity recognition</li>
              <li><strong>2. 관계 식별</strong><br/>엔티티 간 의미적 관계 파악</li>
              <li><strong>3. 그래프 구축</strong><br/>엔티티-관계 그래프 생성</li>
              <li><strong>4. 커뮤니티 탐지</strong><br/>Leiden 알고리즘으로 클러스터링</li>
              <li><strong>5. 계층적 요약</strong><br/>각 커뮤니티의 LLM 기반 요약</li>
            </ol>
          </div>

          <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-xl">
            <h4 className="font-bold text-gray-900 dark:text-white mb-4">🔍 쿼리 단계 (Online)</h4>
            <ol className="space-y-3 text-sm text-gray-700 dark:text-gray-300">
              <li><strong>1. 질문 분석</strong><br/>글로벌 vs 로컬 질문 분류</li>
              <li><strong>2. 관련 커뮤니티 검색</strong><br/>질문과 매칭되는 커뮤니티 탐색</li>
              <li><strong>3. 컨텍스트 구성</strong><br/>커뮤니티 요약 + 관련 엔티티</li>
              <li><strong>4. 답변 생성</strong><br/>LLM 기반 종합 답변 생성</li>
              <li><strong>5. 출처 추적</strong><br/>답변 근거 문서 매핑</li>
            </ol>
          </div>
        </div>
      </div>
    </section>
  )
}
