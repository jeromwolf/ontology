'use client';

// Chapter 5: Answer Generation
export default function Chapter5() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">효과적인 프롬프트 설계</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          RAG 시스템에서 LLM에게 검색된 컨텍스트를 효과적으로 전달하는 것이 중요합니다.
          프롬프트 설계는 답변의 품질을 크게 좌우합니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">RAG 프롬프트 템플릿</h2>
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm">
          <pre className="whitespace-pre-wrap">
{`시스템: 당신은 주어진 컨텍스트를 바탕으로 질문에 답변하는 AI 어시스턴트입니다.

컨텍스트:
{context}

질문: {query}

지침:
1. 컨텍스트에 있는 정보만을 사용하여 답변하세요.
2. 확실하지 않은 경우 "주어진 정보로는 답변할 수 없습니다"라고 말하세요.
3. 답변에 사용한 정보의 출처를 명시하세요.

답변:`}
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">컨텍스트 관리 전략</h2>
        <div className="grid gap-4">
          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-2">
              1. 컨텍스트 순서 최적화
            </h3>
            <p className="text-gray-700 dark:text-gray-300">
              가장 관련성 높은 문서를 앞쪽에 배치합니다. LLM은 프롬프트의 시작과 끝 부분에 더 주의를 기울입니다.
            </p>
          </div>
          
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">
              2. 메타데이터 활용
            </h3>
            <p className="text-gray-700 dark:text-gray-300">
              문서의 출처, 날짜, 저자 등 메타데이터를 포함하여 신뢰성을 높입니다.
            </p>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">
              3. 컨텍스트 압축
            </h3>
            <p className="text-gray-700 dark:text-gray-300">
              토큰 제한을 고려하여 핵심 정보만 추출하거나 요약하여 전달합니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">답변 품질 향상 기법</h2>
        <ul className="space-y-3 text-gray-700 dark:text-gray-300">
          <li className="flex items-start gap-2">
            <span className="text-emerald-600 dark:text-emerald-400">•</span>
            <div>
              <strong>Chain-of-Thought</strong>: 단계별 추론 과정을 포함하도록 유도
            </div>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-emerald-600 dark:text-emerald-400">•</span>
            <div>
              <strong>Self-Consistency</strong>: 여러 번 생성하여 일관된 답변 선택
            </div>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-emerald-600 dark:text-emerald-400">•</span>
            <div>
              <strong>Citation</strong>: 답변에 사용된 소스를 명확히 인용
            </div>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-emerald-600 dark:text-emerald-400">•</span>
            <div>
              <strong>Confidence Score</strong>: 답변의 확신도를 함께 제공
            </div>
          </li>
        </ul>
      </section>
    </div>
  )
}