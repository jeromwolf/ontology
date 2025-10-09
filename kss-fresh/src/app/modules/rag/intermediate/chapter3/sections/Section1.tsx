import { Brain } from 'lucide-react'

export default function Section1() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
          <Brain className="text-purple-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">3.1 검색을 위한 Chain of Thought</h2>
          <p className="text-gray-600 dark:text-gray-400">단계별 사고를 통한 검색 품질 향상</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
          <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">기본 CoT 검색 프롬프트</h3>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border mb-4">
            <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-2">❌ 단순한 접근</p>
            <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded border border-slate-200 dark:border-slate-700">
              <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono whitespace-pre-wrap">
{`사용자 질문: {query}
검색된 문서: {documents}
위 정보를 바탕으로 답변하세요.`}
              </pre>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
            <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-2">✅ Chain of Thought 접근</p>
            <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded border border-slate-200 dark:border-slate-700">
              <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono whitespace-pre-wrap">
{`당신은 검색 증강 AI 어시스턴트입니다. 다음 단계를 따라 사용자 질문에 답변하세요:

사용자 질문: {query}

검색된 문서:
{documents}

답변 프로세스:
1. 먼저 사용자의 핵심 의도를 파악하세요
2. 검색된 각 문서의 관련성을 평가하세요
3. 가장 관련성 높은 정보를 추출하세요
4. 정보 간의 모순이나 차이점이 있는지 확인하세요
5. 종합적인 답변을 구성하세요

단계별 분석:
<thinking>
1. 사용자 의도: [여기에 분석]
2. 문서별 관련성:
   - 문서1: [관련도 및 핵심 정보]
   - 문서2: [관련도 및 핵심 정보]
3. 정보 종합: [추출한 핵심 정보들]
4. 모순 확인: [있다면 설명, 없다면 "없음"]
</thinking>

최종 답변:
[종합적이고 명확한 답변]`}
              </pre>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">고급 CoT 기법: Self-Ask</h3>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
            <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`class SelfAskRAGPrompt:
    def __init__(self):
        self.template = """
사용자 질문을 분석하고 필요한 하위 질문들을 생성한 후,
검색된 정보를 활용해 답변하세요.

원본 질문: {original_query}

<self_ask>
이 질문에 답하려면 어떤 정보가 필요한가?
1. [하위 질문 1]
2. [하위 질문 2]
3. [하위 질문 3]
</self_ask>

각 하위 질문에 대한 검색 결과:
{sub_query_results}

<integration>
하위 답변들을 어떻게 통합할 것인가?
- 정보의 신뢰도 평가
- 시간적 순서나 인과관계 고려
- 모순되는 정보 처리 방법
</integration>

최종 답변:
[통합된 종합 답변]
"""

    def generate_sub_queries(self, original_query):
        # LLM을 사용해 하위 질문 생성
        prompt = f"""
다음 질문을 답하기 위해 필요한 3-5개의 구체적인 하위 질문을 생성하세요:
"{original_query}"

하위 질문들:
"""
        return self.llm.generate(prompt)

    def search_and_answer(self, query):
        # 1. 하위 질문 생성
        sub_queries = self.generate_sub_queries(query)

        # 2. 각 하위 질문에 대해 검색
        sub_results = []
        for sub_q in sub_queries:
            docs = self.retriever.search(sub_q, k=3)
            sub_results.append({
                "question": sub_q,
                "documents": docs
            })

        # 3. 통합 답변 생성
        final_prompt = self.template.format(
            original_query=query,
            sub_query_results=sub_results
        )

        return self.llm.generate(final_prompt)`}
            </pre>
          </div>
        </div>
      </div>
    </section>
  )
}
