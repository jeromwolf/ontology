'use client';

import References from '@/components/common/References';

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

      {/* 학습 요약 */}
      <section className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6 mt-8">
        <h2 className="text-xl font-bold mb-4 text-indigo-800 dark:text-indigo-200">📚 핵심 정리</h2>
        <ul className="space-y-2">
          <li className="flex items-start gap-2">
            <span className="text-indigo-600 dark:text-indigo-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">RAG 프롬프트 템플릿 설계 원칙</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-600 dark:text-indigo-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">컨텍스트 순서 최적화 및 압축 전략</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-600 dark:text-indigo-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">답변 품질 향상 기법 (Chain-of-Thought, Citation)</span>
          </li>
        </ul>
      </section>

      <References
        sections={[
          {
            title: '📚 프롬프트 엔지니어링 논문 (Prompt Engineering Papers)',
            icon: 'paper',
            color: 'border-indigo-500',
            items: [
              {
                title: 'Chain-of-Thought Prompting Elicits Reasoning in LLMs',
                authors: 'Jason Wei, Xuezhi Wang, et al. (Google Research)',
                year: '2022',
                description: '단계별 추론을 유도하여 복잡한 문제 해결 능력 향상',
                link: 'https://arxiv.org/abs/2201.11903'
              },
              {
                title: 'Lost in the Middle: How LLMs Use Long Contexts',
                authors: 'Nelson F. Liu et al.',
                year: '2023',
                description: 'LLM이 긴 컨텍스트를 처리하는 방식과 최적 배치 전략',
                link: 'https://arxiv.org/abs/2307.03172'
              },
              {
                title: 'SELF-RAG: Learning to Retrieve, Generate, and Critique',
                authors: 'Akari Asai et al.',
                year: '2023',
                description: '자기 성찰을 통한 RAG 답변 품질 향상',
                link: 'https://arxiv.org/abs/2310.11511'
              },
              {
                title: 'Measuring and Narrowing the Compositionality Gap in LLMs',
                authors: 'Ofir Press et al.',
                year: '2023',
                description: '복잡한 추론 작업을 분해하여 성능 향상',
                link: 'https://arxiv.org/abs/2210.03350'
              }
            ]
          },
          {
            title: '🛠️ 프롬프트 최적화 도구 (Prompt Optimization Tools)',
            icon: 'tools',
            color: 'border-purple-500',
            items: [
              {
                title: 'LangChain Prompt Templates',
                description: 'RAG용 프롬프트 템플릿 라이브러리 - 10+ 검증된 패턴',
                link: 'https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/'
              },
              {
                title: 'OpenAI Prompt Engineering Guide',
                description: 'OpenAI 공식 프롬프트 엔지니어링 가이드',
                link: 'https://platform.openai.com/docs/guides/prompt-engineering'
              },
              {
                title: 'PromptPerfect',
                description: 'AI 기반 프롬프트 자동 최적화 도구',
                link: 'https://promptperfect.jina.ai/'
              },
              {
                title: 'LlamaIndex Response Synthesis',
                description: '다양한 답변 생성 전략 구현 (refine, tree_summarize, compact)',
                link: 'https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/'
              }
            ]
          },
          {
            title: '📖 컨텍스트 관리 가이드 (Context Management)',
            icon: 'book',
            color: 'border-emerald-500',
            items: [
              {
                title: 'Context Length Optimization',
                description: '토큰 제한 내에서 최대 정보 전달 전략',
                link: 'https://www.anthropic.com/index/100k-context-windows'
              },
              {
                title: 'Context Compression Techniques',
                description: 'LLMLingua를 활용한 컨텍스트 압축 (50% 감소)',
                link: 'https://arxiv.org/abs/2310.06201'
              },
              {
                title: 'Metadata Enrichment Strategies',
                description: '메타데이터로 검색 정확도 30% 향상시키는 방법',
                link: 'https://www.pinecone.io/learn/metadata-filtering/'
              },
              {
                title: 'Citation & Source Attribution',
                description: '답변에 출처 자동 인용 구현 패턴',
                link: 'https://github.com/langchain-ai/langchain/blob/master/docs/docs/use_cases/question_answering/citations.ipynb'
              }
            ]
          },
          {
            title: '⚡ 답변 품질 향상 (Answer Quality)',
            icon: 'web',
            color: 'border-orange-500',
            items: [
              {
                title: 'Self-Consistency Decoding',
                description: '여러 답변 생성 후 투표로 최적 답변 선택',
                link: 'https://arxiv.org/abs/2203.11171'
              },
              {
                title: 'Confidence Scoring',
                description: 'LLM 답변의 확신도 측정 및 표시 방법',
                link: 'https://arxiv.org/abs/2305.14975'
              },
              {
                title: 'Hallucination Detection',
                description: '할루시네이션 자동 감지 및 방지 기법',
                link: 'https://github.com/vectara/hallucination-leaderboard'
              }
            ]
          }
        ]}
      />
    </div>
  )
}