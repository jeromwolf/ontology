import References from '@/components/common/References'

export default function Section8() {
  return (
    <section className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-2xl p-8 text-white">
      <h2 className="text-2xl font-bold mb-6">실습 과제</h2>

      <div className="bg-white/10 rounded-xl p-6 backdrop-blur mb-8">
        <h3 className="font-bold mb-4">RAG 프롬프트 엔지니어링 실습</h3>

        <div className="space-y-4">
          <div className="bg-white/10 p-4 rounded-lg">
            <h4 className="font-medium mb-2">📋 과제 1: 도메인별 프롬프트 템플릿 구축</h4>
            <ol className="space-y-2 text-sm">
              <li>1. 3개 이상의 도메인 선택 (예: 의료, 법률, 기술)</li>
              <li>2. 각 도메인별 시스템 프롬프트 작성</li>
              <li>3. Few-shot 예시 3개씩 준비</li>
              <li>4. 에러 처리 시나리오 정의</li>
              <li>5. 실제 질문으로 테스트 및 평가</li>
            </ol>
          </div>

          <div className="bg-white/10 p-4 rounded-lg">
            <h4 className="font-medium mb-2">🎯 과제 2: Chain of Thought 최적화</h4>
            <ul className="space-y-1 text-sm">
              <li>• Self-Ask 방식으로 복잡한 질문 분해</li>
              <li>• 각 단계별 추론 과정 명시화</li>
              <li>• 검색 효율성과 답변 품질 측정</li>
              <li>• A/B 테스트로 개선 효과 검증</li>
            </ul>
          </div>

          <div className="bg-white/10 p-4 rounded-lg">
            <h4 className="font-medium mb-2">💡 과제 3: 다중 턴 대화 시뮬레이션</h4>
            <p className="text-sm mb-2">
              고객 지원 챗봇 시나리오로 다음을 구현:
            </p>
            <ul className="space-y-1 text-sm">
              <li>• 5턴 이상의 연속 대화 처리</li>
              <li>• 컨텍스트 유지 및 참조 해결</li>
              <li>• 대화 기록 기반 개인화</li>
              <li>• 만족도 평가 시스템 구축</li>
            </ul>
          </div>

          <div className="bg-white/10 p-4 rounded-lg">
            <h4 className="font-medium mb-2">📊 평가 지표</h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="font-medium">정량적 지표:</p>
                <ul className="mt-1">
                  <li>• 응답 정확도</li>
                  <li>• 처리 시간</li>
                  <li>• 토큰 사용량</li>
                </ul>
              </div>
              <div>
                <p className="font-medium">정성적 지표:</p>
                <ul className="mt-1">
                  <li>• 답변의 자연스러움</li>
                  <li>• 컨텍스트 이해도</li>
                  <li>• 에러 처리 품질</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 프롬프트 엔지니어링 가이드',
            icon: 'web' as const,
            color: 'border-teal-500',
            items: [
              {
                title: 'OpenAI Prompt Engineering Guide',
                authors: 'OpenAI',
                year: '2025',
                description: 'GPT 모델을 위한 공식 프롬프트 작성 가이드 - 6가지 전략',
                link: 'https://platform.openai.com/docs/guides/prompt-engineering'
              },
              {
                title: 'Anthropic Prompt Engineering',
                authors: 'Anthropic',
                year: '2025',
                description: 'Claude를 위한 프롬프트 최적화 - Chain of Thought 강조',
                link: 'https://docs.anthropic.com/claude/docs/prompt-engineering'
              },
              {
                title: 'LangChain Prompt Templates',
                authors: 'LangChain',
                year: '2025',
                description: 'RAG용 프롬프트 템플릿 라이브러리 - 재사용 가능',
                link: 'https://python.langchain.com/docs/modules/model_io/prompts/'
              },
              {
                title: 'Prompt Engineering by DAIR.AI',
                authors: 'DAIR.AI',
                year: '2024',
                description: '실전 프롬프트 엔지니어링 가이드 - 40+ 예제',
                link: 'https://www.promptingguide.ai/'
              },
              {
                title: 'AWS RAG Prompt Best Practices',
                authors: 'AWS',
                year: '2025',
                description: 'Bedrock을 위한 RAG 프롬프트 최적화 패턴',
                link: 'https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-engineering-guidelines.html'
              }
            ]
          },
          {
            title: '📖 Chain of Thought & Few-shot 연구',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Chain-of-Thought Prompting',
                authors: 'Wei et al., Google Research',
                year: '2022',
                description: 'CoT 프롬프팅으로 추론 성능 대폭 향상 - 540B 모델',
                link: 'https://arxiv.org/abs/2201.11903'
              },
              {
                title: 'Self-Ask: Decomposing Complex Questions',
                authors: 'Press et al., UW & AI2',
                year: '2023',
                description: '복잡한 질문을 하위 질문으로 분해 - 정확도 30% 향상',
                link: 'https://arxiv.org/abs/2210.03350'
              },
              {
                title: 'Few-Shot Parameter-Efficient Fine-Tuning',
                authors: 'Liu et al., Stanford',
                year: '2022',
                description: 'In-Context Learning의 이론적 기반 - GPT-3',
                link: 'https://arxiv.org/abs/2012.15723'
              },
              {
                title: 'Tree of Thoughts (ToT)',
                authors: 'Yao et al., Princeton',
                year: '2023',
                description: '트리 구조로 사고 확장 - CoT의 고급 버전',
                link: 'https://arxiv.org/abs/2305.10601'
              }
            ]
          },
          {
            title: '🛠️ RAG 프롬프트 실전 도구',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'LlamaIndex Prompt Optimizer',
                authors: 'LlamaIndex',
                year: '2025',
                description: 'RAG 프롬프트 자동 최적화 - A/B 테스트 내장',
                link: 'https://docs.llamaindex.ai/en/stable/module_guides/querying/prompts/'
              },
              {
                title: 'Haystack PromptNode',
                authors: 'deepset',
                year: '2025',
                description: '다양한 LLM용 프롬프트 통합 관리 - 템플릿 시스템',
                link: 'https://docs.haystack.deepset.ai/docs/prompt_node'
              },
              {
                title: 'Guidance (Microsoft)',
                authors: 'Microsoft Research',
                year: '2024',
                description: '구조화된 프롬프트 생성 - 제약 조건 적용',
                link: 'https://github.com/microsoft/guidance'
              },
              {
                title: 'Prompttools',
                authors: 'Hegel AI',
                year: '2024',
                description: '프롬프트 실험 및 평가 프레임워크 - 벤치마킹',
                link: 'https://github.com/hegelai/prompttools'
              },
              {
                title: 'LangSmith Prompt Hub',
                authors: 'LangChain',
                year: '2025',
                description: '커뮤니티 검증된 프롬프트 템플릿 저장소',
                link: 'https://smith.langchain.com/hub'
              }
            ]
          }
        ]}
      />
    </section>
  )
}
