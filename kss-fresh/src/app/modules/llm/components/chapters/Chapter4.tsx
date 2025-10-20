'use client';

import Link from 'next/link';
import { FlaskConical } from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter4() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-indigo-800 dark:text-indigo-200 mb-4">
          프롬프트 엔지니어링 마스터
        </h2>
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-6">
          <p className="text-lg text-gray-700 dark:text-gray-300">
            효과적인 프롬프트 설계는 LLM의 성능을 극대화하는 핵심 기술입니다.
          </p>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">프롬프트 기법들</h3>
        
        {/* Prompt Playground 시뮬레이터 링크 */}
        <div className="mb-6 p-4 bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-900/20 dark:to-teal-900/20 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-semibold text-green-900 dark:text-green-200 mb-1">🎮 Prompt Engineering Playground</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                다양한 프롬프트 기법을 실험하고 결과를 비교해보세요
              </p>
            </div>
            <Link 
              href="/modules/llm/simulators/prompt-playground"
              className="inline-flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
            >
              <FlaskConical className="w-4 h-4" />
              시뮬레이터 실행
            </Link>
          </div>
        </div>
        
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-3">Zero-shot Prompting</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">예시 없이 작업 설명만으로 수행</p>
            <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded border">
              <pre className="text-sm">다음 텍스트를 한국어로 번역해주세요: "Hello, world!"</pre>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-3">Few-shot Prompting</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">몇 개의 예시를 제공하여 패턴 학습</p>
            <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded border">
              <pre className="text-sm whitespace-pre-wrap">{`영어 -> 한국어 번역:
Hello -> 안녕하세요
Thank you -> 감사합니다
Goodbye -> 안녕히 가세요

Good morning -> ?`}</pre>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-3">Chain-of-Thought (CoT)</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">단계별 추론 과정을 명시적으로 안내</p>
            <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded border">
              <pre className="text-sm whitespace-pre-wrap">{`문제를 단계별로 해결해보겠습니다:

1. 주어진 정보 파악
2. 필요한 공식 확인  
3. 계산 수행
4. 결과 검증`}</pre>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">고급 프롬프트 기법</h3>

        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-3">Self-Consistency</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">여러 번 추론 후 다수결로 결정</p>
            <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded border">
              <pre className="text-sm whitespace-pre-wrap">{`동일한 질문을 5번 추론하여 가장 빈번한 답을 선택합니다.
이를 통해 일관성 있는 결과를 얻을 수 있습니다.`}</pre>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-3">ReAct: Reasoning + Acting</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">추론과 행동을 번갈아 수행</p>
            <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded border">
              <pre className="text-sm whitespace-pre-wrap">{`Thought: 날씨 정보가 필요함
Action: Search[서울 날씨]
Observation: 맑음, 15도
Thought: 이제 답변 가능
Answer: 서울은 현재 맑고 15도입니다.`}</pre>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-3">Tree of Thoughts (ToT)</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">여러 사고 경로를 탐색하고 평가</p>
            <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded border">
              <pre className="text-sm whitespace-pre-wrap">{`문제 → [경로1, 경로2, 경로3]
각 경로 평가 후 가장 유망한 경로 선택
재귀적으로 탐색하여 최적 해 도출`}</pre>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-3">Least-to-Most Prompting</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">복잡한 문제를 작은 단위로 분해</p>
            <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded border">
              <pre className="text-sm whitespace-pre-wrap">{`1단계: 가장 간단한 하위 문제 해결
2단계: 이전 해를 활용하여 다음 문제 해결
...
N단계: 전체 문제의 답 도출`}</pre>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">프롬프트 최적화 팁</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg">
            <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-3">명확성</h4>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li>• 구체적인 지시 사용</li>
              <li>• 모호한 표현 피하기</li>
              <li>• 예시 제공하기</li>
            </ul>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg">
            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-3">구조화</h4>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li>• 단계별로 나누기</li>
              <li>• 역할 부여하기</li>
              <li>• 형식 명시하기</li>
            </ul>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg">
            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-3">반복 개선</h4>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li>• A/B 테스트 수행</li>
              <li>• 피드백 반영하기</li>
              <li>• 템플릿 구축하기</li>
            </ul>
          </div>
          <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg">
            <h4 className="font-semibold text-orange-700 dark:text-orange-300 mb-3">컨텍스트 관리</h4>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li>• 관련 정보만 포함</li>
              <li>• 토큰 제한 고려</li>
              <li>• 우선순위 정하기</li>
            </ul>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: '📚 핵심 논문 & 프롬프트 기법',
            icon: 'research' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'Chain-of-Thought Prompting Elicits Reasoning in LLMs (2022)',
                url: 'https://arxiv.org/abs/2201.11903',
                description: '단계별 추론으로 복잡한 문제 해결 - Wei et al., Google Research'
              },
              {
                title: 'ReAct: Synergizing Reasoning and Acting in LLMs (2022)',
                url: 'https://arxiv.org/abs/2210.03629',
                description: '추론과 행동 결합한 에이전트 - Yao et al., Princeton & Google'
              },
              {
                title: 'Tree of Thoughts: Deliberate Problem Solving with LLMs (2023)',
                url: 'https://arxiv.org/abs/2305.10601',
                description: '트리 탐색 기반 사고 확장 - Yao et al., Princeton'
              },
              {
                title: 'Self-Consistency Improves Chain of Thought Reasoning (2022)',
                url: 'https://arxiv.org/abs/2203.11171',
                description: '다수결 기반 일관성 향상 - Wang et al., Google Research'
              },
              {
                title: 'Least-to-Most Prompting Enables Complex Reasoning (2022)',
                url: 'https://arxiv.org/abs/2205.10625',
                description: '하위 문제 분해 전략 - Zhou et al., Google Research'
              }
            ]
          },
          {
            title: '🔬 최신 연구 & 벤치마크',
            icon: 'research' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Large Language Models are Zero-Shot Reasoners (2022)',
                url: 'https://arxiv.org/abs/2205.11916',
                description: '"Let\'s think step by step" 마법 - Kojima et al., University of Tokyo'
              },
              {
                title: 'Automatic Prompt Engineer (APE, 2022)',
                url: 'https://arxiv.org/abs/2211.01910',
                description: 'LLM이 스스로 최적 프롬프트 생성 - Zhou et al., Meta AI'
              },
              {
                title: 'PromptBench: Towards Evaluating Robustness of LLMs (2023)',
                url: 'https://arxiv.org/abs/2306.04528',
                description: '프롬프트 견고성 평가 프레임워크 - Zhu et al., Microsoft Research'
              },
              {
                title: 'DSPy: Programming Foundation Models (2023)',
                url: 'https://arxiv.org/abs/2310.03714',
                description: '프롬프트를 프로그래밍 언어로 - Khattab et al., Stanford'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 가이드',
            icon: 'tools' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'OpenAI Prompt Engineering Guide',
                url: 'https://platform.openai.com/docs/guides/prompt-engineering',
                description: 'GPT 모델 최적화 공식 가이드 - OpenAI'
              },
              {
                title: 'Anthropic Prompt Library',
                url: 'https://docs.anthropic.com/claude/prompt-library',
                description: 'Claude 전용 프롬프트 템플릿 모음 - Anthropic'
              },
              {
                title: 'Prompt Engineering Guide by DAIR.AI',
                url: 'https://www.promptingguide.ai/',
                description: '포괄적인 프롬프트 엔지니어링 가이드 (오픈소스)'
              },
              {
                title: 'LangChain Prompt Templates',
                url: 'https://python.langchain.com/docs/modules/model_io/prompts/',
                description: '재사용 가능한 프롬프트 템플릿 라이브러리'
              },
              {
                title: 'PromptBase Marketplace',
                url: 'https://promptbase.com/',
                description: '검증된 프롬프트 거래 마켓플레이스'
              }
            ]
          }
        ]}
      />
    </div>
  )
}