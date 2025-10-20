'use client';

import Link from 'next/link';
import { FlaskConical } from 'lucide-react';
import dynamic from 'next/dynamic';
import References from '@/components/common/References';

// Dynamic import for TokenizerDemo
const TokenizerDemo = dynamic(() => import('../TokenizerDemo'), {
  ssr: false,
  loading: () => <div className="animate-pulse bg-gray-200 h-64 rounded-lg"></div>
})

export default function Chapter3() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-indigo-800 dark:text-indigo-200 mb-4">
          모델 학습과정과 최적화
        </h2>
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-6">
          <p className="text-lg text-gray-700 dark:text-gray-300">
            LLM의 학습은 사전훈련(Pre-training) → 파인튜닝(Fine-tuning) → RLHF 단계를 거칩니다.
          </p>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">학습 단계별 과정</h3>
        
        {/* Training Lab 시뮬레이터 링크 */}
        <div className="mb-6 p-4 bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-semibold text-orange-900 dark:text-orange-200 mb-1">🎮 LLM Training Lab</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                소규모 언어 모델을 직접 학습시키며 학습 과정을 실시간으로 모니터링해보세요
              </p>
            </div>
            <Link 
              href="/modules/llm/simulators/training-lab"
              className="inline-flex items-center gap-2 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors"
            >
              <FlaskConical className="w-4 h-4" />
              시뮬레이터 실행
            </Link>
          </div>
        </div>
        
        <div className="space-y-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg">
            <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-3">1. 사전훈련 (Pre-training)</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              수조 개의 토큰으로 구성된 대규모 텍스트 데이터로 다음 토큰 예측 학습
            </p>
            <ul className="list-disc list-inside space-y-1 text-gray-600 dark:text-gray-400">
              <li>Common Crawl, Wikipedia, Books 등 웹 데이터</li>
              <li>수천 개의 GPU로 수개월간 학습</li>
              <li>언어의 기본 패턴과 지식 습득</li>
            </ul>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg">
            <h4 className="font-semibold text-green-800 dark:text-green-200 mb-3">2. 지도 파인튜닝 (SFT)</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              고품질의 instruction-following 데이터로 특정 작업 수행 능력 향상
            </p>
            <ul className="list-disc list-inside space-y-1 text-gray-600 dark:text-gray-400">
              <li>질문-답변, 요약, 번역 등 작업별 데이터</li>
              <li>상대적으로 적은 데이터(수만~수십만 개)</li>
              <li>사용자 지시를 따르는 능력 학습</li>
            </ul>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg">
            <h4 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">3. 인간 피드백 강화학습 (RLHF)</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              인간의 선호도를 반영하여 모델의 출력을 인간 가치와 정렬
            </p>
            <ul className="list-disc list-inside space-y-1 text-gray-600 dark:text-gray-400">
              <li>인간 평가자가 출력 품질 평가</li>
              <li>Reward Model 학습 후 PPO 알고리즘 적용</li>
              <li>안전성, 유용성, 정직성 향상</li>
            </ul>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg">
            <h4 className="font-semibold text-orange-800 dark:text-orange-200 mb-3">4. 최신 학습 기법들</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              2024-2025년 등장한 혁신적인 학습 방법들
            </p>
            <ul className="list-disc list-inside space-y-1 text-gray-600 dark:text-gray-400">
              <li><strong>DPO (Direct Preference Optimization)</strong>: RLHF보다 효율적인 선호도 학습</li>
              <li><strong>Constitutional AI</strong>: Anthropic의 헌법 기반 AI 학습</li>
              <li><strong>RLAIF</strong>: AI 피드백을 통한 강화학습</li>
              <li><strong>Chain-of-Thought Fine-tuning</strong>: 추론 능력 향상</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">강화학습 상세 분석</h3>
        <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-xl p-6">
          <h4 className="font-bold text-red-700 dark:text-red-300 mb-4">RLHF vs DPO vs Constitutional AI</h4>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h5 className="font-semibold text-red-600 dark:text-red-400 mb-2">RLHF (PPO 기반)</h5>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 장점: 세밀한 조정 가능, 성능 검증됨</li>
                <li>• 단점: 계산 비용 높음, 불안정한 학습</li>
                <li>• 사용: ChatGPT, Claude 2</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h5 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">DPO (Direct Preference)</h5>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 장점: 간단한 구현, 안정적 학습</li>
                <li>• 단점: 세밀한 조정 어려움</li>
                <li>• 사용: Llama 3, Mixtral</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h5 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">Constitutional AI</h5>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 장점: 명확한 원칙, 투명성</li>
                <li>• 단점: 복잡한 헌법 설계</li>
                <li>• 사용: Claude 3, Claude Opus 4</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">토크나이저와 어휘 구성</h3>
        
        {/* Tokenizer Playground 시뮬레이터 링크 */}
        <div className="mb-6 p-4 bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-semibold text-yellow-900 dark:text-yellow-200 mb-1">🎮 Tokenizer Playground</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                다양한 토크나이저가 텍스트를 어떻게 분해하는지 비교하고 분석해보세요
              </p>
            </div>
            <Link 
              href="/modules/llm/simulators/tokenizer-playground"
              className="inline-flex items-center gap-2 px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 transition-colors"
            >
              <FlaskConical className="w-4 h-4" />
              시뮬레이터 실행
            </Link>
          </div>
        </div>
        
        <TokenizerDemo />
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Scaling Laws와 효율화</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg">
            <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-3">Kaplan Scaling Law</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              모델 크기 10배 → 성능 약 2배 향상
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• 파라미터 수: N</li>
              <li>• 데이터 크기: D</li>
              <li>• 계산량: C</li>
              <li>• Loss ∝ N^(-0.076)</li>
            </ul>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg">
            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-3">Chinchilla Scaling</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              최적 데이터/파라미터 비율 = 20:1
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• 70B 모델 → 1.4T 토큰 필요</li>
              <li>• 데이터 품질이 양보다 중요</li>
              <li>• 효율적 학습 가능</li>
            </ul>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: '📚 핵심 논문 & Fine-tuning 기법',
            icon: 'research' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'LoRA: Low-Rank Adaptation of Large Language Models (2021)',
                url: 'https://arxiv.org/abs/2106.09685',
                description: '저비용 파인튜닝의 혁신 - Hu et al., Microsoft'
              },
              {
                title: 'QLoRA: Efficient Finetuning of Quantized LLMs (2023)',
                url: 'https://arxiv.org/abs/2305.14314',
                description: '4비트 양자화로 메모리 75% 절감 - Dettmers et al., University of Washington'
              },
              {
                title: 'Parameter-Efficient Transfer Learning for NLP (2019)',
                url: 'https://arxiv.org/abs/1902.00751',
                description: 'Adapter 레이어 기법 - Houlsby et al., Google Research'
              },
              {
                title: 'Prefix-Tuning: Optimizing Continuous Prompts (2021)',
                url: 'https://arxiv.org/abs/2101.00190',
                description: '연속적 프롬프트 최적화 - Li & Liang, Stanford'
              },
              {
                title: 'The Power of Scale for Parameter-Efficient Prompt Tuning (2021)',
                url: 'https://arxiv.org/abs/2104.08691',
                description: 'Soft Prompt Tuning - Lester et al., Google Research'
              }
            ]
          },
          {
            title: '🔬 RLHF & Alignment 연구',
            icon: 'research' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Training language models to follow instructions with human feedback (2022)',
                url: 'https://arxiv.org/abs/2203.02155',
                description: 'InstructGPT - RLHF의 표준 - Ouyang et al., OpenAI'
              },
              {
                title: 'Direct Preference Optimization (DPO, 2023)',
                url: 'https://arxiv.org/abs/2305.18290',
                description: 'RL 없이 선호도 학습 - Rafailov et al., Stanford'
              },
              {
                title: 'Constitutional AI: Harmlessness from AI Feedback (2022)',
                url: 'https://arxiv.org/abs/2212.08073',
                description: 'RLAIF 기반 안전성 향상 - Bai et al., Anthropic'
              },
              {
                title: 'Scaling Laws for Reward Model Overoptimization (2022)',
                url: 'https://arxiv.org/abs/2210.10760',
                description: 'RLHF의 한계와 해결책 - Gao et al., OpenAI'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 프레임워크',
            icon: 'tools' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Hugging Face PEFT Library',
                url: 'https://github.com/huggingface/peft',
                description: 'LoRA, QLoRA, Adapter 등 통합 라이브러리 (15k+ stars)'
              },
              {
                title: 'DeepSpeed',
                url: 'https://www.deepspeed.ai/',
                description: 'Microsoft의 대규모 모델 학습 프레임워크 - ZeRO 최적화'
              },
              {
                title: 'TRL (Transformer Reinforcement Learning)',
                url: 'https://github.com/huggingface/trl',
                description: 'RLHF, DPO 구현 라이브러리 - Hugging Face'
              },
              {
                title: 'Axolotl: Fine-tuning Framework',
                url: 'https://github.com/OpenAccess-AI-Collective/axolotl',
                description: '다양한 PEFT 기법 원클릭 적용 (7k+ stars)'
              },
              {
                title: 'Unsloth: 2x Faster Fine-tuning',
                url: 'https://github.com/unslothai/unsloth',
                description: 'LoRA/QLoRA 학습 속도 2배 향상 (11k+ stars)'
              }
            ]
          }
        ]}
      />
    </div>
  )
}