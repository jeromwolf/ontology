'use client'

import dynamic from 'next/dynamic'
import { ArrowLeft } from 'lucide-react'

const PromptPlayground = dynamic(
  () => import('@/components/llm-simulators/PromptPlayground'),
  { ssr: false }
)

export default function PromptPlaygroundSimulator() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <button
            onClick={() => window.history.back()}
            className="inline-flex items-center text-indigo-600 hover:text-indigo-700 dark:text-indigo-400 transition-colors"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            학습 모듈로 돌아가기
          </button>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            Prompt Engineering Playground
          </h1>
          <p className="text-gray-600 dark:text-gray-300 mb-8">
            다양한 프롬프트 기법을 실험하고 결과를 비교하며 최적의 프롬프트를 설계해보세요.
          </p>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 mb-8">
            <h2 className="text-lg font-semibold mb-4">🎮 사용 방법</h2>
            <ul className="space-y-2 text-gray-600 dark:text-gray-300">
              <li>• 다양한 프롬프트 템플릿(Zero-shot, Few-shot, CoT 등)을 선택하세요</li>
              <li>• 시스템 프롬프트와 사용자 프롬프트를 각각 설정할 수 있습니다</li>
              <li>• Temperature, Top-p 등 생성 파라미터를 조정해보세요</li>
              <li>• 여러 프롬프트의 결과를 나란히 비교할 수 있습니다</li>
            </ul>
          </div>

          <div className="border-2 border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden">
            <PromptPlayground />
          </div>

          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-cyan-50 dark:bg-cyan-900/20 rounded-xl p-6">
              <h3 className="font-semibold text-cyan-900 dark:text-cyan-300 mb-2">
                💡 Zero-shot
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                예시 없이 작업을 설명하는 방법. 간단한 작업에 효과적입니다.
              </p>
            </div>
            
            <div className="bg-amber-50 dark:bg-amber-900/20 rounded-xl p-6">
              <h3 className="font-semibold text-amber-900 dark:text-amber-300 mb-2">
                🔬 Few-shot
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                몇 가지 예시를 제공하는 방법. 복잡한 패턴 학습에 유용합니다.
              </p>
            </div>

            <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-xl p-6">
              <h3 className="font-semibold text-emerald-900 dark:text-emerald-300 mb-2">
                🧠 Chain-of-Thought
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                단계별 추론 과정을 보여주는 방법. 논리적 문제에 탁월합니다.
              </p>
            </div>
          </div>

          <div className="mt-6 p-6 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl">
            <h3 className="font-semibold text-indigo-900 dark:text-indigo-300 mb-2">
              🎯 프롬프트 엔지니어링 Best Practices
            </h3>
            <ul className="text-gray-700 dark:text-gray-300 text-sm space-y-1">
              <li>• 명확하고 구체적인 지시사항을 제공하세요</li>
              <li>• 원하는 출력 형식을 명시하세요</li>
              <li>• 제약 조건과 요구사항을 명확히 하세요</li>
              <li>• 필요시 역할(Role)을 부여하세요</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}