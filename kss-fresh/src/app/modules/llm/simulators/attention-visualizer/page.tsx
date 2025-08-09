'use client'

import dynamic from 'next/dynamic'
import { ArrowLeft } from 'lucide-react'

const AttentionVisualizer = dynamic(
  () => import('@/components/llm-simulators/AttentionVisualizer'),
  { ssr: false }
)

export default function AttentionVisualizerSimulator() {
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
            Attention Mechanism Visualizer
          </h1>
          <p className="text-gray-600 dark:text-gray-300 mb-8">
            Self-Attention 메커니즘이 어떻게 단어 간의 관계를 학습하는지 시각적으로 탐구해보세요.
          </p>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 mb-8">
            <h2 className="text-lg font-semibold mb-4">🎮 사용 방법</h2>
            <ul className="space-y-2 text-gray-600 dark:text-gray-300">
              <li>• 문장을 입력하고 토큰화 과정을 확인하세요</li>
              <li>• 각 토큰 간의 Attention 가중치를 히트맵으로 관찰하세요</li>
              <li>• Query, Key, Value 벡터가 어떻게 계산되는지 단계별로 확인하세요</li>
              <li>• Multi-Head Attention의 각 헤드별 패턴을 비교해보세요</li>
            </ul>
          </div>

          <div className="border-2 border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden">
            <AttentionVisualizer />
          </div>

          <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6">
              <h3 className="font-semibold text-blue-900 dark:text-blue-300 mb-2">
                💡 학습 팁
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                문장에서 중요한 단어들이 높은 Attention 가중치를 받는 것을 
                확인하세요. 문맥에 따라 같은 단어도 다른 패턴을 보입니다.
              </p>
            </div>
            
            <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-6">
              <h3 className="font-semibold text-green-900 dark:text-green-300 mb-2">
                🔬 실험해보기
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                한국어와 영어 문장을 각각 입력해보고, 언어별 Attention 
                패턴의 차이를 관찰해보세요.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}