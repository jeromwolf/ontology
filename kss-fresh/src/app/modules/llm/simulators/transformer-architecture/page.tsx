'use client'

import dynamic from 'next/dynamic'
import { ArrowLeft } from 'lucide-react'

const TransformerArchitecture3D = dynamic(
  () => import('@/components/llm-simulators/TransformerArchitecture3D'),
  { ssr: false }
)

export default function TransformerArchitectureSimulator() {
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
            Transformer Architecture 3D 시뮬레이터
          </h1>
          <p className="text-gray-600 dark:text-gray-300 mb-8">
            Transformer 아키텍처의 구조를 3D로 시각화하고 각 구성 요소의 작동 원리를 탐구해보세요.
          </p>


          <div className="border-2 border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden">
            <TransformerArchitecture3D />
          </div>

          <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6">
              <h3 className="font-semibold text-indigo-900 dark:text-indigo-300 mb-2">
                💡 학습 팁
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                Encoder와 Decoder의 차이점에 주목하세요. Self-Attention과 
                Cross-Attention이 어떻게 다른지 관찰해보세요.
              </p>
            </div>
            
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6">
              <h3 className="font-semibold text-purple-900 dark:text-purple-300 mb-2">
                🔬 실험해보기
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                입력 시퀀스의 길이를 변경하면서 Attention 매트릭스가 
                어떻게 변화하는지 확인해보세요.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}