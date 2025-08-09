'use client'

import dynamic from 'next/dynamic'
import { ArrowLeft } from 'lucide-react'

const EnhancedTokenizer = dynamic(
  () => import('@/components/llm-simulators/EnhancedTokenizer'),
  { ssr: false }
)

export default function TokenizerPlaygroundSimulator() {
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
            Tokenizer Playground
          </h1>
          <p className="text-gray-600 dark:text-gray-300 mb-8">
            다양한 토크나이저가 텍스트를 어떻게 분해하는지 실시간으로 비교하고 분석해보세요.
          </p>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 mb-8">
            <h2 className="text-lg font-semibold mb-4">🎮 사용 방법</h2>
            <ul className="space-y-2 text-gray-600 dark:text-gray-300">
              <li>• 텍스트를 입력하면 실시간으로 토큰화 결과를 확인할 수 있습니다</li>
              <li>• GPT, BERT, T5 등 다양한 토크나이저를 선택해보세요</li>
              <li>• 한국어, 영어, 코드 등 다양한 텍스트 타입을 실험해보세요</li>
              <li>• 토큰 ID, 서브워드 분해 과정을 상세히 관찰하세요</li>
            </ul>
          </div>

          <div className="border-2 border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden">
            <EnhancedTokenizer />
          </div>

          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-orange-50 dark:bg-orange-900/20 rounded-xl p-6">
              <h3 className="font-semibold text-orange-900 dark:text-orange-300 mb-2">
                💡 학습 팁
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                한국어는 교착어 특성상 영어보다 더 많은 토큰으로 
                분해되는 경향이 있습니다.
              </p>
            </div>
            
            <div className="bg-pink-50 dark:bg-pink-900/20 rounded-xl p-6">
              <h3 className="font-semibold text-pink-900 dark:text-pink-300 mb-2">
                🔬 실험해보기
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                프로그래밍 코드를 입력하고 일반 텍스트와 토큰화 
                패턴이 어떻게 다른지 비교해보세요.
              </p>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6">
              <h3 className="font-semibold text-yellow-900 dark:text-yellow-300 mb-2">
                📊 토큰 효율성
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                같은 의미의 문장도 토크나이저에 따라 토큰 수가 
                크게 달라질 수 있습니다.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}