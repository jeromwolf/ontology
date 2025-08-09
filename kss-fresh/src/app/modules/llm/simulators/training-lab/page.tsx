'use client'

import dynamic from 'next/dynamic'
import { ArrowLeft } from 'lucide-react'

const TrainingSimulator = dynamic(
  () => import('@/components/llm-simulators/TrainingSimulator'),
  { ssr: false }
)

export default function TrainingLabSimulator() {
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
            LLM Training Lab
          </h1>
          <p className="text-gray-600 dark:text-gray-300 mb-8">
            소규모 언어 모델을 직접 학습시키며 학습 과정을 실시간으로 모니터링해보세요.
          </p>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 mb-8">
            <h2 className="text-lg font-semibold mb-4">🎮 사용 방법</h2>
            <ul className="space-y-2 text-gray-600 dark:text-gray-300">
              <li>• 학습 데이터셋을 선택하거나 직접 입력하세요</li>
              <li>• 모델 크기, 학습률 등 하이퍼파라미터를 조정하세요</li>
              <li>• 학습 시작 버튼을 눌러 실시간 학습 과정을 관찰하세요</li>
              <li>• Loss 그래프와 생성 샘플을 통해 학습 진행 상황을 확인하세요</li>
            </ul>
          </div>

          <div className="border-2 border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden">
            <TrainingSimulator />
          </div>

          <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-red-50 dark:bg-red-900/20 rounded-xl p-6">
              <h3 className="font-semibold text-red-900 dark:text-red-300 mb-2">
                💡 학습 팁
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                학습률이 너무 높으면 Loss가 발산할 수 있고, 너무 낮으면 
                학습이 매우 느려집니다. 적절한 값을 찾아보세요.
              </p>
            </div>
            
            <div className="bg-teal-50 dark:bg-teal-900/20 rounded-xl p-6">
              <h3 className="font-semibold text-teal-900 dark:text-teal-300 mb-2">
                🔬 실험해보기
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                배치 크기를 변경하면서 학습 속도와 안정성이 어떻게 
                변화하는지 관찰해보세요.
              </p>
            </div>
          </div>

          <div className="mt-6 p-6 bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl">
            <h3 className="font-semibold text-purple-900 dark:text-purple-300 mb-2">
              🚀 도전 과제
            </h3>
            <p className="text-gray-700 dark:text-gray-300">
              한국 속담 데이터셋으로 학습시킨 후, 모델이 새로운 속담을 생성할 수 있는지 
              테스트해보세요!
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}