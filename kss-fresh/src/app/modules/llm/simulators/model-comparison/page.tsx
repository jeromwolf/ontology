'use client'

import dynamic from 'next/dynamic'
import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'

const ModelComparison = dynamic(
  () => import('@/components/llm-simulators/ModelComparison'),
  { ssr: false }
)

export default function ModelComparisonSimulator() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <button
            onClick={() => window.history.back()}
            className="inline-flex items-center text-indigo-600 hover:text-indigo-700 dark:text-indigo-400"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            학습 모듈로 돌아가기
          </button>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            LLM Model Comparison Tool
          </h1>
          <p className="text-gray-600 dark:text-gray-300 mb-8">
            GPT, Claude, Gemini 등 주요 LLM 모델들의 특성과 성능을 비교 분석해보세요.
          </p>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 mb-8">
            <h2 className="text-lg font-semibold mb-4">🎮 사용 방법</h2>
            <ul className="space-y-2 text-gray-600 dark:text-gray-300">
              <li>• 비교할 모델들을 선택하세요 (최대 4개)</li>
              <li>• 파라미터 수, 학습 데이터, 특징 등을 한눈에 비교할 수 있습니다</li>
              <li>• 벤치마크 점수와 실제 사용 예시를 확인하세요</li>
              <li>• 각 모델의 장단점과 적합한 사용 사례를 파악하세요</li>
            </ul>
          </div>

          <div className="border-2 border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden">
            <ModelComparison />
          </div>

          <div className="mt-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-4">
              <h3 className="font-semibold text-blue-900 dark:text-blue-300 mb-2 text-sm">
                🤖 GPT Series
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-xs">
                OpenAI의 대표 모델. 범용성과 창의성이 뛰어남
              </p>
            </div>
            
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-4">
              <h3 className="font-semibold text-purple-900 dark:text-purple-300 mb-2 text-sm">
                🧠 Claude
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-xs">
                Anthropic의 안전 중심 모델. 긴 컨텍스트 처리 우수
              </p>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-4">
              <h3 className="font-semibold text-green-900 dark:text-green-300 mb-2 text-sm">
                💎 Gemini
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-xs">
                Google의 멀티모달 모델. 이미지/비디오 이해 가능
              </p>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 rounded-xl p-4">
              <h3 className="font-semibold text-orange-900 dark:text-orange-300 mb-2 text-sm">
                🦙 LLaMA
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-xs">
                Meta의 오픈소스 모델. 경량화와 효율성 중점
              </p>
            </div>
          </div>

          <div className="mt-6 p-6 bg-gradient-to-r from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-700 rounded-xl">
            <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
              📊 비교 기준
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="font-medium">모델 크기</span>
                <p className="text-gray-600 dark:text-gray-400 text-xs">파라미터 수</p>
              </div>
              <div>
                <span className="font-medium">성능</span>
                <p className="text-gray-600 dark:text-gray-400 text-xs">벤치마크 점수</p>
              </div>
              <div>
                <span className="font-medium">속도</span>
                <p className="text-gray-600 dark:text-gray-400 text-xs">추론 시간</p>
              </div>
              <div>
                <span className="font-medium">비용</span>
                <p className="text-gray-600 dark:text-gray-400 text-xs">API 가격</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}