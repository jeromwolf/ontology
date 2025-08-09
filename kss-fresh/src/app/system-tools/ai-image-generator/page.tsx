'use client'

import dynamic from 'next/dynamic'
import { ArrowLeft } from 'lucide-react'

const AIImageGenerator = dynamic(
  () => import('@/components/system-tools/AIImageGenerator'),
  { ssr: false }
)

export default function AIImageGeneratorPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <button
            onClick={() => window.history.back()}
            className="inline-flex items-center text-indigo-600 hover:text-indigo-700 dark:text-indigo-400 transition-colors"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            System Tools로 돌아가기
          </button>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            🎨 AI Image Generator
          </h1>
          <p className="text-gray-600 dark:text-gray-300 mb-8">
            DALL-E 3를 활용해 교육 콘텐츠용 이미지를 생성하고 프로젝트에 바로 사용하세요.
          </p>

          <div className="border-2 border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden">
            <AIImageGenerator />
          </div>

          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6">
              <h3 className="font-semibold text-blue-900 dark:text-blue-300 mb-2">
                🎯 교육용 최적화
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                기술 다이어그램, 플로우차트, 아키텍처 등 교육 콘텐츠에 특화된 프롬프트 제공
              </p>
            </div>
            
            <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-6">
              <h3 className="font-semibold text-green-900 dark:text-green-300 mb-2">
                💾 자동 저장
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                생성된 이미지를 public 폴더에 자동 저장하여 바로 프로젝트에서 사용 가능
              </p>
            </div>
            
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6">
              <h3 className="font-semibold text-purple-900 dark:text-purple-300 mb-2">
                🔧 개발자 도구
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                생성된 이미지의 경로와 코드 예제를 제공하여 개발 워크플로우 최적화
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}