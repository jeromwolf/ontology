'use client'

import Link from 'next/link'
import dynamic from 'next/dynamic'

const TimeSeriesForecasterPyCaret = dynamic(
  () => import('../../components/simulators/TimeSeriesForecasterPyCaret'),
  { 
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    )
  }
)

export default function TimeSeriesForecasterPyCaretPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-black">
      <div className="container mx-auto px-6 py-8">
        {/* 헤더 */}
        <div className="mb-8">
          <Link 
            href="/modules/data-science"
            className="text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 inline-flex items-center gap-2"
          >
            ← 데이터 사이언스로 돌아가기
          </Link>
          <h1 className="text-4xl font-bold mt-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            시계열 예측 with PyCaret
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            여러 시계열 모델을 자동으로 비교하고 최적의 예측을 생성합니다
          </p>
        </div>

        {/* 시뮬레이터 컴포넌트 */}
        <TimeSeriesForecasterPyCaret />
      </div>
    </div>
  )
}