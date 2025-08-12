'use client'

import Link from 'next/link'
import dynamic from 'next/dynamic'

const ClassificationModelComparator = dynamic(
  () => import('../../components/simulators/ClassificationModelComparator'),
  { 
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    )
  }
)

export default function ClassificationModelComparatorPage() {
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
          <h1 className="text-4xl font-bold mt-4 bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            분류 모델 비교 실험실
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            6가지 분류 알고리즘의 성능을 비교하고 혼동행렬을 분석합니다
          </p>
        </div>

        {/* 시뮬레이터 컴포넌트 */}
        <ClassificationModelComparator />
      </div>
    </div>
  )
}