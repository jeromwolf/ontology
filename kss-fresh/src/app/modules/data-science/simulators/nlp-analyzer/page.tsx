'use client'

import Link from 'next/link'
import dynamic from 'next/dynamic'

const NLPAnalyzer = dynamic(
  () => import('../../components/simulators/NLPAnalyzer'),
  { 
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500"></div>
      </div>
    )
  }
)

export default function NLPAnalyzerPage() {
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
          <h1 className="text-4xl font-bold mt-4 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
            자연어 처리 분석기
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            토큰화, 개체명 인식, 감정 분석 등 NLP 기술을 체험해보세요
          </p>
        </div>

        {/* 시뮬레이터 컴포넌트 */}
        <NLPAnalyzer />
      </div>
    </div>
  )
}