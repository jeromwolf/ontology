'use client'

import dynamic from 'next/dynamic'
import { ArrowLeft } from 'lucide-react'
import Link from 'next/link'

// 동적 import로 SSR 문제 해결
const NewsImpactAnalyzer = dynamic(
  () => import('../../components/NewsImpactAnalyzer'),
  { 
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-red-600"></div>
      </div>
    )
  }
)

export default function NewsImpactAnalyzerPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-red-50 via-white to-orange-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      <div className="container mx-auto px-4 py-8">
        {/* 헤더 */}
        <div className="mb-8">
          <Link 
            href="/modules/stock-analysis"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors mb-4"
          >
            <ArrowLeft className="w-4 h-4" />
            주식투자분석으로 돌아가기
          </Link>
          
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            뉴스 영향도 온톨로지 분석
          </h1>
          
          <p className="text-lg text-gray-600 dark:text-gray-400">
            AI 기반 뉴스 분석으로 기업간 관계와 시장 영향도를 실시간으로 파악하세요
          </p>
        </div>

        {/* 시뮬레이터 */}
        <NewsImpactAnalyzer />
        
        {/* 사용 안내 */}
        <div className="mt-8 bg-white dark:bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
            📚 사용 가이드
          </h3>
          
          <div className="grid md:grid-cols-3 gap-6">
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                1. 종목 선택
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                분석하고 싶은 기업을 선택하면 관련 뉴스를 자동으로 수집합니다.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                2. 온톨로지 분석
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                기업간 관계(공급사, 경쟁사, 파트너)를 시각화하여 연쇄 영향을 파악합니다.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                3. 영향도 평가
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                직접/간접/섹터 영향도를 -100~+100 점수로 계량화하여 투자 의사결정을 지원합니다.
              </p>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <strong>💡 Pro Tip:</strong> 뉴스 영향도는 단기적 관점입니다. 
              장기 투자 시에는 기본적 분석과 함께 종합적으로 판단하세요.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}