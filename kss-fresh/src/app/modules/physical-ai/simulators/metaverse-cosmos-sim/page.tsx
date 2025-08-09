'use client'

import { Suspense } from 'react'
import dynamic from 'next/dynamic'
import Navigation from '@/components/Navigation'
import { ArrowLeft } from 'lucide-react'
import Link from 'next/link'

// Dynamic import to avoid SSR issues
const MetaverseCosmosSimulator = dynamic(
  () => import('@/app/modules/physical-ai/components/MetaverseCosmosSimulator'),
  { 
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center min-h-[600px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-slate-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">메타버스 시뮬레이터 로딩 중...</p>
        </div>
      </div>
    )
  }
)

export default function MetaverseCosmosSimPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Navigation />
      
      <main className="container mx-auto px-4 py-8">
        <div className="mb-6">
          <Link 
            href="/modules/physical-ai"
            className="inline-flex items-center gap-2 text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Physical AI 모듈로 돌아가기
          </Link>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            메타버스 COSMOS 시뮬레이터
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mb-8">
            도시 규모 디지털 트윈과 메타버스 통합 환경을 체험해보세요
          </p>
          
          <Suspense fallback={<div>로딩 중...</div>}>
            <MetaverseCosmosSimulator />
          </Suspense>
        </div>
      </main>
    </div>
  )
}