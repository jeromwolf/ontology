'use client'

import { Suspense } from 'react'
import dynamic from 'next/dynamic'
import Navigation from '@/components/Navigation'
import { ArrowLeft } from 'lucide-react'
import Link from 'next/link'

// Dynamic import to avoid SSR issues
const OmniversePhysicsLab = dynamic(
  () => import('@/app/modules/physical-ai/components/OmniversePhysicsLab'),
  { 
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center min-h-[600px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-slate-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">물리 실험실 로딩 중...</p>
        </div>
      </div>
    )
  }
)

export default function OmniversePhysicsLabPage() {
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
            Omniverse 물리 실험실
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mb-8">
            실시간 물리 시뮬레이션과 현실-가상 동기화를 체험해보세요
          </p>
          
          <Suspense fallback={<div>로딩 중...</div>}>
            <OmniversePhysicsLab />
          </Suspense>
        </div>
      </main>
    </div>
  )
}