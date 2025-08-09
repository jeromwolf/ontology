'use client'

import { Suspense } from 'react'
import dynamic from 'next/dynamic'
import Navigation from '@/components/Navigation'
import { ArrowLeft } from 'lucide-react'
import Link from 'next/link'

// Dynamic import to avoid SSR issues
const ImageEnhancementStudio = dynamic(
  () => import('@/app/modules/computer-vision/components/ImageEnhancementStudio'),
  { 
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center min-h-[600px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-teal-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">이미지 향상 스튜디오 로딩 중...</p>
        </div>
      </div>
    )
  }
)

export default function ImageEnhancementStudioPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Navigation />
      
      <main className="container mx-auto px-4 py-8">
        <div className="mb-6">
          <Link 
            href="/modules/computer-vision"
            className="inline-flex items-center gap-2 text-teal-600 dark:text-teal-400 hover:text-teal-700 dark:hover:text-teal-300 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Computer Vision 모듈로 돌아가기
          </Link>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            이미지 향상 스튜디오
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mb-8">
            다양한 필터와 향상 기법으로 이미지를 개선하고 변환해보세요
          </p>
          
          <Suspense fallback={<div>로딩 중...</div>}>
            <ImageEnhancementStudio />
          </Suspense>
        </div>
      </main>
    </div>
  )
}