'use client'

import { Suspense } from 'react'
import dynamic from 'next/dynamic'
import Navigation from '@/components/Navigation'
import { ArrowLeft } from 'lucide-react'
import Link from 'next/link'

// Dynamic import to avoid SSR issues
const GraphRAGExplorer = dynamic(
  () => import('@/app/modules/rag/components/GraphRAGExplorer'),
  { 
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center min-h-[600px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">GraphRAG 탐색기 로딩 중...</p>
        </div>
      </div>
    )
  }
)

export default function GraphRAGExplorerPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Navigation />
      
      <main className="container mx-auto px-4 py-8">
        <div className="mb-6">
          <Link 
            href="/modules/rag"
            className="inline-flex items-center gap-2 text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            RAG 모듈로 돌아가기
          </Link>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            GraphRAG 탐색기
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mb-8">
            문서를 지식 그래프로 변환하고 Neo4j 스타일의 그래프 기반 RAG를 체험해보세요
          </p>
          
          <Suspense fallback={<div>로딩 중...</div>}>
            <GraphRAGExplorer />
          </Suspense>
        </div>
      </main>
    </div>
  )
}