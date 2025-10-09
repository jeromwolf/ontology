'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight } from 'lucide-react'
import dynamic from 'next/dynamic'

// Dynamic imports for sections
const Section1 = dynamic(() => import('./sections/Section1'), { ssr: false })
const Section2 = dynamic(() => import('./sections/Section2'), { ssr: false })
const Section3 = dynamic(() => import('./sections/Section3'), { ssr: false })
const Section4 = dynamic(() => import('./sections/Section4'), { ssr: false })
const Section5 = dynamic(() => import('./sections/Section5'), { ssr: false })
const Section6 = dynamic(() => import('./sections/Section6'), { ssr: false })
const Section7 = dynamic(() => import('./sections/Section7'), { ssr: false })
const Section8 = dynamic(() => import('./sections/Section8'), { ssr: false })

export default function Chapter3Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/rag/intermediate"
          className="inline-flex items-center gap-2 text-emerald-600 hover:text-emerald-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          중급 과정으로 돌아가기
        </Link>

        <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 3: RAG를 위한 프롬프트 엔지니어링</h1>
              <p className="text-pink-100 text-lg">검색 증강 생성을 최적화하는 프롬프트 전략</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        <Section1 />
        <Section2 />
        <Section3 />
        <Section4 />
        <Section5 />
        <Section6 />
        <Section7 />
        <Section8 />
      </div>

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/intermediate/chapter2"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: 하이브리드 검색 전략
          </Link>

          <Link
            href="/modules/rag/intermediate/chapter4"
            className="inline-flex items-center gap-2 bg-purple-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-purple-600 transition-colors"
          >
            다음: Advanced Reranking 전략
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}
