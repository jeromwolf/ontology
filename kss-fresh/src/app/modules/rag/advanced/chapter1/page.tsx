'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, Network } from 'lucide-react'

export default function Chapter1Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/rag/advanced"
          className="inline-flex items-center gap-2 text-emerald-600 hover:text-emerald-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          고급 과정으로 돌아가기
        </Link>
        
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <Network size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 1: GraphRAG 아키텍처</h1>
              <p className="text-blue-100 text-lg">그래프 기반 지식 표현과 검색 시스템</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">GraphRAG 개요</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            GraphRAG는 지식을 그래프 구조로 표현하여 더 정확하고 맥락적인 정보 검색을 가능하게 하는 고급 RAG 아키텍처입니다.
          </p>
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
            <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-3">핵심 특징</h3>
            <ul className="text-blue-700 dark:text-blue-300 space-y-2">
              <li>• 엔티티 기반 지식 그래프 구축</li>
              <li>• 관계 중심의 정보 검색</li>
              <li>• 다중 홉 추론 지원</li>
              <li>• 컨텍스트 보존</li>
            </ul>
          </div>
        </section>
      </div>

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/advanced"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            고급 과정으로
          </Link>
          
          <Link
            href="/modules/rag/advanced/chapter2"
            className="inline-flex items-center gap-2 bg-blue-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-600 transition-colors"
          >
            다음: Multi-hop Reasoning
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}