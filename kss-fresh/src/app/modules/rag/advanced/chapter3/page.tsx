'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, Server } from 'lucide-react'

export default function Chapter3Page() {
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
        
        <div className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <Server size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 3: 분산 시스템 구축</h1>
              <p className="text-indigo-100 text-lg">대규모 RAG 시스템의 분산 처리와 확장성</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">분산 RAG 시스템</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            대규모 데이터와 높은 처리량을 요구하는 환경에서 RAG 시스템을 분산 처리하는 방법을 다룹니다.
          </p>
          <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-xl">
            <h3 className="font-bold text-indigo-800 dark:text-indigo-200 mb-3">핵심 요소</h3>
            <ul className="text-indigo-700 dark:text-indigo-300 space-y-2">
              <li>• 벡터 데이터베이스 클러스터링</li>
              <li>• 로드 밸런싱과 트래픽 분산</li>
              <li>• 캐싱 전략 및 성능 최적화</li>
              <li>• 장애 복구와 고가용성</li>
            </ul>
          </div>
        </section>
      </div>

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/advanced/chapter2"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: Multi-hop Reasoning
          </Link>
          
          <Link
            href="/modules/rag/advanced"
            className="inline-flex items-center gap-2 bg-indigo-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-indigo-600 transition-colors"
          >
            고급 과정 완료
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}