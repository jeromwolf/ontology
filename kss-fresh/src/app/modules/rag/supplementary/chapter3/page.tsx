'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, DollarSign } from 'lucide-react'

export default function Chapter3Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      <div className="mb-8">
        <Link
          href="/modules/rag/supplementary"
          className="inline-flex items-center gap-2 text-emerald-600 hover:text-emerald-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          보충 과정으로 돌아가기
        </Link>
        
        <div className="bg-gradient-to-r from-green-500 to-teal-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <DollarSign size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 3: 비용 최적화</h1>
              <p className="text-green-100 text-lg">RAG 시스템 운영 비용을 80% 절감하는 전략</p>
            </div>
          </div>
        </div>
      </div>

      <div className="space-y-8">
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">비용 최적화 전략</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            RAG 시스템의 운영 비용을 체계적으로 분석하고 최적화하는 방법을 다룹니다.
          </p>
          <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
            <h3 className="font-bold text-green-800 dark:text-green-200 mb-3">주요 절감 포인트</h3>
            <ul className="text-green-700 dark:text-green-300 space-y-2">
              <li>• 벡터 DB 인덱싱 최적화</li>
              <li>• 캐싱 전략 개선</li>
              <li>• 배치 처리 효율화</li>
              <li>• 스토리지 비용 절감</li>
            </ul>
          </div>
        </section>
      </div>

      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/supplementary/chapter2"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: 보안 및 프라이버시
          </Link>
          
          <Link
            href="/modules/rag/supplementary/chapter4"
            className="inline-flex items-center gap-2 bg-green-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-green-600 transition-colors"
          >
            다음: 복구 시스템
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}