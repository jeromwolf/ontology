'use client'

import Link from 'next/link'
import { ArrowRight } from 'lucide-react'

export default function Section6() {
  return (
    <section className="bg-gradient-to-r from-emerald-500 to-green-600 rounded-2xl p-8 text-white">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-2xl font-bold mb-2">🎮 하이브리드 검색 시뮬레이터</h3>
          <p className="text-emerald-100">BM25 + 벡터 검색 가중치를 조정하며 실시간 결과를 확인하세요</p>
        </div>
        <Link
          href="/modules/rag/simulators/hybrid-search-demo"
          className="inline-flex items-center gap-2 bg-white text-emerald-600 px-6 py-3 rounded-lg font-semibold hover:bg-emerald-50 transition-colors shadow-lg"
        >
          시뮬레이터 열기
          <ArrowRight size={20} />
        </Link>
      </div>
    </section>
  )
}
