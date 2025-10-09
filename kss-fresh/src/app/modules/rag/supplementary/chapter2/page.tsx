'use client'

import Link from 'next/link'
import { ArrowLeft, Shield } from 'lucide-react'
import dynamic from 'next/dynamic'

// Dynamic imports for sections
const Section1 = dynamic(() => import('./sections/Section1'), { ssr: false })
const Section2 = dynamic(() => import('./sections/Section2'), { ssr: false })
const Section3 = dynamic(() => import('./sections/Section3'), { ssr: false })
const Section4 = dynamic(() => import('./sections/Section4'), { ssr: false })

export default function Chapter2Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/rag/supplementary"
          className="inline-flex items-center gap-2 text-purple-600 hover:text-purple-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          보충 과정으로 돌아가기
        </Link>

        <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <Shield size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 2: Security & Privacy</h1>
              <p className="text-purple-100 text-lg">Production RAG의 보안과 개인정보 보호 전략</p>
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
      </div>
    </div>
  )
}
