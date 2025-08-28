'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, Server } from 'lucide-react'
import dynamic from 'next/dynamic'

// 동적 임포트로 섹션 컴포넌트들 로드 (성능 최적화)
const Section1MonitoringSystem = dynamic(() => import('./components/Section1MonitoringSystem'), { ssr: false })
const Section2ABTesting = dynamic(() => import('./components/Section2ABTesting'), { ssr: false })
const Section3SecurityPrivacy = dynamic(() => import('./components/Section3SecurityPrivacy'), { ssr: false })
const Section4ScalingStrategies = dynamic(() => import('./components/Section4ScalingStrategies'), { ssr: false })
const Section5APIDeployment = dynamic(() => import('./components/Section5APIDeployment'), { ssr: false })
const Section6MonitoringDashboards = dynamic(() => import('./components/Section6MonitoringDashboards'), { ssr: false })

export default function Chapter6Page() {
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
        
        <div className="bg-gradient-to-r from-emerald-500 to-teal-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <Server size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 6: Production RAG Systems</h1>
              <p className="text-emerald-100 text-lg">실제 운영 환경에서의 RAG 시스템 구축 및 관리</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content - 각 섹션을 별도 컴포넌트로 분리 */}
      <div className="space-y-8">
        <Section1MonitoringSystem />
        <Section2ABTesting />
        <Section3SecurityPrivacy />
        <Section4ScalingStrategies />
        <Section5APIDeployment />
        <Section6MonitoringDashboards />
      </div>

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/intermediate/chapter5"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: 멀티모달 RAG
          </Link>
          
          <Link
            href="/modules/rag/intermediate"
            className="inline-flex items-center gap-2 bg-emerald-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-emerald-600 transition-colors"
          >
            중급 과정 완료!
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}