'use client'

import { useParams } from 'next/navigation'
import Link from 'next/link'
import { ChevronLeft } from 'lucide-react'
import { moduleMetadata } from '../../metadata'

export default function SimulatorPage() {
  const params = useParams()
  const simulatorId = params.simulatorId as string

  // 현재 시뮬레이터 정보 찾기
  const currentSimulator = moduleMetadata.simulators.find(sim => sim.id === simulatorId)

  return (
    <div className="max-w-7xl mx-auto">
      {/* Breadcrumb & Back Button */}
      <div className="mb-6">
        <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-4">
          <Link
            href="/modules/hpc-computing"
            className="hover:text-yellow-600 dark:hover:text-yellow-400 transition-colors"
          >
            HPC Computing
          </Link>
          <span>/</span>
          <span>시뮬레이터</span>
        </div>

        {currentSimulator && (
          <div className="mb-4">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              {currentSimulator.title}
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-400">
              {currentSimulator.description}
            </p>
          </div>
        )}

        <Link
          href="/modules/hpc-computing"
          className="inline-flex items-center gap-2 text-yellow-600 dark:text-yellow-400 hover:text-yellow-700 dark:hover:text-yellow-500 transition-colors text-sm font-medium"
        >
          <ChevronLeft size={16} />
          모듈 홈으로 돌아가기
        </Link>
      </div>

      {/* Simulator Placeholder */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-12 text-center">
        <div className="max-w-md mx-auto">
          <div className="text-6xl mb-4">⚡</div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            시뮬레이터 준비 중
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            {currentSimulator?.title} 시뮬레이터가 곧 추가됩니다.
          </p>
          <Link
            href="/modules/hpc-computing"
            className="inline-flex items-center gap-2 bg-gradient-to-r from-yellow-500 to-orange-600 text-white px-6 py-3 rounded-lg font-semibold hover:shadow-lg transition-all"
          >
            모듈 홈으로
          </Link>
        </div>
      </div>
    </div>
  )
}
