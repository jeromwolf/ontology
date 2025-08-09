'use client'

import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { ChevronLeft, BarChart3 } from 'lucide-react'
import { probabilityStatisticsModule } from '../../metadata'
import Navigation from '@/components/Navigation'
import dynamic from 'next/dynamic'

// 동적 import로 시뮬레이터 컴포넌트 로드
const ProbabilityPlayground = dynamic(() => import('../../components/ProbabilityPlayground'), { 
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-64">로딩 중...</div>
})

const DistributionVisualizer = dynamic(() => import('../../components/DistributionVisualizer'), { 
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-64">로딩 중...</div>
})

const HypothesisTester = dynamic(() => import('../../components/HypothesisTester'), { 
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-64">로딩 중...</div>
})

const RegressionLab = dynamic(() => import('../../components/RegressionLab'), { 
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-64">로딩 중...</div>
})

const MonteCarloSimulator = dynamic(() => import('../../components/MonteCarloSimulator'), { 
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-64">로딩 중...</div>
})

export default function ProbabilityStatisticsSimulatorPage() {
  const params = useParams()
  const router = useRouter()
  const simulatorId = params.simulatorId as string
  
  const currentSimulator = probabilityStatisticsModule.simulators.find(sim => sim.id === simulatorId)

  if (!currentSimulator) {
    router.push('/modules/probability-statistics')
    return null
  }

  const renderSimulator = () => {
    switch (simulatorId) {
      case 'probability-playground':
        return <ProbabilityPlayground />
      case 'distribution-visualizer':
        return <DistributionVisualizer />
      case 'hypothesis-tester':
        return <HypothesisTester />
      case 'regression-lab':
        return <RegressionLab />
      case 'monte-carlo':
        return <MonteCarloSimulator />
      default:
        return <div>시뮬레이터를 찾을 수 없습니다.</div>
    }
  }

  return (
    <div className="min-h-screen">
      <Navigation />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Breadcrumb */}
        <nav className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-8">
          <Link href="/" className="hover:text-purple-600 dark:hover:text-purple-400">
            홈
          </Link>
          <ChevronLeft className="w-4 h-4 rotate-180" />
          <Link href="/modules/probability-statistics" className="hover:text-purple-600 dark:hover:text-purple-400">
            {probabilityStatisticsModule.name}
          </Link>
          <ChevronLeft className="w-4 h-4 rotate-180" />
          <span className="text-gray-900 dark:text-white">{currentSimulator.name}</span>
        </nav>

        {/* Simulator Header */}
        <div className="mb-8">
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 rounded-xl bg-gradient-to-br from-purple-100 to-pink-100 dark:from-purple-900/30 dark:to-pink-900/30">
              <BarChart3 className="w-6 h-6 text-purple-600 dark:text-purple-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                {currentSimulator.name}
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400 mt-2">
                {currentSimulator.description}
              </p>
            </div>
          </div>
        </div>

        {/* Simulator Content */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          {renderSimulator()}
        </div>

        {/* Back Button */}
        <div className="mt-8">
          <Link
            href="/modules/probability-statistics"
            className="inline-flex items-center gap-2 text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300"
          >
            <ChevronLeft className="w-5 h-5" />
            <span>모듈로 돌아가기</span>
          </Link>
        </div>
      </main>
    </div>
  )
}