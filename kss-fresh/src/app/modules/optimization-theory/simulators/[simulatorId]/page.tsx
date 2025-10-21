'use client'

import { useParams } from 'next/navigation'
import Link from 'next/link'
import dynamic from 'next/dynamic'
import { ChevronLeft } from 'lucide-react'
import { moduleMetadata } from '../../metadata'

// 동적 임포트로 시뮬레이터 컴포넌트 로드
const OptimizationVisualizer = dynamic(() => import('@/components/optimization-theory-simulators/OptimizationVisualizer'), { ssr: false })
const ConstraintVisualizer = dynamic(() => import('@/components/optimization-theory-simulators/ConstraintVisualizer'), { ssr: false })
const HyperparameterTuner = dynamic(() => import('@/components/optimization-theory-simulators/HyperparameterTuner'), { ssr: false })
const ParetoFrontier = dynamic(() => import('@/components/optimization-theory-simulators/ParetoFrontier'), { ssr: false })
const GeneticAlgorithm = dynamic(() => import('@/components/optimization-theory-simulators/GeneticAlgorithm'), { ssr: false })
const GradientExplorer = dynamic(() => import('@/components/optimization-theory-simulators/GradientExplorer'), { ssr: false })
const ConvexSolver = dynamic(() => import('@/components/optimization-theory-simulators/ConvexSolver'), { ssr: false })

export default function SimulatorPage() {
  const params = useParams()
  const simulatorId = params.simulatorId as string

  // 현재 시뮬레이터 정보 찾기
  const currentSimulator = moduleMetadata.simulators.find(sim => sim.id === simulatorId)

  const getSimulatorComponent = () => {
    switch (simulatorId) {
      case 'optimization-visualizer':
        return <OptimizationVisualizer />
      case 'constraint-visualizer':
        return <ConstraintVisualizer />
      case 'hyperparameter-tuner':
        return <HyperparameterTuner />
      case 'pareto-frontier':
        return <ParetoFrontier />
      case 'genetic-algorithm':
        return <GeneticAlgorithm />
      case 'gradient-explorer':
        return <GradientExplorer />
      case 'convex-solver':
        return <ConvexSolver />
      default:
        return (
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center">
              <h2 className="text-2xl font-bold mb-2">시뮬레이터를 찾을 수 없습니다</h2>
              <p className="text-gray-600 dark:text-gray-400">
                요청하신 시뮬레이터가 존재하지 않습니다.
              </p>
            </div>
          </div>
        )
    }
  }

  return (
    <div className="max-w-7xl mx-auto">
      {/* Breadcrumb & Back Button */}
      <div className="mb-6">
        <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-4">
          <Link
            href="/modules/optimization-theory"
            className="hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors"
          >
            Optimization Theory
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
          href="/modules/optimization-theory"
          className="inline-flex items-center gap-2 text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-500 transition-colors text-sm font-medium"
        >
          <ChevronLeft size={16} />
          모듈 홈으로 돌아가기
        </Link>
      </div>

      {/* Simulator Component */}
      <div className="mt-8">
        {getSimulatorComponent()}
      </div>
    </div>
  )
}
