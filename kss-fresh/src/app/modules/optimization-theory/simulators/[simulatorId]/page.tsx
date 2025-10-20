'use client'

import { use } from 'react'
import dynamic from 'next/dynamic'

// 동적 임포트로 시뮬레이터 컴포넌트 로드
const OptimizationVisualizer = dynamic(() => import('@/components/optimization-theory-simulators/OptimizationVisualizer'), { ssr: false })
const ConstraintVisualizer = dynamic(() => import('@/components/optimization-theory-simulators/ConstraintVisualizer'), { ssr: false })
const HyperparameterTuner = dynamic(() => import('@/components/optimization-theory-simulators/HyperparameterTuner'), { ssr: false })
const ParetoFrontier = dynamic(() => import('@/components/optimization-theory-simulators/ParetoFrontier'), { ssr: false })
const GeneticAlgorithm = dynamic(() => import('@/components/optimization-theory-simulators/GeneticAlgorithm'), { ssr: false })
const GradientExplorer = dynamic(() => import('@/components/optimization-theory-simulators/GradientExplorer'), { ssr: false })
const ConvexSolver = dynamic(() => import('@/components/optimization-theory-simulators/ConvexSolver'), { ssr: false })

interface PageProps {
  params: Promise<{
    simulatorId: string
  }>
}

export default function SimulatorPage({ params }: PageProps) {
  const { simulatorId } = use(params)

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
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {getSimulatorComponent()}
    </div>
  )
}
