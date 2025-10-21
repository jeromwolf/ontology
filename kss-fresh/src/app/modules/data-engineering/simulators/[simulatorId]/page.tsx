'use client'

import { useParams } from 'next/navigation'
import Link from 'next/link'
import dynamic from 'next/dynamic'
import { ChevronLeft } from 'lucide-react'
import { moduleMetadata } from '../../metadata'

// 동적 임포트로 시뮬레이터 컴포넌트 로드
const EDAPlayground = dynamic(() => import('@/components/data-engineering-simulators/EDAPlayground'), { ssr: false })
const ETLPipelineDesigner = dynamic(() => import('@/components/data-engineering-simulators/ETLPipelineDesigner'), { ssr: false })
const StreamProcessingLab = dynamic(() => import('@/components/data-engineering-simulators/StreamProcessingLab'), { ssr: false })
const DataLakehouseArchitect = dynamic(() => import('@/components/data-engineering-simulators/DataLakehouseArchitect'), { ssr: false })
const AirflowDAGBuilder = dynamic(() => import('@/components/data-engineering-simulators/AirflowDAGBuilder'), { ssr: false })
const SparkOptimizer = dynamic(() => import('@/components/data-engineering-simulators/SparkOptimizer'), { ssr: false })
const DataQualitySuite = dynamic(() => import('@/components/data-engineering-simulators/DataQualitySuite'), { ssr: false })
const CloudCostCalculator = dynamic(() => import('@/components/data-engineering-simulators/CloudCostCalculator'), { ssr: false })
const DataLineageExplorer = dynamic(() => import('@/components/data-engineering-simulators/DataLineageExplorer'), { ssr: false })
const SQLPerformanceTuner = dynamic(() => import('@/components/data-engineering-simulators/SQLPerformanceTuner'), { ssr: false })

export default function SimulatorPage() {
  const params = useParams()
  const simulatorId = params.simulatorId as string

  // 현재 시뮬레이터 정보 찾기
  const currentSimulator = moduleMetadata.simulators.find(sim => sim.id === simulatorId)

  const getSimulatorComponent = () => {
    switch (simulatorId) {
      case 'eda-playground':
        return <EDAPlayground />
      case 'etl-pipeline-designer':
        return <ETLPipelineDesigner />
      case 'stream-processing-lab':
        return <StreamProcessingLab />
      case 'data-lakehouse-architect':
        return <DataLakehouseArchitect />
      case 'airflow-dag-builder':
        return <AirflowDAGBuilder />
      case 'spark-optimizer':
        return <SparkOptimizer />
      case 'data-quality-suite':
        return <DataQualitySuite />
      case 'cloud-cost-calculator':
        return <CloudCostCalculator />
      case 'data-lineage-explorer':
        return <DataLineageExplorer />
      case 'sql-performance-tuner':
        return <SQLPerformanceTuner />
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
            href="/modules/data-engineering"
            className="hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
          >
            Data Engineering
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
          href="/modules/data-engineering"
          className="inline-flex items-center gap-2 text-indigo-600 dark:text-indigo-400 hover:text-indigo-700 dark:hover:text-indigo-500 transition-colors text-sm font-medium"
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
