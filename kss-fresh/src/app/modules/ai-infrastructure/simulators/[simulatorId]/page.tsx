'use client'

import { use } from 'react'
import dynamic from 'next/dynamic'

// 동적 임포트로 시뮬레이터 컴포넌트 로드
const GPUClusterMonitor = dynamic(() => import('@/components/ai-infrastructure-simulators/GPUClusterMonitor'), { ssr: false })
const DistributedTrainingVisualizer = dynamic(() => import('@/components/ai-infrastructure-simulators/DistributedTrainingVisualizer'), { ssr: false })
const ModelServingSimulator = dynamic(() => import('@/components/ai-infrastructure-simulators/ModelServingSimulator'), { ssr: false })
const FeatureStoreExplorer = dynamic(() => import('@/components/ai-infrastructure-simulators/FeatureStoreExplorer'), { ssr: false })
const MLPipelineBuilder = dynamic(() => import('@/components/ai-infrastructure-simulators/MLPipelineBuilder'), { ssr: false })
const ExperimentTracker = dynamic(() => import('@/components/ai-infrastructure-simulators/ExperimentTracker'), { ssr: false })
const DataDriftDetector = dynamic(() => import('@/components/ai-infrastructure-simulators/DataDriftDetector'), { ssr: false })
const ResourceOptimizer = dynamic(() => import('@/components/ai-infrastructure-simulators/ResourceOptimizer'), { ssr: false })
const CICDPipelineVisualizer = dynamic(() => import('@/components/ai-infrastructure-simulators/CICDPipelineVisualizer'), { ssr: false })
const MLOpsArchitectDashboard = dynamic(() => import('@/components/ai-infrastructure-simulators/MLOpsArchitectDashboard'), { ssr: false })

interface PageProps {
  params: Promise<{
    simulatorId: string
  }>
}

export default function SimulatorPage({ params }: PageProps) {
  const { simulatorId } = use(params)

  const getSimulatorComponent = () => {
    switch (simulatorId) {
      case 'gpu-cluster-monitor':
        return <GPUClusterMonitor />
      case 'distributed-training-visualizer':
        return <DistributedTrainingVisualizer />
      case 'model-serving-simulator':
        return <ModelServingSimulator />
      case 'feature-store-explorer':
        return <FeatureStoreExplorer />
      case 'ml-pipeline-builder':
        return <MLPipelineBuilder />
      case 'experiment-tracker':
        return <ExperimentTracker />
      case 'data-drift-detector':
        return <DataDriftDetector />
      case 'resource-optimizer':
        return <ResourceOptimizer />
      case 'cicd-pipeline-visualizer':
        return <CICDPipelineVisualizer />
      case 'mlops-architect-dashboard':
        return <MLOpsArchitectDashboard />
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
