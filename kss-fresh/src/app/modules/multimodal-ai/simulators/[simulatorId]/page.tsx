'use client'

import dynamic from 'next/dynamic'

// 동적 임포트로 시뮬레이터 컴포넌트 로드
const MultimodalArchitect = dynamic(() => import('@/components/multimodal-ai-simulators/MultimodalArchitect'), { ssr: false })
const CLIPExplorer = dynamic(() => import('@/components/multimodal-ai-simulators/CLIPExplorer'), { ssr: false })
const RealtimePipeline = dynamic(() => import('@/components/multimodal-ai-simulators/RealtimePipeline'), { ssr: false })
const CrossmodalSearch = dynamic(() => import('@/components/multimodal-ai-simulators/CrossmodalSearch'), { ssr: false })
const FusionLab = dynamic(() => import('@/components/multimodal-ai-simulators/FusionLab'), { ssr: false })
const VQASystem = dynamic(() => import('@/components/multimodal-ai-simulators/VQASystem'), { ssr: false })

interface PageProps {
  params: {
    simulatorId: string
  }
}

export default function SimulatorPage({ params }: PageProps) {
  const { simulatorId } = params

  const getSimulatorComponent = () => {
    switch (simulatorId) {
      case 'multimodal-architect':
        return <MultimodalArchitect />
      case 'clip-explorer':
        return <CLIPExplorer />
      case 'realtime-pipeline':
        return <RealtimePipeline />
      case 'crossmodal-search':
        return <CrossmodalSearch />
      case 'fusion-lab':
        return <FusionLab />
      case 'vqa-system':
        return <VQASystem />
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
