'use client'

import { useParams } from 'next/navigation'
import Link from 'next/link'
import { ChevronLeft } from 'lucide-react'
import { moduleMetadata } from '../../metadata'
import dynamic from 'next/dynamic'

// Dynamic imports for simulators
const CUDAKernelAnalyzer = dynamic(() => import('@/components/hpc-computing-simulators/CUDAKernelAnalyzer'), { ssr: false })
const ParallelAlgorithmViz = dynamic(() => import('@/components/hpc-computing-simulators/ParallelAlgorithmViz'), { ssr: false })
const GPUMemoryOptimizer = dynamic(() => import('@/components/hpc-computing-simulators/GPUMemoryOptimizer'), { ssr: false })
const ClusterScheduler = dynamic(() => import('@/components/hpc-computing-simulators/ClusterScheduler'), { ssr: false })
const MPIDebugger = dynamic(() => import('@/components/hpc-computing-simulators/MPIDebugger'), { ssr: false })
const PerformanceProfiler = dynamic(() => import('@/components/hpc-computing-simulators/PerformanceProfiler'), { ssr: false })
const LoadBalancer = dynamic(() => import('@/components/hpc-computing-simulators/LoadBalancer'), { ssr: false })
const HPCBenchmark = dynamic(() => import('@/components/hpc-computing-simulators/HPCBenchmark'), { ssr: false })

export default function SimulatorPage() {
  const params = useParams()
  const simulatorId = params.simulatorId as string

  // 현재 시뮬레이터 정보 찾기
  const currentSimulator = moduleMetadata.simulators.find(sim => sim.id === simulatorId)

  const getSimulatorComponent = () => {
    switch (simulatorId) {
      case 'cuda-kernel-analyzer':
        return <CUDAKernelAnalyzer />
      case 'parallel-algorithm-viz':
        return <ParallelAlgorithmViz />
      case 'gpu-memory-optimizer':
        return <GPUMemoryOptimizer />
      case 'cluster-scheduler':
        return <ClusterScheduler />
      case 'mpi-debugger':
        return <MPIDebugger />
      case 'performance-profiler':
        return <PerformanceProfiler />
      case 'load-balancer':
        return <LoadBalancer />
      case 'hpc-benchmark':
        return <HPCBenchmark />
      default:
        return (
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-12 text-center">
            <div className="max-w-md mx-auto">
              <div className="text-6xl mb-4">⚡</div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                시뮬레이터를 찾을 수 없습니다
              </h2>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                요청하신 시뮬레이터가 존재하지 않습니다.
              </p>
              <Link
                href="/modules/hpc-computing"
                className="inline-flex items-center gap-2 bg-gradient-to-r from-yellow-500 to-orange-600 text-white px-6 py-3 rounded-lg font-semibold hover:shadow-lg transition-all"
              >
                모듈 홈으로
              </Link>
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
            href="/modules/hpc-computing"
            className="hover:text-yellow-600 dark:hover:text-yellow-400 transition-colors"
          >
            HPC Computing
          </Link>
          <span>/</span>
          <span>시뮬레이터</span>
        </div>

        <Link
          href="/modules/hpc-computing"
          className="inline-flex items-center gap-2 text-yellow-600 dark:text-yellow-400 hover:text-yellow-700 dark:hover:text-yellow-500 transition-colors text-sm font-medium"
        >
          <ChevronLeft size={16} />
          모듈 홈으로 돌아가기
        </Link>
      </div>

      {/* Simulator Component */}
      {getSimulatorComponent()}
    </div>
  )
}
