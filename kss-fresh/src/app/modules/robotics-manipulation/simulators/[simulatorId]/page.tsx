'use client'

import { useParams } from 'next/navigation'
import dynamic from 'next/dynamic'
import Navigation from '@/components/Navigation'
import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'

// Dynamic imports for all simulators
const ForwardKinematicsLab = dynamic(
  () => import('@/components/robotics-simulators/ForwardKinematicsLab'),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center min-h-[600px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-orange-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">로딩 중...</p>
        </div>
      </div>
    )
  }
)

const InverseKinematicsSolver = dynamic(
  () => import('@/components/robotics-simulators/InverseKinematicsSolver'),
  { ssr: false }
)

const PathPlanningVisualizer = dynamic(
  () => import('@/components/robotics-simulators/PathPlanningVisualizer'),
  { ssr: false }
)

const TrajectoryGenerator = dynamic(
  () => import('@/components/robotics-simulators/TrajectoryGenerator'),
  { ssr: false }
)

const GripperForceSimulator = dynamic(
  () => import('@/components/robotics-simulators/GripperForceSimulator'),
  { ssr: false }
)

const PickAndPlaceLab = dynamic(
  () => import('@/components/robotics-simulators/PickAndPlaceLab'),
  { ssr: false }
)

export default function SimulatorPage() {
  const params = useParams()
  const simulatorId = params?.simulatorId as string

  const getSimulatorComponent = () => {
    switch (simulatorId) {
      case 'forward-kinematics-lab':
        return <ForwardKinematicsLab />
      case 'inverse-kinematics-solver':
        return <InverseKinematicsSolver />
      case 'path-planning-visualizer':
        return <PathPlanningVisualizer />
      case 'trajectory-generator':
        return <TrajectoryGenerator />
      case 'gripper-force-simulator':
        return <GripperForceSimulator />
      case 'pick-and-place-lab':
        return <PickAndPlaceLab />
      default:
        return (
          <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                시뮬레이터를 찾을 수 없습니다
              </h2>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                요청하신 시뮬레이터가 존재하지 않습니다.
              </p>
              <Link
                href="/modules/robotics-manipulation"
                className="inline-flex items-center gap-2 px-6 py-3 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
                모듈로 돌아가기
              </Link>
            </div>
          </div>
        )
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Navigation />
      {getSimulatorComponent()}
    </div>
  )
}
