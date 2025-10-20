'use client'

import dynamic from 'next/dynamic'
import { notFound } from 'next/navigation'

const ProjectileMotion = dynamic(() => import('@/components/physics-simulators/ProjectileMotion'), { ssr: false })
const CollisionLab = dynamic(() => import('@/components/physics-simulators/CollisionLab'), { ssr: false })
const PendulumSimulator = dynamic(() => import('@/components/physics-simulators/PendulumSimulator'), { ssr: false })
const ElectricField = dynamic(() => import('@/components/physics-simulators/ElectricField'), { ssr: false })
const WaveInterference = dynamic(() => import('@/components/physics-simulators/WaveInterference'), { ssr: false })
const ThermodynamicCycles = dynamic(() => import('@/components/physics-simulators/ThermodynamicCycles'), { ssr: false })

export default function PhysicsSimulatorPage({ params }: { params: { simulatorId: string } }) {
  const { simulatorId } = params

  const getSimulator = () => {
    switch (simulatorId) {
      case 'projectile-motion':
        return <ProjectileMotion />
      case 'collision-lab':
        return <CollisionLab />
      case 'pendulum-simulator':
        return <PendulumSimulator />
      case 'electric-field':
        return <ElectricField />
      case 'wave-interference':
        return <WaveInterference />
      case 'thermodynamic-cycles':
        return <ThermodynamicCycles />
      default:
        return notFound()
    }
  }

  return getSimulator()
}
