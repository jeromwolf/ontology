'use client'

import dynamic from 'next/dynamic'
import { notFound } from 'next/navigation'

const LimitCalculator = dynamic(() => import('@/components/calculus-simulators/LimitCalculator'), { ssr: false })
const DerivativeVisualizer = dynamic(() => import('@/components/calculus-simulators/DerivativeVisualizer'), { ssr: false })
const OptimizationLab = dynamic(() => import('@/components/calculus-simulators/OptimizationLab'), { ssr: false })
const IntegralCalculator = dynamic(() => import('@/components/calculus-simulators/IntegralCalculator'), { ssr: false })
const TaylorSeriesExplorer = dynamic(() => import('@/components/calculus-simulators/TaylorSeriesExplorer'), { ssr: false })
const GradientField = dynamic(() => import('@/components/calculus-simulators/GradientField'), { ssr: false })

export default function CalculusSimulatorPage({ params }: { params: { simulatorId: string } }) {
  const { simulatorId } = params

  const getSimulator = () => {
    switch (simulatorId) {
      case 'limit-calculator':
        return <LimitCalculator />
      case 'derivative-visualizer':
        return <DerivativeVisualizer />
      case 'optimization-lab':
        return <OptimizationLab />
      case 'integral-calculator':
        return <IntegralCalculator />
      case 'taylor-series-explorer':
        return <TaylorSeriesExplorer />
      case 'gradient-field':
        return <GradientField />
      default:
        return notFound()
    }
  }

  return getSimulator()
}
