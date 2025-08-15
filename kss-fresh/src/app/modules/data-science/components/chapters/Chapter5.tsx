'use client';

import dynamic from 'next/dynamic';

const Introduction = dynamic(() => import('./chapter5-sections/Introduction'), { ssr: false })
const ClusteringAlgorithms = dynamic(() => import('./chapter5-sections/ClusteringAlgorithms'), { ssr: false })
const DimensionReduction = dynamic(() => import('./chapter5-sections/DimensionReduction'), { ssr: false })
const EvaluationMetrics = dynamic(() => import('./chapter5-sections/EvaluationMetrics'), { ssr: false })
const PracticalGuide = dynamic(() => import('./chapter5-sections/PracticalGuide'), { ssr: false })

interface ChapterProps {
  onComplete?: () => void
}

export default function Chapter5({ onComplete }: ChapterProps) {
  return (
    <div className="space-y-8">
      <Introduction />
      <ClusteringAlgorithms />
      <DimensionReduction />
      <EvaluationMetrics />
      <PracticalGuide onComplete={onComplete} />
    </div>
  )
}