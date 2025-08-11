'use client'

import dynamic from 'next/dynamic'

const Introduction = dynamic(() => import('./chapter6-sections/Introduction'), { ssr: false })
const NeuralNetworkBasics = dynamic(() => import('./chapter6-sections/NeuralNetworkBasics'), { ssr: false })
const DeepLearningFrameworks = dynamic(() => import('./chapter6-sections/DeepLearningFrameworks'), { ssr: false })
const CNNandRNN = dynamic(() => import('./chapter6-sections/CNNandRNN'), { ssr: false })
const PracticalTips = dynamic(() => import('./chapter6-sections/PracticalTips'), { ssr: false })

interface ChapterProps {
  onComplete?: () => void
}

export default function Chapter6({ onComplete }: ChapterProps) {
  return (
    <div className="space-y-8">
      <Introduction />
      <NeuralNetworkBasics />
      <DeepLearningFrameworks />
      <CNNandRNN />
      <PracticalTips onComplete={onComplete} />
    </div>
  )
}