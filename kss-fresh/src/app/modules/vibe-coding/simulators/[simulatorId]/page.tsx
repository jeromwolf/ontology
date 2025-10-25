'use client'

import { use } from 'react'
import Link from 'next/link'
import { ChevronLeft } from 'lucide-react'
import dynamic from 'next/dynamic'

// Dynamic imports for simulators
const AICodeAssistant = dynamic(() => import('@/components/vibe-coding-simulators/AICodeAssistant'), { ssr: false })
const PromptOptimizer = dynamic(() => import('@/components/vibe-coding-simulators/PromptOptimizer'), { ssr: false })
const CodeReviewAI = dynamic(() => import('@/components/vibe-coding-simulators/CodeReviewAI'), { ssr: false })
const RefactoringEngine = dynamic(() => import('@/components/vibe-coding-simulators/RefactoringEngine'), { ssr: false })
const TestGenerator = dynamic(() => import('@/components/vibe-coding-simulators/TestGenerator'), { ssr: false })
const DocGenerator = dynamic(() => import('@/components/vibe-coding-simulators/DocGenerator'), { ssr: false })

interface SimulatorPageProps {
  params: Promise<{ simulatorId: string }>
}

export default function SimulatorPage({ params }: SimulatorPageProps) {
  const { simulatorId } = use(params)

  const getSimulatorComponent = () => {
    switch (simulatorId) {
      case 'ai-code-assistant':
        return <AICodeAssistant />
      case 'prompt-optimizer':
        return <PromptOptimizer />
      case 'code-review-ai':
        return <CodeReviewAI />
      case 'refactoring-engine':
        return <RefactoringEngine />
      case 'test-generator':
        return <TestGenerator />
      case 'doc-generator':
        return <DocGenerator />
      default:
        return <ComingSoon />
    }
  }

  return (
    <div className="min-h-screen">
      {/* Back Button */}
      <div className="mb-6">
        <Link
          href="/modules/vibe-coding"
          className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-300 hover:text-purple-600 dark:hover:text-purple-400 transition-colors"
        >
          <ChevronLeft size={20} />
          <span>Vibe Coding 모듈로 돌아가기</span>
        </Link>
      </div>

      {/* Simulator Content */}
      {getSimulatorComponent()}
    </div>
  )
}

function ComingSoon() {
  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="text-center">
        <div className="text-6xl mb-4">🚧</div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
          시뮬레이터 준비 중
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          이 시뮬레이터는 곧 업데이트될 예정입니다.
        </p>
      </div>
    </div>
  )
}
