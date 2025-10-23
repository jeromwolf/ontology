'use client'

import { useParams } from 'next/navigation'
import dynamic from 'next/dynamic'
import Link from 'next/link'
import { ChevronLeft, Construction } from 'lucide-react'

// Dynamic imports for all simulators
const ChainBuilder = dynamic(() => import('@/components/langchain-simulators/ChainBuilder'), { ssr: false })
const PromptTemplateDesigner = dynamic(() => import('@/components/langchain-simulators/PromptTemplateDesigner'), { ssr: false })
const MemoryPlayground = dynamic(() => import('@/components/langchain-simulators/MemoryPlayground'), { ssr: false })
const AgentToolsWorkshop = dynamic(() => import('@/components/langchain-simulators/AgentToolsWorkshop'), { ssr: false })
const LangGraphFlowDesigner = dynamic(() => import('@/components/langchain-simulators/LangGraphFlowDesigner'), { ssr: false })
const RAGPipelineBuilder = dynamic(() => import('@/components/langchain-simulators/RAGPipelineBuilder'), { ssr: false })
const VectorStoreComparison = dynamic(() => import('@/components/langchain-simulators/VectorStoreComparison'), { ssr: false })
const ChatHistoryManager = dynamic(() => import('@/components/langchain-simulators/ChatHistoryManager'), { ssr: false })
const TokenCostCalculator = dynamic(() => import('@/components/langchain-simulators/TokenCostCalculator'), { ssr: false })
const PerformanceProfiler = dynamic(() => import('@/components/langchain-simulators/PerformanceProfiler'), { ssr: false })
const MultiAgentCoordinator = dynamic(() => import('@/components/langchain-simulators/MultiAgentCoordinator'), { ssr: false })

export default function LangChainSimulatorPage() {
  const params = useParams()
  const simulatorId = params.simulatorId as string

  // Simulator mapping
  const simulatorComponents: Record<string, React.ComponentType> = {
    'chain-builder': ChainBuilder,
    'prompt-optimizer': PromptTemplateDesigner,
    'memory-manager': MemoryPlayground,
    'agent-debugger': AgentToolsWorkshop,
    'langgraph-designer': LangGraphFlowDesigner,
    'tool-integrator': AgentToolsWorkshop, // Reusing AgentToolsWorkshop
    'rag-pipeline': RAGPipelineBuilder,
    'async-executor': VectorStoreComparison, // Placeholder - similar functionality
    'cost-calculator': TokenCostCalculator,
    'performance-profiler': PerformanceProfiler,
    'multi-agent-coordinator': MultiAgentCoordinator
  }

  const SimulatorComponent = simulatorComponents[simulatorId]

  // Get simulator name for header
  const simulatorNames: Record<string, string> = {
    'chain-builder': 'Chain Builder',
    'prompt-optimizer': 'Prompt Template Designer',
    'memory-manager': 'Memory Playground',
    'agent-debugger': 'Agent Tools Workshop',
    'langgraph-designer': 'LangGraph Flow Designer',
    'tool-integrator': 'Tool Integrator',
    'rag-pipeline': 'RAG Pipeline Builder',
    'async-executor': 'Async Executor',
    'cost-calculator': 'Token Cost Calculator',
    'performance-profiler': 'Performance Profiler',
    'multi-agent-coordinator': 'Multi-Agent Coordinator'
  }

  const simulatorName = simulatorNames[simulatorId] || 'Simulator'

  // If simulator exists, render it with header
  if (SimulatorComponent) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-amber-50 via-white to-orange-50 dark:from-gray-900 dark:via-gray-800 dark:to-amber-900/20">
        {/* Header */}
        <header className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-md border-b border-amber-200/30 dark:border-amber-700/30 sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 py-4">
            <div className="flex items-center gap-4">
              <Link
                href="/modules/langchain"
                className="flex items-center gap-2 text-gray-600 dark:text-gray-300 hover:text-amber-600 dark:hover:text-amber-400 transition-colors"
              >
                <ChevronLeft size={20} />
                <span>LangChain 모듈로 돌아가기</span>
              </Link>
              <div className="text-gray-400 dark:text-gray-600">•</div>
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                {simulatorName}
              </span>
            </div>
          </div>
        </header>

        {/* Simulator Content */}
        <main>
          <SimulatorComponent />
        </main>
      </div>
    )
  }

  // Otherwise, show coming soon page
  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 via-white to-orange-50 dark:from-gray-900 dark:via-gray-800 dark:to-amber-900/20">
      {/* Header */}
      <header className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-md border-b border-amber-200/30 dark:border-amber-700/30 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center gap-4">
            <Link
              href="/modules/langchain"
              className="flex items-center gap-2 text-gray-600 dark:text-gray-300 hover:text-amber-600 dark:hover:text-amber-400 transition-colors"
            >
              <ChevronLeft size={20} />
              <span>LangChain 모듈로 돌아가기</span>
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 py-16">
        <div className="text-center">
          <div className="w-20 h-20 mx-auto rounded-3xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center text-white text-4xl mb-6 shadow-lg">
            <Construction size={40} />
          </div>

          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            {simulatorName}
          </h1>

          <p className="text-xl text-gray-600 dark:text-gray-400 mb-8">
            이 시뮬레이터는 현재 개발 중입니다
          </p>

          <div className="bg-amber-50 dark:bg-amber-900/20 rounded-xl p-8 mb-8 border border-amber-200 dark:border-amber-800">
            <h2 className="text-2xl font-bold text-amber-800 dark:text-amber-200 mb-4">
              🚧 Coming Soon
            </h2>
            <p className="text-gray-700 dark:text-gray-300 mb-6">
              LangChain 시뮬레이터는 곧 출시될 예정입니다. <br />
              학습 콘텐츠는 모든 챕터에서 이용 가능합니다.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/modules/langchain"
              className="px-8 py-3 bg-gradient-to-r from-amber-500 to-orange-600 text-white rounded-lg font-medium hover:shadow-lg transition-all duration-200"
            >
              모듈 홈으로
            </Link>
            <Link
              href="/modules/langchain/01-langchain-basics"
              className="px-8 py-3 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg font-medium hover:shadow-lg transition-all duration-200 border border-gray-200 dark:border-gray-700"
            >
              Chapter 1부터 학습
            </Link>
          </div>
        </div>
      </main>
    </div>
  )
}
