'use client'

import React, { useState } from 'react'
import { Activity, Play, AlertTriangle, TrendingUp, Clock } from 'lucide-react'

interface ProfileMetrics {
  totalTime: number
  llmTime: number
  retrievalTime: number
  processingTime: number
  tokenCount: number
  cacheHits: number
  cacheMisses: number
  parallelOps: number
}

interface ChainStep {
  name: string
  time: number
  tokens: number
  cached: boolean
  type: 'llm' | 'retrieval' | 'processing' | 'cache'
}

export default function PerformanceProfiler() {
  const [chainType, setChainType] = useState<'simple' | 'complex' | 'parallel'>('simple')
  const [enableCache, setEnableCache] = useState(true)
  const [enableParallel, setEnableParallel] = useState(false)
  const [metrics, setMetrics] = useState<ProfileMetrics | null>(null)
  const [steps, setSteps] = useState<ChainStep[]>([])
  const [profiling, setProfiling] = useState(false)
  const [progress, setProgress] = useState(0)

  const runProfile = async () => {
    setProfiling(true)
    setProgress(0)
    setSteps([])

    const newSteps: ChainStep[] = []
    let totalTime = 0
    let llmTime = 0
    let retrievalTime = 0
    let processingTime = 0
    let tokenCount = 0
    let cacheHits = 0
    let cacheMisses = 0
    let parallelOps = 0

    // Simulate different chain types
    if (chainType === 'simple') {
      // Simple chain: Prompt -> LLM -> Parse
      const step1: ChainStep = {
        name: 'Prompt Template',
        time: 5,
        tokens: 0,
        cached: false,
        type: 'processing'
      }
      newSteps.push(step1)
      processingTime += step1.time
      totalTime += step1.time
      setSteps([...newSteps])
      setProgress(33)
      await delay(300)

      const step2: ChainStep = {
        name: 'LLM Call',
        time: enableCache && Math.random() > 0.5 ? 50 : 800,
        tokens: 1500,
        cached: enableCache && Math.random() > 0.5,
        type: 'llm'
      }
      newSteps.push(step2)
      llmTime += step2.time
      tokenCount += step2.tokens
      totalTime += step2.time
      if (step2.cached) cacheHits++
      else cacheMisses++
      setSteps([...newSteps])
      setProgress(66)
      await delay(step2.time)

      const step3: ChainStep = {
        name: 'Output Parser',
        time: 10,
        tokens: 0,
        cached: false,
        type: 'processing'
      }
      newSteps.push(step3)
      processingTime += step3.time
      totalTime += step3.time
      setSteps([...newSteps])
      setProgress(100)
      await delay(300)
    } else if (chainType === 'complex') {
      // Complex chain: Retrieval -> Prompt -> LLM -> Parse -> LLM -> Parse
      const retrievalStep: ChainStep = {
        name: 'Vector Retrieval',
        time: enableCache && Math.random() > 0.3 ? 20 : 150,
        tokens: 0,
        cached: enableCache && Math.random() > 0.3,
        type: 'retrieval'
      }
      newSteps.push(retrievalStep)
      retrievalTime += retrievalStep.time
      totalTime += retrievalStep.time
      if (retrievalStep.cached) cacheHits++
      else cacheMisses++
      setSteps([...newSteps])
      setProgress(16)
      await delay(retrievalStep.time)

      for (let i = 0; i < 2; i++) {
        const promptStep: ChainStep = {
          name: `Prompt ${i + 1}`,
          time: 8,
          tokens: 0,
          cached: false,
          type: 'processing'
        }
        newSteps.push(promptStep)
        processingTime += promptStep.time
        totalTime += promptStep.time
        setSteps([...newSteps])
        setProgress(16 + (i * 42) + 7)
        await delay(300)

        const llmStep: ChainStep = {
          name: `LLM Call ${i + 1}`,
          time: enableCache && Math.random() > 0.4 ? 60 : 900,
          tokens: 2000,
          cached: enableCache && Math.random() > 0.4,
          type: 'llm'
        }
        newSteps.push(llmStep)
        llmTime += llmStep.time
        tokenCount += llmStep.tokens
        totalTime += llmStep.time
        if (llmStep.cached) cacheHits++
        else cacheMisses++
        setSteps([...newSteps])
        setProgress(16 + (i * 42) + 21)
        await delay(llmStep.time)

        const parseStep: ChainStep = {
          name: `Parser ${i + 1}`,
          time: 12,
          tokens: 0,
          cached: false,
          type: 'processing'
        }
        newSteps.push(parseStep)
        processingTime += parseStep.time
        totalTime += parseStep.time
        setSteps([...newSteps])
        setProgress(16 + (i * 42) + 35)
        await delay(300)
      }
    } else {
      // Parallel chain: Multiple operations in parallel
      parallelOps = 3

      const parallelSteps: ChainStep[] = [
        { name: 'Retrieval 1', time: 120, tokens: 0, cached: false, type: 'retrieval' },
        { name: 'Retrieval 2', time: 100, tokens: 0, cached: false, type: 'retrieval' },
        { name: 'Retrieval 3', time: 140, tokens: 0, cached: false, type: 'retrieval' }
      ]

      if (enableParallel) {
        // Parallel execution - takes max time
        const maxTime = Math.max(...parallelSteps.map(s => s.time))
        totalTime += maxTime
        retrievalTime += maxTime

        for (const step of parallelSteps) {
          newSteps.push(step)
        }
        setSteps([...newSteps])
        setProgress(40)
        await delay(maxTime)
      } else {
        // Sequential execution
        for (const step of parallelSteps) {
          newSteps.push(step)
          totalTime += step.time
          retrievalTime += step.time
          setSteps([...newSteps])
          await delay(step.time)
        }
        setProgress(40)
      }

      // Aggregate and process
      const aggregateStep: ChainStep = {
        name: 'Aggregate Results',
        time: 15,
        tokens: 0,
        cached: false,
        type: 'processing'
      }
      newSteps.push(aggregateStep)
      processingTime += aggregateStep.time
      totalTime += aggregateStep.time
      setSteps([...newSteps])
      setProgress(60)
      await delay(300)

      // Final LLM call
      const llmStep: ChainStep = {
        name: 'LLM Synthesis',
        time: 1000,
        tokens: 3000,
        cached: false,
        type: 'llm'
      }
      newSteps.push(llmStep)
      llmTime += llmStep.time
      tokenCount += llmStep.tokens
      totalTime += llmStep.time
      cacheMisses++
      setSteps([...newSteps])
      setProgress(100)
      await delay(llmStep.time)
    }

    setMetrics({
      totalTime,
      llmTime,
      retrievalTime,
      processingTime,
      tokenCount,
      cacheHits,
      cacheMisses,
      parallelOps
    })

    setProfiling(false)
  }

  const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

  const getStepColor = (type: string) => {
    switch (type) {
      case 'llm':
        return '#3b82f6'
      case 'retrieval':
        return '#10b981'
      case 'processing':
        return '#8b5cf6'
      case 'cache':
        return '#f59e0b'
      default:
        return '#6b7280'
    }
  }

  const getBottlenecks = (): string[] => {
    if (!metrics) return []

    const bottlenecks: string[] = []

    if (metrics.llmTime > metrics.totalTime * 0.7) {
      bottlenecks.push('LLM calls are the main bottleneck (>70% of time)')
    }

    if (metrics.cacheHits === 0 && metrics.cacheMisses > 0) {
      bottlenecks.push('No cache hits - consider implementing caching')
    }

    if (chainType === 'parallel' && !enableParallel) {
      bottlenecks.push('Parallel operations running sequentially - enable parallelization')
    }

    if (metrics.retrievalTime > 500) {
      bottlenecks.push('Slow retrieval - optimize vector database queries')
    }

    return bottlenecks
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
            ⚡ Performance Profiler
          </h1>
          <p className="text-gray-300 text-lg">
            Analyze and optimize your LangChain application performance.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Configuration */}
          <div className="lg:col-span-1 space-y-4">
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">Configuration</h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Chain Type</label>
                  <select
                    value={chainType}
                    onChange={(e) => setChainType(e.target.value as any)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                  >
                    <option value="simple">Simple Chain</option>
                    <option value="complex">Complex Chain</option>
                    <option value="parallel">Parallel Operations</option>
                  </select>
                </div>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableCache}
                    onChange={(e) => setEnableCache(e.target.checked)}
                    className="w-4 h-4"
                  />
                  <span className="text-sm">Enable Caching</span>
                </label>

                {chainType === 'parallel' && (
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={enableParallel}
                      onChange={(e) => setEnableParallel(e.target.checked)}
                      className="w-4 h-4"
                    />
                    <span className="text-sm">Enable Parallelization</span>
                  </label>
                )}
              </div>

              <button
                onClick={runProfile}
                disabled={profiling}
                className="w-full mt-6 px-6 py-3 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700 disabled:from-gray-600 disabled:to-gray-700 rounded-lg font-medium flex items-center justify-center gap-2"
              >
                <Play className="w-5 h-5" />
                {profiling ? `Profiling... ${progress}%` : 'Run Profile'}
              </button>
            </div>

            {/* Metrics Summary */}
            {metrics && (
              <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
                <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5" />
                  Metrics
                </h3>

                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Total Time</span>
                    <span className="font-bold text-amber-500">{metrics.totalTime}ms</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">LLM Time</span>
                    <span className="font-bold text-blue-500">{metrics.llmTime}ms</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Retrieval Time</span>
                    <span className="font-bold text-green-500">{metrics.retrievalTime}ms</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Processing Time</span>
                    <span className="font-bold text-purple-500">{metrics.processingTime}ms</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Total Tokens</span>
                    <span className="font-bold text-cyan-500">{metrics.tokenCount}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Cache Hit Rate</span>
                    <span className="font-bold text-green-500">
                      {metrics.cacheHits + metrics.cacheMisses > 0
                        ? ((metrics.cacheHits / (metrics.cacheHits + metrics.cacheMisses)) * 100).toFixed(1)
                        : 0}%
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Results */}
          <div className="lg:col-span-2 space-y-4">
            {/* Timeline */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Clock className="w-5 h-5" />
                Execution Timeline
              </h3>

              <div className="space-y-2">
                {steps.length === 0 ? (
                  <div className="text-center py-12 text-gray-500">
                    Run a profile to see execution timeline...
                  </div>
                ) : (
                  steps.map((step, idx) => (
                    <div key={idx} className="flex items-center gap-3">
                      <div className="w-32 text-sm font-medium truncate">{step.name}</div>
                      <div className="flex-1 relative h-8">
                        <div
                          className="absolute inset-y-0 left-0 rounded flex items-center px-2 text-xs font-medium text-white"
                          style={{
                            width: `${Math.min((step.time / (metrics?.totalTime || 1)) * 100, 100)}%`,
                            backgroundColor: getStepColor(step.type)
                          }}
                        >
                          {step.time}ms
                        </div>
                      </div>
                      {step.cached && (
                        <span className="px-2 py-1 bg-green-900/50 border border-green-700 rounded text-xs">
                          Cached
                        </span>
                      )}
                      {step.tokens > 0 && (
                        <span className="text-xs text-gray-400">{step.tokens} tokens</span>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Time Distribution */}
            {metrics && (
              <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
                <h3 className="text-xl font-bold mb-4">Time Distribution</h3>

                <div className="space-y-3">
                  {[
                    { label: 'LLM Calls', value: metrics.llmTime, color: '#3b82f6' },
                    { label: 'Retrieval', value: metrics.retrievalTime, color: '#10b981' },
                    { label: 'Processing', value: metrics.processingTime, color: '#8b5cf6' }
                  ].map(item => (
                    <div key={item.label}>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm">{item.label}</span>
                        <span className="text-sm font-medium">
                          {item.value}ms ({((item.value / metrics.totalTime) * 100).toFixed(1)}%)
                        </span>
                      </div>
                      <div className="h-3 bg-gray-700 rounded overflow-hidden">
                        <div
                          className="h-full transition-all"
                          style={{
                            width: `${(item.value / metrics.totalTime) * 100}%`,
                            backgroundColor: item.color
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Bottlenecks */}
            {metrics && getBottlenecks().length > 0 && (
              <div className="bg-red-900/20 backdrop-blur border border-red-700 rounded-xl p-6">
                <h3 className="text-xl font-bold mb-3 flex items-center gap-2 text-red-400">
                  <AlertTriangle className="w-5 h-5" />
                  Detected Bottlenecks
                </h3>

                <ul className="space-y-2">
                  {getBottlenecks().map((bottleneck, idx) => (
                    <li key={idx} className="text-sm text-gray-300 flex items-start gap-2">
                      <span className="text-red-500 mt-0.5">•</span>
                      <span>{bottleneck}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Optimization Tips */}
            {metrics && (
              <div className="bg-blue-900/20 backdrop-blur border border-blue-700 rounded-xl p-6">
                <h3 className="text-xl font-bold mb-3 flex items-center gap-2 text-blue-400">
                  <TrendingUp className="w-5 h-5" />
                  Optimization Recommendations
                </h3>

                <ul className="space-y-2 text-sm text-gray-300">
                  {metrics.cacheHits / (metrics.cacheHits + metrics.cacheMisses) < 0.5 && (
                    <li>• Enable caching for frequently used prompts and queries</li>
                  )}
                  {metrics.llmTime > 1000 && (
                    <li>• Consider using a faster model for simple tasks</li>
                  )}
                  {chainType === 'parallel' && !enableParallel && (
                    <li>• Use RunnableParallel for independent operations</li>
                  )}
                  {metrics.retrievalTime > 500 && (
                    <li>• Optimize vector database indices and query parameters</li>
                  )}
                  <li>• Use streaming for better perceived performance</li>
                  <li>• Implement request batching where possible</li>
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
