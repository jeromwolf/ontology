'use client'

import React, { useState, useEffect } from 'react'
import { Database, Play, Zap, TrendingUp } from 'lucide-react'

interface VectorStoreMetrics {
  insertTime: number
  queryTime: number
  memoryUsage: number
  accuracy: number
}

type StoreType = 'faiss' | 'chroma' | 'pinecone'

const STORE_INFO = {
  faiss: {
    name: 'FAISS',
    description: 'Facebook AI Similarity Search - In-memory, ultra-fast',
    color: '#3b82f6',
    pros: ['Fastest queries', 'Low latency', 'Free & open-source'],
    cons: ['In-memory only', 'No persistence', 'Manual indexing']
  },
  chroma: {
    name: 'Chroma',
    description: 'Open-source embedding database - Developer-friendly',
    color: '#10b981',
    pros: ['Easy setup', 'Persistent storage', 'Great for dev'],
    cons: ['Slower than FAISS', 'Limited scale', 'Basic features']
  },
  pinecone: {
    name: 'Pinecone',
    description: 'Managed vector database - Production-ready',
    color: '#f59e0b',
    pros: ['Fully managed', 'Scales easily', 'Advanced features'],
    cons: ['Paid service', 'Network latency', 'Vendor lock-in']
  }
}

export default function VectorStoreComparison() {
  const [selectedStores, setSelectedStores] = useState<StoreType[]>(['faiss', 'chroma'])
  const [dataSize, setDataSize] = useState(1000)
  const [queryComplexity, setQueryComplexity] = useState(1)
  const [metrics, setMetrics] = useState<Record<StoreType, VectorStoreMetrics>>({
    faiss: { insertTime: 0, queryTime: 0, memoryUsage: 0, accuracy: 0 },
    chroma: { insertTime: 0, queryTime: 0, memoryUsage: 0, accuracy: 0 },
    pinecone: { insertTime: 0, queryTime: 0, memoryUsage: 0, accuracy: 0 }
  })
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0)

  const toggleStore = (store: StoreType) => {
    setSelectedStores(prev =>
      prev.includes(store)
        ? prev.filter(s => s !== store)
        : [...prev, store]
    )
  }

  const runBenchmark = async () => {
    if (selectedStores.length === 0) return

    setRunning(true)
    setProgress(0)

    const newMetrics = { ...metrics }

    for (const store of selectedStores) {
      // Simulate insertion
      const insertTime = calculateInsertTime(store, dataSize)
      newMetrics[store].insertTime = insertTime

      setProgress(prev => prev + 20)
      await new Promise(resolve => setTimeout(resolve, 300))

      // Simulate query
      const queryTime = calculateQueryTime(store, dataSize, queryComplexity)
      newMetrics[store].queryTime = queryTime

      setProgress(prev => prev + 20)
      await new Promise(resolve => setTimeout(resolve, 300))

      // Calculate memory and accuracy
      newMetrics[store].memoryUsage = calculateMemory(store, dataSize)
      newMetrics[store].accuracy = calculateAccuracy(store)

      setProgress(prev => prev + 10)
      await new Promise(resolve => setTimeout(resolve, 200))
    }

    setMetrics(newMetrics)
    setProgress(100)
    setRunning(false)
  }

  const calculateInsertTime = (store: StoreType, size: number): number => {
    const baseTime = size / 100
    switch (store) {
      case 'faiss':
        return baseTime * 0.8
      case 'chroma':
        return baseTime * 1.2
      case 'pinecone':
        return baseTime * 1.5 // Network overhead
      default:
        return baseTime
    }
  }

  const calculateQueryTime = (store: StoreType, size: number, complexity: number): number => {
    const baseTime = Math.log(size) * complexity * 5
    switch (store) {
      case 'faiss':
        return baseTime * 0.5 // Fastest
      case 'chroma':
        return baseTime * 1.0
      case 'pinecone':
        return baseTime * 1.3 // Network latency
      default:
        return baseTime
    }
  }

  const calculateMemory = (store: StoreType, size: number): number => {
    const baseMemory = size * 1.5 // KB per vector
    switch (store) {
      case 'faiss':
        return baseMemory * 1.2 // Keeps everything in memory
      case 'chroma':
        return baseMemory * 0.8 // Efficient storage
      case 'pinecone':
        return baseMemory * 0.3 // Cloud-based, minimal local
      default:
        return baseMemory
    }
  }

  const calculateAccuracy = (store: StoreType): number => {
    // All are highly accurate, slight differences
    switch (store) {
      case 'faiss':
        return 98.5
      case 'chroma':
        return 97.8
      case 'pinecone':
        return 99.2
      default:
        return 98.0
    }
  }

  const getWinner = (metric: keyof VectorStoreMetrics, lowerIsBetter: boolean = true): StoreType | null => {
    if (!selectedStores.length) return null

    return selectedStores.reduce((best, current) => {
      const currentValue = metrics[current][metric]
      const bestValue = metrics[best][metric]

      if (currentValue === 0 && bestValue === 0) return best

      if (lowerIsBetter) {
        return currentValue < bestValue ? current : best
      } else {
        return currentValue > bestValue ? current : best
      }
    })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
            💾 Vector Store Comparison
          </h1>
          <p className="text-gray-300 text-lg">
            Compare performance metrics across different vector database solutions.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Store Selection */}
          {(Object.keys(STORE_INFO) as StoreType[]).map(store => {
            const info = STORE_INFO[store]
            const isSelected = selectedStores.includes(store)

            return (
              <button
                key={store}
                onClick={() => toggleStore(store)}
                className={`p-6 rounded-xl border-2 transition-all text-left ${
                  isSelected
                    ? 'bg-gray-800 border-amber-500 ring-2 ring-amber-500/50'
                    : 'bg-gray-800/50 border-gray-700 hover:border-gray-600'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-2xl font-bold" style={{ color: info.color }}>
                    {info.name}
                  </h3>
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => {}}
                    className="w-5 h-5"
                  />
                </div>

                <p className="text-sm text-gray-400 mb-4">{info.description}</p>

                <div className="space-y-2">
                  <div>
                    <div className="text-xs font-semibold text-green-400 mb-1">PROS</div>
                    {info.pros.map((pro, idx) => (
                      <div key={idx} className="text-xs text-gray-300">• {pro}</div>
                    ))}
                  </div>

                  <div>
                    <div className="text-xs font-semibold text-red-400 mb-1">CONS</div>
                    {info.cons.map((con, idx) => (
                      <div key={idx} className="text-xs text-gray-300">• {con}</div>
                    ))}
                  </div>
                </div>
              </button>
            )
          })}
        </div>

        {/* Configuration */}
        <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6 mb-6">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Database className="w-5 h-5" />
            Benchmark Configuration
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium mb-2">
                Data Size: {dataSize.toLocaleString()} vectors
              </label>
              <input
                type="range"
                min="100"
                max="10000"
                step="100"
                value={dataSize}
                onChange={(e) => setDataSize(parseInt(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Query Complexity: {queryComplexity}x
              </label>
              <input
                type="range"
                min="1"
                max="5"
                value={queryComplexity}
                onChange={(e) => setQueryComplexity(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
          </div>

          <button
            onClick={runBenchmark}
            disabled={selectedStores.length === 0 || running}
            className="w-full mt-6 px-6 py-3 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700 disabled:from-gray-600 disabled:to-gray-700 rounded-lg font-medium flex items-center justify-center gap-2"
          >
            <Play className="w-5 h-5" />
            {running ? `Running... ${progress}%` : 'Run Benchmark'}
          </button>
        </div>

        {/* Results */}
        {progress > 0 && (
          <div className="space-y-6">
            {/* Metrics Comparison */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Performance Metrics
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Insert Time */}
                <div>
                  <div className="text-sm font-semibold mb-3 text-gray-300">
                    Insertion Time (ms) - Lower is Better
                  </div>
                  <div className="space-y-2">
                    {selectedStores.map(store => {
                      const info = STORE_INFO[store]
                      const value = metrics[store].insertTime
                      const maxValue = Math.max(...selectedStores.map(s => metrics[s].insertTime))
                      const percentage = maxValue > 0 ? (value / maxValue) * 100 : 0
                      const isWinner = getWinner('insertTime') === store

                      return (
                        <div key={store}>
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-sm font-medium">{info.name}</span>
                            <span className="text-sm" style={{ color: info.color }}>
                              {value.toFixed(1)}ms {isWinner && '🏆'}
                            </span>
                          </div>
                          <div className="h-2 bg-gray-700 rounded overflow-hidden">
                            <div
                              className="h-full transition-all"
                              style={{
                                width: `${percentage}%`,
                                backgroundColor: info.color
                              }}
                            />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Query Time */}
                <div>
                  <div className="text-sm font-semibold mb-3 text-gray-300">
                    Query Time (ms) - Lower is Better
                  </div>
                  <div className="space-y-2">
                    {selectedStores.map(store => {
                      const info = STORE_INFO[store]
                      const value = metrics[store].queryTime
                      const maxValue = Math.max(...selectedStores.map(s => metrics[s].queryTime))
                      const percentage = maxValue > 0 ? (value / maxValue) * 100 : 0
                      const isWinner = getWinner('queryTime') === store

                      return (
                        <div key={store}>
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-sm font-medium">{info.name}</span>
                            <span className="text-sm" style={{ color: info.color }}>
                              {value.toFixed(1)}ms {isWinner && '🏆'}
                            </span>
                          </div>
                          <div className="h-2 bg-gray-700 rounded overflow-hidden">
                            <div
                              className="h-full transition-all"
                              style={{
                                width: `${percentage}%`,
                                backgroundColor: info.color
                              }}
                            />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Memory Usage */}
                <div>
                  <div className="text-sm font-semibold mb-3 text-gray-300">
                    Memory Usage (MB) - Lower is Better
                  </div>
                  <div className="space-y-2">
                    {selectedStores.map(store => {
                      const info = STORE_INFO[store]
                      const value = metrics[store].memoryUsage / 1024
                      const maxValue = Math.max(...selectedStores.map(s => metrics[s].memoryUsage)) / 1024
                      const percentage = maxValue > 0 ? (value / maxValue) * 100 : 0
                      const isWinner = getWinner('memoryUsage') === store

                      return (
                        <div key={store}>
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-sm font-medium">{info.name}</span>
                            <span className="text-sm" style={{ color: info.color }}>
                              {value.toFixed(1)}MB {isWinner && '🏆'}
                            </span>
                          </div>
                          <div className="h-2 bg-gray-700 rounded overflow-hidden">
                            <div
                              className="h-full transition-all"
                              style={{
                                width: `${percentage}%`,
                                backgroundColor: info.color
                              }}
                            />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Accuracy */}
                <div>
                  <div className="text-sm font-semibold mb-3 text-gray-300">
                    Accuracy (%) - Higher is Better
                  </div>
                  <div className="space-y-2">
                    {selectedStores.map(store => {
                      const info = STORE_INFO[store]
                      const value = metrics[store].accuracy
                      const isWinner = getWinner('accuracy', false) === store

                      return (
                        <div key={store}>
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-sm font-medium">{info.name}</span>
                            <span className="text-sm" style={{ color: info.color }}>
                              {value.toFixed(1)}% {isWinner && '🏆'}
                            </span>
                          </div>
                          <div className="h-2 bg-gray-700 rounded overflow-hidden">
                            <div
                              className="h-full transition-all"
                              style={{
                                width: `${value}%`,
                                backgroundColor: info.color
                              }}
                            />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>
            </div>

            {/* Recommendation */}
            <div className="bg-blue-900/20 backdrop-blur border border-blue-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-3 text-blue-400">💡 Recommendation</h3>
              <div className="space-y-2 text-sm text-gray-300">
                <p>• <strong>FAISS</strong>: Best for prototyping and low-latency requirements</p>
                <p>• <strong>Chroma</strong>: Great for development and small to medium datasets</p>
                <p>• <strong>Pinecone</strong>: Ideal for production with large-scale data</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
