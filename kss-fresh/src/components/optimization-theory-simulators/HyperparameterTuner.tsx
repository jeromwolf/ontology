'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, Pause, RotateCcw, Download, Sliders, Target } from 'lucide-react'

interface SearchPoint {
  x: number
  y: number
  value: number
  iteration: number
}

interface SearchMethod {
  name: string
  points: SearchPoint[]
  bestPoint: SearchPoint | null
  color: string
  enabled: boolean
}

export default function HyperparameterTuner() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartRef = useRef<HTMLCanvasElement>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [iteration, setIteration] = useState(0)
  const [searchBudget, setSearchBudget] = useState(50)
  const [currentBudget, setCurrentBudget] = useState(50)
  const animationRef = useRef<number>()

  const [searchMethods, setSearchMethods] = useState<SearchMethod[]>([
    {
      name: 'Grid Search',
      points: [],
      bestPoint: null,
      color: '#10b981',
      enabled: true,
    },
    {
      name: 'Random Search',
      points: [],
      bestPoint: null,
      color: '#3b82f6',
      enabled: true,
    },
    {
      name: 'Bayesian Optimization',
      points: [],
      bestPoint: null,
      color: '#8b5cf6',
      enabled: true,
    },
  ])

  // Simulated loss surface (2D function)
  const lossFunction = (x: number, y: number): number => {
    // Complex loss landscape with local minima
    const term1 = Math.sin(x * 2) * Math.cos(y * 2)
    const term2 = 0.3 * ((x - 0.5) ** 2 + (y - 0.7) ** 2)
    const term3 = 0.1 * Math.sin(x * 10) * Math.sin(y * 10)
    return term1 + term2 + term3 + 2
  }

  const performGridSearch = (budget: number): SearchPoint[] => {
    const points: SearchPoint[] = []
    const gridSize = Math.floor(Math.sqrt(budget))
    let iter = 0

    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        if (iter >= budget) break
        const x = i / (gridSize - 1)
        const y = j / (gridSize - 1)
        const value = lossFunction(x, y)
        points.push({ x, y, value, iteration: iter })
        iter++
      }
    }
    return points
  }

  const performRandomSearch = (budget: number): SearchPoint[] => {
    const points: SearchPoint[] = []
    for (let i = 0; i < budget; i++) {
      const x = Math.random()
      const y = Math.random()
      const value = lossFunction(x, y)
      points.push({ x, y, value, iteration: i })
    }
    return points
  }

  const performBayesianOptimization = (budget: number): SearchPoint[] => {
    const points: SearchPoint[] = []

    // Initial random points
    const initialPoints = 5
    for (let i = 0; i < initialPoints; i++) {
      const x = Math.random()
      const y = Math.random()
      const value = lossFunction(x, y)
      points.push({ x, y, value, iteration: i })
    }

    // Bayesian optimization iterations
    for (let i = initialPoints; i < budget; i++) {
      // Find regions with low observed values and high uncertainty
      let bestX = 0
      let bestY = 0
      let bestAcquisition = -Infinity

      // Sample acquisition function
      for (let trial = 0; trial < 100; trial++) {
        const x = Math.random()
        const y = Math.random()

        // Calculate expected improvement (simplified)
        const distances = points.map((p) => Math.sqrt((p.x - x) ** 2 + (p.y - y) ** 2))
        const minDist = Math.min(...distances)
        const nearestPoint = points.reduce((min, p, idx) =>
          distances[idx] === minDist ? p : min
        )

        // Acquisition = exploitation + exploration
        const exploitation = -nearestPoint.value
        const exploration = minDist
        const acquisition = exploitation + 0.3 * exploration

        if (acquisition > bestAcquisition) {
          bestAcquisition = acquisition
          bestX = x
          bestY = y
        }
      }

      const value = lossFunction(bestX, bestY)
      points.push({ x: bestX, y: bestY, value, iteration: i })
    }

    return points
  }

  const runSearch = () => {
    const newMethods = searchMethods.map((method) => {
      if (!method.enabled) return method

      let points: SearchPoint[] = []
      if (method.name === 'Grid Search') {
        points = performGridSearch(searchBudget)
      } else if (method.name === 'Random Search') {
        points = performRandomSearch(searchBudget)
      } else if (method.name === 'Bayesian Optimization') {
        points = performBayesianOptimization(searchBudget)
      }

      const bestPoint = points.reduce((min, p) => (p.value < min.value ? p : min))

      return {
        ...method,
        points,
        bestPoint,
      }
    })

    setSearchMethods(newMethods)
    setCurrentBudget(0)
  }

  const animateSearch = () => {
    if (currentBudget >= searchBudget) {
      setIsRunning(false)
      return
    }

    const newMethods = searchMethods.map((method) => {
      if (!method.enabled) return method

      let newPoint: SearchPoint | null = null
      if (method.name === 'Grid Search') {
        const gridSize = Math.floor(Math.sqrt(searchBudget))
        const i = Math.floor(currentBudget / gridSize)
        const j = currentBudget % gridSize
        const x = i / (gridSize - 1)
        const y = j / (gridSize - 1)
        const value = lossFunction(x, y)
        newPoint = { x, y, value, iteration: currentBudget }
      } else if (method.name === 'Random Search') {
        const x = Math.random()
        const y = Math.random()
        const value = lossFunction(x, y)
        newPoint = { x, y, value, iteration: currentBudget }
      } else if (method.name === 'Bayesian Optimization') {
        if (currentBudget < 5) {
          const x = Math.random()
          const y = Math.random()
          const value = lossFunction(x, y)
          newPoint = { x, y, value, iteration: currentBudget }
        } else {
          let bestX = 0
          let bestY = 0
          let bestAcquisition = -Infinity

          for (let trial = 0; trial < 100; trial++) {
            const x = Math.random()
            const y = Math.random()
            const distances = method.points.map((p) => Math.sqrt((p.x - x) ** 2 + (p.y - y) ** 2))
            const minDist = Math.min(...distances)
            const nearestPoint = method.points.reduce((min, p, idx) =>
              distances[idx] === minDist ? p : min
            )
            const exploitation = -nearestPoint.value
            const exploration = minDist
            const acquisition = exploitation + 0.3 * exploration

            if (acquisition > bestAcquisition) {
              bestAcquisition = acquisition
              bestX = x
              bestY = y
            }
          }

          const value = lossFunction(bestX, bestY)
          newPoint = { x: bestX, y: bestY, value, iteration: currentBudget }
        }
      }

      if (newPoint) {
        const newPoints = [...method.points, newPoint]
        const bestPoint = newPoints.reduce((min, p) => (p.value < min.value ? p : min))
        return {
          ...method,
          points: newPoints,
          bestPoint,
        }
      }

      return method
    })

    setSearchMethods(newMethods)
    setCurrentBudget((prev) => prev + 1)
  }

  const drawLossSurface = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.fillStyle = '#1f2937'
    ctx.fillRect(0, 0, width, height)

    // Draw loss surface
    const resolution = 100
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = i / resolution
        const y = j / resolution
        const value = lossFunction(x, y)
        const normalized = (value - 0) / 4
        const intensity = Math.min(1, Math.max(0, normalized))
        ctx.fillStyle = `rgba(16, 185, 129, ${0.1 + intensity * 0.4})`
        const px = x * width
        const py = height - y * height
        ctx.fillRect(px, py, width / resolution + 1, height / resolution + 1)
      }
    }

    // Draw search points
    searchMethods.forEach((method) => {
      if (!method.enabled) return

      method.points.forEach((point, idx) => {
        const alpha = Math.min(1, 0.3 + (idx / method.points.length) * 0.7)
        ctx.fillStyle = method.color.replace(')', `, ${alpha})`)
        ctx.beginPath()
        ctx.arc(point.x * width, height - point.y * height, 3, 0, Math.PI * 2)
        ctx.fill()
      })

      // Draw best point
      if (method.bestPoint) {
        const px = method.bestPoint.x * width
        const py = height - method.bestPoint.y * height
        ctx.strokeStyle = method.color
        ctx.lineWidth = 3
        ctx.beginPath()
        ctx.arc(px, py, 8, 0, Math.PI * 2)
        ctx.stroke()
        ctx.fillStyle = method.color
        ctx.beginPath()
        ctx.arc(px, py, 4, 0, Math.PI * 2)
        ctx.fill()
      }
    })

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 10; i++) {
      const x = (i / 10) * width
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.stroke()

      const y = (i / 10) * height
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }
  }

  const drawConvergenceChart = () => {
    const canvas = chartRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.fillStyle = '#1f2937'
    ctx.fillRect(0, 0, width, height)

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 5; i++) {
      const y = (i / 5) * height
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    // Draw convergence curves (best value over iterations)
    searchMethods.forEach((method) => {
      if (!method.enabled || method.points.length === 0) return

      ctx.strokeStyle = method.color
      ctx.lineWidth = 2
      ctx.beginPath()

      let bestSoFar = Infinity
      method.points.forEach((point, idx) => {
        bestSoFar = Math.min(bestSoFar, point.value)
        const x = (idx / searchBudget) * width
        const y = height - ((bestSoFar - 0) / 4) * height
        if (idx === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()
    })

    // Draw labels
    ctx.fillStyle = '#9ca3af'
    ctx.font = '12px Inter'
    ctx.fillText('Best Loss', 10, 20)
    ctx.fillText('Iterations', width - 70, height - 10)
  }

  useEffect(() => {
    drawLossSurface()
    drawConvergenceChart()
  }, [searchMethods])

  useEffect(() => {
    if (isRunning && currentBudget < searchBudget) {
      animationRef.current = window.setTimeout(() => {
        animateSearch()
      }, 50)
    } else if (currentBudget >= searchBudget) {
      setIsRunning(false)
    }
    return () => {
      if (animationRef.current) clearTimeout(animationRef.current)
    }
  }, [isRunning, currentBudget, searchBudget])

  const handleReset = () => {
    setIsRunning(false)
    setCurrentBudget(0)
    setSearchMethods((prev) =>
      prev.map((method) => ({
        ...method,
        points: [],
        bestPoint: null,
      }))
    )
  }

  const toggleMethod = (index: number) => {
    setSearchMethods((prev) =>
      prev.map((method, idx) => (idx === index ? { ...method, enabled: !method.enabled } : method))
    )
  }

  const downloadResults = () => {
    const data = searchMethods
      .filter((m) => m.enabled && m.bestPoint)
      .map((method) => ({
        name: method.name,
        bestValue: method.bestPoint?.value,
        bestParams: { x: method.bestPoint?.x, y: method.bestPoint?.y },
        numEvaluations: method.points.length,
      }))
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'hyperparameter-tuning-results.json'
    a.click()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-emerald-600 to-teal-700 rounded-xl">
              <Sliders className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-400 to-teal-400 bg-clip-text text-transparent">
                하이퍼파라미터 튜닝 시뮬레이터
              </h1>
              <p className="text-gray-400 mt-1">Grid Search, Random Search, Bayesian Optimization 비교</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Controls */}
          <div className="lg:col-span-1 space-y-4">
            {/* Search Budget */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                탐색 예산: {searchBudget}
              </label>
              <input
                type="range"
                min="10"
                max="100"
                value={searchBudget}
                onChange={(e) => setSearchBudget(parseInt(e.target.value))}
                className="w-full accent-emerald-500"
              />
              <p className="text-xs text-gray-500 mt-2">평가할 총 하이퍼파라미터 조합 수</p>
            </div>

            {/* Method Selection */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">탐색 방법</label>
              <div className="space-y-3">
                {searchMethods.map((method, idx) => (
                  <label key={method.name} className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={method.enabled}
                      onChange={() => toggleMethod(idx)}
                      className="w-4 h-4 accent-emerald-500"
                    />
                    <div className="flex items-center gap-2 flex-1">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: method.color }}></div>
                      <span className="text-sm text-gray-300">{method.name}</span>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Controls */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 space-y-3">
              <button
                onClick={() => setIsRunning(!isRunning)}
                disabled={currentBudget >= searchBudget}
                className="w-full bg-gradient-to-r from-emerald-600 to-teal-700 hover:from-emerald-700 hover:to-teal-800 disabled:opacity-50 disabled:cursor-not-allowed text-white px-4 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2"
              >
                {isRunning ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                {isRunning ? '일시정지' : '시작'}
              </button>
              <button
                onClick={handleReset}
                className="w-full bg-gray-700 hover:bg-gray-600 text-white px-4 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2"
              >
                <RotateCcw className="w-5 h-5" />
                초기화
              </button>
              <button
                onClick={downloadResults}
                className="w-full bg-gray-700 hover:bg-gray-600 text-white px-4 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2"
              >
                <Download className="w-5 h-5" />
                결과 다운로드
              </button>
            </div>

            {/* Progress */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <div className="text-sm text-gray-400 mb-2">진행률</div>
              <div className="text-3xl font-bold text-emerald-400 mb-2">
                {Math.round((currentBudget / searchBudget) * 100)}%
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-gradient-to-r from-emerald-600 to-teal-700 h-2 rounded-full transition-all"
                  style={{ width: `${(currentBudget / searchBudget) * 100}%` }}
                ></div>
              </div>
              <div className="text-xs text-gray-500 mt-2">
                {currentBudget} / {searchBudget} 평가 완료
              </div>
            </div>

            {/* Best Results */}
            {searchMethods.some((m) => m.bestPoint) && (
              <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <h3 className="text-sm font-semibold text-emerald-400 mb-3">최고 성능</h3>
                <div className="space-y-3">
                  {searchMethods
                    .filter((m) => m.enabled && m.bestPoint)
                    .map((method) => (
                      <div key={method.name} className="bg-gray-700 rounded-lg p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: method.color }}></div>
                          <span className="text-xs text-gray-300">{method.name}</span>
                        </div>
                        <div className="text-lg font-bold" style={{ color: method.color }}>
                          {method.bestPoint?.value.toFixed(4)}
                        </div>
                        <div className="text-xs text-gray-500">
                          x: {method.bestPoint?.x.toFixed(3)}, y: {method.bestPoint?.y.toFixed(3)}
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </div>

          {/* Visualizations */}
          <div className="lg:col-span-3 space-y-6">
            {/* Loss Surface */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 text-emerald-400">하이퍼파라미터 공간 탐색</h3>
              <canvas ref={canvasRef} width={900} height={600} className="w-full rounded-lg" />
              <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="font-medium text-gray-300 mb-1">Grid Search</div>
                  <p className="text-xs text-gray-500">체계적으로 모든 조합 탐색</p>
                </div>
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="font-medium text-gray-300 mb-1">Random Search</div>
                  <p className="text-xs text-gray-500">무작위 샘플링으로 탐색</p>
                </div>
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="font-medium text-gray-300 mb-1">Bayesian Optimization</div>
                  <p className="text-xs text-gray-500">이전 결과를 활용한 지능적 탐색</p>
                </div>
              </div>
            </div>

            {/* Convergence Chart */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 text-emerald-400">수렴 곡선 (Best Loss over Iterations)</h3>
              <canvas ref={chartRef} width={900} height={300} className="w-full rounded-lg" />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
