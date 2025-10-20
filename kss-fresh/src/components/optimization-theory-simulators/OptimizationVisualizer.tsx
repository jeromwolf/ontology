'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, Pause, RotateCcw, Download, TrendingDown } from 'lucide-react'

interface Point {
  x: number
  y: number
}

interface AlgorithmState {
  name: string
  path: Point[]
  currentPoint: Point
  losses: number[]
  color: string
  enabled: boolean
}

type OptimizationFunction = (x: number, y: number) => number

const functions: Record<string, OptimizationFunction> = {
  sphere: (x, y) => x * x + y * y,
  rosenbrock: (x, y) => (1 - x) ** 2 + 100 * (y - x ** 2) ** 2,
  rastrigin: (x, y) => 20 + (x ** 2 - 10 * Math.cos(2 * Math.PI * x)) + (y ** 2 - 10 * Math.cos(2 * Math.PI * y)),
  ackley: (x, y) => -20 * Math.exp(-0.2 * Math.sqrt(0.5 * (x ** 2 + y ** 2))) - Math.exp(0.5 * (Math.cos(2 * Math.PI * x) + Math.cos(2 * Math.PI * y))) + Math.E + 20,
}

export default function OptimizationVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartRef = useRef<HTMLCanvasElement>(null)
  const [selectedFunction, setSelectedFunction] = useState<string>('rosenbrock')
  const [learningRate, setLearningRate] = useState(0.01)
  const [isRunning, setIsRunning] = useState(false)
  const [iteration, setIteration] = useState(0)
  const animationRef = useRef<number>()

  const [algorithms, setAlgorithms] = useState<AlgorithmState[]>([
    {
      name: 'Gradient Descent',
      path: [{ x: -1.5, y: 2 }],
      currentPoint: { x: -1.5, y: 2 },
      losses: [],
      color: '#10b981',
      enabled: true,
    },
    {
      name: 'Momentum',
      path: [{ x: -1.5, y: 2 }],
      currentPoint: { x: -1.5, y: 2 },
      losses: [],
      color: '#3b82f6',
      enabled: true,
    },
    {
      name: 'Adam',
      path: [{ x: -1.5, y: 2 }],
      currentPoint: { x: -1.5, y: 2 },
      losses: [],
      color: '#8b5cf6',
      enabled: true,
    },
    {
      name: 'RMSprop',
      path: [{ x: -1.5, y: 2 }],
      currentPoint: { x: -1.5, y: 2 },
      losses: [],
      color: '#f59e0b',
      enabled: false,
    },
  ])

  const [momentum, setMomentum] = useState<Record<string, { vx: number; vy: number }>>({})
  const [adamState, setAdamState] = useState<Record<string, { m1x: number; m1y: number; m2x: number; m2y: number; t: number }>>({})
  const [rmspropState, setRmspropState] = useState<Record<string, { vx: number; vy: number }>>({})

  const gradient = (fn: OptimizationFunction, x: number, y: number, h = 0.0001): { dx: number; dy: number } => {
    const dx = (fn(x + h, y) - fn(x - h, y)) / (2 * h)
    const dy = (fn(x, y + h) - fn(x, y - h)) / (2 * h)
    return { dx, dy }
  }

  const updateAlgorithms = () => {
    const fn = functions[selectedFunction]
    const newAlgorithms = algorithms.map((algo) => {
      if (!algo.enabled) return algo

      const { x, y } = algo.currentPoint
      const { dx, dy } = gradient(fn, x, y)
      let newX = x
      let newY = y

      if (algo.name === 'Gradient Descent') {
        newX = x - learningRate * dx
        newY = y - learningRate * dy
      } else if (algo.name === 'Momentum') {
        const beta = 0.9
        const v = momentum[algo.name] || { vx: 0, vy: 0 }
        v.vx = beta * v.vx + learningRate * dx
        v.vy = beta * v.vy + learningRate * dy
        newX = x - v.vx
        newY = y - v.vy
        setMomentum((prev) => ({ ...prev, [algo.name]: v }))
      } else if (algo.name === 'Adam') {
        const beta1 = 0.9
        const beta2 = 0.999
        const epsilon = 1e-8
        const state = adamState[algo.name] || { m1x: 0, m1y: 0, m2x: 0, m2y: 0, t: 0 }
        state.t += 1
        state.m1x = beta1 * state.m1x + (1 - beta1) * dx
        state.m1y = beta1 * state.m1y + (1 - beta1) * dy
        state.m2x = beta2 * state.m2x + (1 - beta2) * dx * dx
        state.m2y = beta2 * state.m2y + (1 - beta2) * dy * dy
        const m1xHat = state.m1x / (1 - Math.pow(beta1, state.t))
        const m1yHat = state.m1y / (1 - Math.pow(beta1, state.t))
        const m2xHat = state.m2x / (1 - Math.pow(beta2, state.t))
        const m2yHat = state.m2y / (1 - Math.pow(beta2, state.t))
        newX = x - learningRate * m1xHat / (Math.sqrt(m2xHat) + epsilon)
        newY = y - learningRate * m1yHat / (Math.sqrt(m2yHat) + epsilon)
        setAdamState((prev) => ({ ...prev, [algo.name]: state }))
      } else if (algo.name === 'RMSprop') {
        const beta = 0.9
        const epsilon = 1e-8
        const state = rmspropState[algo.name] || { vx: 0, vy: 0 }
        state.vx = beta * state.vx + (1 - beta) * dx * dx
        state.vy = beta * state.vy + (1 - beta) * dy * dy
        newX = x - learningRate * dx / (Math.sqrt(state.vx) + epsilon)
        newY = y - learningRate * dy / (Math.sqrt(state.vy) + epsilon)
        setRmspropState((prev) => ({ ...prev, [algo.name]: state }))
      }

      const newLoss = fn(newX, newY)
      return {
        ...algo,
        currentPoint: { x: newX, y: newY },
        path: [...algo.path, { x: newX, y: newY }],
        losses: [...algo.losses, newLoss],
      }
    })

    setAlgorithms(newAlgorithms)
    setIteration((prev) => prev + 1)
  }

  const drawContour = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height
    const fn = functions[selectedFunction]

    // Clear canvas
    ctx.fillStyle = '#1f2937'
    ctx.fillRect(0, 0, width, height)

    // Draw contour
    const xMin = -3
    const xMax = 3
    const yMin = -2
    const yMax = 4
    const resolution = 200

    const levels = 20
    const minZ = 0
    const maxZ = selectedFunction === 'rosenbrock' ? 100 : selectedFunction === 'ackley' ? 15 : 50

    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = xMin + (i / resolution) * (xMax - xMin)
        const y = yMin + (j / resolution) * (yMax - yMin)
        const z = fn(x, y)
        const normalized = Math.min(1, (z - minZ) / (maxZ - minZ))
        const intensity = Math.floor(normalized * 255)
        ctx.fillStyle = `rgba(16, 185, 129, ${0.1 + normalized * 0.3})`
        const px = (i / resolution) * width
        const py = height - (j / resolution) * height
        ctx.fillRect(px, py, width / resolution + 1, height / resolution + 1)
      }
    }

    // Draw paths
    algorithms.forEach((algo) => {
      if (!algo.enabled) return
      ctx.strokeStyle = algo.color
      ctx.lineWidth = 2
      ctx.beginPath()
      algo.path.forEach((point, idx) => {
        const px = ((point.x - xMin) / (xMax - xMin)) * width
        const py = height - ((point.y - yMin) / (yMax - yMin)) * height
        if (idx === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      })
      ctx.stroke()

      // Draw current point
      const current = algo.currentPoint
      const px = ((current.x - xMin) / (xMax - xMin)) * width
      const py = height - ((current.y - yMin) / (yMax - yMin)) * height
      ctx.fillStyle = algo.color
      ctx.beginPath()
      ctx.arc(px, py, 6, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = '#1f2937'
      ctx.lineWidth = 2
      ctx.stroke()
    })

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 6; i++) {
      const x = (i / 6) * width
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.stroke()

      const y = (i / 6) * height
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }
  }

  const drawLossChart = () => {
    const canvas = chartRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.fillStyle = '#1f2937'
    ctx.fillRect(0, 0, width, height)

    const maxIterations = Math.max(...algorithms.map((a) => a.losses.length))
    if (maxIterations === 0) return

    const allLosses = algorithms.flatMap((a) => a.losses)
    const maxLoss = Math.max(...allLosses, 1)
    const minLoss = Math.min(...allLosses, 0)

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

    // Draw loss curves
    algorithms.forEach((algo) => {
      if (!algo.enabled || algo.losses.length === 0) return
      ctx.strokeStyle = algo.color
      ctx.lineWidth = 2
      ctx.beginPath()
      algo.losses.forEach((loss, idx) => {
        const x = (idx / maxIterations) * width
        const y = height - ((loss - minLoss) / (maxLoss - minLoss)) * height
        if (idx === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()
    })

    // Draw labels
    ctx.fillStyle = '#9ca3af'
    ctx.font = '12px Inter'
    ctx.fillText(`Max: ${maxLoss.toFixed(2)}`, 10, 20)
    ctx.fillText(`Min: ${minLoss.toFixed(2)}`, 10, height - 10)
  }

  useEffect(() => {
    drawContour()
    drawLossChart()
  }, [algorithms, selectedFunction])

  useEffect(() => {
    if (isRunning) {
      animationRef.current = requestAnimationFrame(() => {
        updateAlgorithms()
      })
    }
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [isRunning, algorithms, learningRate, selectedFunction])

  const handleReset = () => {
    setIsRunning(false)
    setIteration(0)
    setMomentum({})
    setAdamState({})
    setRmspropState({})
    setAlgorithms((prev) =>
      prev.map((algo) => ({
        ...algo,
        path: [{ x: -1.5, y: 2 }],
        currentPoint: { x: -1.5, y: 2 },
        losses: [],
      }))
    )
  }

  const toggleAlgorithm = (index: number) => {
    setAlgorithms((prev) =>
      prev.map((algo, idx) => (idx === index ? { ...algo, enabled: !algo.enabled } : algo))
    )
  }

  const downloadResults = () => {
    const data = algorithms.map((algo) => ({
      name: algo.name,
      enabled: algo.enabled,
      finalLoss: algo.losses[algo.losses.length - 1] || 0,
      iterations: algo.losses.length,
    }))
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'optimization-results.json'
    a.click()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-emerald-600 to-teal-700 rounded-xl">
              <TrendingDown className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-400 to-teal-400 bg-clip-text text-transparent">
                최적화 알고리즘 비교 시각화
              </h1>
              <p className="text-gray-400 mt-1">다양한 최적화 알고리즘의 수렴 과정을 실시간으로 비교합니다</p>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
          <div className="lg:col-span-1 space-y-4">
            {/* Function Selection */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">목적 함수</label>
              <select
                value={selectedFunction}
                onChange={(e) => {
                  setSelectedFunction(e.target.value)
                  handleReset()
                }}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
              >
                <option value="sphere">Sphere (간단)</option>
                <option value="rosenbrock">Rosenbrock (복잡)</option>
                <option value="rastrigin">Rastrigin (다봉)</option>
                <option value="ackley">Ackley (다봉)</option>
              </select>
            </div>

            {/* Learning Rate */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                학습률: {learningRate.toFixed(4)}
              </label>
              <input
                type="range"
                min="0.001"
                max="0.1"
                step="0.001"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                className="w-full accent-emerald-500"
              />
            </div>

            {/* Algorithm Selection */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">알고리즘 선택</label>
              <div className="space-y-3">
                {algorithms.map((algo, idx) => (
                  <label key={algo.name} className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={algo.enabled}
                      onChange={() => toggleAlgorithm(idx)}
                      className="w-4 h-4 accent-emerald-500"
                    />
                    <div className="flex items-center gap-2 flex-1">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: algo.color }}></div>
                      <span className="text-sm text-gray-300">{algo.name}</span>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Controls */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 space-y-3">
              <button
                onClick={() => setIsRunning(!isRunning)}
                className="w-full bg-gradient-to-r from-emerald-600 to-teal-700 hover:from-emerald-700 hover:to-teal-800 text-white px-4 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2"
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

            {/* Stats */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <div className="text-sm text-gray-400 mb-2">반복 횟수</div>
              <div className="text-3xl font-bold text-emerald-400">{iteration}</div>
            </div>
          </div>

          {/* Visualizations */}
          <div className="lg:col-span-3 space-y-6">
            {/* Contour Plot */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 text-emerald-400">최적화 경로 (Contour Plot)</h3>
              <canvas ref={canvasRef} width={800} height={500} className="w-full rounded-lg" />
            </div>

            {/* Loss Chart */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 text-emerald-400">손실 함수 수렴 곡선</h3>
              <canvas ref={chartRef} width={800} height={300} className="w-full rounded-lg" />
            </div>

            {/* Algorithm Stats */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {algorithms.map((algo) => (
                <div key={algo.name} className="bg-gray-800 rounded-xl p-4 border border-gray-700">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: algo.color }}></div>
                    <div className="text-sm font-medium text-gray-300">{algo.name}</div>
                  </div>
                  <div className="text-2xl font-bold" style={{ color: algo.color }}>
                    {algo.enabled && algo.losses.length > 0
                      ? algo.losses[algo.losses.length - 1].toFixed(4)
                      : '—'}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">현재 손실값</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
