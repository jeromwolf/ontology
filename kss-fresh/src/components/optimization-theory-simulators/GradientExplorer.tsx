'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, Pause, RotateCcw, ArrowRight, Compass } from 'lucide-react'

interface PathPoint {
  x: number
  y: number
  gradient: { dx: number; dy: number }
  loss: number
}

interface OptimizationPath {
  name: string
  points: PathPoint[]
  color: string
  enabled: boolean
  learningRate: number
}

export default function GradientExplorer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [selectedFunction, setSelectedFunction] = useState<string>('quadratic')
  const [showGradients, setShowGradients] = useState(true)
  const [animationSpeed, setAnimationSpeed] = useState(1)
  const [currentStep, setCurrentStep] = useState(0)
  const animationRef = useRef<number>()

  const [paths, setPaths] = useState<OptimizationPath[]>([
    {
      name: 'Gradient Descent (LR=0.1)',
      points: [{ x: -2, y: 2, gradient: { dx: 0, dy: 0 }, loss: 0 }],
      color: '#10b981',
      enabled: true,
      learningRate: 0.1,
    },
    {
      name: 'Gradient Descent (LR=0.05)',
      points: [{ x: -2, y: 2, gradient: { dx: 0, dy: 0 }, loss: 0 }],
      color: '#3b82f6',
      enabled: true,
      learningRate: 0.05,
    },
    {
      name: 'Gradient Descent (LR=0.01)',
      points: [{ x: -2, y: 2, gradient: { dx: 0, dy: 0 }, loss: 0 }],
      color: '#8b5cf6',
      enabled: true,
      learningRate: 0.01,
    },
    {
      name: 'Momentum (β=0.9)',
      points: [{ x: -2, y: 2, gradient: { dx: 0, dy: 0 }, loss: 0 }],
      color: '#f59e0b',
      enabled: false,
      learningRate: 0.05,
    },
  ])

  const [momentumState, setMomentumState] = useState<Record<string, { vx: number; vy: number }>>({})

  const functions: Record<string, (x: number, y: number) => number> = {
    quadratic: (x, y) => x * x + y * y,
    rosenbrock: (x, y) => (1 - x) ** 2 + 100 * (y - x ** 2) ** 2,
    beale: (x, y) => (1.5 - x + x * y) ** 2 + (2.25 - x + x * y * y) ** 2 + (2.625 - x + x * y ** 3) ** 2,
    himmelblau: (x, y) => (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2,
  }

  const gradient = (fn: (x: number, y: number) => number, x: number, y: number, h = 0.0001): { dx: number; dy: number } => {
    const dx = (fn(x + h, y) - fn(x - h, y)) / (2 * h)
    const dy = (fn(x, y + h) - fn(x, y - h)) / (2 * h)
    return { dx, dy }
  }

  const updatePaths = () => {
    const fn = functions[selectedFunction]
    const newPaths = paths.map((path) => {
      if (!path.enabled || path.points.length > 200) return path

      const lastPoint = path.points[path.points.length - 1]
      const { dx, dy } = gradient(fn, lastPoint.x, lastPoint.y)

      let newX = lastPoint.x
      let newY = lastPoint.y

      if (path.name.includes('Momentum')) {
        const beta = 0.9
        const state = momentumState[path.name] || { vx: 0, vy: 0 }
        state.vx = beta * state.vx + path.learningRate * dx
        state.vy = beta * state.vy + path.learningRate * dy
        newX = lastPoint.x - state.vx
        newY = lastPoint.y - state.vy
        setMomentumState((prev) => ({ ...prev, [path.name]: state }))
      } else {
        newX = lastPoint.x - path.learningRate * dx
        newY = lastPoint.y - path.learningRate * dy
      }

      const newLoss = fn(newX, newY)
      const newGrad = gradient(fn, newX, newY)

      return {
        ...path,
        points: [
          ...path.points,
          { x: newX, y: newY, gradient: newGrad, loss: newLoss },
        ],
      }
    })

    setPaths(newPaths)
    setCurrentStep((prev) => prev + 1)
  }

  const drawVisualization = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.fillStyle = '#1f2937'
    ctx.fillRect(0, 0, width, height)

    const fn = functions[selectedFunction]
    const xMin = -3
    const xMax = 3
    const yMin = -2
    const yMax = 3

    const toCanvasX = (x: number) => ((x - xMin) / (xMax - xMin)) * width
    const toCanvasY = (y: number) => height - ((y - yMin) / (yMax - yMin)) * height

    // Draw contour plot
    const resolution = 150
    const maxZ = selectedFunction === 'rosenbrock' ? 100 : selectedFunction === 'beale' ? 50 : 50

    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = xMin + (i / resolution) * (xMax - xMin)
        const y = yMin + (j / resolution) * (yMax - yMin)
        const z = fn(x, y)
        const normalized = Math.min(1, z / maxZ)
        ctx.fillStyle = `rgba(16, 185, 129, ${0.05 + normalized * 0.3})`
        const px = toCanvasX(x)
        const py = toCanvasY(y)
        ctx.fillRect(px, py, width / resolution + 1, height / resolution + 1)
      }
    }

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

    // Draw paths
    paths.forEach((path) => {
      if (!path.enabled) return

      ctx.strokeStyle = path.color
      ctx.lineWidth = 2
      ctx.beginPath()
      path.points.forEach((point, idx) => {
        const px = toCanvasX(point.x)
        const py = toCanvasY(point.y)
        if (idx === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      })
      ctx.stroke()

      // Draw gradient vectors
      if (showGradients) {
        path.points.forEach((point, idx) => {
          if (idx % 5 !== 0) return // Show every 5th gradient
          const px = toCanvasX(point.x)
          const py = toCanvasY(point.y)
          const scale = 20
          const gx = -point.gradient.dx * scale
          const gy = point.gradient.dy * scale

          ctx.strokeStyle = path.color + '80'
          ctx.lineWidth = 1.5
          ctx.beginPath()
          ctx.moveTo(px, py)
          ctx.lineTo(px + gx, py + gy)
          ctx.stroke()

          // Arrow head
          const angle = Math.atan2(gy, gx)
          ctx.beginPath()
          ctx.moveTo(px + gx, py + gy)
          ctx.lineTo(
            px + gx - 6 * Math.cos(angle - Math.PI / 6),
            py + gy - 6 * Math.sin(angle - Math.PI / 6)
          )
          ctx.lineTo(
            px + gx - 6 * Math.cos(angle + Math.PI / 6),
            py + gy - 6 * Math.sin(angle + Math.PI / 6)
          )
          ctx.lineTo(px + gx, py + gy)
          ctx.fillStyle = path.color + '80'
          ctx.fill()
        })
      }

      // Draw current point
      const lastPoint = path.points[path.points.length - 1]
      const px = toCanvasX(lastPoint.x)
      const py = toCanvasY(lastPoint.y)
      ctx.fillStyle = path.color
      ctx.beginPath()
      ctx.arc(px, py, 6, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = '#1f2937'
      ctx.lineWidth = 2
      ctx.stroke()

      // Draw starting point
      const startPoint = path.points[0]
      const sx = toCanvasX(startPoint.x)
      const sy = toCanvasY(startPoint.y)
      ctx.strokeStyle = path.color
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(sx, sy, 8, 0, Math.PI * 2)
      ctx.stroke()
    })

    // Draw minimum indicator (for simple functions)
    if (selectedFunction === 'quadratic') {
      const minX = toCanvasX(0)
      const minY = toCanvasY(0)
      ctx.strokeStyle = '#fbbf24'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(minX - 10, minY - 10)
      ctx.lineTo(minX + 10, minY + 10)
      ctx.moveTo(minX + 10, minY - 10)
      ctx.lineTo(minX - 10, minY + 10)
      ctx.stroke()
    }
  }

  useEffect(() => {
    drawVisualization()
  }, [paths, showGradients, selectedFunction])

  useEffect(() => {
    if (isRunning) {
      animationRef.current = window.setTimeout(() => {
        updatePaths()
      }, 100 / animationSpeed)
    }
    return () => {
      if (animationRef.current) clearTimeout(animationRef.current)
    }
  }, [isRunning, paths, selectedFunction, animationSpeed])

  const handleReset = () => {
    setIsRunning(false)
    setCurrentStep(0)
    setMomentumState({})
    setPaths((prev) =>
      prev.map((path) => ({
        ...path,
        points: [{ x: -2, y: 2, gradient: { dx: 0, dy: 0 }, loss: 0 }],
      }))
    )
  }

  const togglePath = (index: number) => {
    setPaths((prev) =>
      prev.map((path, idx) => (idx === index ? { ...path, enabled: !path.enabled } : path))
    )
  }

  const updateLearningRate = (index: number, lr: number) => {
    setPaths((prev) =>
      prev.map((path, idx) => (idx === index ? { ...path, learningRate: lr } : path))
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-emerald-600 to-teal-700 rounded-xl">
              <Compass className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-400 to-teal-400 bg-clip-text text-transparent">
                경사하강법 탐색기
              </h1>
              <p className="text-gray-400 mt-1">다양한 학습률에서의 경사하강 경로를 비교합니다</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Controls */}
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
                <option value="quadratic">Quadratic (간단)</option>
                <option value="rosenbrock">Rosenbrock</option>
                <option value="beale">Beale</option>
                <option value="himmelblau">Himmelblau</option>
              </select>
            </div>

            {/* Animation Speed */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                애니메이션 속도: {animationSpeed}x
              </label>
              <input
                type="range"
                min="0.5"
                max="5"
                step="0.5"
                value={animationSpeed}
                onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
                className="w-full accent-emerald-500"
              />
            </div>

            {/* Show Gradients */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showGradients}
                  onChange={(e) => setShowGradients(e.target.checked)}
                  className="w-4 h-4 accent-emerald-500"
                />
                <span className="text-sm text-gray-300">그라디언트 벡터 표시</span>
              </label>
            </div>

            {/* Path Settings */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">경로 설정</label>
              <div className="space-y-4">
                {paths.map((path, idx) => (
                  <div key={idx} className="bg-gray-700 rounded-lg p-3">
                    <label className="flex items-center gap-2 mb-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={path.enabled}
                        onChange={() => togglePath(idx)}
                        className="w-4 h-4 accent-emerald-500"
                      />
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: path.color }}></div>
                      <span className="text-xs text-gray-300">{path.name}</span>
                    </label>
                    {path.enabled && !path.name.includes('Momentum') && (
                      <div>
                        <label className="text-xs text-gray-400">LR: {path.learningRate.toFixed(3)}</label>
                        <input
                          type="range"
                          min="0.01"
                          max="0.2"
                          step="0.01"
                          value={path.learningRate}
                          onChange={(e) => updateLearningRate(idx, parseFloat(e.target.value))}
                          className="w-full accent-emerald-500 mt-1"
                        />
                      </div>
                    )}
                  </div>
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
            </div>

            {/* Stats */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-sm font-semibold text-emerald-400 mb-3">통계</h3>
              <div className="space-y-2">
                <div>
                  <div className="text-xs text-gray-400">현재 스텝</div>
                  <div className="text-2xl font-bold text-emerald-400">{currentStep}</div>
                </div>
              </div>
            </div>

            {/* Current Losses */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-sm font-semibold text-emerald-400 mb-3">현재 손실값</h3>
              <div className="space-y-2">
                {paths
                  .filter((p) => p.enabled)
                  .map((path) => {
                    const lastPoint = path.points[path.points.length - 1]
                    return (
                      <div key={path.name} className="flex items-center justify-between text-xs">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: path.color }}></div>
                          <span className="text-gray-400 truncate">{path.name.split('(')[0]}</span>
                        </div>
                        <span className="font-mono" style={{ color: path.color }}>
                          {lastPoint.loss.toFixed(4)}
                        </span>
                      </div>
                    )
                  })}
              </div>
            </div>
          </div>

          {/* Visualization */}
          <div className="lg:col-span-3">
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 text-emerald-400">경사하강 경로 시각화</h3>
              <canvas ref={canvasRef} width={900} height={700} className="w-full rounded-lg" />
              <div className="mt-6 grid grid-cols-3 gap-4 text-sm">
                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <ArrowRight className="w-4 h-4 text-emerald-400" />
                    <span className="font-medium text-gray-300">학습률 영향</span>
                  </div>
                  <p className="text-xs text-gray-400">
                    학습률이 크면 빠르지만 진동할 수 있고, 작으면 안정적이지만 느립니다.
                  </p>
                </div>
                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Compass className="w-4 h-4 text-emerald-400" />
                    <span className="font-medium text-gray-300">그라디언트 벡터</span>
                  </div>
                  <p className="text-xs text-gray-400">
                    화살표는 가장 가파른 상승 방향을 나타내며, 경사하강법은 이 반대 방향으로 이동합니다.
                  </p>
                </div>
                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-3 h-3 bg-emerald-500 rounded-full"></div>
                    <span className="font-medium text-gray-300">수렴 과정</span>
                  </div>
                  <p className="text-xs text-gray-400">
                    각 경로는 시작점(빈 원)에서 최솟값을 향해 수렴하는 과정을 보여줍니다.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
