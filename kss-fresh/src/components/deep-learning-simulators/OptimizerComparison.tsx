'use client'

import { useState, useEffect, useRef } from 'react'
import { Play, Pause, RotateCcw } from 'lucide-react'

type OptimizerType = 'sgd' | 'momentum' | 'rmsprop' | 'adam'

interface OptimizerState {
  x: number
  y: number
  vx: number // velocity or moving average
  vy: number
  mx: number // moment (for Adam)
  my: number
  t: number // timestep
  active: boolean
  history: { x: number; y: number; loss: number }[]
}

export default function OptimizerComparison() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartRef = useRef<HTMLCanvasElement>(null)

  const [isRunning, setIsRunning] = useState(false)
  const [learningRate, setLearningRate] = useState(0.1)
  const [iteration, setIteration] = useState(0)

  const [optimizers, setOptimizers] = useState<Record<OptimizerType, OptimizerState>>({
    sgd: {
      x: -2, y: 2, vx: 0, vy: 0, mx: 0, my: 0, t: 0,
      active: true,
      history: []
    },
    momentum: {
      x: -2, y: 2, vx: 0, vy: 0, mx: 0, my: 0, t: 0,
      active: true,
      history: []
    },
    rmsprop: {
      x: -2, y: 2, vx: 0, vy: 0, mx: 0, my: 0, t: 0,
      active: true,
      history: []
    },
    adam: {
      x: -2, y: 2, vx: 0, vy: 0, mx: 0, my: 0, t: 0,
      active: true,
      history: []
    }
  })

  // Loss function (Rosenbrock function)
  const lossFunction = (x: number, y: number): number => {
    const a = 1
    const b = 100
    return Math.pow(a - x, 2) + b * Math.pow(y - x * x, 2)
  }

  // Gradient of loss function
  const gradient = (x: number, y: number): [number, number] => {
    const a = 1
    const b = 100
    const dx = -2 * (a - x) - 4 * b * x * (y - x * x)
    const dy = 2 * b * (y - x * x)
    return [dx, dy]
  }

  // Update optimizer step
  const updateOptimizer = (
    type: OptimizerType,
    state: OptimizerState,
    lr: number
  ): OptimizerState => {
    if (!state.active) return state

    const [gx, gy] = gradient(state.x, state.y)
    const beta1 = 0.9
    const beta2 = 0.999
    const epsilon = 1e-8

    let newState = { ...state, t: state.t + 1 }

    switch (type) {
      case 'sgd':
        newState.x -= lr * gx
        newState.y -= lr * gy
        break

      case 'momentum':
        newState.vx = beta1 * state.vx - lr * gx
        newState.vy = beta1 * state.vy - lr * gy
        newState.x += newState.vx
        newState.y += newState.vy
        break

      case 'rmsprop':
        newState.vx = beta2 * state.vx + (1 - beta2) * gx * gx
        newState.vy = beta2 * state.vy + (1 - beta2) * gy * gy
        newState.x -= lr * gx / (Math.sqrt(newState.vx) + epsilon)
        newState.y -= lr * gy / (Math.sqrt(newState.vy) + epsilon)
        break

      case 'adam':
        newState.mx = beta1 * state.mx + (1 - beta1) * gx
        newState.my = beta1 * state.my + (1 - beta1) * gy
        newState.vx = beta2 * state.vx + (1 - beta2) * gx * gx
        newState.vy = beta2 * state.vy + (1 - beta2) * gy * gy

        const mxHat = newState.mx / (1 - Math.pow(beta1, newState.t))
        const myHat = newState.my / (1 - Math.pow(beta1, newState.t))
        const vxHat = newState.vx / (1 - Math.pow(beta2, newState.t))
        const vyHat = newState.vy / (1 - Math.pow(beta2, newState.t))

        newState.x -= lr * mxHat / (Math.sqrt(vxHat) + epsilon)
        newState.y -= lr * myHat / (Math.sqrt(vyHat) + epsilon)
        break
    }

    const loss = lossFunction(newState.x, newState.y)
    newState.history = [
      ...state.history,
      { x: newState.x, y: newState.y, loss }
    ]

    // Keep only last 100 history points
    if (newState.history.length > 100) {
      newState.history = newState.history.slice(-100)
    }

    return newState
  }

  // Draw contour plot
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    // Clear
    ctx.fillStyle = '#1f2937'
    ctx.fillRect(0, 0, width, height)

    // Draw contour lines
    const xRange = [-3, 3]
    const yRange = [-1, 5]
    const gridSize = 100

    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const x = xRange[0] + (xRange[1] - xRange[0]) * (i / gridSize)
        const y = yRange[0] + (yRange[1] - yRange[0]) * (j / gridSize)
        const loss = lossFunction(x, y)

        const logLoss = Math.log(loss + 1)
        const normalized = Math.min(1, logLoss / 8)

        const r = Math.floor(59 + (239 - 59) * normalized)
        const g = Math.floor(130 + (68 - 130) * normalized)
        const b = Math.floor(246 + (68 - 246) * normalized)

        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.3)`

        const px = ((x - xRange[0]) / (xRange[1] - xRange[0])) * width
        const py = height - ((y - yRange[0]) / (yRange[1] - yRange[0])) * height

        ctx.fillRect(px, py, width / gridSize + 1, height / gridSize + 1)
      }
    }

    // Draw grid
    ctx.strokeStyle = '#374151'
    ctx.lineWidth = 1
    for (let i = 0; i <= 10; i++) {
      ctx.beginPath()
      ctx.moveTo((i / 10) * width, 0)
      ctx.lineTo((i / 10) * width, height)
      ctx.stroke()

      ctx.beginPath()
      ctx.moveTo(0, (i / 10) * height)
      ctx.lineTo(width, (i / 10) * height)
      ctx.stroke()
    }

    // Draw optimizer paths
    const colors: Record<OptimizerType, string> = {
      sgd: '#ef4444',
      momentum: '#f59e0b',
      rmsprop: '#10b981',
      adam: '#3b82f6'
    }

    Object.entries(optimizers).forEach(([type, state]) => {
      if (!state.active || state.history.length === 0) return

      ctx.strokeStyle = colors[type as OptimizerType]
      ctx.lineWidth = 2
      ctx.beginPath()

      state.history.forEach((point, i) => {
        const px = ((point.x - xRange[0]) / (xRange[1] - xRange[0])) * width
        const py = height - ((point.y - yRange[0]) / (yRange[1] - yRange[0])) * height

        if (i === 0) {
          ctx.moveTo(px, py)
        } else {
          ctx.lineTo(px, py)
        }
      })

      ctx.stroke()

      // Draw current position
      if (state.history.length > 0) {
        const last = state.history[state.history.length - 1]
        const px = ((last.x - xRange[0]) / (xRange[1] - xRange[0])) * width
        const py = height - ((last.y - yRange[0]) / (yRange[1] - yRange[0])) * height

        ctx.fillStyle = colors[type as OptimizerType]
        ctx.beginPath()
        ctx.arc(px, py, 6, 0, Math.PI * 2)
        ctx.fill()

        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 2
        ctx.stroke()
      }
    })

    // Draw global minimum
    const minX = 1
    const minY = 1
    const px = ((minX - xRange[0]) / (xRange[1] - xRange[0])) * width
    const py = height - ((minY - yRange[0]) / (yRange[1] - yRange[0])) * height

    ctx.fillStyle = '#ffffff'
    ctx.beginPath()
    ctx.arc(px, py, 8, 0, Math.PI * 2)
    ctx.fill()

    ctx.strokeStyle = '#000000'
    ctx.lineWidth = 2
    ctx.stroke()
  }, [optimizers])

  // Draw loss chart
  useEffect(() => {
    const canvas = chartRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    // Clear
    ctx.fillStyle = '#1f2937'
    ctx.fillRect(0, 0, width, height)

    // Draw grid
    ctx.strokeStyle = '#374151'
    ctx.lineWidth = 1
    for (let i = 0; i <= 10; i++) {
      ctx.beginPath()
      ctx.moveTo((i / 10) * width, 0)
      ctx.lineTo((i / 10) * width, height)
      ctx.stroke()

      ctx.beginPath()
      ctx.moveTo(0, (i / 10) * height)
      ctx.lineTo(width, (i / 10) * height)
      ctx.stroke()
    }

    // Find max loss for normalization
    let maxLoss = 1
    Object.values(optimizers).forEach(state => {
      state.history.forEach(point => {
        maxLoss = Math.max(maxLoss, point.loss)
      })
    })

    const colors: Record<OptimizerType, string> = {
      sgd: '#ef4444',
      momentum: '#f59e0b',
      rmsprop: '#10b981',
      adam: '#3b82f6'
    }

    // Draw loss curves
    Object.entries(optimizers).forEach(([type, state]) => {
      if (!state.active || state.history.length === 0) return

      ctx.strokeStyle = colors[type as OptimizerType]
      ctx.lineWidth = 2
      ctx.beginPath()

      state.history.forEach((point, i) => {
        const x = (i / 100) * width
        const y = height - (Math.log(point.loss + 1) / Math.log(maxLoss + 1)) * height

        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })

      ctx.stroke()
    })
  }, [optimizers])

  // Training loop
  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(() => {
      setOptimizers(prev => {
        const updated: Record<OptimizerType, OptimizerState> = {} as any

        Object.entries(prev).forEach(([type, state]) => {
          updated[type as OptimizerType] = updateOptimizer(
            type as OptimizerType,
            state,
            learningRate
          )
        })

        return updated
      })

      setIteration(prev => prev + 1)

      // Stop after 100 iterations
      if (iteration >= 99) {
        setIsRunning(false)
      }
    }, 50)

    return () => clearInterval(interval)
  }, [isRunning, learningRate, iteration])

  const handleReset = () => {
    setIsRunning(false)
    setIteration(0)
    setOptimizers({
      sgd: {
        x: -2, y: 2, vx: 0, vy: 0, mx: 0, my: 0, t: 0,
        active: optimizers.sgd.active,
        history: []
      },
      momentum: {
        x: -2, y: 2, vx: 0, vy: 0, mx: 0, my: 0, t: 0,
        active: optimizers.momentum.active,
        history: []
      },
      rmsprop: {
        x: -2, y: 2, vx: 0, vy: 0, mx: 0, my: 0, t: 0,
        active: optimizers.rmsprop.active,
        history: []
      },
      adam: {
        x: -2, y: 2, vx: 0, vy: 0, mx: 0, my: 0, t: 0,
        active: optimizers.adam.active,
        history: []
      }
    })
  }

  const toggleOptimizer = (type: OptimizerType) => {
    setOptimizers(prev => ({
      ...prev,
      [type]: {
        ...prev[type],
        active: !prev[type].active
      }
    }))
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Learning Rate */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">
            학습률: {learningRate.toFixed(3)}
          </h3>
          <input
            type="range"
            min="0.01"
            max="0.5"
            step="0.01"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            className="w-full"
            disabled={isRunning}
          />
        </div>

        {/* Optimizer Selection */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-4 lg:col-span-2">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">
            비교할 Optimizer 선택
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {(['sgd', 'momentum', 'rmsprop', 'adam'] as OptimizerType[]).map(type => (
              <button
                key={type}
                onClick={() => toggleOptimizer(type)}
                disabled={isRunning}
                className={`px-3 py-2 rounded-lg font-medium text-sm transition-all ${
                  optimizers[type].active
                    ? 'bg-violet-500 text-white'
                    : 'bg-gray-300 dark:bg-gray-600 text-gray-700 dark:text-gray-300'
                } disabled:opacity-50`}
              >
                {type.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Contour Plot */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Optimization Path (Rosenbrock Function)
            </h3>

            <div className="flex gap-2">
              <button
                onClick={() => setIsRunning(!isRunning)}
                className="flex items-center gap-2 px-4 py-2 bg-violet-500 hover:bg-violet-600 text-white rounded-lg font-medium transition-colors"
              >
                {isRunning ? (
                  <>
                    <Pause size={16} />
                    일시정지
                  </>
                ) : (
                  <>
                    <Play size={16} />
                    시작
                  </>
                )}
              </button>

              <button
                onClick={handleReset}
                className="flex items-center gap-2 px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg font-medium transition-colors"
              >
                <RotateCcw size={16} />
                초기화
              </button>
            </div>
          </div>

          <canvas
            ref={canvasRef}
            width={600}
            height={500}
            className="w-full h-auto border border-gray-300 dark:border-gray-600 rounded-lg mb-4"
          />

          {/* Legend */}
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-red-500" />
              <span className="text-gray-700 dark:text-gray-300">SGD</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-amber-500" />
              <span className="text-gray-700 dark:text-gray-300">Momentum</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-green-500" />
              <span className="text-gray-700 dark:text-gray-300">RMSprop</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-blue-500" />
              <span className="text-gray-700 dark:text-gray-300">Adam</span>
            </div>
          </div>
        </div>

        {/* Loss Chart */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Loss Over Time
          </h3>

          <canvas
            ref={chartRef}
            width={600}
            height={400}
            className="w-full h-auto border border-gray-300 dark:border-gray-600 rounded-lg mb-4"
          />

          {/* Metrics */}
          <div className="grid grid-cols-2 gap-4 mt-4">
            <div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Iteration</div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">{iteration}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Active Optimizers</div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {Object.values(optimizers).filter(o => o.active).length}
              </div>
            </div>
          </div>

          {/* Current Losses */}
          <div className="mt-4 space-y-2">
            {Object.entries(optimizers).map(([type, state]) => {
              if (!state.active) return null

              const currentLoss = state.history[state.history.length - 1]?.loss ?? 0

              return (
                <div key={type} className="flex justify-between items-center text-sm">
                  <span className="text-gray-600 dark:text-gray-400">{type.toUpperCase()}</span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {currentLoss.toFixed(4)}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Info */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-300 mb-3">
          💡 Optimizer 비교 가이드
        </h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm text-blue-800 dark:text-blue-200">
          <div>
            <strong>SGD:</strong> 가장 기본적인 경사하강법. 학습 경로가 단순하지만 느릴 수 있음
          </div>
          <div>
            <strong>Momentum:</strong> 이전 방향을 고려하여 가속. 지그재그 경로 감소
          </div>
          <div>
            <strong>RMSprop:</strong> 학습률을 파라미터별로 적응적으로 조정
          </div>
          <div>
            <strong>Adam:</strong> Momentum + RMSprop 결합. 가장 널리 사용됨
          </div>
        </div>
      </div>
    </div>
  )
}
