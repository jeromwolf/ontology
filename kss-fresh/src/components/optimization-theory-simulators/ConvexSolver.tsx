'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, Pause, RotateCcw, Target, CheckCircle2, AlertCircle } from 'lucide-react'

interface Point {
  x: number
  y: number
}

interface IterationData {
  point: Point
  value: number
  gradient: { dx: number; dy: number }
  kktViolation: number
  iteration: number
}

export default function ConvexSolver() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartRef = useRef<HTMLCanvasElement>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [currentPoint, setCurrentPoint] = useState<Point>({ x: -1.5, y: 1.5 })
  const [history, setHistory] = useState<IterationData[]>([])
  const [iteration, setIteration] = useState(0)
  const [converged, setConverged] = useState(false)
  const [tolerance, setTolerance] = useState(0.001)
  const [barrierParameter, setBarrierParameter] = useState(1)
  const animationRef = useRef<number>()

  // Convex objective function: f(x,y) = x^2 + 2y^2 - 2xy + 4x - 2y
  const objectiveFunction = (x: number, y: number): number => {
    return x * x + 2 * y * y - 2 * x * y + 4 * x - 2 * y
  }

  // Gradient of objective function
  const gradient = (x: number, y: number): { dx: number; dy: number } => {
    const dx = 2 * x - 2 * y + 4
    const dy = 4 * y - 2 * x - 2
    return { dx, dy }
  }

  // Constraints: x >= 0, y >= 0, x + y <= 2
  const constraints = [
    { fn: (x: number, y: number) => x, name: 'x ≥ 0' },
    { fn: (x: number, y: number) => y, name: 'y ≥ 0' },
    { fn: (x: number, y: number) => 2 - x - y, name: 'x + y ≤ 2' },
  ]

  // Check if point is feasible
  const isFeasible = (x: number, y: number): boolean => {
    return constraints.every((c) => c.fn(x, y) >= 0)
  }

  // Barrier function (logarithmic barrier)
  const barrierFunction = (x: number, y: number, mu: number): number => {
    let barrier = 0
    for (const c of constraints) {
      const val = c.fn(x, y)
      if (val <= 0) return Infinity
      barrier -= Math.log(val)
    }
    return objectiveFunction(x, y) + mu * barrier
  }

  // Gradient of barrier function
  const barrierGradient = (x: number, y: number, mu: number): { dx: number; dy: number } => {
    const objGrad = gradient(x, y)
    let barrierDx = 0
    let barrierDy = 0

    // Constraint: x >= 0
    if (x > 0) {
      barrierDx -= mu / x
    }

    // Constraint: y >= 0
    if (y > 0) {
      barrierDy -= mu / y
    }

    // Constraint: x + y <= 2
    const slack = 2 - x - y
    if (slack > 0) {
      barrierDx += mu / slack
      barrierDy += mu / slack
    }

    return {
      dx: objGrad.dx + barrierDx,
      dy: objGrad.dy + barrierDy,
    }
  }

  // KKT conditions violation (simplified check)
  const kktViolation = (x: number, y: number): number => {
    const grad = gradient(x, y)
    let violation = Math.sqrt(grad.dx ** 2 + grad.dy ** 2)

    // Check complementary slackness
    for (const c of constraints) {
      const val = c.fn(x, y)
      if (val < 0) violation += Math.abs(val) * 100
    }

    return violation
  }

  const performIteration = () => {
    if (converged) return

    const grad = barrierGradient(currentPoint.x, currentPoint.y, barrierParameter)
    const learningRate = 0.1

    // Interior point method update
    let newX = currentPoint.x - learningRate * grad.dx
    let newY = currentPoint.y - learningRate * grad.dy

    // Project back to feasible region with small margin
    const margin = 0.01
    newX = Math.max(margin, Math.min(2 - margin, newX))
    newY = Math.max(margin, Math.min(2 - newX - margin, newY))

    const newValue = objectiveFunction(newX, newY)
    const newGrad = gradient(newX, newY)
    const kkt = kktViolation(newX, newY)

    const iterData: IterationData = {
      point: { x: newX, y: newY },
      value: newValue,
      gradient: newGrad,
      kktViolation: kkt,
      iteration: iteration + 1,
    }

    setHistory((prev) => [...prev, iterData])
    setCurrentPoint({ x: newX, y: newY })
    setIteration((prev) => prev + 1)

    // Check convergence
    if (kkt < tolerance) {
      setConverged(true)
      setIsRunning(false)
    }

    // Decrease barrier parameter
    setBarrierParameter((prev) => prev * 0.95)
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

    const xMin = -0.5
    const xMax = 2.5
    const yMin = -0.5
    const yMax = 2.5

    const toCanvasX = (x: number) => ((x - xMin) / (xMax - xMin)) * width
    const toCanvasY = (y: number) => height - ((y - yMin) / (yMax - yMin)) * height

    // Draw objective function contours
    const resolution = 150
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = xMin + (i / resolution) * (xMax - xMin)
        const y = yMin + (j / resolution) * (yMax - yMin)

        if (isFeasible(x, y)) {
          const value = objectiveFunction(x, y)
          const normalized = Math.min(1, Math.max(0, (value + 5) / 20))
          ctx.fillStyle = `rgba(16, 185, 129, ${0.1 + normalized * 0.4})`
        } else {
          ctx.fillStyle = 'rgba(239, 68, 68, 0.1)'
        }

        const px = toCanvasX(x)
        const py = toCanvasY(y)
        ctx.fillRect(px, py, width / resolution + 1, height / resolution + 1)
      }
    }

    // Draw constraints
    ctx.strokeStyle = '#ef4444'
    ctx.lineWidth = 2

    // x = 0
    ctx.beginPath()
    ctx.moveTo(toCanvasX(0), toCanvasY(yMin))
    ctx.lineTo(toCanvasX(0), toCanvasY(yMax))
    ctx.stroke()

    // y = 0
    ctx.beginPath()
    ctx.moveTo(toCanvasX(xMin), toCanvasY(0))
    ctx.lineTo(toCanvasX(xMax), toCanvasY(0))
    ctx.stroke()

    // x + y = 2
    ctx.beginPath()
    ctx.moveTo(toCanvasX(0), toCanvasY(2))
    ctx.lineTo(toCanvasX(2), toCanvasY(0))
    ctx.stroke()

    // Draw feasible region boundary
    ctx.fillStyle = 'rgba(16, 185, 129, 0.1)'
    ctx.strokeStyle = '#10b981'
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.moveTo(toCanvasX(0), toCanvasY(0))
    ctx.lineTo(toCanvasX(2), toCanvasY(0))
    ctx.lineTo(toCanvasX(0), toCanvasY(2))
    ctx.closePath()
    ctx.stroke()

    // Draw optimization path
    if (history.length > 0) {
      ctx.strokeStyle = '#3b82f6'
      ctx.lineWidth = 2
      ctx.beginPath()
      history.forEach((data, idx) => {
        const px = toCanvasX(data.point.x)
        const py = toCanvasY(data.point.y)
        if (idx === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      })
      ctx.stroke()

      // Draw gradient at current point
      const grad = gradient(currentPoint.x, currentPoint.y)
      const px = toCanvasX(currentPoint.x)
      const py = toCanvasY(currentPoint.y)
      const scale = 30
      const gx = -grad.dx * scale
      const gy = grad.dy * scale

      ctx.strokeStyle = '#fbbf24'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(px, py)
      ctx.lineTo(px + gx, py + gy)
      ctx.stroke()

      // Arrow head
      const angle = Math.atan2(gy, gx)
      ctx.beginPath()
      ctx.moveTo(px + gx, py + gy)
      ctx.lineTo(
        px + gx - 8 * Math.cos(angle - Math.PI / 6),
        py + gy - 8 * Math.sin(angle - Math.PI / 6)
      )
      ctx.lineTo(
        px + gx - 8 * Math.cos(angle + Math.PI / 6),
        py + gy - 8 * Math.sin(angle + Math.PI / 6)
      )
      ctx.lineTo(px + gx, py + gy)
      ctx.fillStyle = '#fbbf24'
      ctx.fill()
    }

    // Draw current point
    const px = toCanvasX(currentPoint.x)
    const py = toCanvasY(currentPoint.y)
    ctx.fillStyle = converged ? '#10b981' : '#3b82f6'
    ctx.beginPath()
    ctx.arc(px, py, 8, 0, Math.PI * 2)
    ctx.fill()
    ctx.strokeStyle = '#1f2937'
    ctx.lineWidth = 2
    ctx.stroke()

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 3; i++) {
      const x = toCanvasX(i)
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.stroke()

      const y = toCanvasY(i)
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    // Labels
    ctx.fillStyle = '#9ca3af'
    ctx.font = 'bold 14px Inter'
    ctx.fillText('x', toCanvasX(xMax) - 20, toCanvasY(0) + 30)
    ctx.fillText('y', toCanvasX(0) - 30, toCanvasY(yMax) + 20)

    // Constraint labels
    ctx.fillStyle = '#ef4444'
    ctx.font = '12px Inter'
    ctx.fillText('x ≥ 0', toCanvasX(0.1), toCanvasY(2.2))
    ctx.fillText('y ≥ 0', toCanvasX(2.2), toCanvasY(0.1))
    ctx.fillText('x + y ≤ 2', toCanvasX(1), toCanvasY(1.2))
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

    if (history.length === 0) return

    const maxIter = history[history.length - 1].iteration
    const values = history.map((h) => h.value)
    const maxValue = Math.max(...values)
    const minValue = Math.min(...values)

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

    // Draw objective value curve
    ctx.strokeStyle = '#10b981'
    ctx.lineWidth = 2
    ctx.beginPath()
    history.forEach((data, idx) => {
      const x = (data.iteration / maxIter) * width
      const y = height - ((data.value - minValue) / (maxValue - minValue + 0.001)) * height
      if (idx === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // Draw KKT violation curve
    const kktValues = history.map((h) => h.kktViolation)
    const maxKKT = Math.max(...kktValues, 1)

    ctx.strokeStyle = '#f59e0b'
    ctx.lineWidth = 2
    ctx.beginPath()
    history.forEach((data, idx) => {
      const x = (data.iteration / maxIter) * width
      const y = height - (data.kktViolation / maxKKT) * height
      if (idx === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // Labels
    ctx.fillStyle = '#9ca3af'
    ctx.font = '12px Inter'
    ctx.fillText('반복 횟수', width - 70, height - 10)

    // Legend
    ctx.fillStyle = '#10b981'
    ctx.fillRect(10, 10, 20, 2)
    ctx.fillStyle = '#9ca3af'
    ctx.fillText('목적함수 값', 35, 15)

    ctx.fillStyle = '#f59e0b'
    ctx.fillRect(10, 25, 20, 2)
    ctx.fillStyle = '#9ca3af'
    ctx.fillText('KKT 조건 위반', 35, 30)
  }

  useEffect(() => {
    drawVisualization()
    drawConvergenceChart()
  }, [currentPoint, history, converged])

  useEffect(() => {
    if (isRunning && !converged) {
      animationRef.current = window.setTimeout(() => {
        performIteration()
      }, 100)
    }
    return () => {
      if (animationRef.current) clearTimeout(animationRef.current)
    }
  }, [isRunning, currentPoint, converged, barrierParameter])

  const handleReset = () => {
    setIsRunning(false)
    setCurrentPoint({ x: -1.5, y: 1.5 })
    setHistory([])
    setIteration(0)
    setConverged(false)
    setBarrierParameter(1)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-emerald-600 to-teal-700 rounded-xl">
              <Target className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-400 to-teal-400 bg-clip-text text-transparent">
                볼록 최적화 솔버
              </h1>
              <p className="text-gray-400 mt-1">내부점 방법(Interior Point Method)으로 제약조건 최적화 문제를 해결합니다</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Controls */}
          <div className="lg:col-span-1 space-y-4">
            {/* Problem Definition */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-sm font-semibold text-emerald-400 mb-3">최적화 문제</h3>
              <div className="space-y-2 text-sm">
                <div className="bg-gray-700 rounded p-2">
                  <div className="text-xs text-gray-400 mb-1">목적함수 (최소화):</div>
                  <div className="font-mono text-emerald-400">
                    f(x,y) = x² + 2y² - 2xy + 4x - 2y
                  </div>
                </div>
                <div className="bg-gray-700 rounded p-2">
                  <div className="text-xs text-gray-400 mb-1">제약조건:</div>
                  <div className="font-mono text-xs text-gray-300">
                    x ≥ 0<br />
                    y ≥ 0<br />
                    x + y ≤ 2
                  </div>
                </div>
              </div>
            </div>

            {/* Tolerance */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                수렴 허용오차: {tolerance.toFixed(4)}
              </label>
              <input
                type="range"
                min="0.0001"
                max="0.01"
                step="0.0001"
                value={tolerance}
                onChange={(e) => setTolerance(parseFloat(e.target.value))}
                className="w-full accent-emerald-500"
              />
            </div>

            {/* Controls */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 space-y-3">
              <button
                onClick={() => setIsRunning(!isRunning)}
                disabled={converged}
                className="w-full bg-gradient-to-r from-emerald-600 to-teal-700 hover:from-emerald-700 hover:to-teal-800 disabled:opacity-50 disabled:cursor-not-allowed text-white px-4 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2"
              >
                {isRunning ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                {isRunning ? '일시정지' : '최적화 시작'}
              </button>
              <button
                onClick={handleReset}
                className="w-full bg-gray-700 hover:bg-gray-600 text-white px-4 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2"
              >
                <RotateCcw className="w-5 h-5" />
                초기화
              </button>
            </div>

            {/* Status */}
            <div className={`rounded-xl p-6 border ${
              converged
                ? 'bg-emerald-900/30 border-emerald-600'
                : 'bg-gray-800 border-gray-700'
            }`}>
              <div className="flex items-center gap-2 mb-3">
                {converged ? (
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                ) : (
                  <AlertCircle className="w-5 h-5 text-yellow-400" />
                )}
                <h3 className="text-sm font-semibold text-emerald-400">
                  {converged ? '수렴 완료' : '최적화 진행중'}
                </h3>
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">반복 횟수:</span>
                  <span className="font-mono text-emerald-400">{iteration}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">현재 위치:</span>
                  <span className="font-mono text-emerald-400">
                    ({currentPoint.x.toFixed(3)}, {currentPoint.y.toFixed(3)})
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">목적함수 값:</span>
                  <span className="font-mono text-emerald-400">
                    {objectiveFunction(currentPoint.x, currentPoint.y).toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">KKT 위반:</span>
                  <span className="font-mono text-yellow-400">
                    {kktViolation(currentPoint.x, currentPoint.y).toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">장벽 파라미터:</span>
                  <span className="font-mono text-blue-400">
                    {barrierParameter.toFixed(4)}
                  </span>
                </div>
              </div>
            </div>

            {/* KKT Conditions */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-sm font-semibold text-emerald-400 mb-3">KKT 조건</h3>
              <div className="space-y-2 text-xs text-gray-300">
                <div>1. 정상성 (Stationarity): ∇f(x*) + Σλᵢ∇gᵢ(x*) = 0</div>
                <div>2. 원시 실행가능성: gᵢ(x*) ≥ 0</div>
                <div>3. 쌍대 실행가능성: λᵢ ≥ 0</div>
                <div>4. 상보적 여유성: λᵢgᵢ(x*) = 0</div>
              </div>
            </div>
          </div>

          {/* Visualizations */}
          <div className="lg:col-span-3 space-y-6">
            {/* Main Visualization */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 text-emerald-400">최적화 경로</h3>
              <canvas ref={canvasRef} width={900} height={600} className="w-full rounded-lg" />
              <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-12 h-0.5 bg-emerald-500"></div>
                    <span className="text-gray-300">실행가능 영역</span>
                  </div>
                  <p className="text-xs text-gray-500">모든 제약조건을 만족하는 삼각형 영역</p>
                </div>
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                    <span className="text-gray-300">최적화 경로</span>
                  </div>
                  <p className="text-xs text-gray-500">내부점 방법의 반복 경로</p>
                </div>
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-6 h-0.5 bg-yellow-500"></div>
                    <span className="text-gray-300">그라디언트</span>
                  </div>
                  <p className="text-xs text-gray-500">현재 위치의 경사 방향</p>
                </div>
              </div>
            </div>

            {/* Convergence Chart */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 text-emerald-400">수렴 분석</h3>
              <canvas ref={chartRef} width={900} height={300} className="w-full rounded-lg" />
            </div>

            {/* Method Description */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-3 text-emerald-400">내부점 방법 (Interior Point Method)</h3>
              <p className="text-sm text-gray-300 mb-3">
                제약조건이 있는 최적화 문제를 해결하는 강력한 알고리즘으로, 실행가능 영역의 내부에서 시작하여
                최적해를 향해 이동합니다. 장벽 함수(barrier function)를 사용하여 경계를 피하면서 최적해에 접근합니다.
              </p>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="font-medium text-emerald-400 mb-1">장점</div>
                  <ul className="text-xs text-gray-400 list-disc list-inside space-y-1">
                    <li>다항 시간 복잡도</li>
                    <li>볼록 문제에 매우 효율적</li>
                    <li>대규모 문제에 적합</li>
                  </ul>
                </div>
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="font-medium text-emerald-400 mb-1">응용 분야</div>
                  <ul className="text-xs text-gray-400 list-disc list-inside space-y-1">
                    <li>선형 프로그래밍</li>
                    <li>2차 프로그래밍</li>
                    <li>반정부호 프로그래밍</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
