'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, Plus, Trash2, ZoomIn, ZoomOut, Move, Lock, Unlock } from 'lucide-react'

interface Constraint {
  id: string
  type: 'linear' | 'quadratic'
  a: number
  b: number
  c: number
  inequality: '<=' | '>='
  active: boolean
}

interface Point {
  x: number
  y: number
}

export default function ConstraintVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [constraints, setConstraints] = useState<Constraint[]>([
    { id: '1', type: 'linear', a: 1, b: 1, c: 4, inequality: '<=', active: true },
    { id: '2', type: 'linear', a: 2, b: -1, c: 2, inequality: '<=', active: true },
    { id: '3', type: 'linear', a: -1, b: 2, c: 3, inequality: '<=', active: true },
  ])
  const [objectiveFunction, setObjectiveFunction] = useState({ a: 3, b: 2 })
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [optimalPoint, setOptimalPoint] = useState<Point | null>(null)

  const addConstraint = () => {
    const newConstraint: Constraint = {
      id: Date.now().toString(),
      type: 'linear',
      a: 1,
      b: 1,
      c: 5,
      inequality: '<=',
      active: true,
    }
    setConstraints([...constraints, newConstraint])
  }

  const removeConstraint = (id: string) => {
    setConstraints(constraints.filter((c) => c.id !== id))
  }

  const updateConstraint = (id: string, field: keyof Constraint, value: any) => {
    setConstraints(
      constraints.map((c) => (c.id === id ? { ...c, [field]: value } : c))
    )
  }

  const toggleConstraint = (id: string) => {
    setConstraints(
      constraints.map((c) => (c.id === id ? { ...c, active: !c.active } : c))
    )
  }

  const satisfiesConstraint = (x: number, y: number, constraint: Constraint): boolean => {
    if (!constraint.active) return true
    const value = constraint.a * x + constraint.b * y
    return constraint.inequality === '<=' ? value <= constraint.c : value >= constraint.c
  }

  const findOptimalPoint = () => {
    const activeConstraints = constraints.filter((c) => c.active)
    if (activeConstraints.length < 2) {
      setOptimalPoint(null)
      return
    }

    // Simple linear programming - find intersection points and evaluate
    const intersections: Point[] = []

    // Find all constraint intersections
    for (let i = 0; i < activeConstraints.length; i++) {
      for (let j = i + 1; j < activeConstraints.length; j++) {
        const c1 = activeConstraints[i]
        const c2 = activeConstraints[j]

        // Solve system of linear equations: a1*x + b1*y = c1, a2*x + b2*y = c2
        const det = c1.a * c2.b - c2.a * c1.b
        if (Math.abs(det) > 0.001) {
          const x = (c1.c * c2.b - c2.c * c1.b) / det
          const y = (c1.a * c2.c - c2.a * c1.c) / det

          // Check if point satisfies all constraints
          if (activeConstraints.every((c) => satisfiesConstraint(x, y, c))) {
            intersections.push({ x, y })
          }
        }
      }
    }

    // Also check axis intersections
    activeConstraints.forEach((c) => {
      if (Math.abs(c.b) > 0.001) {
        const y = c.c / c.b
        if (y >= 0 && activeConstraints.every((con) => satisfiesConstraint(0, y, con))) {
          intersections.push({ x: 0, y })
        }
      }
      if (Math.abs(c.a) > 0.001) {
        const x = c.c / c.a
        if (x >= 0 && activeConstraints.every((con) => satisfiesConstraint(x, 0, con))) {
          intersections.push({ x, y: 0 })
        }
      }
    })

    // Add origin if feasible
    if (activeConstraints.every((c) => satisfiesConstraint(0, 0, c))) {
      intersections.push({ x: 0, y: 0 })
    }

    // Find point with maximum objective function
    if (intersections.length > 0) {
      const best = intersections.reduce((max, point) => {
        const value = objectiveFunction.a * point.x + objectiveFunction.b * point.y
        const maxValue = objectiveFunction.a * max.x + objectiveFunction.b * max.y
        return value > maxValue ? point : max
      })
      setOptimalPoint(best)
    } else {
      setOptimalPoint(null)
    }
  }

  const drawVisualization = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.clearRect(0, 0, width, height)
    ctx.fillStyle = '#1f2937'
    ctx.fillRect(0, 0, width, height)

    const xMin = -2 + pan.x / zoom
    const xMax = 8 + pan.x / zoom
    const yMin = -2 + pan.y / zoom
    const yMax = 8 + pan.y / zoom

    const toCanvasX = (x: number) => ((x - xMin) / (xMax - xMin)) * width
    const toCanvasY = (y: number) => height - ((y - yMin) / (yMax - yMin)) * height

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)'
    ctx.lineWidth = 1
    for (let i = Math.floor(xMin); i <= Math.ceil(xMax); i++) {
      if (i >= xMin && i <= xMax) {
        ctx.beginPath()
        ctx.moveTo(toCanvasX(i), 0)
        ctx.lineTo(toCanvasX(i), height)
        ctx.stroke()
      }
    }
    for (let i = Math.floor(yMin); i <= Math.ceil(yMax); i++) {
      if (i >= yMin && i <= yMax) {
        ctx.beginPath()
        ctx.moveTo(0, toCanvasY(i))
        ctx.lineTo(width, toCanvasY(i))
        ctx.stroke()
      }
    }

    // Draw axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(toCanvasX(0), 0)
    ctx.lineTo(toCanvasX(0), height)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(0, toCanvasY(0))
    ctx.lineTo(width, toCanvasY(0))
    ctx.stroke()

    // Draw feasible region
    const resolution = 100
    const activeConstraints = constraints.filter((c) => c.active)

    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = xMin + (i / resolution) * (xMax - xMin)
        const y = yMin + (j / resolution) * (yMax - yMin)

        if (activeConstraints.every((c) => satisfiesConstraint(x, y, c))) {
          ctx.fillStyle = 'rgba(16, 185, 129, 0.15)'
          const px = toCanvasX(x)
          const py = toCanvasY(y)
          ctx.fillRect(px, py, width / resolution + 1, height / resolution + 1)
        }
      }
    }

    // Draw constraint lines
    activeConstraints.forEach((constraint, idx) => {
      const colors = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444']
      ctx.strokeStyle = colors[idx % colors.length]
      ctx.lineWidth = 2
      ctx.beginPath()

      const points: Point[] = []
      for (let x = xMin; x <= xMax; x += 0.1) {
        if (Math.abs(constraint.b) > 0.001) {
          const y = (constraint.c - constraint.a * x) / constraint.b
          if (y >= yMin && y <= yMax) {
            points.push({ x, y })
          }
        }
      }

      points.forEach((point, idx) => {
        const px = toCanvasX(point.x)
        const py = toCanvasY(point.y)
        if (idx === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      })
      ctx.stroke()

      // Draw constraint label
      if (points.length > 0) {
        const midPoint = points[Math.floor(points.length / 2)]
        ctx.fillStyle = colors[idx % colors.length]
        ctx.font = 'bold 12px Inter'
        ctx.fillText(
          `${constraint.a}x ${constraint.b >= 0 ? '+' : ''}${constraint.b}y ${constraint.inequality} ${constraint.c}`,
          toCanvasX(midPoint.x) + 10,
          toCanvasY(midPoint.y) - 10
        )
      }
    })

    // Draw objective function contour lines
    const objColors = ['rgba(251, 191, 36, 0.3)', 'rgba(251, 191, 36, 0.2)', 'rgba(251, 191, 36, 0.1)']
    for (let k = 0; k < 3; k++) {
      const value = (k + 1) * 5
      ctx.strokeStyle = objColors[k]
      ctx.lineWidth = 1
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      for (let x = xMin; x <= xMax; x += 0.1) {
        if (Math.abs(objectiveFunction.b) > 0.001) {
          const y = (value - objectiveFunction.a * x) / objectiveFunction.b
          if (y >= yMin && y <= yMax) {
            const px = toCanvasX(x)
            const py = toCanvasY(y)
            if (x === xMin) ctx.moveTo(px, py)
            else ctx.lineTo(px, py)
          }
        }
      }
      ctx.stroke()
      ctx.setLineDash([])
    }

    // Draw optimal point
    if (optimalPoint) {
      const px = toCanvasX(optimalPoint.x)
      const py = toCanvasY(optimalPoint.y)

      ctx.fillStyle = 'rgba(251, 191, 36, 0.3)'
      ctx.beginPath()
      ctx.arc(px, py, 15, 0, Math.PI * 2)
      ctx.fill()

      ctx.strokeStyle = '#fbbf24'
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.arc(px, py, 8, 0, Math.PI * 2)
      ctx.stroke()

      ctx.fillStyle = '#fbbf24'
      ctx.beginPath()
      ctx.arc(px, py, 4, 0, Math.PI * 2)
      ctx.fill()

      // Draw label
      ctx.fillStyle = '#fbbf24'
      ctx.font = 'bold 14px Inter'
      ctx.fillText(
        `최적해: (${optimalPoint.x.toFixed(2)}, ${optimalPoint.y.toFixed(2)})`,
        px + 20,
        py - 20
      )
      ctx.fillText(
        `값: ${(objectiveFunction.a * optimalPoint.x + objectiveFunction.b * optimalPoint.y).toFixed(2)}`,
        px + 20,
        py - 5
      )
    }
  }

  useEffect(() => {
    findOptimalPoint()
  }, [constraints, objectiveFunction])

  useEffect(() => {
    drawVisualization()
  }, [constraints, objectiveFunction, zoom, pan, optimalPoint])

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true)
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y })
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y })
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-emerald-600 to-teal-700 rounded-xl">
              <Lock className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-400 to-teal-400 bg-clip-text text-transparent">
                제약조건 최적화 시각화
              </h1>
              <p className="text-gray-400 mt-1">선형 제약조건 하에서 최적해를 시각적으로 탐색합니다</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Controls */}
          <div className="lg:col-span-1 space-y-4">
            {/* Objective Function */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-sm font-semibold text-emerald-400 mb-4">목적 함수 (최대화)</h3>
              <div className="space-y-3">
                <div>
                  <label className="text-xs text-gray-400">계수 a (x)</label>
                  <input
                    type="number"
                    value={objectiveFunction.a}
                    onChange={(e) => setObjectiveFunction({ ...objectiveFunction, a: parseFloat(e.target.value) || 0 })}
                    className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 mt-1 text-sm"
                    step="0.1"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-400">계수 b (y)</label>
                  <input
                    type="number"
                    value={objectiveFunction.b}
                    onChange={(e) => setObjectiveFunction({ ...objectiveFunction, b: parseFloat(e.target.value) || 0 })}
                    className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 mt-1 text-sm"
                    step="0.1"
                  />
                </div>
                <div className="text-center text-sm text-gray-300 mt-2 p-2 bg-gray-700 rounded">
                  f(x,y) = {objectiveFunction.a}x + {objectiveFunction.b}y
                </div>
              </div>
            </div>

            {/* Constraints */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-emerald-400">제약조건</h3>
                <button
                  onClick={addConstraint}
                  className="p-1 bg-emerald-600 hover:bg-emerald-700 rounded"
                >
                  <Plus className="w-4 h-4" />
                </button>
              </div>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {constraints.map((constraint, idx) => (
                  <div key={constraint.id} className="bg-gray-700 rounded-lg p-3 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-gray-300">제약 {idx + 1}</span>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => toggleConstraint(constraint.id)}
                          className="p-1 hover:bg-gray-600 rounded"
                        >
                          {constraint.active ? <Lock className="w-3 h-3" /> : <Unlock className="w-3 h-3" />}
                        </button>
                        <button
                          onClick={() => removeConstraint(constraint.id)}
                          className="p-1 hover:bg-gray-600 rounded text-red-400"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      <input
                        type="number"
                        value={constraint.a}
                        onChange={(e) => updateConstraint(constraint.id, 'a', parseFloat(e.target.value) || 0)}
                        className="bg-gray-600 rounded px-2 py-1 text-xs"
                        placeholder="a"
                        step="0.1"
                      />
                      <input
                        type="number"
                        value={constraint.b}
                        onChange={(e) => updateConstraint(constraint.id, 'b', parseFloat(e.target.value) || 0)}
                        className="bg-gray-600 rounded px-2 py-1 text-xs"
                        placeholder="b"
                        step="0.1"
                      />
                      <input
                        type="number"
                        value={constraint.c}
                        onChange={(e) => updateConstraint(constraint.id, 'c', parseFloat(e.target.value) || 0)}
                        className="bg-gray-600 rounded px-2 py-1 text-xs"
                        placeholder="c"
                        step="0.1"
                      />
                    </div>
                    <select
                      value={constraint.inequality}
                      onChange={(e) => updateConstraint(constraint.id, 'inequality', e.target.value)}
                      className="w-full bg-gray-600 rounded px-2 py-1 text-xs"
                    >
                      <option value="<=">≤ (작거나 같음)</option>
                      <option value=">=">&ge; (크거나 같음)</option>
                    </select>
                    <div className="text-center text-xs text-gray-300 bg-gray-600 rounded py-1">
                      {constraint.a}x + {constraint.b}y {constraint.inequality} {constraint.c}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* View Controls */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-sm font-semibold text-emerald-400 mb-4">보기 설정</h3>
              <div className="space-y-3">
                <button
                  onClick={() => setZoom(Math.min(zoom * 1.2, 3))}
                  className="w-full bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded flex items-center justify-center gap-2"
                >
                  <ZoomIn className="w-4 h-4" />
                  확대
                </button>
                <button
                  onClick={() => setZoom(Math.max(zoom / 1.2, 0.5))}
                  className="w-full bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded flex items-center justify-center gap-2"
                >
                  <ZoomOut className="w-4 h-4" />
                  축소
                </button>
                <button
                  onClick={() => {
                    setZoom(1)
                    setPan({ x: 0, y: 0 })
                  }}
                  className="w-full bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded flex items-center justify-center gap-2"
                >
                  <Move className="w-4 h-4" />
                  초기화
                </button>
              </div>
            </div>

            {/* Optimal Solution */}
            {optimalPoint && (
              <div className="bg-gradient-to-br from-yellow-900/30 to-yellow-800/30 border border-yellow-600 rounded-xl p-6">
                <h3 className="text-sm font-semibold text-yellow-400 mb-3">최적해</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-300">x =</span>
                    <span className="font-mono text-yellow-400">{optimalPoint.x.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">y =</span>
                    <span className="font-mono text-yellow-400">{optimalPoint.y.toFixed(4)}</span>
                  </div>
                  <div className="border-t border-yellow-600 pt-2 mt-2">
                    <div className="flex justify-between">
                      <span className="text-gray-300">목적함수 값:</span>
                      <span className="font-mono text-yellow-400 font-bold">
                        {(objectiveFunction.a * optimalPoint.x + objectiveFunction.b * optimalPoint.y).toFixed(4)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Visualization */}
          <div className="lg:col-span-3">
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-emerald-400">제약조건 공간</h3>
                <div className="text-sm text-gray-400">
                  <Move className="w-4 h-4 inline mr-1" />
                  드래그하여 이동
                </div>
              </div>
              <canvas
                ref={canvasRef}
                width={900}
                height={700}
                className="w-full rounded-lg cursor-move"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
              />
              <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-3 h-3 bg-emerald-500/30 rounded"></div>
                    <span className="text-gray-300">실행가능 영역</span>
                  </div>
                  <p className="text-xs text-gray-500">모든 제약조건을 만족하는 영역</p>
                </div>
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-3 h-3 bg-yellow-500 rounded"></div>
                    <span className="text-gray-300">최적해</span>
                  </div>
                  <p className="text-xs text-gray-500">목적함수를 최대화하는 점</p>
                </div>
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-12 h-0.5 border-t border-dashed border-yellow-500"></div>
                    <span className="text-gray-300">등고선</span>
                  </div>
                  <p className="text-xs text-gray-500">목적함수 값이 같은 선</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
