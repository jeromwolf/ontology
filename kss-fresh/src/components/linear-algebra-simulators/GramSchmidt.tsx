'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, RotateCcw, ChevronRight, CheckCircle } from 'lucide-react'

interface Vector3D {
  x: number
  y: number
  z: number
}

export default function GramSchmidt() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [vectors, setVectors] = useState<Vector3D[]>([
    { x: 3, y: 1, z: 0 },
    { x: 2, y: 3, z: 0 },
    { x: 1, y: 2, z: 0 }
  ])
  const [orthogonalVectors, setOrthogonalVectors] = useState<Vector3D[]>([])
  const [currentStep, setCurrentStep] = useState<number>(0)
  const [isAnimating, setIsAnimating] = useState(false)
  const [showOriginal, setShowOriginal] = useState(true)

  useEffect(() => {
    drawCanvas()
  }, [vectors, orthogonalVectors, currentStep, showOriginal])

  const normalize = (v: Vector3D): Vector3D => {
    const mag = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    if (mag === 0) return { x: 0, y: 0, z: 0 }
    return { x: v.x / mag, y: v.y / mag, z: v.z / mag }
  }

  const dotProduct = (a: Vector3D, b: Vector3D): number => {
    return a.x * b.x + a.y * b.y + a.z * b.z
  }

  const scalarMultiply = (v: Vector3D, s: number): Vector3D => {
    return { x: v.x * s, y: v.y * s, z: v.z * s }
  }

  const subtract = (a: Vector3D, b: Vector3D): Vector3D => {
    return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }
  }

  const gramSchmidtStep = (index: number) => {
    if (index >= vectors.length) return null

    let u = vectors[index]

    // Subtract projections onto all previous orthogonal vectors
    for (let i = 0; i < index; i++) {
      const projection = dotProduct(vectors[index], orthogonalVectors[i])
      u = subtract(u, scalarMultiply(orthogonalVectors[i], projection))
    }

    return normalize(u)
  }

  const runAlgorithm = () => {
    setIsAnimating(true)
    setOrthogonalVectors([])
    setCurrentStep(0)

    let step = 0
    const interval = setInterval(() => {
      const result = gramSchmidtStep(step)
      if (result) {
        setOrthogonalVectors((prev) => [...prev, result])
        setCurrentStep(step + 1)
        step++
      }

      if (step >= vectors.length) {
        clearInterval(interval)
        setIsAnimating(false)
      }
    }, 1000)
  }

  const drawCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height
    const centerX = width / 2
    const centerY = height / 2
    const scale = 60

    // Clear canvas
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, width, height)

    // Draw grid
    ctx.strokeStyle = '#1e293b'
    ctx.lineWidth = 1
    for (let i = -10; i <= 10; i++) {
      ctx.beginPath()
      ctx.moveTo(centerX + i * scale, 0)
      ctx.lineTo(centerX + i * scale, height)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(0, centerY + i * scale)
      ctx.lineTo(width, centerY + i * scale)
      ctx.stroke()
    }

    // Draw axes
    ctx.strokeStyle = '#475569'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(0, centerY)
    ctx.lineTo(width, centerY)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(centerX, 0)
    ctx.lineTo(centerX, height)
    ctx.stroke()

    // Draw function
    const drawVector = (v: Vector3D, color: string, label: string, lineWidth = 3) => {
      const endX = centerX + v.x * scale
      const endY = centerY - v.y * scale

      ctx.strokeStyle = color
      ctx.lineWidth = lineWidth
      ctx.beginPath()
      ctx.moveTo(centerX, centerY)
      ctx.lineTo(endX, endY)
      ctx.stroke()

      // Arrow
      const angle = Math.atan2(-v.y, v.x)
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.moveTo(endX, endY)
      ctx.lineTo(
        endX - 12 * Math.cos(angle - Math.PI / 6),
        endY - 12 * Math.sin(angle - Math.PI / 6)
      )
      ctx.lineTo(
        endX - 12 * Math.cos(angle + Math.PI / 6),
        endY - 12 * Math.sin(angle + Math.PI / 6)
      )
      ctx.closePath()
      ctx.fill()

      // Label
      ctx.font = 'bold 14px Inter'
      ctx.fillText(label, endX + 10, endY - 10)
    }

    // Draw original vectors
    if (showOriginal) {
      vectors.forEach((v, i) => {
        const colors = ['#64748b', '#64748b', '#64748b']
        drawVector(v, colors[i], `v${i + 1}`, 2)
      })
    }

    // Draw orthogonal vectors
    orthogonalVectors.forEach((v, i) => {
      const colors = ['#3b82f6', '#10b981', '#f59e0b']
      drawVector(scalarMultiply(v, 2), colors[i], `u${i + 1}`, 3)
    })

    // Draw projection lines for current step
    if (currentStep > 0 && currentStep <= vectors.length && orthogonalVectors.length > 0) {
      const currentVector = vectors[currentStep - 1]

      for (let i = 0; i < currentStep - 1; i++) {
        const u = orthogonalVectors[i]
        const projection = dotProduct(currentVector, u)
        const projVector = scalarMultiply(u, projection)

        // Draw projection
        ctx.strokeStyle = '#ef4444'
        ctx.setLineDash([5, 5])
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(centerX + projVector.x * scale, centerY - projVector.y * scale)
        ctx.lineTo(centerX + currentVector.x * scale, centerY - currentVector.y * scale)
        ctx.stroke()
        ctx.setLineDash([])
      }
    }
  }

  const updateVector = (index: number, axis: 'x' | 'y' | 'z', value: string) => {
    const newVectors = vectors.map((v, i) =>
      i === index ? { ...v, [axis]: parseFloat(value) || 0 } : v
    )
    setVectors(newVectors)
    setOrthogonalVectors([])
    setCurrentStep(0)
  }

  const reset = () => {
    setVectors([
      { x: 3, y: 1, z: 0 },
      { x: 2, y: 3, z: 0 },
      { x: 1, y: 2, z: 0 }
    ])
    setOrthogonalVectors([])
    setCurrentStep(0)
  }

  const presets = [
    {
      name: 'Default',
      vectors: [
        { x: 3, y: 1, z: 0 },
        { x: 2, y: 3, z: 0 },
        { x: 1, y: 2, z: 0 }
      ]
    },
    {
      name: 'Perpendicular',
      vectors: [
        { x: 3, y: 0, z: 0 },
        { x: 0, y: 3, z: 0 },
        { x: 0, y: 0, z: 3 }
      ]
    },
    {
      name: 'Nearly Parallel',
      vectors: [
        { x: 3, y: 1, z: 0 },
        { x: 3.2, y: 1.1, z: 0 },
        { x: 0, y: 0, z: 2 }
      ]
    }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">그람-슈미트 정규직교화</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Canvas */}
          <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">시각화</h2>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={showOriginal}
                  onChange={(e) => setShowOriginal(e.target.checked)}
                  className="rounded"
                />
                원본 벡터 표시
              </label>
            </div>

            <canvas
              ref={canvasRef}
              width={800}
              height={600}
              className="w-full border border-slate-600 rounded-lg"
            />

            <div className="mt-4 space-y-2 text-sm text-slate-300">
              <p>💡 회색: 원본 벡터 (v<sub>1</sub>, v<sub>2</sub>, v<sub>3</sub>)</p>
              <p>💡 파란색/녹색/노란색: 정규직교 벡터 (u<sub>1</sub>, u<sub>2</sub>, u<sub>3</sub>)</p>
              <p>💡 빨간 점선: 투영 제거 과정</p>
            </div>

            {/* Algorithm Steps */}
            <div className="mt-6 bg-slate-800/50 border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <ChevronRight className="w-5 h-5" />
                알고리즘 단계
              </h3>
              <div className="space-y-3">
                {vectors.map((_, i) => (
                  <div
                    key={i}
                    className={`p-4 rounded-lg transition-all ${
                      i < currentStep
                        ? 'bg-green-500/20 border border-green-500/50'
                        : i === currentStep
                        ? 'bg-blue-500/20 border border-blue-500/50'
                        : 'bg-slate-700/30 border border-slate-600'
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      {i < currentStep && <CheckCircle className="w-5 h-5 text-green-400" />}
                      <div className="font-semibold">
                        Step {i + 1}: u<sub>{i + 1}</sub> 계산
                      </div>
                    </div>
                    {i < currentStep && orthogonalVectors[i] && (
                      <div className="text-sm font-mono bg-slate-900/50 p-2 rounded">
                        u<sub>{i + 1}</sub> = ({orthogonalVectors[i].x.toFixed(3)}, {orthogonalVectors[i].y.toFixed(3)}, {orthogonalVectors[i].z.toFixed(3)})
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="space-y-6">
            {/* Run Controls */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">제어</h3>
              <div className="space-y-2">
                <button
                  onClick={runAlgorithm}
                  disabled={isAnimating}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg transition-colors"
                >
                  <Play className="w-5 h-5" />
                  <span>알고리즘 실행</span>
                </button>
                <button
                  onClick={reset}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
                >
                  <RotateCcw className="w-5 h-5" />
                  <span>초기화</span>
                </button>
              </div>
            </div>

            {/* Vector Inputs */}
            {vectors.map((v, idx) => (
              <div
                key={idx}
                className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6"
              >
                <h3 className="text-lg font-semibold mb-4">
                  벡터 v<sub>{idx + 1}</sub>
                </h3>
                <div className="space-y-3">
                  <div>
                    <label className="text-sm text-slate-300 mb-1 block">X: {v.x.toFixed(1)}</label>
                    <input
                      type="range"
                      min="-5"
                      max="5"
                      step="0.1"
                      value={v.x}
                      onChange={(e) => updateVector(idx, 'x', e.target.value)}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-slate-300 mb-1 block">Y: {v.y.toFixed(1)}</label>
                    <input
                      type="range"
                      min="-5"
                      max="5"
                      step="0.1"
                      value={v.y}
                      onChange={(e) => updateVector(idx, 'y', e.target.value)}
                      className="w-full"
                    />
                  </div>
                </div>
              </div>
            ))}

            {/* Presets */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">프리셋</h3>
              <div className="space-y-2">
                {presets.map((preset) => (
                  <button
                    key={preset.name}
                    onClick={() => {
                      setVectors(preset.vectors)
                      setOrthogonalVectors([])
                      setCurrentStep(0)
                    }}
                    className="w-full px-4 py-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors text-sm"
                  >
                    {preset.name}
                  </button>
                ))}
              </div>
            </div>

            {/* Info */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">알고리즘 설명</h3>
              <div className="space-y-2 text-sm text-slate-300">
                <div className="p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                  <div className="font-semibold text-blue-400 mb-1">Step 1</div>
                  <p>u₁ = v₁ / ||v₁||</p>
                </div>
                <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                  <div className="font-semibold text-green-400 mb-1">Step 2</div>
                  <p>v₂′ = v₂ - (v₂·u₁)u₁<br/>u₂ = v₂′ / ||v₂′||</p>
                </div>
                <div className="p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                  <div className="font-semibold text-yellow-400 mb-1">Step 3</div>
                  <p>v₃′ = v₃ - (v₃·u₁)u₁ - (v₃·u₂)u₂<br/>u₃ = v₃′ / ||v₃′||</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
