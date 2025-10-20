'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, RotateCcw, Search } from 'lucide-react'

type Matrix2x2 = [[number, number], [number, number]]

interface Eigenpair {
  value: number
  vector: [number, number]
}

export default function EigenvalueExplorer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [matrix, setMatrix] = useState<Matrix2x2>([[2, 1], [1, 2]])
  const [eigenpairs, setEigenpairs] = useState<Eigenpair[]>([])
  const [animationProgress, setAnimationProgress] = useState(0)
  const [isAnimating, setIsAnimating] = useState(false)

  useEffect(() => {
    calculateEigenpairs()
  }, [matrix])

  useEffect(() => {
    drawCanvas()
  }, [eigenpairs, animationProgress, matrix])

  const calculateEigenpairs = () => {
    const [[a, b], [c, d]] = matrix
    const trace = a + d
    const det = a * d - b * c
    const discriminant = trace * trace - 4 * det

    if (discriminant < 0) {
      setEigenpairs([])
      return
    }

    const sqrt = Math.sqrt(discriminant)
    const lambda1 = (trace + sqrt) / 2
    const lambda2 = (trace - sqrt) / 2

    // Calculate eigenvectors
    const getEigenvector = (lambda: number): [number, number] => {
      if (Math.abs(b) > 1e-10) {
        const y = 1
        const x = (lambda - d) / b
        const mag = Math.sqrt(x * x + y * y)
        return [x / mag, y / mag]
      } else if (Math.abs(c) > 1e-10) {
        const x = 1
        const y = (lambda - a) / c
        const mag = Math.sqrt(x * x + y * y)
        return [x / mag, y / mag]
      } else {
        return lambda === a ? [1, 0] : [0, 1]
      }
    }

    setEigenpairs([
      { value: lambda1, vector: getEigenvector(lambda1) },
      { value: lambda2, vector: getEigenvector(lambda2) }
    ])
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

    // Draw unit circle
    ctx.strokeStyle = '#334155'
    ctx.lineWidth = 1
    ctx.setLineDash([5, 5])
    ctx.beginPath()
    ctx.arc(centerX, centerY, scale, 0, 2 * Math.PI)
    ctx.stroke()
    ctx.setLineDash([])

    // Draw eigenvectors and transformed vectors
    if (eigenpairs.length === 2) {
      const colors = ['#3b82f6', '#10b981']

      eigenpairs.forEach((pair, idx) => {
        const [vx, vy] = pair.vector
        const lambda = pair.value

        // Original eigenvector
        const endX1 = centerX + vx * scale * 2
        const endY1 = centerY - vy * scale * 2

        // Draw original vector
        ctx.strokeStyle = colors[idx]
        ctx.globalAlpha = 0.5
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(centerX, centerY)
        ctx.lineTo(endX1, endY1)
        ctx.stroke()

        // Draw arrow
        const angle1 = Math.atan2(-vy, vx)
        ctx.fillStyle = colors[idx]
        ctx.beginPath()
        ctx.moveTo(endX1, endY1)
        ctx.lineTo(
          endX1 - 10 * Math.cos(angle1 - Math.PI / 6),
          endY1 - 10 * Math.sin(angle1 - Math.PI / 6)
        )
        ctx.lineTo(
          endX1 - 10 * Math.cos(angle1 + Math.PI / 6),
          endY1 - 10 * Math.sin(angle1 + Math.PI / 6)
        )
        ctx.closePath()
        ctx.fill()

        // Transformed vector (Av = λv)
        const scaledVx = vx * lambda * (0.5 + animationProgress * 0.5)
        const scaledVy = vy * lambda * (0.5 + animationProgress * 0.5)
        const endX2 = centerX + scaledVx * scale
        const endY2 = centerY - scaledVy * scale

        ctx.globalAlpha = 1
        ctx.lineWidth = 3
        ctx.beginPath()
        ctx.moveTo(centerX, centerY)
        ctx.lineTo(endX2, endY2)
        ctx.stroke()

        // Draw arrow for transformed vector
        const angle2 = Math.atan2(-scaledVy, scaledVx)
        ctx.fillStyle = colors[idx]
        ctx.beginPath()
        ctx.moveTo(endX2, endY2)
        ctx.lineTo(
          endX2 - 12 * Math.cos(angle2 - Math.PI / 6),
          endY2 - 12 * Math.sin(angle2 - Math.PI / 6)
        )
        ctx.lineTo(
          endX2 - 12 * Math.cos(angle2 + Math.PI / 6),
          endY2 - 12 * Math.sin(angle2 + Math.PI / 6)
        )
        ctx.closePath()
        ctx.fill()

        // Label
        ctx.font = 'bold 14px Inter'
        ctx.fillText(
          `v${idx + 1}`,
          endX1 + 15,
          endY1
        )
        ctx.fillText(
          `λ${idx + 1}v${idx + 1}`,
          endX2 + 15,
          endY2
        )
      })

      ctx.globalAlpha = 1
    }

    // Draw sample vectors and their transformations
    if (!isAnimating) {
      const sampleVectors = [
        [1, 0], [0, 1], [1, 1], [-1, 1]
      ]

      sampleVectors.forEach((v) => {
        const [vx, vy] = v
        const [[a, b], [c, d]] = matrix
        const transformedX = a * vx + b * vy
        const transformedY = c * vx + d * vy

        // Original vector
        ctx.strokeStyle = '#64748b'
        ctx.globalAlpha = 0.3
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(centerX, centerY)
        ctx.lineTo(centerX + vx * scale, centerY - vy * scale)
        ctx.stroke()

        // Transformed vector
        ctx.strokeStyle = '#f59e0b'
        ctx.globalAlpha = 0.5
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(centerX, centerY)
        ctx.lineTo(centerX + transformedX * scale, centerY - transformedY * scale)
        ctx.stroke()
      })

      ctx.globalAlpha = 1
    }
  }

  const updateMatrix = (row: number, col: number, value: string) => {
    const newMatrix: Matrix2x2 = matrix.map((r, i) =>
      i === row ? r.map((c, j) => (j === col ? parseFloat(value) || 0 : c)) as [number, number] : r
    ) as Matrix2x2
    setMatrix(newMatrix)
  }

  const animate = () => {
    setIsAnimating(true)
    let progress = 0
    const interval = setInterval(() => {
      progress += 0.02
      if (progress >= 1) {
        progress = 1
        clearInterval(interval)
        setIsAnimating(false)
      }
      setAnimationProgress(progress)
    }, 50)
  }

  const reset = () => {
    setMatrix([[2, 1], [1, 2]])
    setAnimationProgress(0)
  }

  const presets = [
    { name: 'Identity', matrix: [[1, 0], [0, 1]] as Matrix2x2 },
    { name: 'Scaling', matrix: [[2, 0], [0, 3]] as Matrix2x2 },
    { name: 'Rotation', matrix: [[0, -1], [1, 0]] as Matrix2x2 },
    { name: 'Shear', matrix: [[1, 1], [0, 1]] as Matrix2x2 },
    { name: 'Reflection', matrix: [[1, 0], [0, -1]] as Matrix2x2 }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">고유값 탐색기</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Canvas */}
          <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">시각화</h2>
            <canvas
              ref={canvasRef}
              width={800}
              height={600}
              className="w-full border border-slate-600 rounded-lg"
            />
            <div className="mt-4 space-y-2 text-sm text-slate-300">
              <p>💡 밝은 선: 고유벡터가 λ배로 늘어나는 모습</p>
              <p>💡 흐린 선: 고유벡터의 방향 (단위 벡터)</p>
              <p>💡 노란 선: 일반 벡터들의 변환</p>
            </div>
          </div>

          {/* Controls */}
          <div className="space-y-6">
            {/* Matrix Input */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">행렬 A (2×2)</h3>
              <div className="flex items-center gap-2 mb-6">
                <div className="text-3xl text-slate-600">[</div>
                <div className="space-y-2">
                  {matrix.map((row, i) => (
                    <div key={i} className="flex gap-2">
                      {row.map((val, j) => (
                        <input
                          key={`${i}-${j}`}
                          type="number"
                          value={val}
                          onChange={(e) => updateMatrix(i, j, e.target.value)}
                          className="w-20 px-2 py-2 text-center rounded bg-slate-700 border border-slate-600 text-white font-mono"
                          step="0.1"
                        />
                      ))}
                    </div>
                  ))}
                </div>
                <div className="text-3xl text-slate-600">]</div>
              </div>

              <div className="space-y-2">
                <button
                  onClick={animate}
                  disabled={isAnimating || eigenpairs.length === 0}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg transition-colors"
                >
                  <Play className="w-5 h-5" />
                  <span>애니메이션</span>
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

            {/* Eigenvalues Display */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Search className="w-5 h-5" />
                고유값과 고유벡터
              </h3>
              {eigenpairs.length > 0 ? (
                <div className="space-y-4">
                  {eigenpairs.map((pair, idx) => (
                    <div key={idx} className={`p-4 rounded-lg ${idx === 0 ? 'bg-blue-500/20 border border-blue-500/30' : 'bg-green-500/20 border border-green-500/30'}`}>
                      <div className="font-mono text-lg mb-2">
                        λ<sub>{idx + 1}</sub> = {pair.value.toFixed(4)}
                      </div>
                      <div className="font-mono text-sm text-slate-300">
                        v<sub>{idx + 1}</sub> = [{pair.vector[0].toFixed(3)}, {pair.vector[1].toFixed(3)}]
                      </div>
                    </div>
                  ))}
                  <div className="mt-4 p-4 bg-slate-700/50 rounded-lg">
                    <div className="text-sm text-slate-300">
                      <div>Trace: {(matrix[0][0] + matrix[1][1]).toFixed(2)}</div>
                      <div>Determinant: {(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]).toFixed(2)}</div>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-slate-400 text-sm">복소수 고유값 (실수 고유값 없음)</p>
              )}
            </div>

            {/* Presets */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">프리셋</h3>
              <div className="grid grid-cols-2 gap-2">
                {presets.map((preset) => (
                  <button
                    key={preset.name}
                    onClick={() => {
                      setMatrix(preset.matrix)
                      setAnimationProgress(0)
                    }}
                    className="px-3 py-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors text-sm"
                  >
                    {preset.name}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
