'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, RotateCcw, Grid3x3, RefreshCw } from 'lucide-react'

type Matrix2x2 = [[number, number], [number, number]]

export default function LinearTransformationLab() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [matrix, setMatrix] = useState<Matrix2x2>([[1, 0], [0, 1]])
  const [isAnimating, setIsAnimating] = useState(false)
  const [animationProgress, setAnimationProgress] = useState(0)
  const [showGrid, setShowGrid] = useState(true)

  useEffect(() => {
    drawCanvas()
  }, [matrix, animationProgress, showGrid])

  const drawCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height
    const centerX = width / 2
    const centerY = height / 2
    const scale = 50

    // Clear canvas
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, width, height)

    // Interpolate matrix for animation
    const t = animationProgress
    const [[a, b], [c, d]] = matrix
    const currentMatrix: Matrix2x2 = [
      [1 + (a - 1) * t, b * t],
      [c * t, 1 + (d - 1) * t]
    ]

    // Transform grid points
    const transformPoint = (x: number, y: number) => {
      const [[a, b], [c, d]] = currentMatrix
      return {
        x: a * x + b * y,
        y: c * x + d * y
      }
    }

    // Draw transformed grid
    if (showGrid) {
      ctx.strokeStyle = '#1e3a8a'
      ctx.lineWidth = 1

      for (let i = -10; i <= 10; i++) {
        // Vertical lines
        ctx.beginPath()
        for (let j = -10; j <= 10; j++) {
          const p = transformPoint(i, j)
          const screenX = centerX + p.x * scale
          const screenY = centerY - p.y * scale
          if (j === -10) {
            ctx.moveTo(screenX, screenY)
          } else {
            ctx.lineTo(screenX, screenY)
          }
        }
        ctx.stroke()

        // Horizontal lines
        ctx.beginPath()
        for (let j = -10; j <= 10; j++) {
          const p = transformPoint(j, i)
          const screenX = centerX + p.x * scale
          const screenY = centerY - p.y * scale
          if (j === -10) {
            ctx.moveTo(screenX, screenY)
          } else {
            ctx.lineTo(screenX, screenY)
          }
        }
        ctx.stroke()
      }
    }

    // Draw transformed axes
    ctx.strokeStyle = '#475569'
    ctx.lineWidth = 2

    // X-axis (original [1,0])
    const xAxis = transformPoint(10, 0)
    const xAxisNeg = transformPoint(-10, 0)
    ctx.strokeStyle = '#ef4444'
    ctx.beginPath()
    ctx.moveTo(centerX + xAxisNeg.x * scale, centerY - xAxisNeg.y * scale)
    ctx.lineTo(centerX + xAxis.x * scale, centerY - xAxis.y * scale)
    ctx.stroke()

    // Y-axis (original [0,1])
    const yAxis = transformPoint(0, 10)
    const yAxisNeg = transformPoint(0, -10)
    ctx.strokeStyle = '#3b82f6'
    ctx.beginPath()
    ctx.moveTo(centerX + yAxisNeg.x * scale, centerY - yAxisNeg.y * scale)
    ctx.lineTo(centerX + yAxis.x * scale, centerY - yAxis.y * scale)
    ctx.stroke()

    // Draw unit square and its transformation
    const square = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 }
    ]

    // Original square (faint)
    ctx.strokeStyle = '#64748b'
    ctx.fillStyle = 'rgba(100, 116, 139, 0.1)'
    ctx.lineWidth = 2
    ctx.beginPath()
    square.forEach((p, i) => {
      const screenX = centerX + p.x * scale
      const screenY = centerY - p.y * scale
      if (i === 0) ctx.moveTo(screenX, screenY)
      else ctx.lineTo(screenX, screenY)
    })
    ctx.closePath()
    ctx.stroke()
    ctx.fill()

    // Transformed square
    ctx.strokeStyle = '#10b981'
    ctx.fillStyle = 'rgba(16, 185, 129, 0.2)'
    ctx.lineWidth = 3
    ctx.beginPath()
    square.forEach((p, i) => {
      const transformed = transformPoint(p.x, p.y)
      const screenX = centerX + transformed.x * scale
      const screenY = centerY - transformed.y * scale
      if (i === 0) ctx.moveTo(screenX, screenY)
      else ctx.lineTo(screenX, screenY)
    })
    ctx.closePath()
    ctx.stroke()
    ctx.fill()

    // Draw basis vectors
    const e1 = transformPoint(1, 0)
    const e2 = transformPoint(0, 1)

    // e1 (red)
    const drawVector = (v: { x: number; y: number }, color: string, label: string) => {
      const endX = centerX + v.x * scale
      const endY = centerY - v.y * scale

      ctx.strokeStyle = color
      ctx.lineWidth = 3
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
      ctx.font = 'bold 16px Inter'
      ctx.fillText(label, endX + 15, endY - 5)
    }

    drawVector(e1, '#ef4444', 'e₁')
    drawVector(e2, '#3b82f6', 'e₂')

    // Draw center point
    ctx.fillStyle = '#f59e0b'
    ctx.beginPath()
    ctx.arc(centerX, centerY, 5, 0, Math.PI * 2)
    ctx.fill()

    // Draw labels
    ctx.fillStyle = '#94a3b8'
    ctx.font = '14px Inter'
    ctx.fillText('회색: 원본 단위 정사각형', 20, height - 40)
    ctx.fillText('녹색: 변환된 정사각형', 20, height - 20)
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
    }, 30)
  }

  const reset = () => {
    setMatrix([[1, 0], [0, 1]])
    setAnimationProgress(1)
  }

  const presets = [
    { name: 'Identity', matrix: [[1, 0], [0, 1]] as Matrix2x2, desc: '항등 변환' },
    { name: 'Scale', matrix: [[2, 0], [0, 2]] as Matrix2x2, desc: '2배 확대' },
    { name: 'Stretch', matrix: [[2, 0], [0, 0.5]] as Matrix2x2, desc: 'X축 확대, Y축 축소' },
    { name: 'Rotation 90°', matrix: [[0, -1], [1, 0]] as Matrix2x2, desc: '반시계방향 90도' },
    { name: 'Rotation 45°', matrix: [[0.707, -0.707], [0.707, 0.707]] as Matrix2x2, desc: '반시계방향 45도' },
    { name: 'Shear X', matrix: [[1, 1], [0, 1]] as Matrix2x2, desc: 'X축 방향 전단' },
    { name: 'Shear Y', matrix: [[1, 0], [1, 1]] as Matrix2x2, desc: 'Y축 방향 전단' },
    { name: 'Reflection X', matrix: [[1, 0], [0, -1]] as Matrix2x2, desc: 'X축 대칭' },
    { name: 'Reflection Y', matrix: [[-1, 0], [0, 1]] as Matrix2x2, desc: 'Y축 대칭' },
    { name: 'Projection X', matrix: [[1, 0], [0, 0]] as Matrix2x2, desc: 'X축으로 투영' }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">선형변환 실험실</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Canvas */}
          <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">시각화</h2>
              <button
                onClick={() => setShowGrid(!showGrid)}
                className={`p-2 rounded-lg transition-colors ${
                  showGrid ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-300'
                }`}
                title="Toggle Grid"
              >
                <Grid3x3 className="w-5 h-5" />
              </button>
            </div>

            <canvas
              ref={canvasRef}
              width={800}
              height={600}
              className="w-full border border-slate-600 rounded-lg"
            />

            <div className="mt-4 space-y-2 text-sm text-slate-300">
              <p>💡 빨간색: 변환된 X축 (e₁)</p>
              <p>💡 파란색: 변환된 Y축 (e₂)</p>
              <p>💡 회색 → 녹색: 단위 정사각형의 변환</p>
            </div>

            {/* Transformation Info */}
            <div className="mt-6 grid grid-cols-3 gap-4">
              <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                <div className="text-sm text-slate-400 mb-1">Determinant</div>
                <div className="text-2xl font-bold">
                  {(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]).toFixed(3)}
                </div>
              </div>
              <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                <div className="text-sm text-slate-400 mb-1">Trace</div>
                <div className="text-2xl font-bold">
                  {(matrix[0][0] + matrix[1][1]).toFixed(3)}
                </div>
              </div>
              <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                <div className="text-sm text-slate-400 mb-1">변환 타입</div>
                <div className="text-lg font-bold">
                  {Math.abs(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) < 0.01
                    ? '특이'
                    : '정규'}
                </div>
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="space-y-6">
            {/* Matrix Input */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">변환 행렬</h3>
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
                  disabled={isAnimating}
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

            {/* Presets */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <RefreshCw className="w-5 h-5" />
                변환 프리셋
              </h3>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {presets.map((preset) => (
                  <button
                    key={preset.name}
                    onClick={() => {
                      setMatrix(preset.matrix)
                      setAnimationProgress(1)
                    }}
                    className="w-full text-left px-4 py-3 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors"
                  >
                    <div className="font-semibold text-sm">{preset.name}</div>
                    <div className="text-xs text-slate-400">{preset.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Info */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">선형 변환</h3>
              <div className="space-y-2 text-sm text-slate-300">
                <div className="p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                  <p>선형 변환 T(x) = Ax는 벡터 x를 Ax로 변환합니다.</p>
                </div>
                <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                  <p>기저 벡터 e₁, e₂의 변환이 행렬의 열을 결정합니다.</p>
                </div>
                <div className="p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                  <p>행렬식(det)이 0이면 차원이 감소합니다.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
