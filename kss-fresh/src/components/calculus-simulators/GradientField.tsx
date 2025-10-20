'use client'

import React, { useState, useRef, useEffect } from 'react'
import { RotateCcw } from 'lucide-react'

export default function GradientField() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [functionType, setFunctionType] = useState<'paraboloid' | 'saddle' | 'cone' | 'sine'>('paraboloid')
  const [showContour, setShowContour] = useState(true)
  const [showGradient, setShowGradient] = useState(true)

  useEffect(() => {
    drawCanvas()
  }, [functionType, showContour, showGradient])

  const f = (x: number, y: number): number => {
    switch (functionType) {
      case 'paraboloid': return x * x + y * y
      case 'saddle': return x * x - y * y
      case 'cone': return Math.sqrt(x * x + y * y)
      case 'sine': return Math.sin(x) + Math.cos(y)
      default: return 0
    }
  }

  const gradient = (x: number, y: number): [number, number] => {
    const h = 0.01
    const fx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    const fy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return [fx, fy]
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
    const scale = 40

    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, width, height)

    // Draw contour lines
    if (showContour) {
      const levels = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]

      for (const level of levels) {
        ctx.strokeStyle = level === 0 ? '#ef4444' : '#1e293b'
        ctx.lineWidth = level === 0 ? 2 : 1

        // Simple contour drawing
        for (let y = -10; y < 10; y += 0.1) {
          for (let x = -10; x < 10; x += 0.1) {
            const z1 = f(x, y)
            const z2 = f(x + 0.1, y)
            const z3 = f(x, y + 0.1)

            if ((z1 - level) * (z2 - level) < 0 || (z1 - level) * (z3 - level) < 0) {
              const px = centerX + x * scale
              const py = centerY - y * scale

              if (px >= 0 && px < width && py >= 0 && py < height) {
                ctx.fillStyle = level === 0 ? '#ef4444' : '#334155'
                ctx.fillRect(px, py, 2, 2)
              }
            }
          }
        }
      }
    }

    // Draw gradient vectors
    if (showGradient) {
      const step = 1
      for (let y = -8; y <= 8; y += step) {
        for (let x = -8; x <= 8; x += step) {
          const [gx, gy] = gradient(x, y)
          const mag = Math.sqrt(gx * gx + gy * gy)

          if (mag > 0.01) {
            const normX = gx / mag
            const normY = gy / mag
            const len = Math.min(mag * 0.3, 0.8) * scale

            const startX = centerX + x * scale
            const startY = centerY - y * scale
            const endX = startX + normX * len
            const endY = startY - normY * len

            // Color based on magnitude
            const hue = Math.min(mag * 30, 120)
            ctx.strokeStyle = `hsl(${120 - hue}, 70%, 50%)`
            ctx.lineWidth = 2
            ctx.beginPath()
            ctx.moveTo(startX, startY)
            ctx.lineTo(endX, endY)
            ctx.stroke()

            // Arrow head
            const angle = Math.atan2(-normY, normX)
            ctx.fillStyle = ctx.strokeStyle
            ctx.beginPath()
            ctx.moveTo(endX, endY)
            ctx.lineTo(endX - 8 * Math.cos(angle - Math.PI / 6), endY - 8 * Math.sin(angle - Math.PI / 6))
            ctx.lineTo(endX - 8 * Math.cos(angle + Math.PI / 6), endY - 8 * Math.sin(angle + Math.PI / 6))
            ctx.closePath()
            ctx.fill()
          }
        }
      }
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

    // Labels
    ctx.fillStyle = '#94a3b8'
    ctx.font = '14px Inter'
    ctx.fillText('x', width - 30, centerY - 10)
    ctx.fillText('y', centerX + 10, 30)
  }

  const functions = [
    { id: 'paraboloid', label: 'f(x,y) = x² + y²', desc: '포물면 (최솟값 존재)' },
    { id: 'saddle', label: 'f(x,y) = x² - y²', desc: '안장점 (saddle point)' },
    { id: 'cone', label: 'f(x,y) = √(x² + y²)', desc: '원뿔 (미분 불가능점)' },
    { id: 'sine', label: 'f(x,y) = sin(x) + cos(y)', desc: '삼각함수 (주기적)' }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-green-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">그래디언트 필드 시각화</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">등고선과 그래디언트 벡터</h2>
            <canvas ref={canvasRef} width={800} height={800} className="w-full border border-slate-600 rounded-lg" />
            <div className="mt-4 space-y-2 text-sm text-slate-300">
              {showContour && <p>💡 회색 선: 등고선 (f = 상수)</p>}
              {showContour && <p>💡 빨간 선: f = 0 등고선</p>}
              {showGradient && <p>💡 화살표: 그래디언트 벡터 ∇f</p>}
              {showGradient && <p>💡 색상: 초록(작음) → 노랑(중간) → 빨강(큼)</p>}
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">함수 선택</h3>
              <div className="space-y-2">
                {functions.map((fn) => (
                  <button
                    key={fn.id}
                    onClick={() => setFunctionType(fn.id as any)}
                    className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                      functionType === fn.id ? 'bg-green-600 text-white' : 'bg-slate-700/50 hover:bg-slate-700'
                    }`}
                  >
                    <div className="font-mono text-sm font-semibold">{fn.label}</div>
                    <div className="text-xs text-slate-400">{fn.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">표시 옵션</h3>
              <div className="space-y-3">
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showContour}
                    onChange={(e) => setShowContour(e.target.checked)}
                    className="rounded"
                  />
                  등고선 표시
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showGradient}
                    onChange={(e) => setShowGradient(e.target.checked)}
                    className="rounded"
                  />
                  그래디언트 벡터 표시
                </label>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <button
                onClick={() => {
                  setShowContour(true)
                  setShowGradient(true)
                }}
                className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
              >
                <RotateCcw className="w-5 h-5" />
                <span>초기화</span>
              </button>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">그래디언트란?</h3>
              <div className="text-sm text-slate-300 space-y-2">
                <p className="font-mono text-xs bg-slate-900/50 p-3 rounded">
                  ∇f = (∂f/∂x, ∂f/∂y)
                </p>
                <p className="text-xs">
                  • ∇f는 f가 가장 빠르게 증가하는 방향<br />
                  • |∇f|는 그 방향으로의 변화율<br />
                  • ∇f ⊥ 등고선 (수직)<br />
                  • 경사하강법은 -∇f 방향으로 이동
                </p>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">응용</h3>
              <div className="text-xs text-slate-300 space-y-2">
                <div className="p-2 bg-blue-500/10 border border-blue-500/30 rounded">
                  <span className="font-semibold text-blue-400">최적화:</span> 경사 하강/상승법
                </div>
                <div className="p-2 bg-green-500/10 border border-green-500/30 rounded">
                  <span className="font-semibold text-green-400">머신러닝:</span> 손실 함수 최소화
                </div>
                <div className="p-2 bg-purple-500/10 border border-purple-500/30 rounded">
                  <span className="font-semibold text-purple-400">물리:</span> 전기장, 중력장
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
