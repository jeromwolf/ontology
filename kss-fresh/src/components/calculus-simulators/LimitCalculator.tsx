'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, RotateCcw, Calculator } from 'lucide-react'

export default function LimitCalculator() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [functionType, setFunctionType] = useState<'polynomial' | 'rational' | 'trig'>('polynomial')
  const [limitPoint, setLimitPoint] = useState<number>(2)
  const [epsilon, setEpsilon] = useState<number>(0.5)
  const [showEpsilonDelta, setShowEpsilonDelta] = useState(false)

  useEffect(() => {
    drawCanvas()
  }, [functionType, limitPoint, epsilon, showEpsilonDelta])

  const evaluateFunction = (x: number): number => {
    switch (functionType) {
      case 'polynomial':
        return (x * x - 4) / (x - 2) // Removable discontinuity at x=2
      case 'rational':
        return 1 / (x - 2) // Vertical asymptote at x=2
      case 'trig':
        return Math.sin(x) / x // sinc function
      default:
        return 0
    }
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
    const scale = 50

    // Clear
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, width, height)

    // Grid
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

    // Axes
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
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 2
    ctx.beginPath()
    let started = false

    for (let px = 0; px < width; px++) {
      const x = (px - centerX) / scale

      // Skip near discontinuity for rational function
      if (functionType === 'rational' && Math.abs(x - limitPoint) < 0.1) continue

      const y = evaluateFunction(x)

      if (!isNaN(y) && isFinite(y) && Math.abs(y) < 10) {
        const py = centerY - y * scale
        if (py >= 0 && py <= height) {
          if (!started) {
            ctx.moveTo(px, py)
            started = true
          } else {
            ctx.lineTo(px, py)
          }
        } else {
          started = false
        }
      } else {
        started = false
      }
    }
    ctx.stroke()

    // Calculate limit value
    let limitValue = 0
    if (functionType === 'polynomial') {
      limitValue = limitPoint + 2 // L'Hospital's rule result
    } else if (functionType === 'trig' && limitPoint === 0) {
      limitValue = 1 // lim x->0 sin(x)/x = 1
    }

    // Draw limit point
    if (functionType === 'polynomial' || (functionType === 'trig' && limitPoint === 0)) {
      const lx = centerX + limitPoint * scale
      const ly = centerY - limitValue * scale

      // Hollow circle at discontinuity
      ctx.strokeStyle = '#ef4444'
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.arc(lx, ly, 6, 0, Math.PI * 2)
      ctx.stroke()

      // Limit value indicator
      ctx.fillStyle = '#10b981'
      ctx.beginPath()
      ctx.arc(lx, ly, 4, 0, Math.PI * 2)
      ctx.fill()

      // Label
      ctx.fillStyle = '#10b981'
      ctx.font = 'bold 14px Inter'
      ctx.fillText(`L = ${limitValue.toFixed(2)}`, lx + 15, ly - 10)
    }

    // Epsilon-delta visualization
    if (showEpsilonDelta && (functionType === 'polynomial' || (functionType === 'trig' && limitPoint === 0))) {
      const lx = centerX + limitPoint * scale
      const ly = centerY - limitValue * scale

      // Epsilon band (horizontal)
      ctx.fillStyle = 'rgba(16, 185, 129, 0.1)'
      ctx.fillRect(0, ly - epsilon * scale, width, 2 * epsilon * scale)

      ctx.strokeStyle = '#10b981'
      ctx.setLineDash([5, 5])
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(0, ly - epsilon * scale)
      ctx.lineTo(width, ly - epsilon * scale)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(0, ly + epsilon * scale)
      ctx.lineTo(width, ly + epsilon * scale)
      ctx.stroke()
      ctx.setLineDash([])

      // Delta band (vertical)
      const delta = epsilon * 0.8
      ctx.fillStyle = 'rgba(59, 130, 246, 0.1)'
      ctx.fillRect(lx - delta * scale, 0, 2 * delta * scale, height)

      ctx.strokeStyle = '#3b82f6'
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      ctx.moveTo(lx - delta * scale, 0)
      ctx.lineTo(lx - delta * scale, height)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(lx + delta * scale, 0)
      ctx.lineTo(lx + delta * scale, height)
      ctx.stroke()
      ctx.setLineDash([])

      // Labels
      ctx.fillStyle = '#10b981'
      ctx.font = '12px Inter'
      ctx.fillText(`ε = ${epsilon.toFixed(2)}`, 10, ly - epsilon * scale - 5)
      ctx.fillStyle = '#3b82f6'
      ctx.fillText(`δ = ${delta.toFixed(2)}`, lx - delta * scale + 5, 20)
    }

    // Vertical asymptote for rational function
    if (functionType === 'rational') {
      const ax = centerX + limitPoint * scale
      ctx.strokeStyle = '#ef4444'
      ctx.setLineDash([10, 5])
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(ax, 0)
      ctx.lineTo(ax, height)
      ctx.stroke()
      ctx.setLineDash([])

      ctx.fillStyle = '#ef4444'
      ctx.font = 'bold 14px Inter'
      ctx.fillText('VA', ax + 10, 30)
    }
  }

  const reset = () => {
    setLimitPoint(2)
    setEpsilon(0.5)
    setShowEpsilonDelta(false)
  }

  const functions = [
    { id: 'polynomial', label: '(x²-4)/(x-2)', desc: '제거 가능한 불연속' },
    { id: 'rational', label: '1/(x-2)', desc: '수직 점근선' },
    { id: 'trig', label: 'sin(x)/x', desc: '삼각함수 극한' }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-green-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">극한 계산기</h1>

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
              <p>💡 파란색: 함수 그래프</p>
              <p>💡 초록색: 극한값 (L)</p>
              <p>💡 빨간색: 불연속점 또는 점근선</p>
            </div>
          </div>

          {/* Controls */}
          <div className="space-y-6">
            {/* Function Selection */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Calculator className="w-5 h-5" />
                함수 선택
              </h3>
              <div className="space-y-2">
                {functions.map((fn) => (
                  <button
                    key={fn.id}
                    onClick={() => setFunctionType(fn.id as any)}
                    className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                      functionType === fn.id
                        ? 'bg-green-600 text-white'
                        : 'bg-slate-700/50 hover:bg-slate-700 text-slate-300'
                    }`}
                  >
                    <div className="font-mono text-sm font-semibold">{fn.label}</div>
                    <div className="text-xs text-slate-400">{fn.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Limit Point */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">극한점 (a)</h3>
              <div>
                <label className="text-sm text-slate-300 mb-2 block">
                  x → {limitPoint.toFixed(1)}
                </label>
                <input
                  type="range"
                  min="-5"
                  max="5"
                  step="0.1"
                  value={limitPoint}
                  onChange={(e) => setLimitPoint(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>

            {/* Epsilon-Delta */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">ε-δ 정의</h3>
              <label className="flex items-center gap-2 mb-4 text-sm cursor-pointer">
                <input
                  type="checkbox"
                  checked={showEpsilonDelta}
                  onChange={(e) => setShowEpsilonDelta(e.target.checked)}
                  className="rounded"
                />
                ε-δ 시각화 표시
              </label>
              {showEpsilonDelta && (
                <div>
                  <label className="text-sm text-slate-300 mb-2 block">
                    ε (엡실론): {epsilon.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="2"
                    step="0.1"
                    value={epsilon}
                    onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <p className="text-xs text-slate-400 mt-2">
                    δ = {(epsilon * 0.8).toFixed(2)} (자동 계산)
                  </p>
                </div>
              )}
            </div>

            {/* Controls */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <button
                onClick={reset}
                className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
              >
                <RotateCcw className="w-5 h-5" />
                <span>초기화</span>
              </button>
            </div>

            {/* Info */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">극한의 정의</h3>
              <div className="text-sm text-slate-300 space-y-2">
                <p className="font-mono text-center bg-slate-900/50 p-3 rounded">
                  lim<sub>x→a</sub> f(x) = L
                </p>
                <p className="text-xs">
                  모든 ε {">"} 0에 대해,<br />
                  δ {">"} 0가 존재하여<br />
                  0 {"<"} |x - a| {"<"} δ이면<br />
                  |f(x) - L| {"<"} ε
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
