'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, RotateCcw } from 'lucide-react'

export default function DerivativeVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [functionType, setFunctionType] = useState<'quadratic' | 'cubic' | 'sin' | 'exp'>('quadratic')
  const [xPoint, setXPoint] = useState<number>(1)
  const [showTangent, setShowTangent] = useState(true)
  const [showDerivative, setShowDerivative] = useState(false)

  useEffect(() => {
    drawCanvas()
  }, [functionType, xPoint, showTangent, showDerivative])

  const f = (x: number): number => {
    switch (functionType) {
      case 'quadratic': return x * x
      case 'cubic': return x * x * x - 3 * x
      case 'sin': return 2 * Math.sin(x)
      case 'exp': return Math.exp(x * 0.5)
      default: return 0
    }
  }

  const fPrime = (x: number): number => {
    switch (functionType) {
      case 'quadratic': return 2 * x
      case 'cubic': return 3 * x * x - 3
      case 'sin': return 2 * Math.cos(x)
      case 'exp': return 0.5 * Math.exp(x * 0.5)
      default: return 0
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

    // Draw f(x)
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 3
    ctx.beginPath()
    for (let px = 0; px < width; px++) {
      const x = (px - centerX) / scale
      const y = f(x)
      const py = centerY - y * scale
      if (px === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    ctx.stroke()

    // Draw f'(x) if enabled
    if (showDerivative) {
      ctx.strokeStyle = '#10b981'
      ctx.lineWidth = 2
      ctx.beginPath()
      for (let px = 0; px < width; px++) {
        const x = (px - centerX) / scale
        const y = fPrime(x)
        const py = centerY - y * scale
        if (px === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      }
      ctx.stroke()
    }

    // Point and tangent
    const fx = f(xPoint)
    const slope = fPrime(xPoint)
    const px = centerX + xPoint * scale
    const py = centerY - fx * scale

    // Tangent line
    if (showTangent) {
      ctx.strokeStyle = '#f59e0b'
      ctx.lineWidth = 2
      ctx.beginPath()
      const x1 = (0 - centerX) / scale
      const y1 = slope * (x1 - xPoint) + fx
      const x2 = (width - centerX) / scale
      const y2 = slope * (x2 - xPoint) + fx
      ctx.moveTo(0, centerY - y1 * scale)
      ctx.lineTo(width, centerY - y2 * scale)
      ctx.stroke()
    }

    // Point
    ctx.fillStyle = '#ef4444'
    ctx.beginPath()
    ctx.arc(px, py, 6, 0, Math.PI * 2)
    ctx.fill()

    // Labels
    ctx.fillStyle = '#ef4444'
    ctx.font = 'bold 14px Inter'
    ctx.fillText(`(${xPoint.toFixed(2)}, ${fx.toFixed(2)})`, px + 10, py - 10)

    ctx.fillStyle = '#f59e0b'
    ctx.fillText(`m = ${slope.toFixed(2)}`, px + 10, py + 25)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-green-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">미분 시각화 도구</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">시각화</h2>
            <canvas ref={canvasRef} width={800} height={600} className="w-full border border-slate-600 rounded-lg" />
            <div className="mt-4 space-y-2 text-sm text-slate-300">
              <p>💡 파란색: f(x)</p>
              {showDerivative && <p>💡 초록색: f'(x)</p>}
              {showTangent && <p>💡 노란색: 접선</p>}
              <p>💡 빨간점: 선택한 점</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">함수 선택</h3>
              <div className="space-y-2">
                {[
                  { id: 'quadratic', label: 'f(x) = x²', prime: "f'(x) = 2x" },
                  { id: 'cubic', label: 'f(x) = x³ - 3x', prime: "f'(x) = 3x² - 3" },
                  { id: 'sin', label: 'f(x) = 2sin(x)', prime: "f'(x) = 2cos(x)" },
                  { id: 'exp', label: 'f(x) = e^(x/2)', prime: "f'(x) = 0.5e^(x/2)" }
                ].map((fn) => (
                  <button
                    key={fn.id}
                    onClick={() => setFunctionType(fn.id as any)}
                    className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                      functionType === fn.id ? 'bg-green-600 text-white' : 'bg-slate-700/50 hover:bg-slate-700'
                    }`}
                  >
                    <div className="font-mono text-sm">{fn.label}</div>
                    <div className="text-xs text-slate-400">{fn.prime}</div>
                  </button>
                ))}
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">x 좌표</h3>
              <label className="text-sm text-slate-300 mb-2 block">x = {xPoint.toFixed(2)}</label>
              <input type="range" min="-5" max="5" step="0.1" value={xPoint} onChange={(e) => setXPoint(parseFloat(e.target.value))} className="w-full" />
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">옵션</h3>
              <div className="space-y-3">
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <input type="checkbox" checked={showTangent} onChange={(e) => setShowTangent(e.target.checked)} className="rounded" />
                  접선 표시
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <input type="checkbox" checked={showDerivative} onChange={(e) => setShowDerivative(e.target.checked)} className="rounded" />
                  도함수 f'(x) 표시
                </label>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <button onClick={() => { setXPoint(1); setShowTangent(true); setShowDerivative(false) }} className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors">
                <RotateCcw className="w-5 h-5" />
                초기화
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
