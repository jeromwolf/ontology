'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, RotateCcw } from 'lucide-react'

export default function IntegralCalculator() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [functionType, setFunctionType] = useState<'linear' | 'quadratic' | 'sin'>('quadratic')
  const [a, setA] = useState<number>(-2)
  const [b, setB] = useState<number>(2)
  const [n, setN] = useState<number>(10)
  const [method, setMethod] = useState<'left' | 'right' | 'midpoint' | 'trapezoid'>('midpoint')

  useEffect(() => {
    drawCanvas()
  }, [functionType, a, b, n, method])

  const f = (x: number): number => {
    switch (functionType) {
      case 'linear': return x + 2
      case 'quadratic': return x * x + 1
      case 'sin': return 2 * Math.sin(x) + 3
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
    const centerY = height - 50
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
      ctx.moveTo(0, centerY - i * scale)
      ctx.lineTo(width, centerY - i * scale)
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

    // Riemann sum rectangles
    const dx = (b - a) / n
    let sum = 0

    for (let i = 0; i < n; i++) {
      let xi: number
      switch (method) {
        case 'left': xi = a + i * dx; break
        case 'right': xi = a + (i + 1) * dx; break
        case 'midpoint': xi = a + (i + 0.5) * dx; break
        case 'trapezoid': {
          const left = f(a + i * dx)
          const right = f(a + (i + 1) * dx)
          sum += (left + right) / 2 * dx

          const x1 = centerX + (a + i * dx) * scale
          const x2 = centerX + (a + (i + 1) * dx) * scale
          const y1 = centerY - left * scale
          const y2 = centerY - right * scale

          ctx.fillStyle = 'rgba(16, 185, 129, 0.3)'
          ctx.beginPath()
          ctx.moveTo(x1, centerY)
          ctx.lineTo(x1, y1)
          ctx.lineTo(x2, y2)
          ctx.lineTo(x2, centerY)
          ctx.closePath()
          ctx.fill()

          ctx.strokeStyle = '#10b981'
          ctx.lineWidth = 1
          ctx.stroke()
          continue
        }
        default: xi = a + i * dx
      }

      if (method !== 'trapezoid') {
        const yi = f(xi)
        sum += yi * dx

        const x1 = centerX + (a + i * dx) * scale
        const x2 = centerX + (a + (i + 1) * dx) * scale
        const y = centerY - yi * scale

        ctx.fillStyle = 'rgba(16, 185, 129, 0.3)'
        ctx.fillRect(x1, y, x2 - x1, centerY - y)
        ctx.strokeStyle = '#10b981'
        ctx.lineWidth = 1
        ctx.strokeRect(x1, y, x2 - x1, centerY - y)
      }
    }

    // Draw bounds
    const ax = centerX + a * scale
    const bx = centerX + b * scale
    ctx.strokeStyle = '#ef4444'
    ctx.setLineDash([5, 5])
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(ax, 0)
    ctx.lineTo(ax, height)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(bx, 0)
    ctx.lineTo(bx, height)
    ctx.stroke()
    ctx.setLineDash([])

    // Labels
    ctx.fillStyle = '#ef4444'
    ctx.font = 'bold 14px Inter'
    ctx.fillText(`a = ${a.toFixed(1)}`, ax - 30, height - 10)
    ctx.fillText(`b = ${b.toFixed(1)}`, bx - 30, height - 10)

    ctx.fillStyle = '#10b981'
    ctx.font = 'bold 16px Inter'
    ctx.fillText(`≈ ${sum.toFixed(4)}`, 20, 40)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-green-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">적분 계산기</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">리만 합 시각화</h2>
            <canvas ref={canvasRef} width={800} height={600} className="w-full border border-slate-600 rounded-lg" />
          </div>

          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">함수</h3>
              <div className="space-y-2">
                {[
                  { id: 'linear', label: 'f(x) = x + 2' },
                  { id: 'quadratic', label: 'f(x) = x² + 1' },
                  { id: 'sin', label: 'f(x) = 2sin(x) + 3' }
                ].map((fn) => (
                  <button key={fn.id} onClick={() => setFunctionType(fn.id as any)} className={`w-full px-4 py-2 rounded-lg ${functionType === fn.id ? 'bg-green-600' : 'bg-slate-700/50 hover:bg-slate-700'}`}>
                    <div className="font-mono text-sm">{fn.label}</div>
                  </button>
                ))}
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">적분 구간</h3>
              <div className="space-y-3">
                <div>
                  <label className="text-sm text-slate-300 mb-1 block">a = {a.toFixed(1)}</label>
                  <input type="range" min="-5" max="0" step="0.1" value={a} onChange={(e) => setA(parseFloat(e.target.value))} className="w-full" />
                </div>
                <div>
                  <label className="text-sm text-slate-300 mb-1 block">b = {b.toFixed(1)}</label>
                  <input type="range" min="0" max="5" step="0.1" value={b} onChange={(e) => setB(parseFloat(e.target.value))} className="w-full" />
                </div>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">분할 개수: {n}</h3>
              <input type="range" min="4" max="50" value={n} onChange={(e) => setN(parseInt(e.target.value))} className="w-full" />
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">리만 합 방법</h3>
              <div className="space-y-2">
                {[
                  { id: 'left', label: '좌합' },
                  { id: 'right', label: '우합' },
                  { id: 'midpoint', label: '중점' },
                  { id: 'trapezoid', label: '사다리꼴' }
                ].map((m) => (
                  <button key={m.id} onClick={() => setMethod(m.id as any)} className={`w-full px-4 py-2 rounded-lg text-sm ${method === m.id ? 'bg-blue-600' : 'bg-slate-700/50 hover:bg-slate-700'}`}>
                    {m.label}
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
