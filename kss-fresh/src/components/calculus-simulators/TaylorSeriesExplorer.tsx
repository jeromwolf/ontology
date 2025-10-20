'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play } from 'lucide-react'

export default function TaylorSeriesExplorer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [functionType, setFunctionType] = useState<'exp' | 'sin' | 'cos' | 'ln'>('exp')
  const [terms, setTerms] = useState<number>(5)
  const [center, setCenter] = useState<number>(0)

  useEffect(() => {
    drawCanvas()
  }, [functionType, terms, center])

  const factorial = (n: number): number => {
    if (n <= 1) return 1
    return n * factorial(n - 1)
  }

  const f = (x: number): number => {
    switch (functionType) {
      case 'exp': return Math.exp(x)
      case 'sin': return Math.sin(x)
      case 'cos': return Math.cos(x)
      case 'ln': return x > 0 ? Math.log(x) : NaN
      default: return 0
    }
  }

  const taylor = (x: number, n: number): number => {
    let sum = 0
    const dx = x - center

    for (let k = 0; k < n; k++) {
      let term = 0
      switch (functionType) {
        case 'exp':
          term = Math.exp(center) * Math.pow(dx, k) / factorial(k)
          break
        case 'sin':
          term = k % 4 === 0 ? Math.sin(center) * Math.pow(dx, k) / factorial(k) :
                 k % 4 === 1 ? Math.cos(center) * Math.pow(dx, k) / factorial(k) :
                 k % 4 === 2 ? -Math.sin(center) * Math.pow(dx, k) / factorial(k) :
                 -Math.cos(center) * Math.pow(dx, k) / factorial(k)
          break
        case 'cos':
          term = k % 4 === 0 ? Math.cos(center) * Math.pow(dx, k) / factorial(k) :
                 k % 4 === 1 ? -Math.sin(center) * Math.pow(dx, k) / factorial(k) :
                 k % 4 === 2 ? -Math.cos(center) * Math.pow(dx, k) / factorial(k) :
                 Math.sin(center) * Math.pow(dx, k) / factorial(k)
          break
        case 'ln':
          if (center === 0) return NaN
          term = k === 0 ? Math.log(center) : Math.pow(-1, k + 1) * Math.pow(dx, k) / (k * Math.pow(center, k))
          break
      }
      sum += term
    }
    return sum
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

    // Draw original function
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 3
    ctx.beginPath()
    let started = false
    for (let px = 0; px < width; px++) {
      const x = (px - centerX) / scale
      const y = f(x)
      if (!isNaN(y) && isFinite(y)) {
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
      }
    }
    ctx.stroke()

    // Draw Taylor approximations
    const colors = ['#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
    for (let n = 1; n <= Math.min(terms, 4); n++) {
      ctx.strokeStyle = colors[n - 1]
      ctx.lineWidth = 2
      ctx.globalAlpha = 0.7
      ctx.beginPath()
      started = false
      for (let px = 0; px < width; px++) {
        const x = (px - centerX) / scale
        const y = taylor(x, n * 2)
        if (!isNaN(y) && isFinite(y) && Math.abs(y) < 10) {
          const py = centerY - y * scale
          if (py >= -100 && py <= height + 100) {
            if (!started) {
              ctx.moveTo(px, py)
              started = true
            } else {
              ctx.lineTo(px, py)
            }
          } else {
            started = false
          }
        }
      }
      ctx.stroke()
    }
    ctx.globalAlpha = 1

    // Draw center point
    const cx = centerX + center * scale
    ctx.strokeStyle = '#ef4444'
    ctx.setLineDash([5, 5])
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(cx, 0)
    ctx.lineTo(cx, height)
    ctx.stroke()
    ctx.setLineDash([])

    ctx.fillStyle = '#ef4444'
    ctx.font = 'bold 14px Inter'
    ctx.fillText(`a = ${center.toFixed(1)}`, cx + 10, 30)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-green-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">테일러 급수 탐색기</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">시각화</h2>
            <canvas ref={canvasRef} width={800} height={600} className="w-full border border-slate-600 rounded-lg" />
            <div className="mt-4 space-y-2 text-sm text-slate-300">
              <p>💡 파란색: 원래 함수</p>
              <p>💡 초록/노랑/빨강/보라: 테일러 근사 (2, 4, 6, 8차)</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">함수</h3>
              <div className="space-y-2">
                {[
                  { id: 'exp', label: 'e^x', series: '1 + x + x²/2! + x³/3! + ...' },
                  { id: 'sin', label: 'sin(x)', series: 'x - x³/3! + x⁵/5! - ...' },
                  { id: 'cos', label: 'cos(x)', series: '1 - x²/2! + x⁴/4! - ...' },
                  { id: 'ln', label: 'ln(x)', series: '(x-1) - (x-1)²/2 + (x-1)³/3 - ...' }
                ].map((fn) => (
                  <button key={fn.id} onClick={() => setFunctionType(fn.id as any)} className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${functionType === fn.id ? 'bg-green-600 text-white' : 'bg-slate-700/50 hover:bg-slate-700'}`}>
                    <div className="font-mono text-sm font-semibold">{fn.label}</div>
                    <div className="text-xs text-slate-400">{fn.series}</div>
                  </button>
                ))}
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">항 개수: {terms}</h3>
              <input type="range" min="1" max="10" value={terms} onChange={(e) => setTerms(parseInt(e.target.value))} className="w-full" />
              <p className="text-xs text-slate-400 mt-2">최대 {terms * 2}차 항까지 표시</p>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">중심점 a</h3>
              <label className="text-sm text-slate-300 mb-2 block">a = {center.toFixed(1)}</label>
              <input type="range" min="-5" max="5" step="0.5" value={center} onChange={(e) => setCenter(parseFloat(e.target.value))} className="w-full" />
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">테일러 급수</h3>
              <div className="text-sm text-slate-300 space-y-2">
                <p className="font-mono text-xs bg-slate-900/50 p-3 rounded">
                  f(x) = Σ [f<sup>(n)</sup>(a)/n!]·(x-a)<sup>n</sup>
                </p>
                <p className="text-xs">
                  x = a 근처에서 함수를 무한급수로 표현합니다.<br />
                  항이 많을수록 정확도가 높아집니다.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
