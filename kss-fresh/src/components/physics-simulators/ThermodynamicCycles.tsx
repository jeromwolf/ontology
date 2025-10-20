'use client'

import React, { useState, useRef, useEffect } from 'react'

export default function ThermodynamicCycles() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [cycle, setCycle] = useState<'carnot' | 'otto' | 'diesel'>('carnot')
  const [TH, setTH] = useState(600) // High temperature (K)
  const [TC, setTC] = useState(300) // Low temperature (K)

  useEffect(() => {
    drawCanvas()
  }, [cycle, TH, TC])

  const drawCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, width, height)

    // Draw axes
    ctx.strokeStyle = '#475569'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(80, height - 80)
    ctx.lineTo(width - 40, height - 80)
    ctx.moveTo(80, height - 80)
    ctx.lineTo(80, 40)
    ctx.stroke()

    // Labels
    ctx.fillStyle = '#94a3b8'
    ctx.font = '16px Inter'
    ctx.fillText('V (Volume)', width / 2, height - 30)
    ctx.save()
    ctx.translate(30, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText('P (Pressure)', 0, 0)
    ctx.restore()

    if (cycle === 'carnot') {
      drawCarnotCycle(ctx, width, height)
    } else if (cycle === 'otto') {
      drawOttoCycle(ctx, width, height)
    } else {
      drawDieselCycle(ctx, width, height)
    }
  }

  const drawCarnotCycle = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const x1 = 150, x2 = 350, x3 = 550, x4 = 250
    const y1 = 200, y2 = 200, y3 = 400, y4 = 400

    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2) // Isothermal expansion (TH)
    ctx.stroke()

    ctx.strokeStyle = '#10b981'
    ctx.beginPath()
    ctx.moveTo(x2, y2)
    ctx.lineTo(x3, y3) // Adiabatic expansion
    ctx.stroke()

    ctx.strokeStyle = '#ef4444'
    ctx.beginPath()
    ctx.moveTo(x3, y3)
    ctx.lineTo(x4, y4) // Isothermal compression (TC)
    ctx.stroke()

    ctx.strokeStyle = '#a78bfa'
    ctx.beginPath()
    ctx.moveTo(x4, y4)
    ctx.lineTo(x1, y1) // Adiabatic compression
    ctx.stroke()

    // Labels
    ctx.fillStyle = '#94a3b8'
    ctx.font = '14px Inter'
    ctx.fillText('1→2: 등온 팽창 (TH)', x1 + 50, y1 - 20)
    ctx.fillText('2→3: 단열 팽창', x2 + 50, y2 + 100)
    ctx.fillText('3→4: 등온 압축 (TC)', x3 - 100, y3 + 30)
    ctx.fillText('4→1: 단열 압축', x4 - 100, y4 - 100)
  }

  const drawOttoCycle = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const x1 = 200, x2 = 200, x3 = 500, x4 = 500
    const y1 = 400, y2 = 200, y3 = 200, y4 = 400

    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2) // Adiabatic compression
    ctx.stroke()

    ctx.strokeStyle = '#ef4444'
    ctx.beginPath()
    ctx.moveTo(x2, y2)
    ctx.lineTo(x3, y3) // Constant volume heating
    ctx.stroke()

    ctx.strokeStyle = '#10b981'
    ctx.beginPath()
    ctx.moveTo(x3, y3)
    ctx.lineTo(x4, y4) // Adiabatic expansion
    ctx.stroke()

    ctx.strokeStyle = '#a78bfa'
    ctx.beginPath()
    ctx.moveTo(x4, y4)
    ctx.lineTo(x1, y1) // Constant volume cooling
    ctx.stroke()

    ctx.fillStyle = '#94a3b8'
    ctx.font = '14px Inter'
    ctx.fillText('1→2: 단열 압축', x1 - 120, y1 - 100)
    ctx.fillText('2→3: 등적 가열', x2 + 150, y2 - 20)
    ctx.fillText('3→4: 단열 팽창', x3 + 30, y3 + 100)
    ctx.fillText('4→1: 등적 냉각', x4 - 150, y4 + 30)
  }

  const drawDieselCycle = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const x1 = 200, x2 = 200, x3 = 350, x4 = 500, x5 = 500
    const y1 = 400, y2 = 200, y3 = 200, y4 = 300, y5 = 400

    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.stroke()

    ctx.strokeStyle = '#ef4444'
    ctx.beginPath()
    ctx.moveTo(x2, y2)
    ctx.lineTo(x3, y3)
    ctx.stroke()

    ctx.strokeStyle = '#10b981'
    ctx.beginPath()
    ctx.moveTo(x3, y3)
    ctx.lineTo(x4, y4)
    ctx.stroke()

    ctx.strokeStyle = '#f59e0b'
    ctx.beginPath()
    ctx.moveTo(x4, y4)
    ctx.lineTo(x5, y5)
    ctx.stroke()

    ctx.strokeStyle = '#a78bfa'
    ctx.beginPath()
    ctx.moveTo(x5, y5)
    ctx.lineTo(x1, y1)
    ctx.stroke()

    ctx.fillStyle = '#94a3b8'
    ctx.font = '12px Inter'
    ctx.fillText('1→2: 단열 압축', x1 - 100, y1 - 100)
    ctx.fillText('2→3: 등압 가열', x2 + 50, y2 - 20)
    ctx.fillText('3→4: 단열 팽창', x3 + 80, y3 + 50)
    ctx.fillText('4→5: 등적 냉각', x4 + 30, y4 + 50)
  }

  const efficiency_carnot = ((TH - TC) / TH) * 100
  const efficiency_otto = (1 - Math.pow(TC / TH, 0.4)) * 100
  const efficiency_diesel = (1 - Math.pow(TC / TH, 0.4) * 0.9) * 100

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">열역학 사이클</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">P-V 다이어그램</h2>
            <canvas ref={canvasRef} width={800} height={600} className="w-full border border-purple-600 rounded-lg" />
          </div>

          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">사이클 선택</h3>
              <div className="space-y-2">
                {[
                  { id: 'carnot', label: '카르노 사이클', desc: '이론적 최대 효율' },
                  { id: 'otto', label: '오토 사이클', desc: '가솔린 엔진' },
                  { id: 'diesel', label: '디젤 사이클', desc: '디젤 엔진' }
                ].map((c) => (
                  <button
                    key={c.id}
                    onClick={() => setCycle(c.id as any)}
                    className={`w-full text-left px-4 py-3 rounded-lg ${
                      cycle === c.id ? 'bg-purple-600' : 'bg-slate-700 hover:bg-slate-600'
                    }`}
                  >
                    <div className="font-semibold">{c.label}</div>
                    <div className="text-xs text-slate-400">{c.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">온도 설정</h3>
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-slate-300 mb-2 block">고온부 (T_H): {TH} K</label>
                  <input
                    type="range"
                    min="400"
                    max="800"
                    value={TH}
                    onChange={(e) => setTH(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="text-sm text-slate-300 mb-2 block">저온부 (T_C): {TC} K</label>
                  <input
                    type="range"
                    min="200"
                    max="400"
                    value={TC}
                    onChange={(e) => setTC(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">효율</h3>
              <div className="space-y-3">
                {cycle === 'carnot' && (
                  <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                    <p className="text-sm text-slate-400 mb-1">카르노 효율</p>
                    <p className="text-2xl font-bold text-purple-400">{efficiency_carnot.toFixed(1)}%</p>
                    <p className="text-xs text-slate-400 mt-2 font-mono">η = 1 - T_C/T_H</p>
                  </div>
                )}
                {cycle === 'otto' && (
                  <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                    <p className="text-sm text-slate-400 mb-1">오토 효율</p>
                    <p className="text-2xl font-bold text-blue-400">{efficiency_otto.toFixed(1)}%</p>
                    <p className="text-xs text-slate-400 mt-2">실제: 20-30%</p>
                  </div>
                )}
                {cycle === 'diesel' && (
                  <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                    <p className="text-sm text-slate-400 mb-1">디젤 효율</p>
                    <p className="text-2xl font-bold text-green-400">{efficiency_diesel.toFixed(1)}%</p>
                    <p className="text-xs text-slate-400 mt-2">실제: 30-40%</p>
                  </div>
                )}
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">과정 설명</h3>
              <div className="text-xs text-slate-300 space-y-2">
                <div className="p-2 bg-blue-500/10 border border-blue-500/30 rounded">
                  <span className="font-semibold text-blue-400">등온:</span> ΔT = 0, Q = W
                </div>
                <div className="p-2 bg-green-500/10 border border-green-500/30 rounded">
                  <span className="font-semibold text-green-400">단열:</span> Q = 0, ΔU = -W
                </div>
                <div className="p-2 bg-red-500/10 border border-red-500/30 rounded">
                  <span className="font-semibold text-red-400">등적:</span> ΔV = 0, W = 0
                </div>
                <div className="p-2 bg-purple-500/10 border border-purple-500/30 rounded">
                  <span className="font-semibold text-purple-400">등압:</span> ΔP = 0
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
