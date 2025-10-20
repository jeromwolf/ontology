'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Plus, Minus } from 'lucide-react'

export default function ElectricField() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [charges, setCharges] = useState([
    { x: 300, y: 300, q: 1 },
    { x: 500, y: 300, q: -1 }
  ])

  useEffect(() => {
    drawCanvas()
  }, [charges])

  const electricField = (x: number, y: number) => {
    let Ex = 0,
      Ey = 0
    const k = 50

    charges.forEach((charge) => {
      const dx = x - charge.x
      const dy = y - charge.y
      const r2 = dx * dx + dy * dy
      if (r2 < 1) return

      const E = (k * charge.q) / r2
      const r = Math.sqrt(r2)
      Ex += (E * dx) / r
      Ey += (E * dy) / r
    })

    return { Ex, Ey }
  }

  const drawCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw field lines
    const step = 30
    for (let y = step; y < canvas.height; y += step) {
      for (let x = step; x < canvas.width; x += step) {
        const { Ex, Ey } = electricField(x, y)
        const mag = Math.sqrt(Ex * Ex + Ey * Ey)
        if (mag < 0.1) continue

        const len = Math.min(mag * 0.8, 20)
        const angle = Math.atan2(Ey, Ex)

        const hue = Math.min(mag * 3, 120)
        ctx.strokeStyle = `hsl(${120 - hue}, 70%, 50%)`
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(x, y)
        ctx.lineTo(x + Math.cos(angle) * len, y + Math.sin(angle) * len)
        ctx.stroke()

        // Arrow head
        ctx.fillStyle = ctx.strokeStyle
        ctx.beginPath()
        ctx.moveTo(x + Math.cos(angle) * len, y + Math.sin(angle) * len)
        ctx.lineTo(
          x + Math.cos(angle) * len - 6 * Math.cos(angle - Math.PI / 6),
          y + Math.sin(angle) * len - 6 * Math.sin(angle - Math.PI / 6)
        )
        ctx.lineTo(
          x + Math.cos(angle) * len - 6 * Math.cos(angle + Math.PI / 6),
          y + Math.sin(angle) * len - 6 * Math.sin(angle + Math.PI / 6)
        )
        ctx.closePath()
        ctx.fill()
      }
    }

    // Draw charges
    charges.forEach((charge) => {
      ctx.fillStyle = charge.q > 0 ? '#ef4444' : '#3b82f6'
      ctx.beginPath()
      ctx.arc(charge.x, charge.y, Math.abs(charge.q) * 20, 0, Math.PI * 2)
      ctx.fill()

      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 20px Inter'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(charge.q > 0 ? '+' : '-', charge.x, charge.y)
    })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">전기장 시각화</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">전기장 벡터</h2>
            <canvas
              ref={canvasRef}
              width={800}
              height={600}
              className="w-full border border-purple-600 rounded-lg cursor-crosshair"
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect()
                const x = ((e.clientX - rect.left) / rect.width) * 800
                const y = ((e.clientY - rect.top) / rect.height) * 600
                setCharges([...charges, { x, y, q: 1 }])
              }}
            />
            <p className="mt-4 text-sm text-slate-300">💡 클릭하여 전하 추가</p>
          </div>

          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">전하 목록</h3>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {charges.map((charge, i) => (
                  <div key={i} className="flex items-center gap-2 bg-slate-900/50 p-3 rounded-lg">
                    <div className={`w-4 h-4 rounded-full ${charge.q > 0 ? 'bg-red-500' : 'bg-blue-500'}`}></div>
                    <span className="text-sm flex-1">
                      {charge.q > 0 ? '+' : '-'}{Math.abs(charge.q)} C at ({charge.x.toFixed(0)}, {charge.y.toFixed(0)})
                    </span>
                    <button
                      onClick={() => setCharges(charges.filter((_, idx) => idx !== i))}
                      className="px-2 py-1 bg-red-600 hover:bg-red-500 rounded text-xs"
                    >
                      삭제
                    </button>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">전하 추가</h3>
              <div className="space-y-3">
                <button
                  onClick={() => {
                    const x = 200 + Math.random() * 400
                    const y = 200 + Math.random() * 200
                    setCharges([...charges, { x, y, q: 1 }])
                  }}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-red-600 hover:bg-red-500 rounded-lg"
                >
                  <Plus className="w-5 h-5" />
                  <span>양전하 (+)</span>
                </button>
                <button
                  onClick={() => {
                    const x = 200 + Math.random() * 400
                    const y = 200 + Math.random() * 200
                    setCharges([...charges, { x, y, q: -1 }])
                  }}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded-lg"
                >
                  <Minus className="w-5 h-5" />
                  <span>음전하 (-)</span>
                </button>
                <button
                  onClick={() => setCharges([])}
                  className="w-full px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg"
                >
                  모두 삭제
                </button>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">공식</h3>
              <div className="text-xs text-slate-300 space-y-2 font-mono">
                <p>E = kQ/r²</p>
                <p>F = qE</p>
                <p>k = 8.99×10⁹ N·m²/C²</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
