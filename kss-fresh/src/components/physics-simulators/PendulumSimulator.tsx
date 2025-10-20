'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, Pause, RotateCcw } from 'lucide-react'

export default function PendulumSimulator() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [length, setLength] = useState(200)
  const [angle, setAngle] = useState(45)
  const [isRunning, setIsRunning] = useState(false)
  const [currentAngle, setCurrentAngle] = useState((45 * Math.PI) / 180)
  const [angularVelocity, setAngularVelocity] = useState(0)
  const animationRef = useRef<number>()

  const g = 9.8
  const dt = 0.016 // ~60fps

  useEffect(() => {
    drawCanvas()
  }, [currentAngle, length])

  useEffect(() => {
    if (isRunning) {
      const animate = () => {
        setAngularVelocity((omega) => {
          const alpha = -(g / (length / 100)) * Math.sin(currentAngle)
          return omega + alpha * dt
        })
        setCurrentAngle((theta) => theta + angularVelocity * dt)
        animationRef.current = requestAnimationFrame(animate)
      }
      animationRef.current = requestAnimationFrame(animate)
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning, currentAngle, angularVelocity, length])

  const drawCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height
    const pivotX = width / 2
    const pivotY = 100

    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, width, height)

    // Draw pivot
    ctx.fillStyle = '#64748b'
    ctx.beginPath()
    ctx.arc(pivotX, pivotY, 8, 0, Math.PI * 2)
    ctx.fill()

    // Calculate bob position
    const bobX = pivotX + length * Math.sin(currentAngle)
    const bobY = pivotY + length * Math.cos(currentAngle)

    // Draw string
    ctx.strokeStyle = '#94a3b8'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(pivotX, pivotY)
    ctx.lineTo(bobX, bobY)
    ctx.stroke()

    // Draw bob
    ctx.fillStyle = '#3b82f6'
    ctx.beginPath()
    ctx.arc(bobX, bobY, 20, 0, Math.PI * 2)
    ctx.fill()
    ctx.strokeStyle = '#60a5fa'
    ctx.lineWidth = 3
    ctx.stroke()

    // Draw trajectory arc
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.3)'
    ctx.lineWidth = 1
    ctx.setLineDash([5, 5])
    ctx.beginPath()
    const startAngle = -(angle * Math.PI) / 180
    const endAngle = (angle * Math.PI) / 180
    ctx.arc(pivotX, pivotY, length, startAngle, endAngle)
    ctx.stroke()
    ctx.setLineDash([])

    // Labels
    ctx.fillStyle = '#94a3b8'
    ctx.font = '14px Inter'
    ctx.fillText(`θ = ${((currentAngle * 180) / Math.PI).toFixed(1)}°`, pivotX + length + 30, pivotY)
    ctx.fillText(`L = ${length} px`, pivotX - 60, pivotY - 20)
  }

  const period = 2 * Math.PI * Math.sqrt((length / 100) / g)

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">진자 시뮬레이터</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">시뮬레이션</h2>
            <canvas ref={canvasRef} width={800} height={600} className="w-full border border-purple-600 rounded-lg" />
          </div>

          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">파라미터</h3>
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-slate-300 mb-2 block">줄 길이: {length} px</label>
                  <input
                    type="range"
                    min="50"
                    max="300"
                    value={length}
                    onChange={(e) => setLength(parseInt(e.target.value))}
                    className="w-full"
                    disabled={isRunning}
                  />
                </div>
                <div>
                  <label className="text-sm text-slate-300 mb-2 block">초기 각도: {angle}°</label>
                  <input
                    type="range"
                    min="5"
                    max="90"
                    value={angle}
                    onChange={(e) => setAngle(parseInt(e.target.value))}
                    className="w-full"
                    disabled={isRunning}
                  />
                </div>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">제어</h3>
              <div className="space-y-3">
                <button
                  onClick={() => setIsRunning(!isRunning)}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-500 rounded-lg"
                >
                  {isRunning ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                  <span>{isRunning ? '일시정지' : '시작'}</span>
                </button>
                <button
                  onClick={() => {
                    setIsRunning(false)
                    setCurrentAngle((angle * Math.PI) / 180)
                    setAngularVelocity(0)
                  }}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg"
                >
                  <RotateCcw className="w-5 h-5" />
                  <span>초기화</span>
                </button>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">계산</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400">주기:</span>
                  <span className="font-mono text-purple-300">{period.toFixed(2)} s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">진동수:</span>
                  <span className="font-mono text-purple-300">{(1 / period).toFixed(2)} Hz</span>
                </div>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">공식</h3>
              <div className="text-xs text-slate-300 space-y-2 font-mono">
                <p>T = 2π√(L/g)</p>
                <p>α = -(g/L)sin(θ)</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
