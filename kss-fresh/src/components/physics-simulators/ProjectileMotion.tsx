'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, RotateCcw, Pause } from 'lucide-react'

export default function ProjectileMotion() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [velocity, setVelocity] = useState<number>(50)
  const [angle, setAngle] = useState<number>(45)
  const [isRunning, setIsRunning] = useState(false)
  const [time, setTime] = useState(0)
  const animationRef = useRef<number>()

  const g = 9.8 // gravity

  useEffect(() => {
    drawCanvas()
  }, [velocity, angle, time])

  useEffect(() => {
    if (isRunning) {
      const startTime = Date.now() - time * 1000
      const animate = () => {
        const elapsed = (Date.now() - startTime) / 1000
        const maxTime = (2 * velocity * Math.sin((angle * Math.PI) / 180)) / g

        if (elapsed >= maxTime) {
          setIsRunning(false)
          setTime(maxTime)
        } else {
          setTime(elapsed)
          animationRef.current = requestAnimationFrame(animate)
        }
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
  }, [isRunning, velocity, angle])

  const calculatePosition = (t: number) => {
    const angleRad = (angle * Math.PI) / 180
    const vx = velocity * Math.cos(angleRad)
    const vy = velocity * Math.sin(angleRad)

    const x = vx * t
    const y = vy * t - 0.5 * g * t * t

    return { x, y, vx, vy: vy - g * t }
  }

  const drawCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height
    const scale = 2

    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, width, height)

    // Draw ground
    ctx.strokeStyle = '#475569'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(0, height - 50)
    ctx.lineTo(width, height - 50)
    ctx.stroke()

    // Draw trajectory
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 2
    ctx.beginPath()
    const angleRad = (angle * Math.PI) / 180
    const maxTime = (2 * velocity * Math.sin(angleRad)) / g

    for (let t = 0; t <= maxTime; t += 0.05) {
      const pos = calculatePosition(t)
      const px = 50 + pos.x * scale
      const py = height - 50 - pos.y * scale

      if (t === 0) {
        ctx.moveTo(px, py)
      } else {
        ctx.lineTo(px, py)
      }
    }
    ctx.stroke()

    // Draw current position
    if (time > 0) {
      const currentPos = calculatePosition(time)
      const px = 50 + currentPos.x * scale
      const py = height - 50 - currentPos.y * scale

      ctx.fillStyle = '#10b981'
      ctx.beginPath()
      ctx.arc(px, py, 8, 0, Math.PI * 2)
      ctx.fill()

      // Draw velocity vector
      const vectorScale = 0.5
      ctx.strokeStyle = '#ef4444'
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.moveTo(px, py)
      ctx.lineTo(px + currentPos.vx * vectorScale, py - currentPos.vy * vectorScale)
      ctx.stroke()

      // Arrow head
      const vAngle = Math.atan2(-currentPos.vy, currentPos.vx)
      ctx.fillStyle = '#ef4444'
      ctx.beginPath()
      ctx.moveTo(px + currentPos.vx * vectorScale, py - currentPos.vy * vectorScale)
      ctx.lineTo(
        px + currentPos.vx * vectorScale - 10 * Math.cos(vAngle - Math.PI / 6),
        py - currentPos.vy * vectorScale - 10 * Math.sin(vAngle - Math.PI / 6)
      )
      ctx.lineTo(
        px + currentPos.vx * vectorScale - 10 * Math.cos(vAngle + Math.PI / 6),
        py - currentPos.vy * vectorScale - 10 * Math.sin(vAngle + Math.PI / 6)
      )
      ctx.closePath()
      ctx.fill()
    }

    // Draw angle arc
    ctx.strokeStyle = '#a78bfa'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.arc(50, height - 50, 30, -angleRad, 0)
    ctx.stroke()

    // Labels
    ctx.fillStyle = '#94a3b8'
    ctx.font = '14px Inter'
    ctx.fillText(`θ = ${angle}°`, 85, height - 35)
    ctx.fillText(`v₀ = ${velocity} m/s`, 50, 30)

    if (time > 0) {
      const currentPos = calculatePosition(time)
      ctx.fillText(`t = ${time.toFixed(2)} s`, width - 150, 30)
      ctx.fillText(`x = ${currentPos.x.toFixed(1)} m`, width - 150, 50)
      ctx.fillText(`y = ${currentPos.y.toFixed(1)} m`, width - 150, 70)
    }
  }

  const handleStart = () => {
    setTime(0)
    setIsRunning(true)
  }

  const handleReset = () => {
    setIsRunning(false)
    setTime(0)
  }

  const angleRad = (angle * Math.PI) / 180
  const maxHeight = (velocity * velocity * Math.sin(angleRad) * Math.sin(angleRad)) / (2 * g)
  const range = (velocity * velocity * Math.sin(2 * angleRad)) / g
  const totalTime = (2 * velocity * Math.sin(angleRad)) / g

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">포물선 운동 시뮬레이터</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">시각화</h2>
            <canvas ref={canvasRef} width={800} height={500} className="w-full border border-purple-600 rounded-lg" />
            <div className="mt-4 space-y-2 text-sm text-slate-300">
              <p>💡 파란색 선: 포물선 궤적</p>
              <p>💡 초록색 점: 현재 위치</p>
              <p>💡 빨간색 화살표: 속도 벡터</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">초기 조건</h3>
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-slate-300 mb-2 block">초기 속력: {velocity} m/s</label>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    value={velocity}
                    onChange={(e) => {
                      setVelocity(parseInt(e.target.value))
                      setTime(0)
                      setIsRunning(false)
                    }}
                    className="w-full"
                    disabled={isRunning}
                  />
                </div>
                <div>
                  <label className="text-sm text-slate-300 mb-2 block">발사 각도: {angle}°</label>
                  <input
                    type="range"
                    min="0"
                    max="90"
                    value={angle}
                    onChange={(e) => {
                      setAngle(parseInt(e.target.value))
                      setTime(0)
                      setIsRunning(false)
                    }}
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
                  onClick={handleStart}
                  disabled={isRunning}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-500 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg transition-colors"
                >
                  <Play className="w-5 h-5" />
                  <span>시작</span>
                </button>
                <button
                  onClick={() => setIsRunning(!isRunning)}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-yellow-600 hover:bg-yellow-500 rounded-lg transition-colors"
                >
                  <Pause className="w-5 h-5" />
                  <span>{isRunning ? '일시정지' : '재개'}</span>
                </button>
                <button
                  onClick={handleReset}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
                >
                  <RotateCcw className="w-5 h-5" />
                  <span>초기화</span>
                </button>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">계산 결과</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400">최대 높이:</span>
                  <span className="font-mono text-purple-300">{maxHeight.toFixed(2)} m</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">수평 도달 거리:</span>
                  <span className="font-mono text-purple-300">{range.toFixed(2)} m</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">비행 시간:</span>
                  <span className="font-mono text-purple-300">{totalTime.toFixed(2)} s</span>
                </div>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">공식</h3>
              <div className="text-xs text-slate-300 space-y-2 font-mono">
                <p>x(t) = v₀ cos(θ) · t</p>
                <p>y(t) = v₀ sin(θ) · t - ½gt²</p>
                <p>H = v₀² sin²(θ) / (2g)</p>
                <p>R = v₀² sin(2θ) / g</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
