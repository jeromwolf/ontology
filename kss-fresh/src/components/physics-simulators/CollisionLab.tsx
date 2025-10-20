'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, RotateCcw } from 'lucide-react'

export default function CollisionLab() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [mass1, setMass1] = useState(2)
  const [mass2, setMass2] = useState(1)
  const [velocity1, setVelocity1] = useState(5)
  const [velocity2, setVelocity2] = useState(-2)
  const [collisionType, setCollisionType] = useState<'elastic' | 'inelastic'>('elastic')
  const [isRunning, setIsRunning] = useState(false)
  const [time, setTime] = useState(0)
  const animationRef = useRef<number>()

  const [pos1, setPos1] = useState(200)
  const [pos2, setPos2] = useState(600)
  const [vel1, setVel1] = useState(velocity1)
  const [vel2, setVel2] = useState(velocity2)
  const [collided, setCollided] = useState(false)

  useEffect(() => {
    drawCanvas()
  }, [pos1, pos2, mass1, mass2])

  useEffect(() => {
    if (isRunning) {
      const animate = () => {
        setPos1((p) => p + vel1)
        setPos2((p) => p + vel2)

        const distance = Math.abs(pos2 - pos1)
        const radius1 = Math.sqrt(mass1) * 20
        const radius2 = Math.sqrt(mass2) * 20

        if (distance <= radius1 + radius2 && !collided) {
          setCollided(true)

          if (collisionType === 'elastic') {
            const v1f = ((mass1 - mass2) * vel1 + 2 * mass2 * vel2) / (mass1 + mass2)
            const v2f = ((mass2 - mass1) * vel2 + 2 * mass1 * vel1) / (mass1 + mass2)
            setVel1(v1f)
            setVel2(v2f)
          } else {
            const vf = (mass1 * vel1 + mass2 * vel2) / (mass1 + mass2)
            setVel1(vf)
            setVel2(vf)
          }
        }

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
  }, [isRunning, vel1, vel2, pos1, pos2, collided, mass1, mass2, collisionType])

  const drawCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, width, height)

    // Draw center line
    ctx.strokeStyle = '#475569'
    ctx.lineWidth = 1
    ctx.setLineDash([5, 5])
    ctx.beginPath()
    ctx.moveTo(width / 2, 0)
    ctx.lineTo(width / 2, height)
    ctx.stroke()
    ctx.setLineDash([])

    // Draw ground
    ctx.strokeStyle = '#475569'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(0, height / 2 + 80)
    ctx.lineTo(width, height / 2 + 80)
    ctx.stroke()

    // Draw ball 1
    const radius1 = Math.sqrt(mass1) * 20
    ctx.fillStyle = '#3b82f6'
    ctx.beginPath()
    ctx.arc(pos1, height / 2, radius1, 0, Math.PI * 2)
    ctx.fill()
    ctx.strokeStyle = '#60a5fa'
    ctx.lineWidth = 3
    ctx.stroke()

    // Draw ball 2
    const radius2 = Math.sqrt(mass2) * 20
    ctx.fillStyle = '#ef4444'
    ctx.beginPath()
    ctx.arc(pos2, height / 2, radius2, 0, Math.PI * 2)
    ctx.fill()
    ctx.strokeStyle = '#f87171'
    ctx.lineWidth = 3
    ctx.stroke()

    // Labels
    ctx.fillStyle = '#94a3b8'
    ctx.font = 'bold 14px Inter'
    ctx.fillText(`m₁ = ${mass1} kg`, pos1 - 30, height / 2 - radius1 - 10)
    ctx.fillText(`m₂ = ${mass2} kg`, pos2 - 30, height / 2 - radius2 - 10)
  }

  const handleStart = () => {
    setPos1(200)
    setPos2(600)
    setVel1(velocity1)
    setVel2(velocity2)
    setCollided(false)
    setIsRunning(true)
  }

  const handleReset = () => {
    setIsRunning(false)
    setPos1(200)
    setPos2(600)
    setVel1(velocity1)
    setVel2(velocity2)
    setCollided(false)
    setTime(0)
  }

  const initialMomentum = mass1 * velocity1 + mass2 * velocity2
  const initialKE = 0.5 * mass1 * velocity1 ** 2 + 0.5 * mass2 * velocity2 ** 2
  const finalMomentum = mass1 * vel1 + mass2 * vel2
  const finalKE = 0.5 * mass1 * vel1 ** 2 + 0.5 * mass2 * vel2 ** 2

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">충돌 실험실</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">시뮬레이션</h2>
            <canvas ref={canvasRef} width={800} height={400} className="w-full border border-purple-600 rounded-lg" />
          </div>

          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">물체 1 (파란색)</h3>
              <div className="space-y-3">
                <div>
                  <label className="text-sm text-slate-300 mb-1 block">질량: {mass1} kg</label>
                  <input
                    type="range"
                    min="0.5"
                    max="5"
                    step="0.5"
                    value={mass1}
                    onChange={(e) => setMass1(parseFloat(e.target.value))}
                    className="w-full"
                    disabled={isRunning}
                  />
                </div>
                <div>
                  <label className="text-sm text-slate-300 mb-1 block">속도: {velocity1} m/s</label>
                  <input
                    type="range"
                    min="-10"
                    max="10"
                    value={velocity1}
                    onChange={(e) => setVelocity1(parseInt(e.target.value))}
                    className="w-full"
                    disabled={isRunning}
                  />
                </div>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">물체 2 (빨간색)</h3>
              <div className="space-y-3">
                <div>
                  <label className="text-sm text-slate-300 mb-1 block">질량: {mass2} kg</label>
                  <input
                    type="range"
                    min="0.5"
                    max="5"
                    step="0.5"
                    value={mass2}
                    onChange={(e) => setMass2(parseFloat(e.target.value))}
                    className="w-full"
                    disabled={isRunning}
                  />
                </div>
                <div>
                  <label className="text-sm text-slate-300 mb-1 block">속도: {velocity2} m/s</label>
                  <input
                    type="range"
                    min="-10"
                    max="10"
                    value={velocity2}
                    onChange={(e) => setVelocity2(parseInt(e.target.value))}
                    className="w-full"
                    disabled={isRunning}
                  />
                </div>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">충돌 유형</h3>
              <div className="space-y-2">
                <button
                  onClick={() => setCollisionType('elastic')}
                  disabled={isRunning}
                  className={`w-full px-4 py-2 rounded-lg ${
                    collisionType === 'elastic' ? 'bg-green-600' : 'bg-slate-700 hover:bg-slate-600'
                  } disabled:opacity-50`}
                >
                  탄성 충돌
                </button>
                <button
                  onClick={() => setCollisionType('inelastic')}
                  disabled={isRunning}
                  className={`w-full px-4 py-2 rounded-lg ${
                    collisionType === 'inelastic' ? 'bg-green-600' : 'bg-slate-700 hover:bg-slate-600'
                  } disabled:opacity-50`}
                >
                  완전 비탄성 충돌
                </button>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">제어</h3>
              <div className="space-y-3">
                <button
                  onClick={handleStart}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-500 rounded-lg"
                >
                  <Play className="w-5 h-5" />
                  <span>시작</span>
                </button>
                <button
                  onClick={handleReset}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg"
                >
                  <RotateCcw className="w-5 h-5" />
                  <span>초기화</span>
                </button>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">보존량</h3>
              <div className="space-y-3 text-sm">
                <div>
                  <p className="text-slate-400 mb-1">초기 운동량: {initialMomentum.toFixed(2)} kg·m/s</p>
                  <p className="text-green-400">최종 운동량: {finalMomentum.toFixed(2)} kg·m/s</p>
                </div>
                <div>
                  <p className="text-slate-400 mb-1">초기 에너지: {initialKE.toFixed(2)} J</p>
                  <p className={collisionType === 'elastic' ? 'text-green-400' : 'text-yellow-400'}>
                    최종 에너지: {finalKE.toFixed(2)} J
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
