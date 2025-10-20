'use client'

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Play, Pause, RotateCcw, Settings, Zap, Wind } from 'lucide-react'

interface Ball {
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  mass: number
  color: string
}

interface Pendulum {
  angle: number
  angularVelocity: number
  length: number
  mass: number
}

type SimulationType = 'pendulum' | 'collision' | 'projectile' | 'springs'

export default function NewtonMechanicsLab() {
  const [isRunning, setIsRunning] = useState(false)
  const [simType, setSimType] = useState<SimulationType>('pendulum')
  const [gravity, setGravity] = useState(9.81)
  const [damping, setDamping] = useState(0.99)
  const [restitution, setRestitution] = useState(0.85)

  // Pendulum state
  const [pendulum, setPendulum] = useState<Pendulum>({
    angle: Math.PI / 4,
    angularVelocity: 0,
    length: 200,
    mass: 1
  })

  // Collision balls
  const [balls, setBalls] = useState<Ball[]>([
    { x: 200, y: 200, vx: 50, vy: 30, radius: 30, mass: 1, color: '#3b82f6' },
    { x: 500, y: 300, vx: -30, vy: -20, radius: 40, mass: 1.5, color: '#ef4444' },
    { x: 350, y: 150, vx: 20, vy: 50, radius: 25, mass: 0.8, color: '#10b981' }
  ])

  // Projectile state
  const [projectile, setProjectile] = useState({
    x: 50,
    y: 400,
    vx: 150,
    vy: -200,
    angle: 45,
    velocity: 250,
    trail: [] as { x: number; y: number }[]
  })

  // Springs state
  const [springs, setSprings] = useState([
    { x: 400, y: 100, vy: 0, k: 0.1, equilibrium: 100 }
  ])

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const lastTimeRef = useRef<number>(0)

  // Physics calculations
  const updatePendulum = useCallback((dt: number) => {
    setPendulum(prev => {
      const angularAcceleration = (-gravity / prev.length) * Math.sin(prev.angle)
      const newAngularVelocity = prev.angularVelocity + angularAcceleration * dt
      const newAngle = prev.angle + newAngularVelocity * dt

      return {
        ...prev,
        angle: newAngle,
        angularVelocity: newAngularVelocity * damping
      }
    })
  }, [gravity, damping])

  const updateCollisions = useCallback((dt: number, canvasWidth: number, canvasHeight: number) => {
    setBalls(prevBalls => {
      const newBalls = prevBalls.map(ball => {
        let newX = ball.x + ball.vx * dt
        let newY = ball.y + ball.vy * dt
        let newVx = ball.vx
        let newVy = ball.vy + gravity * 10 * dt // Apply gravity

        // Wall collisions
        if (newX - ball.radius < 0) {
          newX = ball.radius
          newVx = -newVx * restitution
        } else if (newX + ball.radius > canvasWidth) {
          newX = canvasWidth - ball.radius
          newVx = -newVx * restitution
        }

        if (newY - ball.radius < 0) {
          newY = ball.radius
          newVy = -newVy * restitution
        } else if (newY + ball.radius > canvasHeight) {
          newY = canvasHeight - ball.radius
          newVy = -newVy * restitution
        }

        // Apply damping
        newVx *= damping
        newVy *= damping

        return { ...ball, x: newX, y: newY, vx: newVx, vy: newVy }
      })

      // Ball-to-ball collisions
      for (let i = 0; i < newBalls.length; i++) {
        for (let j = i + 1; j < newBalls.length; j++) {
          const b1 = newBalls[i]
          const b2 = newBalls[j]

          const dx = b2.x - b1.x
          const dy = b2.y - b1.y
          const distance = Math.sqrt(dx * dx + dy * dy)

          if (distance < b1.radius + b2.radius) {
            // Normalize collision vector
            const nx = dx / distance
            const ny = dy / distance

            // Relative velocity
            const dvx = b2.vx - b1.vx
            const dvy = b2.vy - b1.vy
            const dvn = dvx * nx + dvy * ny

            // Don't collide if moving apart
            if (dvn < 0) continue

            // Calculate impulse (elastic collision)
            const totalMass = b1.mass + b2.mass
            const impulse = (2 * dvn) / totalMass

            // Apply impulse
            newBalls[i].vx += impulse * b2.mass * nx * restitution
            newBalls[i].vy += impulse * b2.mass * ny * restitution
            newBalls[j].vx -= impulse * b1.mass * nx * restitution
            newBalls[j].vy -= impulse * b1.mass * ny * restitution

            // Separate balls to prevent overlap
            const overlap = (b1.radius + b2.radius - distance) / 2
            newBalls[i].x -= overlap * nx
            newBalls[i].y -= overlap * ny
            newBalls[j].x += overlap * nx
            newBalls[j].y += overlap * ny
          }
        }
      }

      return newBalls
    })
  }, [gravity, damping, restitution])

  const updateProjectile = useCallback((dt: number) => {
    setProjectile(prev => {
      const newVy = prev.vy + gravity * 10 * dt
      const newX = prev.x + prev.vx * dt
      const newY = prev.y + newVy * dt

      // Add to trail
      const newTrail = [...prev.trail, { x: newX, y: newY }]
      if (newTrail.length > 100) newTrail.shift()

      // Reset if hits ground
      if (newY > 400) {
        const angleRad = (prev.angle * Math.PI) / 180
        return {
          ...prev,
          x: 50,
          y: 400,
          vx: prev.velocity * Math.cos(angleRad),
          vy: -prev.velocity * Math.sin(angleRad),
          trail: []
        }
      }

      return {
        ...prev,
        x: newX,
        y: newY,
        vy: newVy,
        trail: newTrail
      }
    })
  }, [gravity])

  const updateSprings = useCallback((dt: number) => {
    setSprings(prevSprings =>
      prevSprings.map(spring => {
        const displacement = spring.y - spring.equilibrium
        const force = -spring.k * displacement
        const acceleration = force / 1 // mass = 1
        const newVy = spring.vy + acceleration * dt + gravity * 10 * dt
        const newY = spring.y + newVy * dt

        return {
          ...spring,
          y: newY,
          vy: newVy * damping
        }
      })
    )
  }, [gravity, damping])

  // Drawing functions
  const drawPendulum = useCallback((ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const pivotX = width / 2
    const pivotY = 50
    const bobX = pivotX + pendulum.length * Math.sin(pendulum.angle)
    const bobY = pivotY + pendulum.length * Math.cos(pendulum.angle)

    // Rod
    ctx.strokeStyle = '#64748b'
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.moveTo(pivotX, pivotY)
    ctx.lineTo(bobX, bobY)
    ctx.stroke()

    // Pivot
    ctx.fillStyle = '#94a3b8'
    ctx.beginPath()
    ctx.arc(pivotX, pivotY, 8, 0, Math.PI * 2)
    ctx.fill()

    // Bob
    const gradient = ctx.createRadialGradient(bobX - 10, bobY - 10, 5, bobX, bobY, 30)
    gradient.addColorStop(0, '#fbbf24')
    gradient.addColorStop(1, '#f59e0b')
    ctx.fillStyle = gradient
    ctx.beginPath()
    ctx.arc(bobX, bobY, 30, 0, Math.PI * 2)
    ctx.fill()

    // Shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.2)'
    ctx.beginPath()
    ctx.ellipse(bobX, height - 20, 40, 10, 0, 0, Math.PI * 2)
    ctx.fill()

    // Info
    ctx.font = '14px Inter, sans-serif'
    ctx.fillStyle = '#e5e7eb'
    ctx.fillText(`Angle: ${(pendulum.angle * 180 / Math.PI).toFixed(1)}°`, 20, 30)
    ctx.fillText(`Angular Velocity: ${pendulum.angularVelocity.toFixed(2)} rad/s`, 20, 50)
    ctx.fillText(`Period: ${(2 * Math.PI * Math.sqrt(pendulum.length / (gravity * 100))).toFixed(2)}s`, 20, 70)
  }, [pendulum, gravity])

  const drawCollisions = useCallback((ctx: CanvasRenderingContext2D) => {
    balls.forEach((ball, index) => {
      // Shadow
      ctx.fillStyle = 'rgba(0, 0, 0, 0.2)'
      ctx.beginPath()
      ctx.ellipse(ball.x, 450, ball.radius, ball.radius * 0.3, 0, 0, Math.PI * 2)
      ctx.fill()

      // Ball
      const gradient = ctx.createRadialGradient(
        ball.x - ball.radius * 0.3,
        ball.y - ball.radius * 0.3,
        ball.radius * 0.1,
        ball.x,
        ball.y,
        ball.radius
      )
      gradient.addColorStop(0, '#ffffff')
      gradient.addColorStop(1, ball.color)
      ctx.fillStyle = gradient
      ctx.beginPath()
      ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2)
      ctx.fill()

      // Outline
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)'
      ctx.lineWidth = 2
      ctx.stroke()

      // Velocity vector
      ctx.strokeStyle = '#22c55e'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(ball.x, ball.y)
      ctx.lineTo(ball.x + ball.vx * 0.5, ball.y + ball.vy * 0.5)
      ctx.stroke()

      // Info
      ctx.font = '12px Inter, sans-serif'
      ctx.fillStyle = '#e5e7eb'
      ctx.fillText(`Ball ${index + 1}`, ball.x - 20, ball.y - ball.radius - 10)

      const speed = Math.sqrt(ball.vx * ball.vx + ball.vy * ball.vy)
      ctx.fillText(`${speed.toFixed(0)} px/s`, ball.x - 20, ball.y - ball.radius - 25)
    })

    // Energy info
    const totalKE = balls.reduce((sum, ball) => {
      const speed = Math.sqrt(ball.vx * ball.vx + ball.vy * ball.vy)
      return sum + 0.5 * ball.mass * speed * speed
    }, 0)

    ctx.font = '14px Inter, sans-serif'
    ctx.fillStyle = '#e5e7eb'
    ctx.fillText(`Total Kinetic Energy: ${totalKE.toFixed(0)} J`, 20, 30)
  }, [balls])

  const drawProjectile = useCallback((ctx: CanvasRenderingContext2D) => {
    // Ground
    ctx.fillStyle = '#374151'
    ctx.fillRect(0, 400, 800, 100)

    // Trail
    if (projectile.trail.length > 1) {
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.5)'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(projectile.trail[0].x, projectile.trail[0].y)
      projectile.trail.forEach(point => {
        ctx.lineTo(point.x, point.y)
      })
      ctx.stroke()
    }

    // Projectile
    const gradient = ctx.createRadialGradient(
      projectile.x - 5,
      projectile.y - 5,
      2,
      projectile.x,
      projectile.y,
      15
    )
    gradient.addColorStop(0, '#fef08a')
    gradient.addColorStop(1, '#eab308')
    ctx.fillStyle = gradient
    ctx.beginPath()
    ctx.arc(projectile.x, projectile.y, 15, 0, Math.PI * 2)
    ctx.fill()

    // Velocity vector
    const vMag = Math.sqrt(projectile.vx * projectile.vx + projectile.vy * projectile.vy)
    const arrowScale = 0.3
    ctx.strokeStyle = '#ef4444'
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.moveTo(projectile.x, projectile.y)
    ctx.lineTo(projectile.x + projectile.vx * arrowScale, projectile.y + projectile.vy * arrowScale)
    ctx.stroke()

    // Info
    ctx.font = '14px Inter, sans-serif'
    ctx.fillStyle = '#e5e7eb'
    ctx.fillText(`Velocity: ${vMag.toFixed(0)} px/s`, 20, 30)
    ctx.fillText(`Height: ${(400 - projectile.y).toFixed(0)} px`, 20, 50)
    ctx.fillText(`Range: ${projectile.x.toFixed(0)} px`, 20, 70)

    // Calculate max range and height
    const angleRad = (projectile.angle * Math.PI) / 180
    const v0 = projectile.velocity
    const maxRange = (v0 * v0 * Math.sin(2 * angleRad)) / (gravity * 10)
    const maxHeight = (v0 * v0 * Math.sin(angleRad) * Math.sin(angleRad)) / (2 * gravity * 10)

    ctx.fillText(`Theoretical Range: ${maxRange.toFixed(0)} px`, 20, 90)
    ctx.fillText(`Max Height: ${maxHeight.toFixed(0)} px`, 20, 110)
  }, [projectile, gravity])

  const drawSprings = useCallback((ctx: CanvasRenderingContext2D) => {
    springs.forEach((spring, index) => {
      const x = 200 + index * 200
      const topY = 50

      // Ceiling
      ctx.fillStyle = '#6b7280'
      ctx.fillRect(x - 50, topY - 20, 100, 20)

      // Spring
      const springSegments = 20
      const segmentHeight = (spring.y - topY) / springSegments

      ctx.strokeStyle = '#a855f7'
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.moveTo(x, topY)

      for (let i = 1; i <= springSegments; i++) {
        const y = topY + i * segmentHeight
        const xOffset = (i % 2 === 0 ? 20 : -20)
        ctx.lineTo(x + xOffset, y)
      }
      ctx.stroke()

      // Mass
      const massSize = 40
      const gradient = ctx.createRadialGradient(
        x - 10,
        spring.y - 10,
        5,
        x,
        spring.y,
        massSize
      )
      gradient.addColorStop(0, '#fde047')
      gradient.addColorStop(1, '#facc15')
      ctx.fillStyle = gradient
      ctx.fillRect(x - massSize / 2, spring.y - massSize / 2, massSize, massSize)

      // Info
      ctx.font = '12px Inter, sans-serif'
      ctx.fillStyle = '#e5e7eb'
      ctx.fillText(`k = ${spring.k}`, x - 20, topY - 30)
      ctx.fillText(`y = ${(spring.y - spring.equilibrium).toFixed(0)}`, x - 20, spring.y + massSize)
    })
  }, [springs])

  // Main draw function
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw based on simulation type
    switch (simType) {
      case 'pendulum':
        drawPendulum(ctx, canvas.width, canvas.height)
        break
      case 'collision':
        drawCollisions(ctx)
        break
      case 'projectile':
        drawProjectile(ctx)
        break
      case 'springs':
        drawSprings(ctx)
        break
    }
  }, [simType, drawPendulum, drawCollisions, drawProjectile, drawSprings])

  // Animation loop
  useEffect(() => {
    if (!isRunning) return

    const animate = (currentTime: number) => {
      if (lastTimeRef.current === 0) {
        lastTimeRef.current = currentTime
      }

      const dt = Math.min((currentTime - lastTimeRef.current) / 1000, 0.02) // Cap at 20ms
      lastTimeRef.current = currentTime

      // Update physics
      switch (simType) {
        case 'pendulum':
          updatePendulum(dt)
          break
        case 'collision':
          updateCollisions(dt, 800, 500)
          break
        case 'projectile':
          updateProjectile(dt)
          break
        case 'springs':
          updateSprings(dt)
          break
      }

      draw()
      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning, simType, updatePendulum, updateCollisions, updateProjectile, updateSprings, draw])

  // Initial draw
  useEffect(() => {
    draw()
  }, [simType, draw])

  const handleStart = () => {
    setIsRunning(true)
    lastTimeRef.current = 0
  }

  const handlePause = () => {
    setIsRunning(false)
  }

  const handleReset = () => {
    setIsRunning(false)
    lastTimeRef.current = 0

    switch (simType) {
      case 'pendulum':
        setPendulum({
          angle: Math.PI / 4,
          angularVelocity: 0,
          length: 200,
          mass: 1
        })
        break
      case 'collision':
        setBalls([
          { x: 200, y: 200, vx: 50, vy: 30, radius: 30, mass: 1, color: '#3b82f6' },
          { x: 500, y: 300, vx: -30, vy: -20, radius: 40, mass: 1.5, color: '#ef4444' },
          { x: 350, y: 150, vx: 20, vy: 50, radius: 25, mass: 0.8, color: '#10b981' }
        ])
        break
      case 'projectile':
        const angleRad = (projectile.angle * Math.PI) / 180
        setProjectile({
          ...projectile,
          x: 50,
          y: 400,
          vx: projectile.velocity * Math.cos(angleRad),
          vy: -projectile.velocity * Math.sin(angleRad),
          trail: []
        })
        break
      case 'springs':
        setSprings([{ x: 400, y: 100, vy: 0, k: 0.1, equilibrium: 100 }])
        break
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-indigo-900 to-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Zap className="w-10 h-10 text-yellow-400" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
              Newton Mechanics Lab
            </h1>
          </div>
          <p className="text-slate-300 text-lg">
            고전 역학의 기본 원리를 인터랙티브하게 탐구하는 실험실
          </p>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Canvas */}
          <div className="lg:col-span-3 bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
            <canvas
              ref={canvasRef}
              width={800}
              height={500}
              className="w-full bg-slate-950 rounded-lg"
            />
          </div>

          {/* Controls */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 p-6 space-y-6">
            <div>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Settings className="w-5 h-5 text-blue-400" />
                Simulation Type
              </h2>
              <select
                value={simType}
                onChange={(e) => {
                  setSimType(e.target.value as SimulationType)
                  setIsRunning(false)
                }}
                className="w-full bg-slate-900/50 border border-slate-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="pendulum">Simple Pendulum</option>
                <option value="collision">Elastic Collisions</option>
                <option value="projectile">Projectile Motion</option>
                <option value="springs">Spring Oscillation</option>
              </select>
            </div>

            {/* Physics Parameters */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Wind className="w-4 h-4 text-green-400" />
                Physics Parameters
              </h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2 text-slate-300">
                    Gravity: {gravity.toFixed(2)} m/s²
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="20"
                    step="0.1"
                    value={gravity}
                    onChange={(e) => setGravity(Number(e.target.value))}
                    className="w-full accent-blue-500"
                  />
                </div>

                {(simType === 'collision' || simType === 'pendulum' || simType === 'springs') && (
                  <div>
                    <label className="block text-sm font-medium mb-2 text-slate-300">
                      Damping: {damping.toFixed(2)}
                    </label>
                    <input
                      type="range"
                      min="0.9"
                      max="1"
                      step="0.01"
                      value={damping}
                      onChange={(e) => setDamping(Number(e.target.value))}
                      className="w-full accent-green-500"
                    />
                  </div>
                )}

                {simType === 'collision' && (
                  <div>
                    <label className="block text-sm font-medium mb-2 text-slate-300">
                      Restitution: {restitution.toFixed(2)}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={restitution}
                      onChange={(e) => setRestitution(Number(e.target.value))}
                      className="w-full accent-purple-500"
                    />
                  </div>
                )}

                {simType === 'projectile' && (
                  <>
                    <div>
                      <label className="block text-sm font-medium mb-2 text-slate-300">
                        Angle: {projectile.angle}°
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="90"
                        step="5"
                        value={projectile.angle}
                        onChange={(e) => setProjectile(prev => ({ ...prev, angle: Number(e.target.value) }))}
                        className="w-full accent-yellow-500"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-2 text-slate-300">
                        Velocity: {projectile.velocity}
                      </label>
                      <input
                        type="range"
                        min="50"
                        max="400"
                        step="10"
                        value={projectile.velocity}
                        onChange={(e) => setProjectile(prev => ({ ...prev, velocity: Number(e.target.value) }))}
                        className="w-full accent-red-500"
                      />
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Control Buttons */}
            <div className="flex gap-2">
              {!isRunning ? (
                <button
                  onClick={handleStart}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  Start
                </button>
              ) : (
                <button
                  onClick={handlePause}
                  className="flex-1 bg-yellow-600 hover:bg-yellow-700 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  <Pause className="w-4 h-4" />
                  Pause
                </button>
              )}
              <button
                onClick={handleReset}
                className="bg-slate-700 hover:bg-slate-600 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>

            {/* Info */}
            <div className="bg-slate-900/50 rounded-lg p-4 text-sm text-slate-300 space-y-2">
              <h4 className="font-semibold text-white mb-2">
                {simType === 'pendulum' && 'Simple Pendulum'}
                {simType === 'collision' && 'Elastic Collisions'}
                {simType === 'projectile' && 'Projectile Motion'}
                {simType === 'springs' && 'Spring Oscillation'}
              </h4>
              <p>
                {simType === 'pendulum' && '중력과 길이에 따른 진자의 주기적 운동을 관찰합니다.'}
                {simType === 'collision' && '운동량 보존과 에너지 보존 법칙을 시각화합니다.'}
                {simType === 'projectile' && '포물선 운동과 최대 도달 거리를 계산합니다.'}
                {simType === 'springs' && '훅의 법칙(F = -kx)과 조화 진동을 시뮬레이션합니다.'}
              </p>
            </div>
          </div>
        </div>

        {/* Theory Panel */}
        <div className="mt-6 bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
          <h3 className="text-lg font-semibold mb-3 text-purple-400">Classic Mechanics Principles</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm text-slate-300">
            <div>
              <h4 className="font-semibold text-white mb-2">Newton's Laws</h4>
              <p>F = ma, 작용-반작용, 관성의 법칙</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">Energy Conservation</h4>
              <p>운동 에너지와 위치 에너지의 보존</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">Momentum Conservation</h4>
              <p>충돌 전후 운동량 합은 일정</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">Harmonic Motion</h4>
              <p>주기적 운동과 복원력</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
