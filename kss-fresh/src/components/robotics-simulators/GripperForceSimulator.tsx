'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Play, Pause, RotateCcw, Maximize, Hand, Info, Zap, ArrowLeft } from 'lucide-react'
import Link from 'next/link'

interface ObjectProperties {
  mass: number // kg
  friction: number // coefficient
  fragility: number // 0-1 (0 = sturdy, 1 = fragile)
  shape: 'box' | 'cylinder' | 'sphere'
}

type GripType = 'parallel' | 'suction' | 'three-finger'

export default function GripperForceSimulator() {
  const [gripType, setGripType] = useState<GripType>('parallel')
  const [object, setObject] = useState<ObjectProperties>({
    mass: 1.5,
    friction: 0.6,
    fragility: 0.3,
    shape: 'box'
  })

  const [gripForce, setGripForce] = useState(30) // Newtons
  const [gripWidth, setGripWidth] = useState(80) // mm
  const [isGripping, setIsGripping] = useState(false)
  const [acceleration, setAcceleration] = useState(2.0) // m/s²

  const [forceBalance, setForceBalance] = useState<{
    normal: number
    friction: number
    gravity: number
    required: number
    safety: number
  }>({ normal: 0, friction: 0, gravity: 0, required: 0, safety: 0 })

  const [status, setStatus] = useState<'safe' | 'warning' | 'slip' | 'crush'>('safe')

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()

  // Calculate forces
  const calculateForces = useCallback(() => {
    const gravity = object.mass * 9.81 // Weight in Newtons
    const inertialForce = object.mass * acceleration // F = ma

    let normalForce = 0
    let frictionForce = 0
    let requiredForce = 0

    if (gripType === 'parallel') {
      // Two-finger parallel gripper
      normalForce = gripForce / 2 // Force per finger
      frictionForce = normalForce * object.friction * 2 // Total friction from both fingers
      requiredForce = gravity + inertialForce
    } else if (gripType === 'suction') {
      // Suction gripper (vacuum)
      normalForce = gripForce // Vacuum force
      frictionForce = normalForce // Suction provides direct holding force
      requiredForce = gravity + inertialForce
    } else if (gripType === 'three-finger') {
      // Three-finger gripper (more contact points)
      normalForce = gripForce / 3 // Force per finger
      frictionForce = normalForce * object.friction * 3 // Total friction
      requiredForce = gravity + inertialForce
    }

    const safetyFactor = frictionForce / requiredForce

    // Fragility check (excessive force causes crushing)
    const maxSafeForce = (1 - object.fragility) * 100 // Max force before damage
    const crushRisk = gripForce > maxSafeForce

    let newStatus: 'safe' | 'warning' | 'slip' | 'crush' = 'safe'
    if (crushRisk) {
      newStatus = 'crush'
    } else if (safetyFactor < 1.0) {
      newStatus = 'slip'
    } else if (safetyFactor < 1.5) {
      newStatus = 'warning'
    }

    setForceBalance({
      normal: normalForce,
      friction: frictionForce,
      gravity: gravity,
      required: requiredForce,
      safety: safetyFactor
    })

    setStatus(newStatus)
  }, [gripType, object, gripForce, acceleration])

  useEffect(() => {
    calculateForces()
  }, [calculateForces])

  // Draw canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Grid
    ctx.strokeStyle = 'rgba(71, 85, 105, 0.2)'
    ctx.lineWidth = 1
    for (let x = 0; x < canvas.width; x += 50) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, canvas.height)
      ctx.stroke()
    }
    for (let y = 0; y < canvas.height; y += 50) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(canvas.width, y)
      ctx.stroke()
    }

    const centerX = 400
    const centerY = 300

    // Draw object
    const objectWidth = 100
    const objectHeight = 120

    let statusColor = '#10b981' // green
    if (status === 'warning') statusColor = '#f59e0b' // orange
    if (status === 'slip') statusColor = '#ef4444' // red
    if (status === 'crush') statusColor = '#dc2626' // dark red

    if (object.shape === 'box') {
      ctx.fillStyle = 'rgba(59, 130, 246, 0.3)'
      ctx.strokeStyle = statusColor
      ctx.lineWidth = 3
      ctx.fillRect(
        centerX - objectWidth / 2,
        centerY - objectHeight / 2,
        objectWidth,
        objectHeight
      )
      ctx.strokeRect(
        centerX - objectWidth / 2,
        centerY - objectHeight / 2,
        objectWidth,
        objectHeight
      )
    } else if (object.shape === 'cylinder') {
      ctx.fillStyle = 'rgba(59, 130, 246, 0.3)'
      ctx.strokeStyle = statusColor
      ctx.lineWidth = 3

      // Top ellipse
      ctx.beginPath()
      ctx.ellipse(centerX, centerY - objectHeight / 2, objectWidth / 2, 20, 0, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()

      // Body
      ctx.fillRect(
        centerX - objectWidth / 2,
        centerY - objectHeight / 2,
        objectWidth,
        objectHeight
      )
      ctx.strokeRect(
        centerX - objectWidth / 2,
        centerY - objectHeight / 2,
        objectWidth,
        objectHeight
      )

      // Bottom ellipse
      ctx.beginPath()
      ctx.ellipse(centerX, centerY + objectHeight / 2, objectWidth / 2, 20, 0, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
    } else if (object.shape === 'sphere') {
      ctx.fillStyle = 'rgba(59, 130, 246, 0.3)'
      ctx.strokeStyle = statusColor
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.arc(centerX, centerY, objectWidth / 2, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
    }

    // Draw gripper
    if (gripType === 'parallel') {
      const fingerWidth = 30
      const fingerHeight = 150
      const gap = gripWidth

      // Left finger
      ctx.fillStyle = '#64748b'
      ctx.fillRect(centerX - gap / 2 - fingerWidth, centerY - fingerHeight / 2, fingerWidth, fingerHeight)

      // Right finger
      ctx.fillRect(centerX + gap / 2, centerY - fingerHeight / 2, fingerWidth, fingerHeight)

      // Contact points
      if (isGripping && gap <= objectWidth + 10) {
        ctx.fillStyle = statusColor
        ctx.beginPath()
        ctx.arc(centerX - objectWidth / 2, centerY, 8, 0, Math.PI * 2)
        ctx.fill()
        ctx.beginPath()
        ctx.arc(centerX + objectWidth / 2, centerY, 8, 0, Math.PI * 2)
        ctx.fill()
      }

      // Force arrows (normal force)
      if (isGripping && gap <= objectWidth + 10) {
        const arrowLength = (gripForce / 100) * 80
        drawArrow(ctx, centerX - gap / 2, centerY, centerX - gap / 2 + arrowLength, centerY, '#f59e0b')
        drawArrow(ctx, centerX + gap / 2, centerY, centerX + gap / 2 - arrowLength, centerY, '#f59e0b')

        // Labels
        ctx.font = '12px Inter, sans-serif'
        ctx.fillStyle = '#f59e0b'
        ctx.textAlign = 'center'
        ctx.fillText(`${(gripForce / 2).toFixed(1)}N`, centerX - gap / 2 - 40, centerY - 10)
        ctx.fillText(`${(gripForce / 2).toFixed(1)}N`, centerX + gap / 2 + 40, centerY - 10)
      }
    } else if (gripType === 'suction') {
      // Suction cup
      const cupRadius = 50
      ctx.fillStyle = '#64748b'
      ctx.beginPath()
      ctx.arc(centerX, centerY - objectHeight / 2 - 20, cupRadius, 0, Math.PI)
      ctx.fill()

      // Vacuum indicator
      if (isGripping) {
        ctx.fillStyle = statusColor
        ctx.beginPath()
        ctx.arc(centerX, centerY - objectHeight / 2, 10, 0, Math.PI * 2)
        ctx.fill()

        // Suction force arrow
        const arrowLength = (gripForce / 100) * 80
        drawArrow(
          ctx,
          centerX,
          centerY - objectHeight / 2 - 30,
          centerX,
          centerY - objectHeight / 2 - 30 + arrowLength,
          '#f59e0b'
        )

        ctx.font = '12px Inter, sans-serif'
        ctx.fillStyle = '#f59e0b'
        ctx.textAlign = 'center'
        ctx.fillText(`${gripForce.toFixed(1)}N`, centerX + 50, centerY - objectHeight / 2 - 25)
      }
    } else if (gripType === 'three-finger') {
      // Three fingers at 120° apart
      const fingerLength = 60
      const angles = [0, (2 * Math.PI) / 3, (4 * Math.PI) / 3]

      angles.forEach((angle, idx) => {
        const x = centerX + Math.cos(angle) * (objectWidth / 2 + 30)
        const y = centerY + Math.sin(angle) * (objectWidth / 2 + 30)

        ctx.fillStyle = '#64748b'
        ctx.beginPath()
        ctx.arc(x, y, 15, 0, Math.PI * 2)
        ctx.fill()

        if (isGripping) {
          // Contact point
          const contactX = centerX + Math.cos(angle) * (objectWidth / 2)
          const contactY = centerY + Math.sin(angle) * (objectWidth / 2)

          ctx.fillStyle = statusColor
          ctx.beginPath()
          ctx.arc(contactX, contactY, 6, 0, Math.PI * 2)
          ctx.fill()

          // Force arrow
          const arrowLength = (gripForce / 100) * 40
          drawArrow(
            ctx,
            x,
            y,
            x - Math.cos(angle) * arrowLength,
            y - Math.sin(angle) * arrowLength,
            '#f59e0b'
          )
        }
      })
    }

    // Draw gravity vector
    const gravityArrowLength = (forceBalance.gravity / 50) * 60
    drawArrow(
      ctx,
      centerX,
      centerY + objectHeight / 2 + 20,
      centerX,
      centerY + objectHeight / 2 + 20 + gravityArrowLength,
      '#ef4444'
    )

    ctx.font = '12px Inter, sans-serif'
    ctx.fillStyle = '#ef4444'
    ctx.textAlign = 'center'
    ctx.fillText(`Gravity: ${forceBalance.gravity.toFixed(1)}N`, centerX, centerY + objectHeight / 2 + gravityArrowLength + 40)

    // Status indicator
    ctx.font = 'bold 18px Inter, sans-serif'
    ctx.fillStyle = statusColor
    ctx.textAlign = 'left'
    const statusText = {
      safe: '✓ Safe Grip',
      warning: '⚠ Low Safety Margin',
      slip: '✗ Object Slipping!',
      crush: '✗ Crushing Object!'
    }
    ctx.fillText(statusText[status], 20, 30)

    // Force balance info
    ctx.font = '14px Inter, sans-serif'
    ctx.fillStyle = '#e5e7eb'
    let yOffset = 60
    ctx.fillText(`Friction Force: ${forceBalance.friction.toFixed(2)} N`, 20, yOffset)
    yOffset += 25
    ctx.fillText(`Required Force: ${forceBalance.required.toFixed(2)} N`, 20, yOffset)
    yOffset += 25
    ctx.fillText(`Safety Factor: ${forceBalance.safety.toFixed(2)}x`, 20, yOffset)
  }, [object, gripType, gripForce, gripWidth, isGripping, status, forceBalance])

  // Helper to draw arrows
  const drawArrow = (
    ctx: CanvasRenderingContext2D,
    fromX: number,
    fromY: number,
    toX: number,
    toY: number,
    color: string
  ) => {
    const headLength = 15
    const angle = Math.atan2(toY - fromY, toX - fromX)

    ctx.strokeStyle = color
    ctx.fillStyle = color
    ctx.lineWidth = 3

    // Line
    ctx.beginPath()
    ctx.moveTo(fromX, fromY)
    ctx.lineTo(toX, toY)
    ctx.stroke()

    // Arrowhead
    ctx.beginPath()
    ctx.moveTo(toX, toY)
    ctx.lineTo(
      toX - headLength * Math.cos(angle - Math.PI / 6),
      toY - headLength * Math.sin(angle - Math.PI / 6)
    )
    ctx.lineTo(
      toX - headLength * Math.cos(angle + Math.PI / 6),
      toY - headLength * Math.sin(angle + Math.PI / 6)
    )
    ctx.closePath()
    ctx.fill()
  }

  // Animation loop
  useEffect(() => {
    const animate = () => {
      draw()
      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [draw])

  const handleReset = () => {
    setIsGripping(false)
    setGripForce(30)
    setGripWidth(80)
    setAcceleration(2.0)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-between mb-4">
            <Link
              href="/modules/robotics-manipulation"
              className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700"
            >
              <ArrowLeft className="w-4 h-4 text-slate-400" />
              <span className="text-slate-300 text-sm">모듈로 돌아가기</span>
            </Link>

            <div className="flex items-center gap-3">
              <Hand className="w-10 h-10 text-purple-400" />
              <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                Gripper Force Simulator
              </h1>
            </div>

            <Link
              href="/modules/robotics-manipulation/simulators/gripper-force-simulator"
              className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors"
              title="전체화면으로 보기"
            >
              <Maximize className="w-5 h-5 text-slate-400 hover:text-purple-400" />
            </Link>
          </div>
          <p className="text-slate-300 text-lg">
            그리퍼 파지력과 마찰력을 계산하고 물체의 미끄러짐 및 파손을 시뮬레이션합니다
          </p>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Canvas */}
          <div className="lg:col-span-3 bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
            <canvas ref={canvasRef} width={800} height={600} className="w-full bg-slate-950 rounded-lg" />
          </div>

          {/* Controls */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 p-6 space-y-6">
            <div>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Hand className="w-5 h-5 text-purple-400" />
                Gripper Type
              </h2>

              <div className="space-y-2">
                {[
                  { id: 'parallel', name: 'Parallel Jaw' },
                  { id: 'suction', name: 'Suction Cup' },
                  { id: 'three-finger', name: 'Three-Finger' }
                ].map((type) => (
                  <label
                    key={type.id}
                    className="flex items-center gap-3 cursor-pointer p-2 hover:bg-slate-700/50 rounded-lg"
                  >
                    <input
                      type="radio"
                      name="gripType"
                      value={type.id}
                      checked={gripType === type.id}
                      onChange={(e) => setGripType(e.target.value as GripType)}
                      className="accent-purple-500"
                    />
                    <span className="text-sm text-slate-300">{type.name}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Object Properties */}
            <div>
              <h3 className="text-lg font-semibold mb-3">Object Properties</h3>
              <div className="space-y-3">
                <div>
                  <label className="text-sm text-slate-300 mb-1 block">
                    Mass: {object.mass.toFixed(1)} kg
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="5"
                    step="0.1"
                    value={object.mass}
                    onChange={(e) => setObject({ ...object, mass: Number(e.target.value) })}
                    className="w-full accent-purple-500"
                  />
                </div>

                <div>
                  <label className="text-sm text-slate-300 mb-1 block">
                    Friction: {object.friction.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={object.friction}
                    onChange={(e) => setObject({ ...object, friction: Number(e.target.value) })}
                    className="w-full accent-purple-500"
                  />
                </div>

                <div>
                  <label className="text-sm text-slate-300 mb-1 block">
                    Fragility: {object.fragility.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={object.fragility}
                    onChange={(e) => setObject({ ...object, fragility: Number(e.target.value) })}
                    className="w-full accent-purple-500"
                  />
                </div>

                <div>
                  <label className="text-sm text-slate-300 mb-2 block">Shape</label>
                  <div className="flex gap-2">
                    {['box', 'cylinder', 'sphere'].map((shape) => (
                      <button
                        key={shape}
                        onClick={() => setObject({ ...object, shape: shape as any })}
                        className={`flex-1 px-3 py-2 rounded-lg text-xs font-semibold transition-colors ${
                          object.shape === shape
                            ? 'bg-purple-600 text-white'
                            : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                        }`}
                      >
                        {shape}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Gripper Controls */}
            <div>
              <h3 className="text-lg font-semibold mb-3">Gripper Controls</h3>
              <div className="space-y-3">
                <div>
                  <label className="text-sm text-slate-300 mb-1 block">
                    Grip Force: {gripForce.toFixed(1)} N
                  </label>
                  <input
                    type="range"
                    min="5"
                    max="100"
                    step="1"
                    value={gripForce}
                    onChange={(e) => setGripForce(Number(e.target.value))}
                    className="w-full accent-purple-500"
                  />
                </div>

                {gripType === 'parallel' && (
                  <div>
                    <label className="text-sm text-slate-300 mb-1 block">
                      Grip Width: {gripWidth.toFixed(0)} mm
                    </label>
                    <input
                      type="range"
                      min="60"
                      max="150"
                      step="5"
                      value={gripWidth}
                      onChange={(e) => setGripWidth(Number(e.target.value))}
                      className="w-full accent-purple-500"
                    />
                  </div>
                )}

                <div>
                  <label className="text-sm text-slate-300 mb-1 block">
                    Acceleration: {acceleration.toFixed(1)} m/s²
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="10"
                    step="0.5"
                    value={acceleration}
                    onChange={(e) => setAcceleration(Number(e.target.value))}
                    className="w-full accent-purple-500"
                  />
                </div>
              </div>
            </div>

            {/* Controls */}
            <div className="space-y-3">
              <button
                onClick={() => setIsGripping(!isGripping)}
                className={`w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-semibold transition-colors ${
                  isGripping
                    ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
                    : 'bg-purple-600 hover:bg-purple-700 text-white'
                }`}
              >
                <Hand className="w-5 h-5" />
                {isGripping ? 'Release' : 'Grip Object'}
              </button>

              <button
                onClick={handleReset}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-semibold transition-colors"
              >
                <RotateCcw className="w-5 h-5" />
                Reset
              </button>
            </div>

            {/* Info */}
            <div className="bg-slate-900/50 rounded-lg p-4 text-sm text-slate-300">
              <div className="flex items-start gap-2 mb-2">
                <Info className="w-4 h-4 text-purple-400 mt-0.5" />
                <p className="font-semibold text-white">Force Analysis</p>
              </div>
              <p className="text-xs">
                그리퍼가 물체를 안전하게 파지하려면 마찰력이 중력과 관성력의 합보다 커야 합니다.
                안전계수(Safety Factor)가 1.5 이상이면 안전합니다.
              </p>
            </div>
          </div>
        </div>

        {/* Theory Panel */}
        <div className="mt-6 bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
          <h3 className="text-lg font-semibold mb-3 text-purple-400">그리퍼 역학 이론</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-slate-300">
            <div>
              <h4 className="font-semibold text-white mb-2">마찰력</h4>
              <p>F_friction = μ × N (μ: 마찰계수, N: 법선력). 파지를 유지하는 핵심 힘</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">필요 조건</h4>
              <p>F_friction ≥ F_gravity + F_inertial. 미끄러지지 않으려면 마찰력이 중력과 관성력보다 커야 함</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">안전계수</h4>
              <p>SF = F_friction / F_required. 1.5 이상 권장. 너무 크면 물체 파손 위험</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
