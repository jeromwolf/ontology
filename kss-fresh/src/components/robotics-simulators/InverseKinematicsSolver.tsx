'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Target, RotateCcw, Maximize, Info, ArrowLeft } from 'lucide-react'
import Link from 'next/link'

interface Joint {
  angle: number
  length: number
}

export default function InverseKinematicsSolver() {
  const [joints, setJoints] = useState<Joint[]>([
    { angle: 0, length: 150 },
    { angle: 0, length: 120 },
    { angle: 0, length: 100 }
  ])

  const [targetPos, setTargetPos] = useState({ x: 300, y: 200 })
  const [reachable, setReachable] = useState(true)
  const [isDragging, setIsDragging] = useState(false)
  const [solverMethod, setSolverMethod] = useState<'jacobian' | 'analytical'>('analytical')
  const [iterations, setIterations] = useState(0)

  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Analytical IK for 3-link planar arm
  const solveIKAnalytical = useCallback((target: { x: number; y: number }) => {
    const L1 = joints[0].length
    const L2 = joints[1].length
    const L3 = joints[2].length

    // Target relative to base
    const tx = target.x - 400
    const ty = 400 - target.y

    // Check if target is reachable
    const totalLength = L1 + L2 + L3
    const distance = Math.sqrt(tx * tx + ty * ty)

    if (distance > totalLength || distance < Math.abs(L1 - L2 - L3)) {
      setReachable(false)
      return
    }

    setReachable(true)

    // Simplified 2-link IK (treat last link as fixed orientation)
    const targetDist = Math.sqrt(tx * tx + ty * ty) - L3
    const targetAngle = Math.atan2(ty, tx)

    // Law of cosines for 2-link arm
    const cosTheta2 = (targetDist * targetDist - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    const theta2 = Math.acos(Math.max(-1, Math.min(1, cosTheta2)))

    const k1 = L1 + L2 * Math.cos(theta2)
    const k2 = L2 * Math.sin(theta2)
    const theta1 = targetAngle - Math.atan2(k2, k1)

    // Convert to degrees
    const angle1 = (theta1 * 180) / Math.PI
    const angle2 = (theta2 * 180) / Math.PI
    const angle3 = 0 // Keep end-effector horizontal

    setJoints([
      { ...joints[0], angle: angle1 },
      { ...joints[1], angle: angle2 },
      { ...joints[2], angle: angle3 }
    ])
  }, [joints])

  // Jacobian-based IK (iterative)
  const solveIKJacobian = useCallback((target: { x: number; y: number }) => {
    let currentJoints = [...joints]
    const maxIterations = 50
    const threshold = 2.0

    for (let iter = 0; iter < maxIterations; iter++) {
      // Forward kinematics to get current end-effector position
      let x = 400
      let y = 400
      let angle = 0

      currentJoints.forEach((joint) => {
        angle += (joint.angle * Math.PI) / 180
        x += joint.length * Math.cos(angle)
        y -= joint.length * Math.sin(angle)
      })

      // Error
      const dx = target.x - x
      const dy = target.y - y
      const error = Math.sqrt(dx * dx + dy * dy)

      if (error < threshold) {
        setIterations(iter + 1)
        break
      }

      // Compute Jacobian (simplified for 3-DOF planar arm)
      const J = []
      let cumAngle = 0
      let cumX = 400
      let cumY = 400

      for (let i = 0; i < currentJoints.length; i++) {
        cumAngle += (currentJoints[i].angle * Math.PI) / 180

        // Position of this joint
        const jx = cumX
        const jy = cumY

        cumX += currentJoints[i].length * Math.cos(cumAngle)
        cumY -= currentJoints[i].length * Math.sin(cumAngle)

        // Jacobian column: derivative of end-effector pos w.r.t this joint
        // For revolute joint: J = [-sin(θ)*(ex-jx) + cos(θ)*(ey-jy), ...]
        const toEndX = x - jx
        const toEndY = y - jy

        J.push({
          dx: -toEndY,
          dy: toEndX
        })
      }

      // Pseudo-inverse and update (simplified)
      const alpha = 0.5 // Step size
      const deltaE = { x: dx, y: dy }

      currentJoints = currentJoints.map((joint, i) => {
        const dTheta = (J[i].dx * deltaE.x + J[i].dy * deltaE.y) * alpha * 0.01
        return {
          ...joint,
          angle: joint.angle + (dTheta * 180) / Math.PI
        }
      })
    }

    setJoints(currentJoints)
    setReachable(true)
  }, [joints])

  // Draw visualization
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Grid
    ctx.strokeStyle = 'rgba(71, 85, 105, 0.3)'
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

    // Workspace circle (reachable area)
    const totalLength = joints.reduce((sum, j) => sum + j.length, 0)
    ctx.strokeStyle = 'rgba(34, 197, 94, 0.2)'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.arc(400, 400, totalLength, 0, Math.PI * 2)
    ctx.stroke()

    // Draw robot arm
    let x = 400
    let y = 400
    let cumulativeAngle = 0

    const positions = [{ x, y }]

    joints.forEach((joint) => {
      cumulativeAngle += (joint.angle * Math.PI) / 180
      x += joint.length * Math.cos(cumulativeAngle)
      y -= joint.length * Math.sin(cumulativeAngle)
      positions.push({ x, y })
    })

    // Links
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 8
    for (let i = 0; i < positions.length - 1; i++) {
      ctx.beginPath()
      ctx.moveTo(positions[i].x, positions[i].y)
      ctx.lineTo(positions[i + 1].x, positions[i + 1].y)
      ctx.stroke()
    }

    // Joints
    positions.forEach((pos, i) => {
      ctx.fillStyle = i === 0 ? '#ef4444' : '#60a5fa'
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, i === 0 ? 8 : 10, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = '#1e293b'
      ctx.lineWidth = 2
      ctx.stroke()
    })

    // End-effector
    const endPos = positions[positions.length - 1]
    ctx.fillStyle = '#22c55e'
    ctx.beginPath()
    ctx.arc(endPos.x, endPos.y, 12, 0, Math.PI * 2)
    ctx.fill()
    ctx.strokeStyle = '#16a34a'
    ctx.lineWidth = 3
    ctx.stroke()

    // Target
    ctx.fillStyle = reachable ? 'rgba(249, 115, 22, 0.3)' : 'rgba(239, 68, 68, 0.3)'
    ctx.strokeStyle = reachable ? '#f97316' : '#ef4444'
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.arc(targetPos.x, targetPos.y, 15, 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()

    // Crosshair on target
    ctx.beginPath()
    ctx.moveTo(targetPos.x - 20, targetPos.y)
    ctx.lineTo(targetPos.x + 20, targetPos.y)
    ctx.moveTo(targetPos.x, targetPos.y - 20)
    ctx.lineTo(targetPos.x, targetPos.y + 20)
    ctx.stroke()

    // Distance line
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.5)'
    ctx.lineWidth = 2
    ctx.setLineDash([5, 5])
    ctx.beginPath()
    ctx.moveTo(endPos.x, endPos.y)
    ctx.lineTo(targetPos.x, targetPos.y)
    ctx.stroke()
    ctx.setLineDash([])

    // Error distance text
    const errorDist = Math.sqrt(
      Math.pow(targetPos.x - endPos.x, 2) + Math.pow(targetPos.y - endPos.y, 2)
    )
    ctx.fillStyle = '#94a3b8'
    ctx.font = '14px monospace'
    ctx.fillText(
      `Error: ${errorDist.toFixed(1)}px`,
      (endPos.x + targetPos.x) / 2 + 10,
      (endPos.y + targetPos.y) / 2
    )
  }, [joints, targetPos, reachable])

  useEffect(() => {
    draw()
  }, [draw])

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height

    setTargetPos({ x, y })
  }

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height

    const dist = Math.sqrt(Math.pow(x - targetPos.x, 2) + Math.pow(y - targetPos.y, 2))
    if (dist < 20) {
      setIsDragging(true)
    }
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging) return

    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height

    setTargetPos({ x, y })
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const handleSolve = () => {
    if (solverMethod === 'analytical') {
      solveIKAnalytical(targetPos)
    } else {
      solveIKJacobian(targetPos)
    }
  }

  const handleReset = () => {
    setJoints([
      { angle: 0, length: 150 },
      { angle: 0, length: 120 },
      { angle: 0, length: 100 }
    ])
    setTargetPos({ x: 300, y: 200 })
    setReachable(true)
    setIterations(0)
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
              <Target className="w-10 h-10 text-purple-400" />
              <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                Inverse Kinematics Solver
              </h1>
            </div>

            <Link
              href="/modules/robotics-manipulation/simulators/inverse-kinematics-solver"
              className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors"
              title="전체화면으로 보기"
            >
              <Maximize className="w-5 h-5 text-slate-400 hover:text-purple-400" />
            </Link>
          </div>
          <p className="text-slate-300 text-lg">
            목표 위치를 설정하고 관절 각도를 자동으로 계산합니다
          </p>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Canvas */}
          <div className="lg:col-span-3 bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
            <canvas
              ref={canvasRef}
              width={800}
              height={600}
              className="w-full bg-slate-950 rounded-lg cursor-crosshair"
              onClick={handleCanvasClick}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            />
            <p className="text-sm text-slate-400 mt-3 text-center">
              💡 클릭하거나 드래그하여 목표 위치 설정
            </p>
          </div>

          {/* Controls */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 p-6 space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Target className="w-5 h-5 text-purple-400" />
                Solver Method
              </h3>
              <select
                value={solverMethod}
                onChange={(e) => setSolverMethod(e.target.value as 'jacobian' | 'analytical')}
                className="w-full px-3 py-2 bg-slate-700 rounded-lg border border-slate-600 text-white"
              >
                <option value="analytical">Analytical (2-link)</option>
                <option value="jacobian">Jacobian (Iterative)</option>
              </select>
            </div>

            <button
              onClick={handleSolve}
              className="w-full px-4 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
            >
              <Target className="w-5 h-5" />
              Solve IK
            </button>

            <button
              onClick={handleReset}
              className="w-full px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <RotateCcw className="w-4 h-4" />
              Reset
            </button>

            <div className="pt-4 border-t border-slate-700">
              <h3 className="text-sm font-semibold mb-3 text-slate-300">Status</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400">Reachable:</span>
                  <span className={reachable ? 'text-green-400' : 'text-red-400'}>
                    {reachable ? '✓ Yes' : '✗ No'}
                  </span>
                </div>
                {solverMethod === 'jacobian' && (
                  <div className="flex justify-between">
                    <span className="text-slate-400">Iterations:</span>
                    <span className="text-blue-400">{iterations}</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-slate-400">Target:</span>
                  <span className="text-orange-400">
                    ({targetPos.x.toFixed(0)}, {targetPos.y.toFixed(0)})
                  </span>
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-slate-700">
              <h3 className="text-sm font-semibold mb-3 text-slate-300 flex items-center gap-2">
                <Info className="w-4 h-4" />
                Joint Angles
              </h3>
              <div className="space-y-2 text-sm font-mono">
                {joints.map((joint, i) => (
                  <div key={i} className="flex justify-between">
                    <span className="text-slate-400">θ{i + 1}:</span>
                    <span className="text-blue-400">{joint.angle.toFixed(1)}°</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="pt-4 border-t border-slate-700">
              <div className="bg-purple-900/30 rounded-lg p-4 border border-purple-700/50">
                <h4 className="text-sm font-semibold mb-2 text-purple-300">💡 Inverse Kinematics</h4>
                <p className="text-xs text-slate-300 leading-relaxed">
                  목표 위치(x, y)가 주어졌을 때 각 관절의 각도(θ₁, θ₂, θ₃)를 계산합니다.
                  해석적 방법은 빠르지만 2-link에만 적용되고, Jacobian 방법은 반복적으로 수렴합니다.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
