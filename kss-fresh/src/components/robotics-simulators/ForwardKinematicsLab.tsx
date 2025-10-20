'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Play, Pause, RotateCcw, Settings, Info, Maximize, ArrowLeft } from 'lucide-react'
import Link from 'next/link'

interface Joint {
  angle: number // degrees
  length: number // pixels
  min: number
  max: number
}

export default function ForwardKinematicsLab() {
  const [joints, setJoints] = useState<Joint[]>([
    { angle: 0, length: 150, min: -180, max: 180 },
    { angle: 30, length: 120, min: -135, max: 135 },
    { angle: -45, length: 100, min: -120, max: 120 },
    { angle: 0, length: 80, min: -90, max: 90 }
  ])

  const [isAnimating, setIsAnimating] = useState(false)
  const [showTrace, setShowTrace] = useState(true)
  const [tracePoints, setTracePoints] = useState<{ x: number; y: number }[]>([])
  const [endEffectorPos, setEndEffectorPos] = useState({ x: 0, y: 0 })

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()

  // Calculate forward kinematics
  const calculateFK = useCallback((jointAngles: Joint[]) => {
    const baseX = 400
    const baseY = 400
    let x = baseX
    let y = baseY
    let cumulativeAngle = 0

    const positions: { x: number; y: number }[] = [{ x, y }]

    jointAngles.forEach((joint) => {
      cumulativeAngle += joint.angle
      const radians = (cumulativeAngle * Math.PI) / 180
      x += joint.length * Math.cos(radians)
      y -= joint.length * Math.sin(radians) // Y축 반전 (Canvas 좌표계)
      positions.push({ x, y })
    })

    return positions
  }, [])

  // Draw robot arm
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw grid
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

    // Draw trace
    if (showTrace && tracePoints.length > 1) {
      ctx.strokeStyle = 'rgba(249, 115, 22, 0.4)'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(tracePoints[0].x, tracePoints[0].y)
      tracePoints.forEach((point) => {
        ctx.lineTo(point.x, point.y)
      })
      ctx.stroke()
    }

    // Calculate positions
    const positions = calculateFK(joints)

    // Draw links
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 8
    for (let i = 0; i < positions.length - 1; i++) {
      const start = positions[i]
      const end = positions[i + 1]

      // Link
      ctx.beginPath()
      ctx.moveTo(start.x, start.y)
      ctx.lineTo(end.x, end.y)
      ctx.stroke()

      // Joint circle
      ctx.fillStyle = '#1e40af'
      ctx.beginPath()
      ctx.arc(start.x, start.y, 12, 0, Math.PI * 2)
      ctx.fill()

      ctx.strokeStyle = '#60a5fa'
      ctx.lineWidth = 2
      ctx.stroke()
    }

    // End effector
    const endPos = positions[positions.length - 1]
    const gradient = ctx.createRadialGradient(endPos.x, endPos.y, 5, endPos.x, endPos.y, 20)
    gradient.addColorStop(0, '#fbbf24')
    gradient.addColorStop(1, '#f59e0b')
    ctx.fillStyle = gradient
    ctx.beginPath()
    ctx.arc(endPos.x, endPos.y, 20, 0, Math.PI * 2)
    ctx.fill()

    ctx.strokeStyle = '#fef3c7'
    ctx.lineWidth = 3
    ctx.stroke()

    // Update end effector position
    setEndEffectorPos({ x: endPos.x, y: endPos.y })

    // Draw coordinates
    ctx.font = 'bold 14px Inter, sans-serif'
    ctx.fillStyle = '#e5e7eb'
    ctx.fillText(`End Effector: (${(endPos.x - 400).toFixed(0)}, ${(400 - endPos.y).toFixed(0)})`, 20, 30)

    // Draw joint info
    let yOffset = 60
    joints.forEach((joint, i) => {
      ctx.font = '12px Inter, sans-serif'
      ctx.fillStyle = '#94a3b8'
      ctx.fillText(`Joint ${i + 1}: ${joint.angle.toFixed(1)}° | Length: ${joint.length}px`, 20, yOffset)
      yOffset += 20
    })
  }, [joints, showTrace, tracePoints, calculateFK])

  // Animation loop
  useEffect(() => {
    if (!isAnimating) {
      draw()
      return
    }

    let animationTime = 0

    const animate = () => {
      animationTime += 0.02

      setJoints((prevJoints) =>
        prevJoints.map((joint, idx) => ({
          ...joint,
          angle: 30 * Math.sin(animationTime * (idx + 1) * 0.5)
        }))
      )

      draw()
      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isAnimating, draw])

  // Update trace
  useEffect(() => {
    if (showTrace && isAnimating) {
      setTracePoints((prev) => {
        const newPoints = [...prev, endEffectorPos]
        if (newPoints.length > 200) {
          return newPoints.slice(-200)
        }
        return newPoints
      })
    }
  }, [endEffectorPos, showTrace, isAnimating])

  // Initial draw
  useEffect(() => {
    draw()
  }, [draw])

  const handleJointChange = (index: number, value: number) => {
    setJoints((prevJoints) =>
      prevJoints.map((joint, idx) => (idx === index ? { ...joint, angle: value } : joint))
    )
  }

  const handleReset = () => {
    setIsAnimating(false)
    setJoints([
      { angle: 0, length: 150, min: -180, max: 180 },
      { angle: 30, length: 120, min: -135, max: 135 },
      { angle: -45, length: 100, min: -120, max: 120 },
      { angle: 0, length: 80, min: -90, max: 90 }
    ])
    setTracePoints([])
  }

  const handleClearTrace = () => {
    setTracePoints([])
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-8">
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
              <Settings className="w-10 h-10 text-blue-400" />
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                Forward Kinematics Lab
              </h1>
            </div>

            <Link
              href="/modules/robotics-manipulation/simulators/forward-kinematics-lab"
              className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors"
              title="전체화면으로 보기"
            >
              <Maximize className="w-5 h-5 text-slate-400 hover:text-blue-400" />
            </Link>
          </div>
          <p className="text-slate-300 text-lg">
            관절 각도를 조절하여 로봇 팔의 순기구학을 실시간으로 시각화합니다
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
              className="w-full bg-slate-950 rounded-lg"
            />
          </div>

          {/* Controls */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 p-6 space-y-6">
            <div>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Settings className="w-5 h-5 text-blue-400" />
                Joint Controls
              </h2>

              <div className="space-y-4">
                {joints.map((joint, index) => (
                  <div key={index}>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm font-medium text-slate-300">
                        Joint {index + 1}
                      </label>
                      <span className="text-sm text-blue-400 font-mono">
                        {joint.angle.toFixed(1)}°
                      </span>
                    </div>
                    <input
                      type="range"
                      min={joint.min}
                      max={joint.max}
                      step="1"
                      value={joint.angle}
                      onChange={(e) => handleJointChange(index, Number(e.target.value))}
                      disabled={isAnimating}
                      className="w-full accent-blue-500"
                    />
                    <div className="flex justify-between text-xs text-slate-400 mt-1">
                      <span>{joint.min}°</span>
                      <span>{joint.max}°</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Animation Controls */}
            <div className="space-y-3">
              <button
                onClick={() => setIsAnimating(!isAnimating)}
                className={`w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-semibold transition-colors ${
                  isAnimating
                    ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                }`}
              >
                {isAnimating ? (
                  <>
                    <Pause className="w-5 h-5" />
                    Pause Animation
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Start Animation
                  </>
                )}
              </button>

              <button
                onClick={handleReset}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-semibold transition-colors"
              >
                <RotateCcw className="w-5 h-5" />
                Reset
              </button>
            </div>

            {/* Options */}
            <div>
              <h3 className="text-lg font-semibold mb-3">Options</h3>
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showTrace}
                  onChange={(e) => {
                    setShowTrace(e.target.checked)
                    if (!e.target.checked) {
                      setTracePoints([])
                    }
                  }}
                  className="w-5 h-5 accent-blue-500"
                />
                <span className="text-sm text-slate-300">Show End Effector Trace</span>
              </label>

              {showTrace && (
                <button
                  onClick={handleClearTrace}
                  className="mt-3 w-full px-3 py-2 bg-slate-700 hover:bg-slate-600 text-white text-sm rounded-lg transition-colors"
                >
                  Clear Trace
                </button>
              )}
            </div>

            {/* Info */}
            <div className="bg-slate-900/50 rounded-lg p-4 text-sm text-slate-300">
              <div className="flex items-start gap-2 mb-2">
                <Info className="w-4 h-4 text-blue-400 mt-0.5" />
                <p className="font-semibold text-white">Forward Kinematics</p>
              </div>
              <p className="text-xs">
                관절 각도(θ₁, θ₂, θ₃, θ₄)를 입력하여 엔드이펙터의 위치(x, y)를 계산합니다.
                각 링크의 변환 행렬을 순차적으로 곱하여 최종 위치를 도출합니다.
              </p>
            </div>

            {/* Formula */}
            <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4 text-xs font-mono">
              <p className="text-blue-300 mb-2">Transformation:</p>
              <p className="text-slate-300">
                T = T₁ × T₂ × T₃ × T₄
              </p>
              <p className="text-slate-400 mt-2 text-[10px]">
                각 Tᵢ는 회전(θᵢ)과 이동(Lᵢ)을 포함하는 4×4 동차 변환 행렬
              </p>
            </div>
          </div>
        </div>

        {/* Theory Panel */}
        <div className="mt-6 bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
          <h3 className="text-lg font-semibold mb-3 text-blue-400">순기구학(Forward Kinematics) 이론</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-slate-300">
            <div>
              <h4 className="font-semibold text-white mb-2">입력</h4>
              <p>관절 각도 θ₁, θ₂, ..., θₙ과 링크 길이 L₁, L₂, ..., Lₙ</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">출력</h4>
              <p>엔드이펙터의 위치(x, y, z)와 방향(roll, pitch, yaw)</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">방법</h4>
              <p>DH 파라미터 기반 동차 변환 행렬의 순차적 곱셈</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
