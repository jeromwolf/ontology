'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Play, Pause, RotateCcw, Maximize, TrendingUp, Info, ArrowLeft } from 'lucide-react'
import Link from 'next/link'

interface Waypoint {
  position: number
  velocity: number
  time: number
}

type InterpolationType = 'linear' | 'cubic' | 'quintic' | 'spline'
type ProfileType = 'trapezoidal' | 's-curve'

export default function TrajectoryGenerator() {
  const [waypoints, setWaypoints] = useState<Waypoint[]>([
    { position: 0, velocity: 0, time: 0 },
    { position: 90, velocity: 0, time: 2 },
    { position: 45, velocity: 0, time: 4 },
    { position: 180, velocity: 0, time: 6 }
  ])

  const [interpolation, setInterpolation] = useState<InterpolationType>('cubic')
  const [profile, setProfile] = useState<ProfileType>('trapezoidal')
  const [maxVelocity, setMaxVelocity] = useState(100)
  const [maxAcceleration, setMaxAcceleration] = useState(80)
  const [jerkLimit, setJerkLimit] = useState(200)

  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const lastTimeRef = useRef<number>(0)

  // Generate trajectory points
  const generateTrajectory = useCallback((): {
    time: number[]
    position: number[]
    velocity: number[]
    acceleration: number[]
  } => {
    const totalTime = waypoints[waypoints.length - 1].time
    const dt = 0.02
    const steps = Math.ceil(totalTime / dt)

    const time: number[] = []
    const position: number[] = []
    const velocity: number[] = []
    const acceleration: number[] = []

    for (let i = 0; i <= steps; i++) {
      const t = (i * totalTime) / steps
      time.push(t)

      // Find segment
      let segmentIdx = 0
      for (let j = 0; j < waypoints.length - 1; j++) {
        if (t >= waypoints[j].time && t <= waypoints[j + 1].time) {
          segmentIdx = j
          break
        }
      }

      const w0 = waypoints[segmentIdx]
      const w1 = waypoints[segmentIdx + 1]
      const segmentDuration = w1.time - w0.time
      const tau = (t - w0.time) / segmentDuration // Normalized time [0, 1]

      let pos = 0
      let vel = 0
      let acc = 0

      if (interpolation === 'linear') {
        pos = w0.position + (w1.position - w0.position) * tau
        vel = (w1.position - w0.position) / segmentDuration
        acc = 0
      } else if (interpolation === 'cubic') {
        // Cubic polynomial: p(t) = a₀ + a₁t + a₂t² + a₃t³
        const p0 = w0.position
        const p1 = w1.position
        const v0 = w0.velocity
        const v1 = w1.velocity
        const T = segmentDuration

        const a0 = p0
        const a1 = v0
        const a2 = (3 * (p1 - p0) - (2 * v0 + v1) * T) / (T * T)
        const a3 = (2 * (p0 - p1) + (v0 + v1) * T) / (T * T * T)

        const tRel = tau * T
        pos = a0 + a1 * tRel + a2 * tRel * tRel + a3 * tRel * tRel * tRel
        vel = a1 + 2 * a2 * tRel + 3 * a3 * tRel * tRel
        acc = 2 * a2 + 6 * a3 * tRel
      } else if (interpolation === 'quintic') {
        // Quintic polynomial for smooth acceleration
        const p0 = w0.position
        const p1 = w1.position
        const v0 = w0.velocity
        const v1 = w1.velocity

        // Quintic coefficients (assuming zero initial/final acceleration)
        const a0 = p0
        const a1 = v0
        const a2 = 0
        const a3 = 10 * (p1 - p0) - 6 * v0 - 4 * v1
        const a4 = -15 * (p1 - p0) + 8 * v0 + 7 * v1
        const a5 = 6 * (p1 - p0) - 3 * v0 - 3 * v1

        pos = a0 + a1 * tau + a2 * tau ** 2 + a3 * tau ** 3 + a4 * tau ** 4 + a5 * tau ** 5
        vel =
          (a1 + 2 * a2 * tau + 3 * a3 * tau ** 2 + 4 * a4 * tau ** 3 + 5 * a5 * tau ** 4) /
          segmentDuration
        acc =
          (2 * a2 + 6 * a3 * tau + 12 * a4 * tau ** 2 + 20 * a5 * tau ** 3) /
          (segmentDuration * segmentDuration)
      } else if (interpolation === 'spline') {
        // Catmull-Rom spline (for smoothness across multiple waypoints)
        const getWaypoint = (idx: number) => {
          if (idx < 0) return waypoints[0]
          if (idx >= waypoints.length) return waypoints[waypoints.length - 1]
          return waypoints[idx]
        }

        const p_1 = getWaypoint(segmentIdx - 1).position
        const p0 = w0.position
        const p1 = w1.position
        const p2 = getWaypoint(segmentIdx + 2).position

        // Catmull-Rom basis
        const t2 = tau * tau
        const t3 = t2 * tau

        pos =
          0.5 *
          (2 * p0 +
            (-p_1 + p1) * tau +
            (2 * p_1 - 5 * p0 + 4 * p1 - p2) * t2 +
            (-p_1 + 3 * p0 - 3 * p1 + p2) * t3)

        vel =
          (0.5 *
            (-p_1 +
              p1 +
              2 * (2 * p_1 - 5 * p0 + 4 * p1 - p2) * tau +
              3 * (-p_1 + 3 * p0 - 3 * p1 + p2) * t2)) /
          segmentDuration

        acc =
          (0.5 * (2 * (2 * p_1 - 5 * p0 + 4 * p1 - p2) + 6 * (-p_1 + 3 * p0 - 3 * p1 + p2) * tau)) /
          (segmentDuration * segmentDuration)
      }

      position.push(pos)
      velocity.push(vel)
      acceleration.push(acc)
    }

    return { time, position, velocity, acceleration }
  }, [waypoints, interpolation])

  // Draw canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const { time, position, velocity, acceleration } = generateTrajectory()

    // Clear canvas
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    const chartHeight = canvas.height / 3
    const padding = 40

    // Helper to draw chart
    const drawChart = (
      yOffset: number,
      data: number[],
      color: string,
      label: string,
      unit: string
    ) => {
      // Background
      ctx.fillStyle = 'rgba(30, 41, 59, 0.5)'
      ctx.fillRect(0, yOffset, canvas.width, chartHeight)

      // Grid
      ctx.strokeStyle = 'rgba(71, 85, 105, 0.3)'
      ctx.lineWidth = 1
      for (let x = padding; x < canvas.width; x += 50) {
        ctx.beginPath()
        ctx.moveTo(x, yOffset)
        ctx.lineTo(x, yOffset + chartHeight)
        ctx.stroke()
      }
      for (let y = yOffset; y < yOffset + chartHeight; y += 40) {
        ctx.beginPath()
        ctx.moveTo(padding, y)
        ctx.lineTo(canvas.width, y)
        ctx.stroke()
      }

      // Axes
      ctx.strokeStyle = '#475569'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(padding, yOffset)
      ctx.lineTo(padding, yOffset + chartHeight)
      ctx.lineTo(canvas.width, yOffset + chartHeight)
      ctx.stroke()

      // Data
      const maxVal = Math.max(...data.map((v) => Math.abs(v)))
      const scale = (chartHeight - 20) / (2 * maxVal)

      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.beginPath()

      data.forEach((val, i) => {
        const x = padding + (i / data.length) * (canvas.width - padding)
        const y = yOffset + chartHeight / 2 - val * scale

        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })

      ctx.stroke()

      // Current time marker
      const currentIdx = Math.floor((currentTime / time[time.length - 1]) * data.length)
      if (currentIdx < data.length) {
        const x = padding + (currentIdx / data.length) * (canvas.width - padding)
        const y = yOffset + chartHeight / 2 - data[currentIdx] * scale

        ctx.fillStyle = color
        ctx.beginPath()
        ctx.arc(x, y, 6, 0, Math.PI * 2)
        ctx.fill()

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(x, yOffset)
        ctx.lineTo(x, yOffset + chartHeight)
        ctx.stroke()
      }

      // Label
      ctx.font = 'bold 14px Inter, sans-serif'
      ctx.fillStyle = color
      ctx.textAlign = 'left'
      ctx.fillText(label, 10, yOffset + 20)

      // Current value
      if (currentIdx < data.length) {
        ctx.font = '12px Inter, sans-serif'
        ctx.fillStyle = '#e5e7eb'
        ctx.textAlign = 'right'
        ctx.fillText(`${data[currentIdx].toFixed(2)} ${unit}`, canvas.width - 10, yOffset + 20)
      }

      // Zero line
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)'
      ctx.lineWidth = 1
      ctx.beginPath()
      const zeroY = yOffset + chartHeight / 2
      ctx.moveTo(padding, zeroY)
      ctx.lineTo(canvas.width, zeroY)
      ctx.stroke()
    }

    // Draw three charts
    drawChart(0, position, '#3b82f6', 'Position', '°')
    drawChart(chartHeight, velocity, '#10b981', 'Velocity', '°/s')
    drawChart(chartHeight * 2, acceleration, '#f59e0b', 'Acceleration', '°/s²')

    // Time axis label
    ctx.font = '12px Inter, sans-serif'
    ctx.fillStyle = '#94a3b8'
    ctx.textAlign = 'center'
    ctx.fillText('Time (s)', canvas.width / 2, canvas.height - 5)
    ctx.fillText(
      `t = ${currentTime.toFixed(2)}s / ${time[time.length - 1].toFixed(2)}s`,
      canvas.width / 2,
      20
    )
  }, [currentTime, generateTrajectory])

  // Animation loop
  useEffect(() => {
    if (!isPlaying) {
      draw()
      return
    }

    const animate = (timestamp: number) => {
      if (lastTimeRef.current === 0) {
        lastTimeRef.current = timestamp
      }

      const deltaTime = (timestamp - lastTimeRef.current) / 1000
      lastTimeRef.current = timestamp

      setCurrentTime((prev) => {
        const maxTime = waypoints[waypoints.length - 1].time
        const newTime = prev + deltaTime

        if (newTime >= maxTime) {
          setIsPlaying(false)
          return maxTime
        }

        return newTime
      })

      draw()
      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      lastTimeRef.current = 0
    }
  }, [isPlaying, draw, waypoints])

  // Initial draw
  useEffect(() => {
    draw()
  }, [draw])

  const handleReset = () => {
    setIsPlaying(false)
    setCurrentTime(0)
    lastTimeRef.current = 0
  }

  const handleAddWaypoint = () => {
    const lastWp = waypoints[waypoints.length - 1]
    setWaypoints([
      ...waypoints,
      { position: (lastWp.position + 45) % 360, velocity: 0, time: lastWp.time + 2 }
    ])
    handleReset()
  }

  const handleRemoveWaypoint = () => {
    if (waypoints.length > 2) {
      setWaypoints(waypoints.slice(0, -1))
      handleReset()
    }
  }

  const handleWaypointChange = (idx: number, field: keyof Waypoint, value: number) => {
    const newWaypoints = [...waypoints]
    newWaypoints[idx] = { ...newWaypoints[idx], [field]: value }
    setWaypoints(newWaypoints)
    handleReset()
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
              <TrendingUp className="w-10 h-10 text-blue-400" />
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-green-400 bg-clip-text text-transparent">
                Trajectory Generator
              </h1>
            </div>

            <Link
              href="/modules/robotics-manipulation/simulators/trajectory-generator"
              className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors"
              title="전체화면으로 보기"
            >
              <Maximize className="w-5 h-5 text-slate-400 hover:text-blue-400" />
            </Link>
          </div>
          <p className="text-slate-300 text-lg">
            웨이포인트 간 부드러운 궤적을 생성하고 위치, 속도, 가속도를 시각화합니다
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
                <TrendingUp className="w-5 h-5 text-blue-400" />
                Interpolation
              </h2>

              <div className="space-y-2">
                {[
                  { id: 'linear', name: 'Linear' },
                  { id: 'cubic', name: 'Cubic' },
                  { id: 'quintic', name: 'Quintic' },
                  { id: 'spline', name: 'Catmull-Rom Spline' }
                ].map((type) => (
                  <label
                    key={type.id}
                    className="flex items-center gap-3 cursor-pointer p-2 hover:bg-slate-700/50 rounded-lg"
                  >
                    <input
                      type="radio"
                      name="interpolation"
                      value={type.id}
                      checked={interpolation === type.id}
                      onChange={(e) => {
                        setInterpolation(e.target.value as InterpolationType)
                        handleReset()
                      }}
                      className="accent-blue-500"
                    />
                    <span className="text-sm text-slate-300">{type.name}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Controls */}
            <div className="space-y-3">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-semibold transition-colors ${
                  isPlaying
                    ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                }`}
              >
                {isPlaying ? (
                  <>
                    <Pause className="w-5 h-5" />
                    Pause
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Play
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

            {/* Waypoints */}
            <div>
              <h3 className="text-lg font-semibold mb-3">Waypoints ({waypoints.length})</h3>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {waypoints.map((wp, idx) => (
                  <div key={idx} className="bg-slate-900/50 rounded-lg p-2 text-xs">
                    <div className="font-semibold text-blue-400 mb-1">WP {idx}</div>
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="text-slate-400 w-12">Pos:</span>
                        <input
                          type="number"
                          value={wp.position}
                          onChange={(e) =>
                            handleWaypointChange(idx, 'position', Number(e.target.value))
                          }
                          disabled={idx === 0}
                          className="flex-1 bg-slate-800 px-2 py-1 rounded text-white disabled:opacity-50"
                        />
                        <span className="text-slate-500">°</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-slate-400 w-12">Time:</span>
                        <input
                          type="number"
                          value={wp.time}
                          onChange={(e) => handleWaypointChange(idx, 'time', Number(e.target.value))}
                          disabled={idx === 0}
                          step="0.5"
                          className="flex-1 bg-slate-800 px-2 py-1 rounded text-white disabled:opacity-50"
                        />
                        <span className="text-slate-500">s</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="flex gap-2 mt-3">
                <button
                  onClick={handleAddWaypoint}
                  className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors"
                >
                  + Add
                </button>
                <button
                  onClick={handleRemoveWaypoint}
                  disabled={waypoints.length <= 2}
                  className="flex-1 px-3 py-2 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition-colors disabled:opacity-50"
                >
                  - Remove
                </button>
              </div>
            </div>

            {/* Info */}
            <div className="bg-slate-900/50 rounded-lg p-4 text-sm text-slate-300">
              <div className="flex items-start gap-2 mb-2">
                <Info className="w-4 h-4 text-blue-400 mt-0.5" />
                <p className="font-semibold text-white">Trajectory Generation</p>
              </div>
              <p className="text-xs">
                여러 웨이포인트를 부드럽게 연결하는 궤적을 생성합니다.
                Linear는 직선 보간, Cubic/Quintic은 다항식 보간, Spline은 전역 부드러움을 보장합니다.
              </p>
            </div>
          </div>
        </div>

        {/* Theory Panel */}
        <div className="mt-6 bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
          <h3 className="text-lg font-semibold mb-3 text-blue-400">보간 방법 비교</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm text-slate-300">
            <div>
              <h4 className="font-semibold text-white mb-2">Linear</h4>
              <p>직선 보간. 속도 불연속, 가속도 무한대 (실제 로봇에는 부적합)</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">Cubic</h4>
              <p>3차 다항식. 위치와 속도 연속. 가속도는 불연속 가능</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">Quintic</h4>
              <p>5차 다항식. 위치, 속도, 가속도 모두 연속. 부드러운 움직임</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">Spline</h4>
              <p>Catmull-Rom 스플라인. 전역 부드러움, 모든 웨이포인트 통과</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
