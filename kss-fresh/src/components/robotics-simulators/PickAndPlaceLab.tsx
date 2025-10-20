'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Play, Pause, RotateCcw, Maximize, Package, Info, CheckCircle, XCircle, ArrowLeft } from 'lucide-react'
import Link from 'next/link'

interface Position {
  x: number
  y: number
}

interface Box {
  id: number
  position: Position
  color: string
  picked: boolean
  placed: boolean
}

type TaskPhase =
  | 'idle'
  | 'moving-to-pick'
  | 'gripping'
  | 'lifting'
  | 'moving-to-place'
  | 'releasing'
  | 'returning'
  | 'complete'

export default function PickAndPlaceLab() {
  const [boxes, setBoxes] = useState<Box[]>([
    { id: 1, position: { x: 150, y: 450 }, color: '#ef4444', picked: false, placed: false },
    { id: 2, position: { x: 250, y: 450 }, color: '#3b82f6', picked: false, placed: false },
    { id: 3, position: { x: 350, y: 450 }, color: '#10b981', picked: false, placed: false }
  ])

  const [targetPositions] = useState<Position[]>([
    { x: 550, y: 200 },
    { x: 650, y: 200 },
    { x: 750, y: 200 }
  ])

  const [gripperPos, setGripperPos] = useState<Position>({ x: 400, y: 100 })
  const [gripperOpen, setGripperOpen] = useState(true)
  const [heldBox, setHeldBox] = useState<Box | null>(null)

  const [currentTask, setCurrentTask] = useState(0)
  const [phase, setPhase] = useState<TaskPhase>('idle')
  const [isRunning, setIsRunning] = useState(false)

  const [stats, setStats] = useState({
    completed: 0,
    failed: 0,
    totalTime: 0
  })

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const phaseTimerRef = useRef<number>(0)

  // Execute pick-and-place task
  const executeTask = useCallback(() => {
    if (currentTask >= boxes.length) {
      setPhase('complete')
      setIsRunning(false)
      return
    }

    const box = boxes[currentTask]
    const target = targetPositions[currentTask]

    phaseTimerRef.current += 0.016 // ~60fps

    switch (phase) {
      case 'idle':
        setPhase('moving-to-pick')
        phaseTimerRef.current = 0
        break

      case 'moving-to-pick':
        // Move to box position
        const dx1 = box.position.x - gripperPos.x
        const dy1 = box.position.y - 80 - gripperPos.y

        if (Math.abs(dx1) < 2 && Math.abs(dy1) < 2) {
          setPhase('gripping')
          phaseTimerRef.current = 0
        } else {
          setGripperPos({
            x: gripperPos.x + dx1 * 0.05,
            y: gripperPos.y + dy1 * 0.05
          })
        }
        break

      case 'gripping':
        if (phaseTimerRef.current > 0.5) {
          setGripperOpen(false)
          setHeldBox(box)
          setBoxes((prev) =>
            prev.map((b) => (b.id === box.id ? { ...b, picked: true } : b))
          )
          setPhase('lifting')
          phaseTimerRef.current = 0
        }
        break

      case 'lifting':
        if (phaseTimerRef.current > 0.3) {
          setGripperPos({ ...gripperPos, y: 150 })
          setPhase('moving-to-place')
          phaseTimerRef.current = 0
        } else {
          setGripperPos({ ...gripperPos, y: gripperPos.y - 3 })
        }
        break

      case 'moving-to-place':
        // Move to target position
        const dx2 = target.x - gripperPos.x
        const dy2 = target.y - 80 - gripperPos.y

        if (Math.abs(dx2) < 2 && Math.abs(dy2) < 2) {
          setPhase('releasing')
          phaseTimerRef.current = 0
        } else {
          setGripperPos({
            x: gripperPos.x + dx2 * 0.05,
            y: gripperPos.y + dy2 * 0.05
          })
        }
        break

      case 'releasing':
        if (phaseTimerRef.current > 0.5) {
          setGripperOpen(true)
          if (heldBox) {
            setBoxes((prev) =>
              prev.map((b) =>
                b.id === heldBox.id
                  ? { ...b, position: target, placed: true }
                  : b
              )
            )
            setStats((prev) => ({
              ...prev,
              completed: prev.completed + 1,
              totalTime: prev.totalTime + phaseTimerRef.current
            }))
          }
          setHeldBox(null)
          setPhase('returning')
          phaseTimerRef.current = 0
        }
        break

      case 'returning':
        const dx3 = 400 - gripperPos.x
        const dy3 = 100 - gripperPos.y

        if (Math.abs(dx3) < 2 && Math.abs(dy3) < 2) {
          setCurrentTask((prev) => prev + 1)
          setPhase('idle')
          phaseTimerRef.current = 0
        } else {
          setGripperPos({
            x: gripperPos.x + dx3 * 0.05,
            y: gripperPos.y + dy3 * 0.05
          })
        }
        break

      case 'complete':
        setIsRunning(false)
        break
    }
  }, [phase, currentTask, boxes, targetPositions, gripperPos, heldBox])

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

    // Workbench
    ctx.fillStyle = 'rgba(100, 116, 139, 0.3)'
    ctx.fillRect(50, 420, 400, 150)
    ctx.strokeStyle = '#64748b'
    ctx.lineWidth = 2
    ctx.strokeRect(50, 420, 400, 150)

    // Target area
    ctx.fillStyle = 'rgba(16, 185, 129, 0.1)'
    ctx.fillRect(500, 150, 300, 100)
    ctx.strokeStyle = '#10b981'
    ctx.lineWidth = 2
    ctx.setLineDash([10, 5])
    ctx.strokeRect(500, 150, 300, 100)
    ctx.setLineDash([])

    ctx.font = '14px Inter, sans-serif'
    ctx.fillStyle = '#10b981'
    ctx.textAlign = 'center'
    ctx.fillText('TARGET ZONE', 650, 140)

    // Target position markers
    targetPositions.forEach((pos, idx) => {
      ctx.strokeStyle = '#94a3b8'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 3])
      ctx.strokeRect(pos.x - 25, pos.y - 25, 50, 50)
      ctx.setLineDash([])

      ctx.font = '12px Inter, sans-serif'
      ctx.fillStyle = '#94a3b8'
      ctx.textAlign = 'center'
      ctx.fillText(`${idx + 1}`, pos.x, pos.y - 35)
    })

    // Boxes
    boxes.forEach((box) => {
      if (box.picked && !box.placed && heldBox?.id === box.id) {
        // Box is held by gripper
        const boxPos = {
          x: gripperPos.x,
          y: gripperPos.y + 50
        }
        ctx.fillStyle = box.color
        ctx.fillRect(boxPos.x - 25, boxPos.y - 25, 50, 50)
        ctx.strokeStyle = '#fff'
        ctx.lineWidth = 2
        ctx.strokeRect(boxPos.x - 25, boxPos.y - 25, 50, 50)
      } else if (!box.picked || box.placed) {
        // Box on surface
        ctx.fillStyle = box.color
        ctx.fillRect(box.position.x - 25, box.position.y - 25, 50, 50)
        ctx.strokeStyle = box.placed ? '#10b981' : '#fff'
        ctx.lineWidth = box.placed ? 3 : 2
        ctx.strokeRect(box.position.x - 25, box.position.y - 25, 50, 50)

        if (box.placed) {
          ctx.fillStyle = '#10b981'
          ctx.font = 'bold 24px Inter, sans-serif'
          ctx.textAlign = 'center'
          ctx.fillText('✓', box.position.x, box.position.y + 8)
        }
      }
    })

    // Gripper arm (simple representation)
    ctx.strokeStyle = '#94a3b8'
    ctx.lineWidth = 4
    ctx.beginPath()
    ctx.moveTo(gripperPos.x, 0)
    ctx.lineTo(gripperPos.x, gripperPos.y)
    ctx.stroke()

    // Gripper
    const gripperWidth = gripperOpen ? 60 : 30
    ctx.fillStyle = '#475569'
    ctx.fillRect(gripperPos.x - gripperWidth / 2 - 5, gripperPos.y, 5, 40)
    ctx.fillRect(gripperPos.x + gripperWidth / 2, gripperPos.y, 5, 40)

    // Gripper status indicator
    ctx.fillStyle = gripperOpen ? '#ef4444' : '#10b981'
    ctx.beginPath()
    ctx.arc(gripperPos.x, gripperPos.y + 20, 8, 0, Math.PI * 2)
    ctx.fill()

    // Phase indicator
    ctx.font = 'bold 16px Inter, sans-serif'
    ctx.fillStyle = '#e5e7eb'
    ctx.textAlign = 'left'
    ctx.fillText(`Phase: ${phase.replace(/-/g, ' ').toUpperCase()}`, 20, 30)

    ctx.font = '14px Inter, sans-serif'
    ctx.fillText(`Task: ${currentTask + 1} / ${boxes.length}`, 20, 55)

    // Stats
    ctx.fillStyle = '#10b981'
    ctx.fillText(`✓ Completed: ${stats.completed}`, 20, 85)

    if (phase === 'complete') {
      ctx.font = 'bold 32px Inter, sans-serif'
      ctx.fillStyle = '#10b981'
      ctx.textAlign = 'center'
      ctx.fillText('ALL TASKS COMPLETED!', 400, 300)
    }
  }, [boxes, targetPositions, gripperPos, gripperOpen, heldBox, phase, currentTask, stats])

  // Animation loop
  useEffect(() => {
    if (!isRunning) {
      draw()
      return
    }

    const animate = () => {
      executeTask()
      draw()

      if (isRunning) {
        animationRef.current = requestAnimationFrame(animate)
      }
    }

    animationRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning, draw, executeTask])

  // Initial draw
  useEffect(() => {
    draw()
  }, [draw])

  const handleReset = () => {
    setIsRunning(false)
    setBoxes([
      { id: 1, position: { x: 150, y: 450 }, color: '#ef4444', picked: false, placed: false },
      { id: 2, position: { x: 250, y: 450 }, color: '#3b82f6', picked: false, placed: false },
      { id: 3, position: { x: 350, y: 450 }, color: '#10b981', picked: false, placed: false }
    ])
    setGripperPos({ x: 400, y: 100 })
    setGripperOpen(true)
    setHeldBox(null)
    setCurrentTask(0)
    setPhase('idle')
    phaseTimerRef.current = 0
    setStats({ completed: 0, failed: 0, totalTime: 0 })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-green-900 to-slate-900 text-white p-8">
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
              <Package className="w-10 h-10 text-green-400" />
              <h1 className="text-4xl font-bold bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent">
                Pick-and-Place Lab
              </h1>
            </div>

            <Link
              href="/modules/robotics-manipulation/simulators/pick-and-place-lab"
              className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors"
              title="전체화면으로 보기"
            >
              <Maximize className="w-5 h-5 text-slate-400 hover:text-green-400" />
            </Link>
          </div>
          <p className="text-slate-300 text-lg">
            완전한 픽앤플레이스 작업을 시뮬레이션하고 각 단계를 시각화합니다
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
                <Package className="w-5 h-5 text-green-400" />
                Task Status
              </h2>

              <div className="space-y-3">
                <div className="bg-slate-900/50 rounded-lg p-3">
                  <div className="text-sm text-slate-400 mb-2">Current Phase</div>
                  <div className="text-lg font-semibold text-white">
                    {phase.replace(/-/g, ' ').toUpperCase()}
                  </div>
                </div>

                <div className="bg-slate-900/50 rounded-lg p-3">
                  <div className="text-sm text-slate-400 mb-2">Progress</div>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 bg-slate-700 rounded-full h-3 overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-green-500 to-blue-500 h-full transition-all"
                        style={{ width: `${(stats.completed / boxes.length) * 100}%` }}
                      />
                    </div>
                    <span className="text-sm font-mono text-white">
                      {stats.completed}/{boxes.length}
                    </span>
                  </div>
                </div>

                <div className="bg-slate-900/50 rounded-lg p-3">
                  <div className="text-sm text-slate-400 mb-2">Gripper State</div>
                  <div className="flex items-center gap-2">
                    {gripperOpen ? (
                      <>
                        <XCircle className="w-5 h-5 text-red-400" />
                        <span className="text-white">Open</span>
                      </>
                    ) : (
                      <>
                        <CheckCircle className="w-5 h-5 text-green-400" />
                        <span className="text-white">Closed</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Task Sequence */}
            <div>
              <h3 className="text-lg font-semibold mb-3">Task Sequence</h3>
              <div className="space-y-2 text-xs">
                {[
                  'Move to Pick',
                  'Grip Object',
                  'Lift Up',
                  'Move to Place',
                  'Release Object',
                  'Return Home'
                ].map((step, idx) => {
                  const phaseMap = [
                    'moving-to-pick',
                    'gripping',
                    'lifting',
                    'moving-to-place',
                    'releasing',
                    'returning'
                  ]
                  const isActive = phase === phaseMap[idx]
                  const isPast = phaseMap.indexOf(phase) > idx

                  return (
                    <div
                      key={idx}
                      className={`flex items-center gap-2 p-2 rounded-lg ${
                        isActive
                          ? 'bg-green-900/30 border border-green-500'
                          : isPast
                          ? 'bg-slate-900/30 border border-slate-600'
                          : 'bg-slate-900/10 border border-slate-700'
                      }`}
                    >
                      {isPast || isActive ? (
                        <CheckCircle className="w-4 h-4 text-green-400" />
                      ) : (
                        <div className="w-4 h-4 border-2 border-slate-600 rounded-full" />
                      )}
                      <span className={isActive ? 'text-white font-semibold' : 'text-slate-400'}>
                        {idx + 1}. {step}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Controls */}
            <div className="space-y-3">
              <button
                onClick={() => setIsRunning(!isRunning)}
                disabled={phase === 'complete'}
                className={`w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-semibold transition-colors ${
                  isRunning
                    ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
                    : 'bg-green-600 hover:bg-green-700 text-white disabled:opacity-50'
                }`}
              >
                {isRunning ? (
                  <>
                    <Pause className="w-5 h-5" />
                    Pause
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Start
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

            {/* Info */}
            <div className="bg-slate-900/50 rounded-lg p-4 text-sm text-slate-300">
              <div className="flex items-start gap-2 mb-2">
                <Info className="w-4 h-4 text-green-400 mt-0.5" />
                <p className="font-semibold text-white">Pick-and-Place</p>
              </div>
              <p className="text-xs">
                완전한 픽앤플레이스 작업은 위치 제어, 경로 계획, 파지 제어를 통합합니다.
                각 박스를 작업대에서 목표 구역으로 이동시킵니다.
              </p>
            </div>

            {/* Stats */}
            <div className="bg-green-900/20 border border-green-700 rounded-lg p-4 text-xs font-mono">
              <p className="text-green-300 mb-2">Statistics:</p>
              <p className="text-slate-300">Completed: {stats.completed}</p>
              <p className="text-slate-300">Failed: {stats.failed}</p>
              {phase === 'complete' && (
                <p className="text-green-400 mt-2 font-bold">SUCCESS! All tasks done.</p>
              )}
            </div>
          </div>
        </div>

        {/* Theory Panel */}
        <div className="mt-6 bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
          <h3 className="text-lg font-semibold mb-3 text-green-400">픽앤플레이스 작업 구성 요소</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-slate-300">
            <div>
              <h4 className="font-semibold text-white mb-2">인식 (Perception)</h4>
              <p>카메라/센서로 물체 위치와 자세를 감지. 비전 시스템 필수</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">계획 (Planning)</h4>
              <p>충돌 없는 경로 생성. IK로 관절 각도 계산, 궤적 생성</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">제어 (Control)</h4>
              <p>경로 추종 제어, 파지력 제어, 안전 모니터링</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
