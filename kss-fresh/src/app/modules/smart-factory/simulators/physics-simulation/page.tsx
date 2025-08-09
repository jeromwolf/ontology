'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'
import { ArrowLeft, Play, Pause, RotateCcw, Zap, Box, CircuitBoard, Waves, Gauge, Settings, Download, Upload, Sparkles } from 'lucide-react'

interface PhysicsObject {
  id: string
  type: 'box' | 'cylinder' | 'conveyor' | 'robotic-arm'
  x: number
  y: number
  z: number
  vx: number
  vy: number
  vz: number
  mass: number
  friction: number
  color: string
  rotation: number
}

interface SimulationSettings {
  gravity: number
  airResistance: number
  timeScale: number
  collisionDamping: number
}

interface Force {
  id: string
  type: 'push' | 'magnetic' | 'pneumatic' | 'gravity'
  x: number
  y: number
  magnitude: number
  angle: number
}

export default function PhysicsSimulationPage() {
  const searchParams = useSearchParams()
  const backUrl = searchParams.get('from') || '/modules/smart-factory'
  const [isRunning, setIsRunning] = useState(false)
  const [selectedScenario, setSelectedScenario] = useState('conveyor')
  const [showForceVectors, setShowForceVectors] = useState(true)
  const [showTrajectory, setShowTrajectory] = useState(true)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  
  const [objects, setObjects] = useState<PhysicsObject[]>([
    {
      id: 'box1',
      type: 'box',
      x: 100,
      y: 200,
      z: 0,
      vx: 50,
      vy: 0,
      vz: 0,
      mass: 5,
      friction: 0.3,
      color: '#3B82F6',
      rotation: 0
    }
  ])

  const [settings, setSettings] = useState<SimulationSettings>({
    gravity: 9.81,
    airResistance: 0.02,
    timeScale: 1.0,
    collisionDamping: 0.8
  })

  const [forces, setForces] = useState<Force[]>([])
  const [trajectory, setTrajectory] = useState<{x: number, y: number}[]>([])

  const scenarios = [
    {
      id: 'conveyor',
      name: '컨베이어 벨트 물리',
      description: '제품 이송 시 마찰력과 관성',
      icon: '📦'
    },
    {
      id: 'robotic-pick',
      name: '로봇 피킹 동역학',
      description: '로봇 암의 토크와 가속도',
      icon: '🤖'
    },
    {
      id: 'collision',
      name: '충돌 시뮬레이션',
      description: '제품 간 충돌과 에너지 손실',
      icon: '💥'
    },
    {
      id: 'vibration',
      name: '진동 분석',
      description: '장비 진동과 공진 현상',
      icon: '〰️'
    }
  ]

  // 물리 시뮬레이션 엔진
  useEffect(() => {
    if (!canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = 800
    canvas.height = 500

    let lastTime = 0
    const dt = 0.016 // 60 FPS

    const updatePhysics = (deltaTime: number) => {
      if (!isRunning) return

      setObjects(prev => prev.map(obj => {
        let newObj = { ...obj }
        
        // 중력 적용
        newObj.vy += settings.gravity * deltaTime * 10

        // 공기 저항
        newObj.vx *= (1 - settings.airResistance)
        newObj.vy *= (1 - settings.airResistance)

        // 속도에 따른 위치 업데이트
        newObj.x += newObj.vx * deltaTime * settings.timeScale
        newObj.y += newObj.vy * deltaTime * settings.timeScale

        // 바닥 충돌
        if (newObj.y > canvas.height - 50) {
          newObj.y = canvas.height - 50
          newObj.vy = -newObj.vy * settings.collisionDamping
          
          // 마찰력
          newObj.vx *= (1 - obj.friction)
        }

        // 벽 충돌
        if (newObj.x < 50 || newObj.x > canvas.width - 50) {
          newObj.vx = -newObj.vx * settings.collisionDamping
          newObj.x = newObj.x < 50 ? 50 : canvas.width - 50
        }

        // 회전
        if (obj.type === 'box' || obj.type === 'cylinder') {
          newObj.rotation += (newObj.vx / 50) * deltaTime
        }

        return newObj
      }))

      // 궤적 기록
      if (objects.length > 0 && showTrajectory) {
        setTrajectory(prev => [...prev.slice(-100), { x: objects[0].x, y: objects[0].y }])
      }
    }

    const render = (currentTime: number) => {
      const deltaTime = (currentTime - lastTime) / 1000
      lastTime = currentTime

      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // 배경 그리드
      ctx.strokeStyle = '#E5E7EB'
      ctx.lineWidth = 1
      for (let i = 0; i < canvas.width; i += 50) {
        ctx.beginPath()
        ctx.moveTo(i, 0)
        ctx.lineTo(i, canvas.height)
        ctx.stroke()
      }
      for (let i = 0; i < canvas.height; i += 50) {
        ctx.beginPath()
        ctx.moveTo(0, i)
        ctx.lineTo(canvas.width, i)
        ctx.stroke()
      }

      // 바닥
      ctx.fillStyle = '#6B7280'
      ctx.fillRect(0, canvas.height - 40, canvas.width, 40)

      // 궤적 그리기
      if (showTrajectory && trajectory.length > 1) {
        ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)'
        ctx.lineWidth = 2
        ctx.beginPath()
        trajectory.forEach((point, index) => {
          if (index === 0) ctx.moveTo(point.x, point.y)
          else ctx.lineTo(point.x, point.y)
        })
        ctx.stroke()
      }

      // 오브젝트 렌더링
      objects.forEach(obj => {
        ctx.save()
        ctx.translate(obj.x, obj.y)
        ctx.rotate(obj.rotation)

        if (obj.type === 'box') {
          ctx.fillStyle = obj.color
          ctx.fillRect(-25, -25, 50, 50)
          ctx.strokeStyle = '#000'
          ctx.lineWidth = 2
          ctx.strokeRect(-25, -25, 50, 50)
        } else if (obj.type === 'cylinder') {
          ctx.fillStyle = obj.color
          ctx.beginPath()
          ctx.arc(0, 0, 25, 0, Math.PI * 2)
          ctx.fill()
          ctx.stroke()
        }

        ctx.restore()

        // 힘 벡터 표시
        if (showForceVectors) {
          // 속도 벡터
          ctx.strokeStyle = '#10B981'
          ctx.lineWidth = 3
          ctx.beginPath()
          ctx.moveTo(obj.x, obj.y)
          ctx.lineTo(obj.x + obj.vx * 0.5, obj.y + obj.vy * 0.5)
          ctx.stroke()

          // 화살표
          const angle = Math.atan2(obj.vy, obj.vx)
          ctx.save()
          ctx.translate(obj.x + obj.vx * 0.5, obj.y + obj.vy * 0.5)
          ctx.rotate(angle)
          ctx.beginPath()
          ctx.moveTo(0, 0)
          ctx.lineTo(-10, -5)
          ctx.lineTo(-10, 5)
          ctx.closePath()
          ctx.fillStyle = '#10B981'
          ctx.fill()
          ctx.restore()

          // 중력 벡터
          ctx.strokeStyle = '#EF4444'
          ctx.lineWidth = 2
          ctx.beginPath()
          ctx.moveTo(obj.x, obj.y)
          ctx.lineTo(obj.x, obj.y + settings.gravity * 5)
          ctx.stroke()
        }
      })

      // 정보 표시
      ctx.fillStyle = '#000'
      ctx.font = '14px Arial'
      ctx.fillText(`중력: ${settings.gravity.toFixed(1)} m/s²`, 10, 20)
      ctx.fillText(`시간 배속: ${settings.timeScale.toFixed(1)}x`, 10, 40)
      
      if (objects.length > 0) {
        const obj = objects[0]
        ctx.fillText(`속도: ${Math.sqrt(obj.vx * obj.vx + obj.vy * obj.vy).toFixed(1)} m/s`, 10, 60)
        ctx.fillText(`위치: (${obj.x.toFixed(0)}, ${obj.y.toFixed(0)})`, 10, 80)
      }

      updatePhysics(deltaTime)
      animationRef.current = requestAnimationFrame(render)
    }

    render(0)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning, objects, settings, showForceVectors, showTrajectory, trajectory])

  const loadScenario = (scenarioId: string) => {
    setSelectedScenario(scenarioId)
    setTrajectory([])

    switch (scenarioId) {
      case 'conveyor':
        setObjects([
          { id: 'box1', type: 'box', x: 100, y: 200, z: 0, vx: 100, vy: 0, vz: 0, mass: 5, friction: 0.3, color: '#3B82F6', rotation: 0 },
          { id: 'box2', type: 'box', x: 200, y: 200, z: 0, vx: 80, vy: 0, vz: 0, mass: 3, friction: 0.3, color: '#10B981', rotation: 0 },
          { id: 'box3', type: 'box', x: 300, y: 200, z: 0, vx: 60, vy: 0, vz: 0, mass: 7, friction: 0.3, color: '#F59E0B', rotation: 0 }
        ])
        setSettings(prev => ({ ...prev, gravity: 9.81, airResistance: 0.01 }))
        break

      case 'robotic-pick':
        setObjects([
          { id: 'arm', type: 'cylinder', x: 400, y: 100, z: 0, vx: 0, vy: 50, vz: 0, mass: 10, friction: 0.1, color: '#8B5CF6', rotation: 0 },
          { id: 'object', type: 'box', x: 400, y: 300, z: 0, vx: 0, vy: 0, vz: 0, mass: 2, friction: 0.5, color: '#EF4444', rotation: 0 }
        ])
        setSettings(prev => ({ ...prev, gravity: 9.81, airResistance: 0.02 }))
        break

      case 'collision':
        setObjects([
          { id: 'ball1', type: 'cylinder', x: 100, y: 250, z: 0, vx: 150, vy: 0, vz: 0, mass: 5, friction: 0.1, color: '#3B82F6', rotation: 0 },
          { id: 'ball2', type: 'cylinder', x: 700, y: 250, z: 0, vx: -100, vy: 0, vz: 0, mass: 3, friction: 0.1, color: '#EF4444', rotation: 0 }
        ])
        setSettings(prev => ({ ...prev, gravity: 0, airResistance: 0 }))
        break

      case 'vibration':
        setObjects([
          { id: 'vibrator', type: 'box', x: 400, y: 300, z: 0, vx: 0, vy: 0, vz: 0, mass: 10, friction: 0.8, color: '#F59E0B', rotation: 0 }
        ])
        setSettings(prev => ({ ...prev, gravity: 9.81, airResistance: 0.05 }))
        // 진동 시뮬레이션용 주기적 힘 적용
        break
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Link 
                href={backUrl}
                className="flex items-center gap-2 text-amber-600 dark:text-amber-400 hover:text-amber-700 dark:hover:text-amber-300"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>학습 페이지로 돌아가기</span>
              </Link>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowForceVectors(!showForceVectors)}
                className={`px-3 py-1 rounded text-sm ${
                  showForceVectors 
                    ? 'bg-green-600 text-white' 
                    : 'bg-gray-300 dark:bg-gray-600 text-gray-700 dark:text-gray-300'
                }`}
              >
                <Zap className="w-4 h-4 inline mr-1" />
                힘 벡터
              </button>
              <button
                onClick={() => setShowTrajectory(!showTrajectory)}
                className={`px-3 py-1 rounded text-sm ${
                  showTrajectory 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-300 dark:bg-gray-600 text-gray-700 dark:text-gray-300'
                }`}
              >
                <Waves className="w-4 h-4 inline mr-1" />
                궤적
              </button>
              <button
                onClick={() => setIsRunning(!isRunning)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium ${
                  isRunning 
                    ? 'bg-red-600 text-white hover:bg-red-700' 
                    : 'bg-green-600 text-white hover:bg-green-700'
                }`}
              >
                {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isRunning ? '일시정지' : '시뮬레이션 시작'}
              </button>
              <button
                onClick={() => {
                  setIsRunning(false)
                  loadScenario(selectedScenario)
                }}
                className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
              >
                <RotateCcw className="w-4 h-4" />
                리셋
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Title */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center">
              <CircuitBoard className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                물리 시뮬레이션 실험실
                <Sparkles className="w-6 h-6 text-yellow-500" />
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400">제조 현장의 물리 현상을 실시간으로 시뮬레이션</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Canvas */}
          <div className="lg:col-span-2">
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">물리 시뮬레이션</h2>
              
              <canvas 
                ref={canvasRef}
                className="w-full border border-gray-300 dark:border-gray-600 rounded-lg"
                style={{ maxWidth: '800px', height: '500px' }}
              />
              
              <div className="mt-4 grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    중력 (m/s²)
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="20"
                    step="0.1"
                    value={settings.gravity}
                    onChange={(e) => setSettings(prev => ({ ...prev, gravity: parseFloat(e.target.value) }))}
                    className="w-full"
                  />
                  <span className="text-sm text-gray-600 dark:text-gray-400">{settings.gravity.toFixed(1)}</span>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    시간 배속
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="3"
                    step="0.1"
                    value={settings.timeScale}
                    onChange={(e) => setSettings(prev => ({ ...prev, timeScale: parseFloat(e.target.value) }))}
                    className="w-full"
                  />
                  <span className="text-sm text-gray-600 dark:text-gray-400">{settings.timeScale.toFixed(1)}x</span>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    공기 저항
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="0.1"
                    step="0.01"
                    value={settings.airResistance}
                    onChange={(e) => setSettings(prev => ({ ...prev, airResistance: parseFloat(e.target.value) }))}
                    className="w-full"
                  />
                  <span className="text-sm text-gray-600 dark:text-gray-400">{settings.airResistance.toFixed(2)}</span>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    충돌 감쇠
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={settings.collisionDamping}
                    onChange={(e) => setSettings(prev => ({ ...prev, collisionDamping: parseFloat(e.target.value) }))}
                    className="w-full"
                  />
                  <span className="text-sm text-gray-600 dark:text-gray-400">{settings.collisionDamping.toFixed(2)}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Control Panel */}
          <div className="space-y-6">
            {/* Scenarios */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">시뮬레이션 시나리오</h3>
              
              <div className="space-y-3">
                {scenarios.map((scenario) => (
                  <button
                    key={scenario.id}
                    onClick={() => loadScenario(scenario.id)}
                    className={`w-full p-3 rounded-lg border text-left transition-all ${
                      selectedScenario === scenario.id
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">{scenario.icon}</span>
                      <div>
                        <div className="font-medium text-gray-900 dark:text-white">
                          {scenario.name}
                        </div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">
                          {scenario.description}
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Physics Info */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">물리 법칙</h3>
              
              <div className="space-y-3 text-sm">
                <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className="font-medium text-blue-700 dark:text-blue-300">뉴턴의 운동 법칙</div>
                  <div className="text-blue-600 dark:text-blue-400 mt-1">F = ma (힘 = 질량 × 가속도)</div>
                </div>
                
                <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <div className="font-medium text-green-700 dark:text-green-300">운동 에너지</div>
                  <div className="text-green-600 dark:text-green-400 mt-1">KE = ½mv² (운동에너지 = ½ × 질량 × 속도²)</div>
                </div>
                
                <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <div className="font-medium text-purple-700 dark:text-purple-300">위치 에너지</div>
                  <div className="text-purple-600 dark:text-purple-400 mt-1">PE = mgh (위치에너지 = 질량 × 중력 × 높이)</div>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">빠른 작업</h3>
              
              <div className="space-y-3">
                <button className="w-full p-3 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors text-sm font-medium">
                  <Upload className="w-4 h-4 inline mr-2" />
                  시뮬레이션 데이터 가져오기
                </button>
                
                <button className="w-full p-3 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded-lg hover:bg-green-100 dark:hover:bg-green-900/30 transition-colors text-sm font-medium">
                  <Download className="w-4 h-4 inline mr-2" />
                  결과 데이터 내보내기
                </button>
                
                <button className="w-full p-3 bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors text-sm font-medium">
                  <Settings className="w-4 h-4 inline mr-2" />
                  고급 설정
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}