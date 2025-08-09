'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import { ArrowLeft, Play, Pause, RotateCcw, Box, Monitor, Zap, Thermometer, Gauge, Activity, Settings, RefreshCw, AlertTriangle, CheckCircle, Sparkles, Rocket, TrendingUp } from 'lucide-react'

interface Machine3D {
  id: string
  name: string
  x: number
  y: number
  z: number
  status: 'running' | 'idle' | 'maintenance' | 'error'
  temperature: number
  efficiency: number
  output: number
  particles?: Particle[]
}

interface Particle {
  id: number
  x: number
  y: number
  vx: number
  vy: number
  life: number
  color: string
}

interface FactoryMetrics {
  totalOutput: number
  energyConsumption: number
  efficiency: number
  temperature: number
  activeMachines: number
  productionRate: number
  defectRate: number
}

interface Scenario {
  id: string
  name: string
  description: string
  parameters: {
    speed: number
    load: number
    temperature: number
  }
  sparkEffect?: boolean
  boostAnimation?: boolean
}

interface EnhancedDigitalTwinProps {
  backUrl?: string
}

export default function EnhancedDigitalTwinPage({ backUrl = '/modules/smart-factory' }: EnhancedDigitalTwinProps) {
  const [isRunning, setIsRunning] = useState(false)
  const [selectedScenario, setSelectedScenario] = useState('normal')
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('3d')
  const [syncStatus, setSyncStatus] = useState<'connected' | 'disconnected' | 'syncing'>('connected')
  const [showPresentationEffect, setShowPresentationEffect] = useState(false)
  const [explosionEffect, setExplosionEffect] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  
  const [machines, setMachines] = useState<Machine3D[]>([
    { id: 'M1', name: '사출기 A', x: 2, y: 1, z: 0, status: 'running', temperature: 45, efficiency: 85, output: 120, particles: [] },
    { id: 'M2', name: '로봇 B', x: 4, y: 1, z: 0, status: 'running', temperature: 38, efficiency: 92, output: 95, particles: [] },
    { id: 'M3', name: '컨베이어 C', x: 6, y: 1, z: 0, status: 'running', temperature: 25, efficiency: 88, output: 150, particles: [] },
    { id: 'M4', name: '포장기 D', x: 8, y: 1, z: 0, status: 'idle', temperature: 22, efficiency: 0, output: 0, particles: [] },
    { id: 'M5', name: '검사기 E', x: 2, y: 3, z: 0, status: 'maintenance', temperature: 28, efficiency: 0, output: 0, particles: [] },
    { id: 'M6', name: '용접기 F', x: 4, y: 3, z: 0, status: 'error', temperature: 62, efficiency: 0, output: 0, particles: [] }
  ])

  const [metrics, setMetrics] = useState<FactoryMetrics>({
    totalOutput: 365,
    energyConsumption: 1250,
    efficiency: 78,
    temperature: 34,
    activeMachines: 3,
    productionRate: 0,
    defectRate: 2.3
  })

  const scenarios: Scenario[] = [
    {
      id: 'normal',
      name: '⚙️ 정상 운영',
      description: '평소처럼 안정적으로 운영 (추천)',
      parameters: { speed: 1.0, load: 0.8, temperature: 45 }
    },
    {
      id: 'peak',
      name: '🚀 피크 생산',
      description: '급한 주문! 최대한 빨리 생산 (뜨거워짐⚠️)',
      parameters: { speed: 1.5, load: 1.0, temperature: 55 },
      sparkEffect: true,
      boostAnimation: true
    },
    {
      id: 'maintenance',
      name: '🔧 정비 모드',
      description: '기계 점검 중... 천천히 운영',
      parameters: { speed: 0.6, load: 0.5, temperature: 35 }
    },
    {
      id: 'emergency',
      name: '🚨 비상 상황',
      description: '문제 발생! 최소한만 운영',
      parameters: { speed: 0.2, load: 0.3, temperature: 30 }
    },
    {
      id: 'presentation',
      name: '🎭 발표 모드',
      description: '화려한 효과로 시선 집중!',
      parameters: { speed: 1.2, load: 0.9, temperature: 48 },
      sparkEffect: true,
      boostAnimation: true
    }
  ]

  // 파티클 효과를 위한 Canvas 애니메이션
  useEffect(() => {
    if (!canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight

    let particles: Particle[] = []

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // 파티클 업데이트 및 렌더링
      particles = particles.filter(particle => {
        particle.x += particle.vx
        particle.y += particle.vy
        particle.life -= 0.02

        if (particle.life <= 0) return false

        ctx.globalAlpha = particle.life
        ctx.fillStyle = particle.color
        ctx.beginPath()
        ctx.arc(particle.x, particle.y, 3, 0, Math.PI * 2)
        ctx.fill()

        return true
      })

      // 실행 중이고 특정 시나리오일 때 파티클 생성
      if (isRunning && (selectedScenario === 'peak' || selectedScenario === 'presentation')) {
        machines.forEach(machine => {
          if (machine.status === 'running' && Math.random() < 0.1) {
            const rect = canvas.getBoundingClientRect()
            const x = (machine.x * 12 + 10) * rect.width / 100
            const y = (machine.y * 25 + 15) * rect.height / 100

            for (let i = 0; i < 5; i++) {
              particles.push({
                id: Date.now() + i,
                x,
                y,
                vx: (Math.random() - 0.5) * 4,
                vy: -Math.random() * 3 - 1,
                life: 1,
                color: selectedScenario === 'presentation' ? 
                  ['#FFD700', '#FF69B4', '#00CED1', '#32CD32'][Math.floor(Math.random() * 4)] : 
                  '#FFA500'
              })
            }
          }
        })
      }

      animationRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning, selectedScenario, machines])

  useEffect(() => {
    let interval: NodeJS.Timeout
    
    if (isRunning) {
      interval = setInterval(() => {
        const scenario = scenarios.find(s => s.id === selectedScenario)!
        
        setMachines(prev => prev.map(machine => {
          if (machine.status === 'running') {
            const speedMultiplier = scenario.parameters.speed
            const newOutput = Math.min(
              machine.output + (Math.random() * 10 * speedMultiplier),
              200
            )
            const newTemp = scenario.parameters.temperature + (Math.random() - 0.5) * 10
            const newEfficiency = Math.min(100, machine.efficiency + (Math.random() - 0.5) * 5)
            
            return {
              ...machine,
              output: Math.round(newOutput),
              temperature: Math.round(newTemp * 10) / 10,
              efficiency: Math.round(newEfficiency)
            }
          }
          return machine
        }))
        
        setMetrics(prev => {
          const runningMachines = machines.filter(m => m.status === 'running').length
          const totalOutputIncrease = runningMachines * scenario.parameters.speed * 5
          const productionRate = totalOutputIncrease * 60 // per hour
          
          return {
            totalOutput: prev.totalOutput + Math.floor(totalOutputIncrease),
            energyConsumption: Math.round(1000 + (runningMachines * 200 * scenario.parameters.load)),
            efficiency: Math.round((runningMachines / machines.length) * scenario.parameters.speed * 100),
            temperature: Math.round(scenario.parameters.temperature + (Math.random() - 0.5) * 5),
            activeMachines: runningMachines,
            productionRate: Math.round(productionRate),
            defectRate: Math.max(0, Math.min(10, 2.3 + (scenario.parameters.speed - 1) * 3))
          }
        })

        // Simulate sync status changes
        if (Math.random() < 0.1) {
          setSyncStatus('syncing')
          setTimeout(() => setSyncStatus('connected'), 1000)
        }
      }, 2000)
    }
    
    return () => clearInterval(interval)
  }, [isRunning, selectedScenario, machines])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-green-500'
      case 'idle': return 'bg-yellow-500'
      case 'maintenance': return 'bg-blue-500'
      case 'error': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  const getMachineEmoji = (name: string) => {
    if (name.includes('사출')) return '🏭'
    if (name.includes('로봇')) return '🤖'
    if (name.includes('컨베이어')) return '📦'
    if (name.includes('포장')) return '📮'
    if (name.includes('검사')) return '🔍'
    if (name.includes('용접')) return '⚡'
    return '⚙️'
  }
  
  const getMachineDescription = (name: string) => {
    if (name.includes('사출')) return '플라스틱 제품 생산'
    if (name.includes('로봇')) return '제품 조립'
    if (name.includes('컨베이어')) return '제품 운반'
    if (name.includes('포장')) return '완제품 포장'
    if (name.includes('검사')) return '품질 검사'
    if (name.includes('용접')) return '금속 접합'
    return '제조 작업'
  }

  const getSyncStatusIcon = () => {
    switch (syncStatus) {
      case 'connected': return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'disconnected': return <AlertTriangle className="w-4 h-4 text-red-500" />
      case 'syncing': return <RefreshCw className="w-4 h-4 text-blue-500 animate-spin" />
    }
  }

  const handleScenarioChange = (scenarioId: string) => {
    setSelectedScenario(scenarioId)
    const scenario = scenarios.find(s => s.id === scenarioId)
    
    if (scenario?.sparkEffect) {
      setExplosionEffect(true)
      setTimeout(() => setExplosionEffect(false), 1000)
    }
    
    if (scenarioId === 'presentation') {
      setShowPresentationEffect(true)
      setTimeout(() => setShowPresentationEffect(false), 3000)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 relative overflow-hidden">
      {/* 발표 모드 특별 효과 */}
      {showPresentationEffect && (
        <div className="fixed inset-0 pointer-events-none z-50">
          <div className="absolute inset-0 bg-gradient-to-r from-yellow-400/20 via-pink-500/20 to-purple-600/20 animate-pulse"></div>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
            <div className="text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 to-pink-600 animate-bounce">
              🎉 Smart Factory in Action! 🎉
            </div>
          </div>
        </div>
      )}

      {/* 폭발 효과 */}
      {explosionEffect && (
        <div className="fixed inset-0 pointer-events-none z-40">
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
            <div className="w-96 h-96 bg-gradient-to-r from-orange-500 to-red-500 rounded-full animate-ping"></div>
          </div>
        </div>
      )}

      {/* 파티클 캔버스 */}
      <canvas 
        ref={canvasRef}
        className="fixed inset-0 pointer-events-none z-30"
        style={{ width: '100%', height: '100%' }}
      />

      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 relative z-20">
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
              <div className="flex items-center gap-2">
                {getSyncStatusIcon()}
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  {syncStatus === 'connected' ? '실시간 동기화' : 
                   syncStatus === 'syncing' ? '동기화 중...' : '연결 끊김'}
                </span>
              </div>
              <button
                onClick={() => setIsRunning(!isRunning)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                  isRunning 
                    ? 'bg-red-600 text-white hover:bg-red-700 animate-pulse' 
                    : 'bg-green-600 text-white hover:bg-green-700'
                }`}
              >
                {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isRunning ? '시뮬레이션 정지' : '시뮬레이션 시작'}
              </button>
              <button
                onClick={() => {
                  setIsRunning(false)
                  setMachines(prev => prev.map(m => ({ ...m, output: 0, efficiency: Math.random() * 30 + 70 })))
                  setMetrics({
                    totalOutput: 365,
                    energyConsumption: 1250,
                    efficiency: 78,
                    temperature: 34,
                    activeMachines: 3,
                    productionRate: 0,
                    defectRate: 2.3
                  })
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

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        {/* Title */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center animate-pulse">
              <Box className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                디지털 트윈 팩토리 
                <Sparkles className="inline-block w-6 h-6 ml-2 text-yellow-500 animate-spin" />
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400">3D 가상 공장과 실시간 시뮬레이션 환경</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-4 gap-8">
          {/* 3D Factory View */}
          <div className="xl:col-span-3">
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6 mb-6 relative overflow-hidden">
              {/* 피크/발표 모드일 때 번쩍이는 효과 */}
              {isRunning && (selectedScenario === 'peak' || selectedScenario === 'presentation') && (
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer pointer-events-none"></div>
              )}
              
              <div className="mb-6">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                    가상 공장 레이아웃
                    {selectedScenario === 'presentation' && (
                      <Rocket className="inline-block w-5 h-5 ml-2 text-purple-500 animate-bounce" />
                    )}
                  </h2>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setViewMode('2d')}
                      className={`px-3 py-1 rounded ${viewMode === '2d' ? 'bg-blue-500 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'}`}
                    >
                      2D
                    </button>
                    <button
                      onClick={() => setViewMode('3d')}
                      className={`px-3 py-1 rounded ${viewMode === '3d' ? 'bg-blue-500 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'}`}
                    >
                      3D
                    </button>
                  </div>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  💡 <strong>사용법:</strong> ① 우측 상단 "시뮬레이션 시작" 클릭 ② 오른쪽 시나리오 선택 ③ 숫자 변화 관찰
                </p>
              </div>

              {/* Factory Layout Grid */}
              <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-6 h-96 relative overflow-hidden">
                {/* Grid Background */}
                <div className="absolute inset-0 opacity-20">
                  <svg width="100%" height="100%" className="text-gray-400">
                    <defs>
                      <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                        <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="1"/>
                      </pattern>
                    </defs>
                    <rect width="100%" height="100%" fill="url(#grid)" />
                  </svg>
                </div>

                {/* Machines */}
                {machines.map((machine) => (
                  <div
                    key={machine.id}
                    className={`absolute transform -translate-x-1/2 -translate-y-1/2 cursor-pointer group ${
                      isRunning && machine.status === 'running' && (selectedScenario === 'peak' || selectedScenario === 'presentation') 
                        ? 'animate-bounce' 
                        : ''
                    }`}
                    style={{
                      left: `${machine.x * 12 + 10}%`,
                      top: `${machine.y * 25 + 15}%`,
                    }}
                  >
                    <div className={`relative p-3 rounded-lg border-2 transition-all duration-300 ${
                      machine.status === 'running' ? 'border-green-500 bg-green-50 dark:bg-green-900/20' :
                      machine.status === 'idle' ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20' :
                      machine.status === 'maintenance' ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' :
                      'border-red-500 bg-red-50 dark:bg-red-900/20'
                    } group-hover:scale-110 ${
                      viewMode === '3d' ? 'shadow-xl transform-gpu perspective-1000 rotate-x-12' : ''
                    }`}>
                      
                      {/* Status Indicator */}
                      <div className={`absolute -top-1 -right-1 w-3 h-3 rounded-full ${getStatusColor(machine.status)} ${
                        machine.status === 'running' ? 'animate-pulse' : ''
                      }`}></div>

                      {/* Machine Icon */}
                      <div className="text-2xl text-center">
                        {getMachineEmoji(machine.name)}
                      </div>
                      
                      {/* Machine Info */}
                      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity bg-black text-white text-xs rounded px-2 py-1 whitespace-nowrap z-10">
                        <div className="font-semibold">{machine.name}</div>
                        <div className="text-yellow-300 mb-1">{getMachineDescription(machine.name)}</div>
                        <div>효율: {machine.efficiency}%</div>
                        <div>온도: {machine.temperature}°C</div>
                        <div>생산: {machine.output}개/시간</div>
                      </div>
                    </div>
                  </div>
                ))}

                {/* Conveyor Lines */}
                <svg className="absolute inset-0 pointer-events-none" width="100%" height="100%">
                  <defs>
                    <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                            refX="0" refY="3.5" orient="auto">
                      <polygon points="0 0, 10 3.5, 0 7" fill="#6B7280" />
                    </marker>
                  </defs>
                  
                  {/* Production Flow Lines */}
                  <line x1="22%" y1="40%" x2="58%" y2="40%" 
                        stroke="#6B7280" strokeWidth="2" markerEnd="url(#arrowhead)"
                        strokeDasharray={isRunning ? "5,5" : "none"}
                        className={isRunning ? "animate-pulse" : ""} />
                  <line x1="70%" y1="40%" x2="98%" y2="40%" 
                        stroke="#6B7280" strokeWidth="2" markerEnd="url(#arrowhead)"
                        strokeDasharray={isRunning ? "5,5" : "none"}
                        className={isRunning ? "animate-pulse" : ""} />
                </svg>

                {/* Legend */}
                <div className="absolute bottom-4 left-4 bg-white dark:bg-gray-800 rounded-lg p-3 text-xs">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      <span>가동</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                      <span>대기</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                      <span>정비</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                      <span>오류</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Real-time Metrics */}
            <div className="mb-3">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                실시간 공장 상태
                {metrics.productionRate > 500 && (
                  <TrendingUp className="inline-block w-5 h-5 ml-2 text-green-500 animate-bounce" />
                )}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">📊 아래 숫자들이 실시간으로 변합니다!</p>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-5 lg:grid-cols-7 gap-4">
              <div className={`bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700 ${
                isRunning && metrics.totalOutput > 500 ? 'animate-pulse ring-2 ring-green-500' : ''
              }`}>
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-5 h-5 text-blue-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">총 생산량</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metrics.totalOutput.toLocaleString()}
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="w-5 h-5 text-yellow-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">에너지 소비</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metrics.energyConsumption} kW
                </div>
              </div>

              <div className={`bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700 ${
                metrics.efficiency > 90 ? 'ring-2 ring-green-500' : ''
              }`}>
                <div className="flex items-center gap-2 mb-2">
                  <Gauge className="w-5 h-5 text-green-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">효율성</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metrics.efficiency}%
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-2 mb-2">
                  <Thermometer className="w-5 h-5 text-red-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">평균 온도</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metrics.temperature}°C
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-2 mb-2">
                  <Monitor className="w-5 h-5 text-purple-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">가동 장비</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metrics.activeMachines}/{machines.length}
                </div>
              </div>

              <div className={`bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700 ${
                isRunning && metrics.productionRate > 0 ? 'animate-pulse' : ''
              }`}>
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="w-5 h-5 text-indigo-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">생산 속도</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metrics.productionRate}/h
                </div>
              </div>

              <div className={`bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700 ${
                metrics.defectRate > 5 ? 'ring-2 ring-red-500' : ''
              }`}>
                <div className="flex items-center gap-2 mb-2">
                  <AlertTriangle className="w-5 h-5 text-orange-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">불량률</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metrics.defectRate.toFixed(1)}%
                </div>
              </div>
            </div>
          </div>

          {/* Control Panel */}
          <div className="space-y-6">
            {/* Scenario Control */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                시나리오 제어
                {selectedScenario === 'presentation' && (
                  <Sparkles className="inline-block w-5 h-5 ml-2 text-yellow-500" />
                )}
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                🎮 클릭하면 공장 운영 방식이 바뀝니다!
              </p>
              
              <div className="space-y-4">
                {scenarios.map((scenario) => (
                  <button
                    key={scenario.id}
                    onClick={() => handleScenarioChange(scenario.id)}
                    disabled={false}
                    className={`w-full p-3 rounded-lg border text-left transition-all ${
                      selectedScenario === scenario.id
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 ring-2 ring-blue-500 ring-opacity-50'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                    } ${isRunning ? 'opacity-50 cursor-not-allowed' : ''} ${
                      scenario.sparkEffect && selectedScenario === scenario.id ? 'animate-pulse' : ''
                    }`}
                  >
                    <div className="font-semibold text-gray-900 dark:text-white">
                      {scenario.name}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      {scenario.description}
                    </div>
                    <div className="flex justify-between mt-2 text-xs text-gray-500 dark:text-gray-400">
                      <span>속도: {(scenario.parameters.speed * 100).toFixed(0)}%</span>
                      <span>부하: {(scenario.parameters.load * 100).toFixed(0)}%</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Machine Status */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">장비 상태</h2>
              
              <div className="space-y-3">
                {machines.map((machine) => (
                  <div key={machine.id} className={`flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg ${
                    isRunning && machine.status === 'running' ? 'animate-pulse' : ''
                  }`}>
                    <div className="flex items-center gap-3">
                      <div className={`w-3 h-3 rounded-full ${getStatusColor(machine.status)}`}></div>
                      <div>
                        <div className="font-medium text-gray-900 dark:text-white">
                          {machine.name}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          효율: {machine.efficiency}%
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-medium text-gray-900 dark:text-white">
                        {machine.output}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        개/시간
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* What-if Analysis */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">What-if 분석</h2>
              
              <div className="space-y-4">
                <button className="w-full p-3 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors">
                  <Settings className="w-4 h-4 inline mr-2" />
                  생산 라인 최적화
                </button>
                
                <button className="w-full p-3 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded-lg hover:bg-green-100 dark:hover:bg-green-900/30 transition-colors">
                  <Zap className="w-4 h-4 inline mr-2" />
                  에너지 효율 시뮬레이션
                </button>
                
                <button className="w-full p-3 bg-orange-50 dark:bg-orange-900/20 text-orange-700 dark:text-orange-300 rounded-lg hover:bg-orange-100 dark:hover:bg-orange-900/30 transition-colors">
                  <AlertTriangle className="w-4 h-4 inline mr-2" />
                  고장 시나리오 테스트
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        
        .animate-shimmer {
          animation: shimmer 2s infinite;
        }
        
        .perspective-1000 {
          perspective: 1000px;
        }
        
        .rotate-x-12 {
          transform: rotateX(12deg);
        }
      `}</style>
    </div>
  )
}