'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import { ArrowLeft, Play, Pause, RotateCcw, AlertTriangle, TrendingDown, TrendingUp, Zap, Thermometer, Activity, Gauge, Calendar, Clock, Brain, Shield, Wrench, BarChart3, AlertCircle, CheckCircle, XCircle, Sparkles, Flame, DollarSign } from 'lucide-react'

interface SensorData {
  timestamp: number
  temperature: number
  vibration: number
  pressure: number
  current: number
  rpm: number
  oilLevel: number
}

interface EquipmentStatus {
  id: string
  name: string
  type: string
  health: number
  rul: number // Remaining Useful Life in days
  status: 'healthy' | 'warning' | 'critical'
  lastMaintenance: string
  nextMaintenance: string
  sensorData: SensorData[]
  anomalyScore: number
  maintenanceCost: number
  downTimeCost: number
  predictedFailure?: {
    component: string
    probability: number
    timeframe: string
  }
}

interface PredictionResult {
  component: string
  failureProbability: number
  confidence: number
  recommendation: string
  estimatedCost: number
}

interface Alert {
  id: string
  type: 'info' | 'warning' | 'danger' | 'critical'
  message: string
  timestamp: Date
  equipment?: string
}

export default function EnhancedPredictiveMaintenancePage() {
  const [isRunning, setIsRunning] = useState(false)
  const [selectedEquipment, setSelectedEquipment] = useState('CNC_001')
  const [analysisMode, setAnalysisMode] = useState<'realtime' | 'prediction' | 'optimization'>('realtime')
  const [showAnomalyDetection, setShowAnomalyDetection] = useState(true)
  const [predictionAccuracy, setPredictionAccuracy] = useState(94.5)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartCanvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  
  const [equipment, setEquipment] = useState<EquipmentStatus[]>([
    {
      id: 'CNC_001',
      name: 'CNC 머시닝 센터 A',
      type: 'CNC Machine',
      health: 85,
      rul: 45,
      status: 'healthy',
      lastMaintenance: '2024-11-15',
      nextMaintenance: '2025-01-15',
      sensorData: [],
      anomalyScore: 0.12,
      maintenanceCost: 15000,
      downTimeCost: 50000
    },
    {
      id: 'ROBOT_002',
      name: '용접 로봇 B',
      type: 'Welding Robot',
      health: 62,
      rul: 15,
      status: 'warning',
      lastMaintenance: '2024-10-20',
      nextMaintenance: '2024-12-20',
      sensorData: [],
      anomalyScore: 0.68,
      maintenanceCost: 8000,
      downTimeCost: 35000,
      predictedFailure: {
        component: '모터 베어링',
        probability: 78,
        timeframe: '2주 이내'
      }
    },
    {
      id: 'PUMP_003',
      name: '유압 펌프 C',
      type: 'Hydraulic Pump',
      health: 38,
      rul: 5,
      status: 'critical',
      lastMaintenance: '2024-09-01',
      nextMaintenance: '2024-12-01',
      sensorData: [],
      anomalyScore: 0.92,
      maintenanceCost: 5000,
      downTimeCost: 25000,
      predictedFailure: {
        component: '씰 누출',
        probability: 95,
        timeframe: '5일 이내'
      }
    },
    {
      id: 'CONV_004',
      name: '컨베이어 시스템 D',
      type: 'Conveyor',
      health: 92,
      rul: 120,
      status: 'healthy',
      lastMaintenance: '2024-11-01',
      nextMaintenance: '2025-02-01',
      sensorData: [],
      anomalyScore: 0.08,
      maintenanceCost: 3000,
      downTimeCost: 15000
    }
  ])

  const [alerts, setAlerts] = useState<Alert[]>([
    { id: '1', type: 'info', message: '예측 정비 시스템 가동 중', timestamp: new Date() }
  ])

  const [totalSavings, setTotalSavings] = useState(0)
  const [preventedFailures, setPreventedFailures] = useState(0)

  // 실시간 센서 데이터 차트
  useEffect(() => {
    if (!chartCanvasRef.current) return
    const canvas = chartCanvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight

    let dataPoints: number[] = new Array(50).fill(0)
    let anomalyPoints: number[] = []

    const drawChart = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // 그리드
      ctx.strokeStyle = '#333'
      ctx.lineWidth = 0.5
      for (let i = 0; i < 5; i++) {
        const y = (canvas.height / 4) * i
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(canvas.width, y)
        ctx.stroke()
      }

      // 데이터 라인
      if (isRunning) {
        const selected = equipment.find(e => e.id === selectedEquipment)
        if (selected) {
          // 새 데이터 추가
          const newValue = 50 + Math.sin(Date.now() / 1000) * 20 + (Math.random() - 0.5) * 10
          if (selected.anomalyScore > 0.7) {
            // 이상 징후 시 급격한 변화
            dataPoints.push(newValue + (Math.random() - 0.5) * 30)
            anomalyPoints.push(dataPoints.length - 1)
          } else {
            dataPoints.push(newValue)
          }
          dataPoints.shift()

          // 라인 그리기
          ctx.strokeStyle = selected.status === 'critical' ? '#FF0000' :
                           selected.status === 'warning' ? '#FFA500' : '#00FF00'
          ctx.lineWidth = 2
          ctx.beginPath()
          dataPoints.forEach((value, index) => {
            const x = (canvas.width / (dataPoints.length - 1)) * index
            const y = canvas.height - (value / 100) * canvas.height
            if (index === 0) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)
          })
          ctx.stroke()

          // 이상 징후 포인트 표시
          if (showAnomalyDetection) {
            ctx.fillStyle = '#FF0000'
            anomalyPoints.forEach(index => {
              if (index < dataPoints.length) {
                const x = (canvas.width / (dataPoints.length - 1)) * index
                const y = canvas.height - (dataPoints[index] / 100) * canvas.height
                ctx.beginPath()
                ctx.arc(x, y, 5, 0, Math.PI * 2)
                ctx.fill()
              }
            })
          }
        }
      }
    }

    const interval = setInterval(drawChart, 100)
    return () => clearInterval(interval)
  }, [isRunning, selectedEquipment, equipment, showAnomalyDetection])

  // 파티클 효과
  useEffect(() => {
    if (!canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight

    let particles: Array<{
      x: number, y: number, vx: number, vy: number, 
      life: number, color: string, size: number
    }> = []
    let warnings: Array<{x: number, y: number, scale: number, opacity: number}> = []

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // 파티클 업데이트
      particles = particles.filter(particle => {
        particle.x += particle.vx
        particle.y += particle.vy
        particle.life -= 0.01

        if (particle.life <= 0) return false

        ctx.globalAlpha = particle.life
        ctx.fillStyle = particle.color
        ctx.beginPath()
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2)
        ctx.fill()

        return true
      })

      // 경고 아이콘 애니메이션
      warnings = warnings.filter(warning => {
        warning.scale += 0.05
        warning.opacity -= 0.02
        if (warning.opacity <= 0) return false

        ctx.globalAlpha = warning.opacity
        ctx.save()
        ctx.translate(warning.x, warning.y)
        ctx.scale(warning.scale, warning.scale)
        ctx.fillStyle = '#FF0000'
        ctx.font = '20px Arial'
        ctx.fillText('⚠️', -10, 10)
        ctx.restore()

        return true
      })

      // 장비 상태에 따른 파티클 생성
      if (isRunning) {
        equipment.forEach((eq, index) => {
          if (eq.status === 'critical' && Math.random() < 0.1) {
            const x = (index + 1) * (canvas.width / (equipment.length + 1))
            const y = canvas.height / 2

            // 위험 파티클
            for (let i = 0; i < 5; i++) {
              particles.push({
                x,
                y,
                vx: (Math.random() - 0.5) * 4,
                vy: (Math.random() - 0.5) * 4,
                life: 1,
                color: '#FF0000',
                size: Math.random() * 3 + 1
              })
            }

            // 경고 아이콘
            if (Math.random() < 0.05) {
              warnings.push({x, y: y - 50, scale: 1, opacity: 1})
            }
          } else if (eq.status === 'warning' && Math.random() < 0.05) {
            const x = (index + 1) * (canvas.width / (equipment.length + 1))
            const y = canvas.height / 2

            // 주의 파티클
            for (let i = 0; i < 3; i++) {
              particles.push({
                x,
                y,
                vx: (Math.random() - 0.5) * 3,
                vy: -Math.random() * 2,
                life: 1,
                color: '#FFA500',
                size: Math.random() * 2 + 1
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
  }, [isRunning, equipment])

  // 센서 데이터 시뮬레이션
  useEffect(() => {
    let interval: NodeJS.Timeout
    
    if (isRunning) {
      interval = setInterval(() => {
        setEquipment(prev => prev.map(eq => {
          // 센서 데이터 업데이트
          const newSensorData: SensorData = {
            timestamp: Date.now(),
            temperature: eq.health > 50 ? 45 + Math.random() * 10 : 55 + Math.random() * 15,
            vibration: eq.health > 50 ? 2 + Math.random() * 1 : 3 + Math.random() * 2,
            pressure: 100 + Math.random() * 20,
            current: 10 + Math.random() * 5,
            rpm: eq.type === 'CNC Machine' ? 3000 + Math.random() * 500 : 1500 + Math.random() * 300,
            oilLevel: Math.max(20, eq.health - Math.random() * 5)
          }

          // 건강도 감소
          let newHealth = eq.health - Math.random() * 0.5
          if (eq.status === 'critical') newHealth -= Math.random() * 1
          if (eq.status === 'warning') newHealth -= Math.random() * 0.7

          // RUL 감소
          let newRul = eq.rul - 0.1
          if (eq.status === 'critical') newRul -= 0.2
          
          // 이상 점수 계산
          const newAnomalyScore = Math.min(1, 
            (100 - newHealth) / 100 * 0.5 +
            (newSensorData.temperature > 60 ? 0.3 : 0) +
            (newSensorData.vibration > 4 ? 0.2 : 0)
          )

          // 상태 업데이트
          let newStatus: 'healthy' | 'warning' | 'critical' = 'healthy'
          if (newHealth < 40 || newAnomalyScore > 0.8) newStatus = 'critical'
          else if (newHealth < 70 || newAnomalyScore > 0.5) newStatus = 'warning'

          return {
            ...eq,
            health: Math.max(0, Math.round(newHealth)),
            rul: Math.max(0, Math.round(newRul)),
            status: newStatus,
            sensorData: [...eq.sensorData.slice(-49), newSensorData],
            anomalyScore: Math.round(newAnomalyScore * 100) / 100
          }
        }))

        // 알림 생성
        if (Math.random() < 0.15) {
          const criticalEquipment = equipment.find(eq => eq.status === 'critical')
          const warningEquipment = equipment.find(eq => eq.status === 'warning')
          
          if (criticalEquipment && Math.random() < 0.5) {
            setAlerts(prev => [...prev.slice(-4), {
              id: Date.now().toString(),
              type: 'critical',
              message: `🚨 ${criticalEquipment.name} 즉시 정비 필요!`,
              timestamp: new Date(),
              equipment: criticalEquipment.id
            }])
          } else if (warningEquipment && Math.random() < 0.3) {
            setAlerts(prev => [...prev.slice(-4), {
              id: Date.now().toString(),
              type: 'warning',
              message: `⚠️ ${warningEquipment.name} 이상 징후 감지`,
              timestamp: new Date(),
              equipment: warningEquipment.id
            }])
          } else {
            setAlerts(prev => [...prev.slice(-4), {
              id: Date.now().toString(),
              type: 'info',
              message: '✅ 모든 장비 정상 작동 중',
              timestamp: new Date()
            }])
          }
        }

        // 비용 절감 계산
        const prevented = equipment.filter(eq => eq.status === 'warning' && eq.anomalyScore > 0.6).length
        if (prevented > 0) {
          setPreventedFailures(prev => prev + prevented)
          setTotalSavings(prev => prev + prevented * 20000)
        }
      }, 1000)
    }
    
    return () => clearInterval(interval)
  }, [isRunning, equipment])

  const getHealthColor = (health: number) => {
    if (health > 70) return 'text-green-500'
    if (health > 40) return 'text-yellow-500'
    return 'text-red-500'
  }

  const getHealthBg = (health: number) => {
    if (health > 70) return 'bg-green-500'
    if (health > 40) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  const getAlertBg = (type: string) => {
    switch (type) {
      case 'critical': return 'bg-red-100 dark:bg-red-900/20 border-red-300 dark:border-red-800'
      case 'danger': return 'bg-orange-100 dark:bg-orange-900/20 border-orange-300 dark:border-orange-800'
      case 'warning': return 'bg-yellow-100 dark:bg-yellow-900/20 border-yellow-300 dark:border-yellow-800'
      default: return 'bg-blue-100 dark:bg-blue-900/20 border-blue-300 dark:border-blue-800'
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 relative overflow-hidden">
      {/* 파티클 효과 캔버스 */}
      <canvas 
        ref={canvasRef}
        className="fixed inset-0 pointer-events-none z-30"
        style={{ width: '100%', height: '100%' }}
      />

      {/* 위험 상태 배경 효과 */}
      {equipment.some(eq => eq.status === 'critical') && (
        <div className="fixed inset-0 pointer-events-none z-20">
          <div className="absolute inset-0 bg-gradient-to-br from-red-500/10 to-orange-500/10 animate-pulse"></div>
        </div>
      )}

      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 relative z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Link 
                href="/modules/smart-factory"
                className="flex items-center gap-2 text-amber-600 dark:text-amber-400 hover:text-amber-700 dark:hover:text-amber-300"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>Smart Factory로 돌아가기</span>
              </Link>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowAnomalyDetection(!showAnomalyDetection)}
                className={`px-3 py-1 rounded text-sm flex items-center gap-1 ${
                  showAnomalyDetection 
                    ? 'bg-purple-600 text-white' 
                    : 'bg-gray-300 dark:bg-gray-600 text-gray-700 dark:text-gray-300'
                }`}
              >
                <Brain className="w-4 h-4" />
                AI 이상 감지
              </button>
              <button
                onClick={() => setIsRunning(!isRunning)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium ${
                  isRunning 
                    ? 'bg-red-600 text-white hover:bg-red-700 animate-pulse' 
                    : 'bg-green-600 text-white hover:bg-green-700'
                }`}
              >
                {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isRunning ? '모니터링 중지' : '모니터링 시작'}
              </button>
              <button
                onClick={() => {
                  setIsRunning(false)
                  setAlerts([{ id: '1', type: 'info', message: '시스템 초기화 완료', timestamp: new Date() }])
                  setTotalSavings(0)
                  setPreventedFailures(0)
                  setEquipment(prev => prev.map(eq => ({
                    ...eq,
                    health: 85 + Math.random() * 10,
                    rul: 30 + Math.random() * 90,
                    status: 'healthy',
                    anomalyScore: Math.random() * 0.2,
                    sensorData: []
                  })))
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
            <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center animate-pulse">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                AI 예측 정비 연구소
                <Sparkles className="w-6 h-6 text-yellow-500" />
                {equipment.some(eq => eq.status === 'critical') && (
                  <Flame className="w-6 h-6 text-red-500 animate-bounce" />
                )}
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400">머신러닝 기반 장비 고장 예측 및 최적화</p>
            </div>
          </div>
        </div>

        {/* Analysis Mode Tabs */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-1 mb-6">
          <div className="flex">
            {[
              { id: 'realtime', name: '실시간 모니터링', icon: Activity },
              { id: 'prediction', name: 'AI 고장 예측', icon: Brain },
              { id: 'optimization', name: '비용 최적화', icon: DollarSign }
            ].map((mode) => (
              <button
                key={mode.id}
                onClick={() => setAnalysisMode(mode.id as any)}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md transition-all ${
                  analysisMode === mode.id
                    ? 'bg-purple-600 text-white'
                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                <mode.icon className="w-4 h-4" />
                {mode.name}
              </button>
            ))}
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Equipment Status */}
          <div className="lg:col-span-2">
            {/* Real-time Sensor Chart */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6 mb-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                  실시간 센서 데이터
                </h2>
                <select
                  value={selectedEquipment}
                  onChange={(e) => setSelectedEquipment(e.target.value)}
                  className="px-3 py-1 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-sm"
                >
                  {equipment.map(eq => (
                    <option key={eq.id} value={eq.id}>{eq.name}</option>
                  ))}
                </select>
              </div>
              
              <div className="bg-black rounded-lg overflow-hidden">
                <canvas 
                  ref={chartCanvasRef}
                  className="w-full"
                  style={{ height: '200px' }}
                />
              </div>
              
              <div className="grid grid-cols-3 gap-4 mt-4">
                {equipment.find(eq => eq.id === selectedEquipment) && (
                  <>
                    <div className="text-center">
                      <Thermometer className="w-5 h-5 text-red-500 mx-auto mb-1" />
                      <div className="text-sm text-gray-600 dark:text-gray-400">온도</div>
                      <div className="font-bold text-gray-900 dark:text-white">
                        {equipment.find(eq => eq.id === selectedEquipment)?.sensorData.slice(-1)[0]?.temperature.toFixed(1) || '0'}°C
                      </div>
                    </div>
                    <div className="text-center">
                      <Activity className="w-5 h-5 text-blue-500 mx-auto mb-1" />
                      <div className="text-sm text-gray-600 dark:text-gray-400">진동</div>
                      <div className="font-bold text-gray-900 dark:text-white">
                        {equipment.find(eq => eq.id === selectedEquipment)?.sensorData.slice(-1)[0]?.vibration.toFixed(1) || '0'} mm/s
                      </div>
                    </div>
                    <div className="text-center">
                      <Gauge className="w-5 h-5 text-green-500 mx-auto mb-1" />
                      <div className="text-sm text-gray-600 dark:text-gray-400">압력</div>
                      <div className="font-bold text-gray-900 dark:text-white">
                        {equipment.find(eq => eq.id === selectedEquipment)?.sensorData.slice(-1)[0]?.pressure.toFixed(0) || '0'} PSI
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Equipment Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {equipment.map((eq) => (
                <div 
                  key={eq.id} 
                  className={`bg-white dark:bg-gray-800 rounded-xl border p-4 ${
                    eq.status === 'critical' ? 'border-red-500 ring-2 ring-red-500 ring-opacity-50 animate-pulse' :
                    eq.status === 'warning' ? 'border-yellow-500' :
                    'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h3 className="font-semibold text-gray-900 dark:text-white">{eq.name}</h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">{eq.type}</p>
                    </div>
                    <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                      eq.status === 'critical' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300' :
                      eq.status === 'warning' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300' :
                      'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300'
                    }`}>
                      {eq.status === 'critical' ? '위험' :
                       eq.status === 'warning' ? '주의' : '정상'}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-600 dark:text-gray-400">건강도</span>
                        <span className={`font-medium ${getHealthColor(eq.health)}`}>{eq.health}%</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full transition-all duration-500 ${getHealthBg(eq.health)}`}
                          style={{ width: `${eq.health}%` }}
                        ></div>
                      </div>
                    </div>

                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">잔여 수명</span>
                      <span className={`font-medium ${eq.rul < 10 ? 'text-red-500' : 'text-gray-900 dark:text-white'}`}>
                        {eq.rul}일
                      </span>
                    </div>

                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">이상 점수</span>
                      <span className={`font-medium ${
                        eq.anomalyScore > 0.7 ? 'text-red-500' :
                        eq.anomalyScore > 0.4 ? 'text-yellow-500' :
                        'text-green-500'
                      }`}>
                        {eq.anomalyScore.toFixed(2)}
                      </span>
                    </div>

                    {eq.predictedFailure && (
                      <div className="mt-3 p-2 bg-red-50 dark:bg-red-900/20 rounded-lg">
                        <div className="flex items-start gap-2">
                          <AlertTriangle className="w-4 h-4 text-red-500 mt-0.5" />
                          <div className="text-xs">
                            <div className="font-medium text-red-700 dark:text-red-300">
                              예측된 고장: {eq.predictedFailure.component}
                            </div>
                            <div className="text-red-600 dark:text-red-400">
                              확률 {eq.predictedFailure.probability}% ({eq.predictedFailure.timeframe})
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Side Panel */}
          <div className="space-y-6">
            {/* AI Performance */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                AI 성능 지표
                <Brain className="w-5 h-5 text-purple-500" />
              </h3>
              
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm text-gray-600 dark:text-gray-400">예측 정확도</span>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {predictionAccuracy}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-purple-500 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${predictionAccuracy}%` }}
                    ></div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <Shield className="w-8 h-8 text-green-500 mx-auto mb-1" />
                    <div className="text-2xl font-bold text-gray-900 dark:text-white">
                      {preventedFailures}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">예방된 고장</div>
                  </div>
                  <div className="text-center">
                    <DollarSign className="w-8 h-8 text-blue-500 mx-auto mb-1" />
                    <div className="text-2xl font-bold text-gray-900 dark:text-white">
                      ${totalSavings.toLocaleString()}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">절감 비용</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Real-time Alerts */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                실시간 알림
              </h3>
              
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {alerts.slice().reverse().map((alert) => (
                  <div 
                    key={alert.id} 
                    className={`p-3 rounded-lg text-sm border ${getAlertBg(alert.type)} animate-fade-in`}
                  >
                    <p className="font-medium">{alert.message}</p>
                    <p className="text-xs opacity-70 mt-1">
                      {alert.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">빠른 작업</h3>
              
              <div className="space-y-3">
                <button className="w-full p-3 bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors text-sm font-medium">
                  <Calendar className="w-4 h-4 inline mr-2" />
                  정비 일정 최적화
                </button>
                
                <button className="w-full p-3 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors text-sm font-medium">
                  <Wrench className="w-4 h-4 inline mr-2" />
                  즉시 정비 요청
                </button>
                
                <button className="w-full p-3 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded-lg hover:bg-green-100 dark:hover:bg-green-900/30 transition-colors text-sm font-medium">
                  <BarChart3 className="w-4 h-4 inline mr-2" />
                  상세 분석 리포트
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fade-in {
          0% { opacity: 0; transform: translateY(-10px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        
        .animate-fade-in {
          animation: fade-in 0.3s ease-out;
        }
      `}</style>
    </div>
  )
}