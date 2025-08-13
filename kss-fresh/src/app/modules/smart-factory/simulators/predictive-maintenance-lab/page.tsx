'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'
import { ArrowLeft, Play, Pause, RotateCcw, AlertTriangle, TrendingDown, TrendingUp, Zap, Thermometer, Activity, Gauge, Calendar, Clock, Brain, Shield, Wrench, BarChart3, AlertCircle, CheckCircle, XCircle } from 'lucide-react'

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
}

interface PredictionResult {
  component: string
  failureProbability: number
  confidence: number
  recommendation: string
}

export default function PredictiveMaintenanceLabPage() {
  const searchParams = useSearchParams()
  const backUrl = searchParams.get('from') || '/modules/smart-factory'
  
  const [isRunning, setIsRunning] = useState(false)
  const [selectedEquipment, setSelectedEquipment] = useState('CNC_001')
  const [analysisMode, setAnalysisMode] = useState<'realtime' | 'prediction' | 'optimization'>('realtime')
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  
  const [equipment, setEquipment] = useState<EquipmentStatus[]>([
    {
      id: 'CNC_001',
      name: 'CNC 머시닝 센터 A',
      type: 'CNC Machine',
      health: 92,
      rul: 145,
      status: 'healthy',
      lastMaintenance: '2024-10-15',
      nextMaintenance: '2025-01-15',
      sensorData: [],
      anomalyScore: 0.12,
      maintenanceCost: 15000000,
      downTimeCost: 50000000
    },
    {
      id: 'ROBOT_002',
      name: '용접 로봇 B',
      type: 'Welding Robot',
      health: 76,
      rul: 62,
      status: 'warning',
      lastMaintenance: '2024-09-20',
      nextMaintenance: '2024-12-20',
      sensorData: [],
      anomalyScore: 0.38,
      maintenanceCost: 12000000,
      downTimeCost: 40000000
    },
    {
      id: 'PRESS_003',
      name: '프레스 장비 C',
      type: 'Press Machine',
      health: 48,
      rul: 18,
      status: 'critical',
      lastMaintenance: '2024-07-10',
      nextMaintenance: '2024-11-25',
      sensorData: [],
      anomalyScore: 0.72,
      maintenanceCost: 18000000,
      downTimeCost: 60000000
    },
    {
      id: 'CONVEYOR_004',
      name: '컨베이어 시스템 D',
      type: 'Conveyor System',
      health: 88,
      rul: 98,
      status: 'healthy',
      lastMaintenance: '2024-11-01',
      nextMaintenance: '2025-02-01',
      sensorData: [],
      anomalyScore: 0.15,
      maintenanceCost: 8000000,
      downTimeCost: 30000000
    }
  ])

  const [predictions, setPredictions] = useState<PredictionResult[]>([])

  // Real-time data generation and anomaly detection
  useEffect(() => {
    let interval: NodeJS.Timeout
    
    if (isRunning) {
      interval = setInterval(() => {
        const now = Date.now()
        
        setEquipment(prev => prev.map(eq => {
          // Generate realistic sensor data based on equipment health
          const healthFactor = eq.health / 100
          const degradationFactor = 1 - healthFactor
          
          const baseTemp = 45 + degradationFactor * 25
          const baseVibration = 1.5 + degradationFactor * 3.5
          const basePressure = 2.8 + degradationFactor * 1.2
          const baseCurrent = 12 + degradationFactor * 8
          const baseRpm = 3000 - degradationFactor * 500
          const baseOilLevel = 85 - degradationFactor * 30
          
          // Add realistic noise and trends
          const timeVariation = Math.sin(now / 10000) * 0.1
          
          const newSensorData: SensorData = {
            timestamp: now,
            temperature: baseTemp + (Math.random() - 0.5) * 5 + timeVariation * 10,
            vibration: baseVibration + (Math.random() - 0.5) * 1 + Math.abs(timeVariation * 2),
            pressure: basePressure + (Math.random() - 0.5) * 0.5,
            current: baseCurrent + (Math.random() - 0.5) * 2 + timeVariation * 3,
            rpm: baseRpm + (Math.random() - 0.5) * 100,
            oilLevel: Math.max(0, Math.min(100, baseOilLevel + (Math.random() - 0.5) * 5))
          }
          
          const updatedSensorData = [...eq.sensorData, newSensorData].slice(-50)
          
          // Advanced health degradation model
          const vibrationFactor = newSensorData.vibration > 3 ? 0.3 : 0.1
          const tempFactor = newSensorData.temperature > 60 ? 0.2 : 0.05
          const currentFactor = newSensorData.current > 16 ? 0.15 : 0.05
          
          const healthDegradation = (vibrationFactor + tempFactor + currentFactor) * 
            (eq.status === 'critical' ? 2 : eq.status === 'warning' ? 1.5 : 1)
          
          const newHealth = Math.max(0, eq.health - healthDegradation)
          const newRul = Math.max(0, eq.rul - 0.2)
          
          // Calculate anomaly score using multiple factors
          const tempAnomaly = Math.abs(newSensorData.temperature - 50) / 50
          const vibAnomaly = Math.abs(newSensorData.vibration - 2) / 4
          const currAnomaly = Math.abs(newSensorData.current - 14) / 10
          const anomalyScore = (tempAnomaly + vibAnomaly + currAnomaly) / 3
          
          let newStatus: 'healthy' | 'warning' | 'critical' = 'healthy'
          if (newHealth < 50 || anomalyScore > 0.6) newStatus = 'critical'
          else if (newHealth < 75 || anomalyScore > 0.3) newStatus = 'warning'
          
          return {
            ...eq,
            health: Math.round(newHealth * 10) / 10,
            rul: Math.round(newRul * 10) / 10,
            status: newStatus,
            sensorData: updatedSensorData,
            anomalyScore: Math.round(anomalyScore * 100) / 100
          }
        }))
        
        // Generate AI predictions
        if (Math.random() > 0.7) {
          generatePredictions()
        }
      }, 1000)
    }
    
    return () => clearInterval(interval)
  }, [isRunning])

  // Draw real-time charts
  useEffect(() => {
    if (!canvasRef.current || !isRunning) return
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const selectedEq = equipment.find(eq => eq.id === selectedEquipment)
    if (!selectedEq || selectedEq.sensorData.length < 2) return
    
    const draw = () => {
      // Set canvas size
      canvas.width = canvas.offsetWidth * window.devicePixelRatio
      canvas.height = canvas.offsetHeight * window.devicePixelRatio
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
      
      const width = canvas.offsetWidth
      const height = canvas.offsetHeight
      
      // Clear canvas
      ctx.fillStyle = '#1f2937'
      ctx.fillRect(0, 0, width, height)
      
      // Draw grid
      ctx.strokeStyle = '#374151'
      ctx.lineWidth = 0.5
      for (let i = 0; i <= 5; i++) {
        const y = (height / 5) * i
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(width, y)
        ctx.stroke()
      }
      
      // Draw sensor data lines
      const data = selectedEq.sensorData
      const pointsPerLine = data.length
      const xStep = width / (pointsPerLine - 1)
      
      // Temperature (Red)
      ctx.strokeStyle = '#ef4444'
      ctx.lineWidth = 2
      ctx.beginPath()
      data.forEach((d, i) => {
        const x = i * xStep
        const y = height - (d.temperature / 80) * height
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()
      
      // Vibration (Blue)
      ctx.strokeStyle = '#3b82f6'
      ctx.lineWidth = 2
      ctx.beginPath()
      data.forEach((d, i) => {
        const x = i * xStep
        const y = height - (d.vibration / 6) * height
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()
      
      // Current (Yellow)
      ctx.strokeStyle = '#eab308'
      ctx.lineWidth = 2
      ctx.beginPath()
      data.forEach((d, i) => {
        const x = i * xStep
        const y = height - (d.current / 25) * height
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()
      
      // Draw legend
      ctx.font = '12px Inter'
      ctx.fillStyle = '#ef4444'
      ctx.fillText('온도', 10, 20)
      ctx.fillStyle = '#3b82f6'
      ctx.fillText('진동', 60, 20)
      ctx.fillStyle = '#eab308'
      ctx.fillText('전류', 110, 20)
      
      animationRef.current = requestAnimationFrame(draw)
    }
    
    draw()
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [equipment, selectedEquipment, isRunning])

  const generatePredictions = () => {
    const components = ['베어링', '모터', '기어박스', '컨트롤러', '센서']
    const newPredictions: PredictionResult[] = []
    
    equipment.forEach(eq => {
      if (eq.status !== 'healthy') {
        const component = components[Math.floor(Math.random() * components.length)]
        const probability = eq.status === 'critical' ? 
          60 + Math.random() * 30 : 
          20 + Math.random() * 30
        
        newPredictions.push({
          component: `${eq.name} - ${component}`,
          failureProbability: Math.round(probability),
          confidence: Math.round(75 + Math.random() * 20),
          recommendation: probability > 50 ? 
            '즉시 점검 필요' : 
            '예방 정비 계획 수립'
        })
      }
    })
    
    setPredictions(newPredictions)
  }

  const selectedEq = equipment.find(eq => eq.id === selectedEquipment)
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/30'
      case 'warning': return 'text-yellow-600 bg-yellow-100 dark:text-yellow-400 dark:bg-yellow-900/30'
      case 'critical': return 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/30'
      default: return 'text-gray-600 bg-gray-100 dark:text-gray-400 dark:bg-gray-900/30'
    }
  }

  const getHealthColor = (health: number) => {
    if (health >= 70) return 'text-green-600'
    if (health >= 50) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getRulColor = (rul: number) => {
    if (rul >= 30) return 'text-green-600'
    if (rul >= 15) return 'text-yellow-600'
    return 'text-red-600'
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
                onClick={() => setIsRunning(!isRunning)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium ${
                  isRunning 
                    ? 'bg-red-600 text-white hover:bg-red-700' 
                    : 'bg-green-600 text-white hover:bg-green-700'
                }`}
              >
                {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isRunning ? '일시정지' : '시작'}
              </button>
              <button
                onClick={() => {
                  setIsRunning(false)
                  setEquipment(prev => prev.map(eq => ({
                    ...eq,
                    health: 85 + Math.random() * 15,
                    rul: 80 + Math.random() * 100,
                    status: 'healthy' as const,
                    sensorData: [],
                    anomalyScore: Math.random() * 0.2
                  })))
                  setPredictions([])
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
        {/* Title and Mode Selector */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">예측 유지보수 실험실</h1>
                <p className="text-lg text-gray-600 dark:text-gray-400">AI 기반 장비 고장 예측과 최적 정비 계획</p>
              </div>
            </div>
            
            {/* Analysis Mode Tabs */}
            <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
              <button
                onClick={() => setAnalysisMode('realtime')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  analysisMode === 'realtime' 
                    ? 'bg-white dark:bg-gray-800 text-gray-900 dark:text-white shadow-sm' 
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                실시간 모니터링
              </button>
              <button
                onClick={() => setAnalysisMode('prediction')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  analysisMode === 'prediction' 
                    ? 'bg-white dark:bg-gray-800 text-gray-900 dark:text-white shadow-sm' 
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                AI 예측 분석
              </button>
              <button
                onClick={() => setAnalysisMode('optimization')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  analysisMode === 'optimization' 
                    ? 'bg-white dark:bg-gray-800 text-gray-900 dark:text-white shadow-sm' 
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                정비 최적화
              </button>
            </div>
          </div>
        </div>

        {/* Equipment Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {equipment.map((eq) => (
            <div 
              key={eq.id}
              onClick={() => setSelectedEquipment(eq.id)}
              className={`bg-white dark:bg-gray-800 rounded-xl p-6 border cursor-pointer transition-all hover:shadow-lg ${
                selectedEquipment === eq.id 
                  ? 'border-blue-500 ring-2 ring-blue-200 dark:ring-blue-800' 
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
              }`}
            >
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white">{eq.name}</h3>
                  <p className="text-xs text-gray-500 dark:text-gray-400">{eq.type}</p>
                </div>
                <div className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(eq.status)}`}>
                  {eq.status === 'healthy' && <CheckCircle className="w-3 h-3" />}
                  {eq.status === 'warning' && <AlertCircle className="w-3 h-3" />}
                  {eq.status === 'critical' && <XCircle className="w-3 h-3" />}
                  {eq.status === 'healthy' && '정상'}
                  {eq.status === 'warning' && '주의'}
                  {eq.status === 'critical' && '위험'}
                </div>
              </div>
              
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-sm text-gray-600 dark:text-gray-400">건전성</span>
                    <span className={`text-sm font-bold ${getHealthColor(eq.health)}`}>{eq.health}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all duration-300 ${
                        eq.health >= 70 ? 'bg-green-500' : eq.health >= 50 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${eq.health}%` }}
                    ></div>
                  </div>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">잔여 수명</span>
                  <span className={`text-sm font-bold ${getRulColor(eq.rul)}`}>{eq.rul}일</span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">이상치 점수</span>
                  <span className={`text-sm font-bold ${
                    eq.anomalyScore < 0.3 ? 'text-green-600' : 
                    eq.anomalyScore < 0.6 ? 'text-yellow-600' : 'text-red-600'
                  }`}>{eq.anomalyScore.toFixed(2)}</span>
                </div>

                <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-500 dark:text-gray-400">정비 예정</span>
                    <span className="text-gray-700 dark:text-gray-300">{eq.nextMaintenance}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Main Content Area */}
        {selectedEq && (
          <>
            {/* Real-time Monitoring Mode */}
            {analysisMode === 'realtime' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Real-time Chart */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
                  <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">실시간 센서 데이터</h2>
                  
                  <div className="space-y-6">
                    {/* Chart Canvas */}
                    <div className="relative h-64 bg-gray-900 rounded-lg overflow-hidden">
                      <canvas 
                        ref={canvasRef}
                        className="w-full h-full"
                        style={{ imageRendering: 'crisp-edges' }}
                      />
                    </div>

                    {/* Sensor Values Grid */}
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <Thermometer className="w-4 h-4 text-red-500" />
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">온도</span>
                        </div>
                        <div className="text-xl font-bold text-red-600 dark:text-red-400">
                          {selectedEq.sensorData.length > 0 
                            ? `${selectedEq.sensorData[selectedEq.sensorData.length - 1].temperature.toFixed(1)}°C`
                            : '--°C'
                          }
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          정상: 40-55°C
                        </div>
                      </div>

                      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <Activity className="w-4 h-4 text-blue-500" />
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">진동</span>
                        </div>
                        <div className="text-xl font-bold text-blue-600 dark:text-blue-400">
                          {selectedEq.sensorData.length > 0 
                            ? `${selectedEq.sensorData[selectedEq.sensorData.length - 1].vibration.toFixed(1)}`
                            : '--'
                          } mm/s
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          정상: 0-2.5
                        </div>
                      </div>

                      <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <Gauge className="w-4 h-4 text-green-500" />
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">압력</span>
                        </div>
                        <div className="text-xl font-bold text-green-600 dark:text-green-400">
                          {selectedEq.sensorData.length > 0 
                            ? `${selectedEq.sensorData[selectedEq.sensorData.length - 1].pressure.toFixed(1)}`
                            : '--'
                          } bar
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          정상: 2.5-3.2
                        </div>
                      </div>

                      <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <Zap className="w-4 h-4 text-yellow-500" />
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">전류</span>
                        </div>
                        <div className="text-xl font-bold text-yellow-600 dark:text-yellow-400">
                          {selectedEq.sensorData.length > 0 
                            ? `${selectedEq.sensorData[selectedEq.sensorData.length - 1].current.toFixed(1)}`
                            : '--'
                          } A
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          정상: 10-15
                        </div>
                      </div>

                      <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <Gauge className="w-4 h-4 text-purple-500" />
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">RPM</span>
                        </div>
                        <div className="text-xl font-bold text-purple-600 dark:text-purple-400">
                          {selectedEq.sensorData.length > 0 
                            ? Math.round(selectedEq.sensorData[selectedEq.sensorData.length - 1].rpm)
                            : '--'
                          }
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          정상: 2800-3200
                        </div>
                      </div>

                      <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <Shield className="w-4 h-4 text-orange-500" />
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">윤활유</span>
                        </div>
                        <div className="text-xl font-bold text-orange-600 dark:text-orange-400">
                          {selectedEq.sensorData.length > 0 
                            ? `${Math.round(selectedEq.sensorData[selectedEq.sensorData.length - 1].oilLevel)}%`
                            : '--%'
                          }
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          정상: {'>'}70%
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Equipment Details & Status */}
                <div className="space-y-6">
                  {/* Current Status */}
                  <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">장비 상태 분석</h2>
                    
                    <div className="space-y-4">
                      <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                        <div>
                          <h3 className="font-semibold text-gray-900 dark:text-white">전체 건전성 지수</h3>
                          <p className="text-sm text-gray-600 dark:text-gray-400">Multi-factor health score</p>
                        </div>
                        <div className={`text-3xl font-bold ${getHealthColor(selectedEq.health)}`}>
                          {selectedEq.health}%
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                          <div className="flex items-center gap-2 mb-2">
                            <Clock className="w-4 h-4 text-blue-500" />
                            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">가동 시간</span>
                          </div>
                          <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
                            2,847 시간
                          </div>
                          <div className="text-xs text-gray-500 dark:text-gray-400">
                            마지막 정비 이후
                          </div>
                        </div>

                        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                          <div className="flex items-center gap-2 mb-2">
                            <TrendingUp className="w-4 h-4 text-green-500" />
                            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">가동률</span>
                          </div>
                          <div className="text-lg font-bold text-green-600 dark:text-green-400">
                            94.2%
                          </div>
                          <div className="text-xs text-gray-500 dark:text-gray-400">
                            이번 달 평균
                          </div>
                        </div>
                      </div>

                      {/* Alert Messages */}
                      {selectedEq.status === 'critical' && (
                        <div className="flex items-start gap-3 p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                          <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                          <div>
                            <h3 className="font-semibold text-red-700 dark:text-red-300">긴급 정비 필요</h3>
                            <p className="text-sm text-red-600 dark:text-red-400 mt-1">
                              센서 데이터가 위험 수준을 나타내고 있습니다. 
                              즉시 정비 팀에 연락하여 점검을 받으세요.
                            </p>
                          </div>
                        </div>
                      )}

                      {selectedEq.status === 'warning' && (
                        <div className="flex items-start gap-3 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
                          <AlertCircle className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
                          <div>
                            <h3 className="font-semibold text-yellow-700 dark:text-yellow-300">예방 정비 권장</h3>
                            <p className="text-sm text-yellow-600 dark:text-yellow-400 mt-1">
                              일부 센서 값이 정상 범위를 벗어나고 있습니다. 
                              2주 내 정비 계획을 수립하세요.
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Maintenance Schedule */}
                  <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">정비 일정</h2>
                    
                    <div className="space-y-4">
                      <div className="flex items-center gap-3 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                        <Calendar className="w-5 h-5 text-blue-500" />
                        <div className="flex-1">
                          <h3 className="font-semibold text-gray-900 dark:text-white">마지막 정비</h3>
                          <p className="text-sm text-gray-600 dark:text-gray-400">{selectedEq.lastMaintenance}</p>
                        </div>
                        <CheckCircle className="w-5 h-5 text-green-500" />
                      </div>

                      <div className="flex items-center gap-3 p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                        <Wrench className="w-5 h-5 text-orange-500" />
                        <div className="flex-1">
                          <h3 className="font-semibold text-gray-900 dark:text-white">다음 정비 예정</h3>
                          <p className="text-sm text-gray-600 dark:text-gray-400">{selectedEq.nextMaintenance}</p>
                        </div>
                        <span className="text-sm font-medium text-orange-600 dark:text-orange-400">
                          D-{Math.max(0, Math.floor(selectedEq.rul))}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* AI Prediction Mode */}
            {analysisMode === 'prediction' && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* AI Analysis Results */}
                <div className="lg:col-span-2 space-y-6">
                  <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                      <Brain className="w-6 h-6 text-purple-500" />
                      AI 고장 예측 분석
                    </h2>
                    
                    <div className="space-y-4">
                      {/* Failure Probability Chart */}
                      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
                        <h3 className="font-semibold text-gray-900 dark:text-white mb-4">고장 확률 예측 (30일)</h3>
                        <div className="space-y-3">
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm text-gray-600 dark:text-gray-400">베어링 시스템</span>
                              <span className="text-sm font-medium text-red-600">72%</span>
                            </div>
                            <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                              <div className="bg-red-500 h-2 rounded-full" style={{ width: '72%' }}></div>
                            </div>
                          </div>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm text-gray-600 dark:text-gray-400">모터 권선</span>
                              <span className="text-sm font-medium text-yellow-600">45%</span>
                            </div>
                            <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                              <div className="bg-yellow-500 h-2 rounded-full" style={{ width: '45%' }}></div>
                            </div>
                          </div>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm text-gray-600 dark:text-gray-400">기어박스</span>
                              <span className="text-sm font-medium text-green-600">18%</span>
                            </div>
                            <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                              <div className="bg-green-500 h-2 rounded-full" style={{ width: '18%' }}></div>
                            </div>
                          </div>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm text-gray-600 dark:text-gray-400">컨트롤러</span>
                              <span className="text-sm font-medium text-green-600">12%</span>
                            </div>
                            <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                              <div className="bg-green-500 h-2 rounded-full" style={{ width: '12%' }}></div>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* RUL Prediction */}
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                          <h3 className="font-semibold text-purple-900 dark:text-purple-100 mb-2">예측 RUL</h3>
                          <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                            {selectedEq.rul.toFixed(0)}일
                          </div>
                          <p className="text-sm text-purple-700 dark:text-purple-300 mt-1">
                            신뢰도: 87%
                          </p>
                        </div>
                        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                          <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">이상치 점수</h3>
                          <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                            {selectedEq.anomalyScore.toFixed(2)}
                          </div>
                          <p className="text-sm text-blue-700 dark:text-blue-300 mt-1">
                            임계값: 0.60
                          </p>
                        </div>
                      </div>

                      {/* Prediction Details */}
                      {predictions.length > 0 && (
                        <div className="space-y-3">
                          <h3 className="font-semibold text-gray-900 dark:text-white">상세 예측 결과</h3>
                          {predictions.map((pred, idx) => (
                            <div key={idx} className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                              <div className="flex items-center justify-between mb-2">
                                <span className="font-medium text-gray-900 dark:text-white">{pred.component}</span>
                                <span className={`text-sm font-bold ${
                                  pred.failureProbability > 50 ? 'text-red-600' : 'text-yellow-600'
                                }`}>
                                  {pred.failureProbability}% 위험
                                </span>
                              </div>
                              <div className="flex items-center justify-between text-sm">
                                <span className="text-gray-600 dark:text-gray-400">
                                  신뢰도: {pred.confidence}%
                                </span>
                                <span className="text-blue-600 dark:text-blue-400 font-medium">
                                  {pred.recommendation}
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* AI Model Info */}
                <div className="space-y-6">
                  <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">AI 모델 정보</h2>
                    
                    <div className="space-y-4">
                      <div className="text-center py-4">
                        <div className="w-20 h-20 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
                          <Brain className="w-10 h-10 text-white" />
                        </div>
                        <h3 className="font-semibold text-gray-900 dark:text-white">LSTM + Attention</h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                          시계열 예측 특화 모델
                        </p>
                      </div>

                      <div className="space-y-3">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600 dark:text-gray-400">모델 버전</span>
                          <span className="font-medium text-gray-900 dark:text-white">v3.2.1</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600 dark:text-gray-400">학습 데이터</span>
                          <span className="font-medium text-gray-900 dark:text-white">2.4M 샘플</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600 dark:text-gray-400">정확도</span>
                          <span className="font-medium text-green-600 dark:text-green-400">94.7%</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600 dark:text-gray-400">마지막 업데이트</span>
                          <span className="font-medium text-gray-900 dark:text-white">2024-11-15</span>
                        </div>
                      </div>

                      <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                        <h4 className="font-medium text-gray-900 dark:text-white mb-2">입력 특성</h4>
                        <div className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                          <div>• 온도, 진동, 압력, 전류</div>
                          <div>• RPM, 윤활유 레벨</div>
                          <div>• 가동 시간, 정비 이력</div>
                          <div>• 환경 조건 (습도, 온도)</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Optimization Mode */}
            {analysisMode === 'optimization' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Cost Analysis */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
                  <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">비용 분석 & 최적화</h2>
                  
                  <div className="space-y-6">
                    {/* Cost Comparison */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                        <h3 className="font-medium text-green-900 dark:text-green-100 mb-2">예방 정비 비용</h3>
                        <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                          ₩{(selectedEq.maintenanceCost / 1000000).toFixed(0)}M
                        </div>
                        <p className="text-sm text-green-700 dark:text-green-300 mt-1">
                          계획된 정비
                        </p>
                      </div>
                      <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
                        <h3 className="font-medium text-red-900 dark:text-red-100 mb-2">긴급 정비 비용</h3>
                        <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                          ₩{(selectedEq.downTimeCost / 1000000).toFixed(0)}M
                        </div>
                        <p className="text-sm text-red-700 dark:text-red-300 mt-1">
                          생산 중단 포함
                        </p>
                      </div>
                    </div>

                    {/* Optimization Chart */}
                    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
                      <h3 className="font-semibold text-gray-900 dark:text-white mb-4">최적 정비 시점</h3>
                      <div className="relative h-48 flex items-end justify-between gap-2">
                        {[70, 85, 100, 95, 80, 65, 50, 40, 35, 30].map((value, idx) => (
                          <div key={idx} className="flex-1 flex flex-col items-center gap-2">
                            <div 
                              className={`w-full rounded-t transition-all ${
                                idx === 5 ? 'bg-blue-500' : 'bg-gray-300 dark:bg-gray-600'
                              }`}
                              style={{ height: `${value}%` }}
                            ></div>
                            <span className="text-xs text-gray-500 dark:text-gray-400">
                              {idx * 10}일
                            </span>
                          </div>
                        ))}
                      </div>
                      <div className="mt-4 text-center">
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          최적 정비 시점: <span className="font-bold text-blue-600 dark:text-blue-400">50일 후</span>
                        </p>
                      </div>
                    </div>

                    {/* ROI Calculation */}
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                      <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-3">예측 정비 ROI</h3>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-blue-700 dark:text-blue-300">연간 절감액</span>
                          <span className="font-bold text-blue-900 dark:text-blue-100">
                            ₩{((selectedEq.downTimeCost - selectedEq.maintenanceCost) * 2 / 1000000).toFixed(0)}M
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-blue-700 dark:text-blue-300">다운타임 감소</span>
                          <span className="font-bold text-blue-900 dark:text-blue-100">75%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-blue-700 dark:text-blue-300">정비 효율 향상</span>
                          <span className="font-bold text-blue-900 dark:text-blue-100">40%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Maintenance Strategy */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
                  <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">정비 전략 추천</h2>
                  
                  <div className="space-y-6">
                    {/* Strategy Options */}
                    <div className="space-y-3">
                      <div className={`p-4 rounded-lg border-2 ${
                        selectedEq.health > 70 
                          ? 'border-green-500 bg-green-50 dark:bg-green-900/20' 
                          : 'border-gray-200 dark:border-gray-700'
                      }`}>
                        <div className="flex items-start gap-3">
                          <Shield className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                          <div>
                            <h3 className="font-semibold text-gray-900 dark:text-white">상태 기반 정비 (CBM)</h3>
                            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                              센서 데이터 기반 실시간 모니터링. 현재 장비에 최적.
                            </p>
                            {selectedEq.health > 70 && (
                              <span className="inline-block mt-2 text-xs font-medium text-green-600 dark:text-green-400">
                                ✓ 추천
                              </span>
                            )}
                          </div>
                        </div>
                      </div>

                      <div className={`p-4 rounded-lg border-2 ${
                        selectedEq.health <= 70 && selectedEq.health > 50
                          ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20' 
                          : 'border-gray-200 dark:border-gray-700'
                      }`}>
                        <div className="flex items-start gap-3">
                          <BarChart3 className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
                          <div>
                            <h3 className="font-semibold text-gray-900 dark:text-white">예측 정비 (PdM)</h3>
                            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                              AI 모델 기반 고장 예측. 정비 일정 최적화.
                            </p>
                            {selectedEq.health <= 70 && selectedEq.health > 50 && (
                              <span className="inline-block mt-2 text-xs font-medium text-yellow-600 dark:text-yellow-400">
                                ✓ 추천
                              </span>
                            )}
                          </div>
                        </div>
                      </div>

                      <div className={`p-4 rounded-lg border-2 ${
                        selectedEq.health <= 50
                          ? 'border-red-500 bg-red-50 dark:bg-red-900/20' 
                          : 'border-gray-200 dark:border-gray-700'
                      }`}>
                        <div className="flex items-start gap-3">
                          <Wrench className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                          <div>
                            <h3 className="font-semibold text-gray-900 dark:text-white">즉시 정비</h3>
                            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                              긴급 정비 필요. 생산 중단 위험 높음.
                            </p>
                            {selectedEq.health <= 50 && (
                              <span className="inline-block mt-2 text-xs font-medium text-red-600 dark:text-red-400">
                                ✓ 추천
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Action Items */}
                    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                      <h3 className="font-semibold text-gray-900 dark:text-white mb-3">권장 조치사항</h3>
                      <ul className="space-y-2 text-sm">
                        <li className="flex items-start gap-2">
                          <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0 mt-0.5" />
                          <span className="text-gray-700 dark:text-gray-300">
                            베어링 점검 및 윤활유 보충 (이번 주)
                          </span>
                        </li>
                        <li className="flex items-start gap-2">
                          <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0 mt-0.5" />
                          <span className="text-gray-700 dark:text-gray-300">
                            진동 센서 캘리브레이션 (2주 내)
                          </span>
                        </li>
                        <li className="flex items-start gap-2">
                          <Clock className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" />
                          <span className="text-gray-700 dark:text-gray-300">
                            예비 부품 재고 확인 및 주문
                          </span>
                        </li>
                        <li className="flex items-start gap-2">
                          <BarChart3 className="w-4 h-4 text-purple-500 flex-shrink-0 mt-0.5" />
                          <span className="text-gray-700 dark:text-gray-300">
                            월간 성능 리포트 생성 및 분석
                          </span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}