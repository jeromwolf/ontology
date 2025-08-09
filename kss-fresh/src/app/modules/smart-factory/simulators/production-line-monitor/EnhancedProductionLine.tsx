'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import { ArrowLeft, Play, Pause, RotateCcw, Package, AlertCircle, TrendingUp, Clock, Activity, Zap, CheckCircle, AlertTriangle, Sparkles, Flame } from 'lucide-react'

interface Station {
  id: string
  name: string
  status: 'active' | 'idle' | 'warning' | 'error'
  speed: number
  quality: number
  items: number
  efficiency: number
}

interface Product {
  id: number
  x: number
  y: number
  status: 'good' | 'defect'
  station: number
}

interface Alert {
  id: string
  type: 'warning' | 'error' | 'info' | 'success'
  message: string
  timestamp: Date
}

export default function EnhancedProductionLinePage() {
  const [isRunning, setIsRunning] = useState(false)
  const [simulationSpeed, setSimulationSpeed] = useState(1)
  const [showEffects, setShowEffects] = useState(true)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  
  const [stations, setStations] = useState<Station[]>([
    { id: 'S1', name: '재료 투입', status: 'active', speed: 85, quality: 98, items: 0, efficiency: 92 },
    { id: 'S2', name: '성형 공정', status: 'active', speed: 78, quality: 96, items: 0, efficiency: 88 },
    { id: 'S3', name: '조립 라인', status: 'active', speed: 82, quality: 97, items: 0, efficiency: 90 },
    { id: 'S4', name: '품질 검사', status: 'warning', speed: 70, quality: 99, items: 0, efficiency: 75 },
    { id: 'S5', name: '포장 공정', status: 'active', speed: 90, quality: 100, items: 0, efficiency: 95 }
  ])

  const [products, setProducts] = useState<Product[]>([])
  const [totalProduction, setTotalProduction] = useState(0)
  const [defectRate, setDefectRate] = useState(1.2)
  const [overallEfficiency, setOverallEfficiency] = useState(88)
  const [productionRate, setProductionRate] = useState(0)
  
  const [alerts, setAlerts] = useState<Alert[]>([
    { id: '1', type: 'info', message: '생산 라인 모니터링 시작', timestamp: new Date() }
  ])

  // 파티클 및 특수 효과 애니메이션
  useEffect(() => {
    if (!canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight

    let particles: Array<{x: number, y: number, vx: number, vy: number, life: number, color: string}> = []
    let sparkles: Array<{x: number, y: number, size: number, life: number}> = []

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // 파티클 업데이트
      particles = particles.filter(particle => {
        particle.x += particle.vx
        particle.y += particle.vy
        particle.vy += 0.1 // 중력 효과
        particle.life -= 0.02

        if (particle.life <= 0) return false

        ctx.globalAlpha = particle.life
        ctx.fillStyle = particle.color
        ctx.beginPath()
        ctx.arc(particle.x, particle.y, 2, 0, Math.PI * 2)
        ctx.fill()

        return true
      })

      // 스파클 효과
      sparkles = sparkles.filter(sparkle => {
        sparkle.life -= 0.03
        if (sparkle.life <= 0) return false

        ctx.globalAlpha = sparkle.life
        ctx.strokeStyle = '#FFD700'
        ctx.lineWidth = 2
        
        const size = sparkle.size * sparkle.life
        ctx.beginPath()
        ctx.moveTo(sparkle.x - size, sparkle.y)
        ctx.lineTo(sparkle.x + size, sparkle.y)
        ctx.moveTo(sparkle.x, sparkle.y - size)
        ctx.lineTo(sparkle.x, sparkle.y + size)
        ctx.stroke()

        return true
      })

      // 효과 생성
      if (isRunning && showEffects) {
        stations.forEach((station, index) => {
          if (station.status === 'active' && Math.random() < 0.05) {
            const x = (index + 1) * (canvas.width / 6)
            const y = canvas.height / 2

            // 파티클 생성
            for (let i = 0; i < 3; i++) {
              particles.push({
                x,
                y,
                vx: (Math.random() - 0.5) * 3,
                vy: -Math.random() * 3 - 1,
                life: 1,
                color: station.efficiency > 90 ? '#10B981' : '#F59E0B'
              })
            }

            // 고효율일 때 스파클 효과
            if (station.efficiency > 90 && Math.random() < 0.3) {
              sparkles.push({
                x,
                y: y - 20,
                size: 15,
                life: 1
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
  }, [isRunning, showEffects, stations])

  // 생산 시뮬레이션
  useEffect(() => {
    let interval: NodeJS.Timeout
    
    if (isRunning) {
      interval = setInterval(() => {
        // 제품 생성 및 이동
        setProducts(prev => {
          let newProducts = [...prev]
          
          // 새 제품 생성
          if (Math.random() < 0.3 * simulationSpeed) {
            newProducts.push({
              id: Date.now(),
              x: 0,
              y: 50,
              status: Math.random() > 0.02 ? 'good' : 'defect',
              station: 0
            })
          }

          // 제품 이동
          newProducts = newProducts.map(product => {
            const newX = product.x + 2 * simulationSpeed
            const newStation = Math.floor(newX / 20)
            
            return {
              ...product,
              x: newX,
              station: newStation
            }
          }).filter(product => product.x < 100)

          return newProducts
        })

        // 스테이션 업데이트
        setStations(prev => prev.map(station => {
          const randomChange = (Math.random() - 0.5) * 5
          const newSpeed = Math.max(50, Math.min(100, station.speed + randomChange))
          const newQuality = Math.max(90, Math.min(100, station.quality + (Math.random() - 0.5) * 2))
          const newEfficiency = Math.max(60, Math.min(100, station.efficiency + (Math.random() - 0.5) * 3))
          
          // 상태 업데이트
          let newStatus = station.status
          if (newEfficiency < 70) newStatus = 'warning'
          else if (newEfficiency < 50) newStatus = 'error'
          else if (newEfficiency > 85) newStatus = 'active'
          
          return {
            ...station,
            speed: Math.round(newSpeed),
            quality: Math.round(newQuality * 10) / 10,
            efficiency: Math.round(newEfficiency),
            status: newStatus,
            items: products.filter(p => Math.floor(p.x / 20) === stations.indexOf(station)).length
          }
        }))

        // 메트릭 업데이트
        setTotalProduction(prev => prev + Math.floor(Math.random() * 3 * simulationSpeed))
        setProductionRate(Math.round(60 + Math.random() * 40 * simulationSpeed))
        setDefectRate(Math.max(0, Math.min(5, 1.2 + (Math.random() - 0.5) * 0.5)))
        
        const avgEfficiency = stations.reduce((acc, s) => acc + s.efficiency, 0) / stations.length
        setOverallEfficiency(Math.round(avgEfficiency))

        // 알림 생성
        if (Math.random() < 0.1) {
          const alertTypes = [
            { type: 'success' as const, message: '생산 목표 달성! 🎉' },
            { type: 'warning' as const, message: '품질 검사 대기 중' },
            { type: 'info' as const, message: '새로운 주문 접수' }
          ]
          const alert = alertTypes[Math.floor(Math.random() * alertTypes.length)]
          
          setAlerts(prev => [...prev.slice(-4), {
            id: Date.now().toString(),
            ...alert,
            timestamp: new Date()
          }])
        }
      }, 1000 / simulationSpeed)
    }
    
    return () => clearInterval(interval)
  }, [isRunning, simulationSpeed, products])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-500'
      case 'idle': return 'text-gray-500'
      case 'warning': return 'text-yellow-500'
      case 'error': return 'text-red-500'
      default: return 'text-gray-500'
    }
  }

  const getStatusBg = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500'
      case 'idle': return 'bg-gray-500'
      case 'warning': return 'bg-yellow-500'
      case 'error': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 relative overflow-hidden">
      {/* 특수 효과 캔버스 */}
      <canvas 
        ref={canvasRef}
        className="fixed inset-0 pointer-events-none z-30"
        style={{ width: '100%', height: '100%' }}
      />

      {/* 고효율 모드 배경 효과 */}
      {isRunning && overallEfficiency > 90 && (
        <div className="fixed inset-0 pointer-events-none z-20">
          <div className="absolute inset-0 bg-gradient-to-br from-green-400/10 to-blue-500/10 animate-pulse"></div>
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
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-600 dark:text-gray-400">속도:</label>
                <select 
                  value={simulationSpeed} 
                  onChange={(e) => setSimulationSpeed(Number(e.target.value))}
                  className="px-2 py-1 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-sm"
                >
                  <option value={0.5}>0.5x</option>
                  <option value={1}>1x</option>
                  <option value={2}>2x</option>
                  <option value={3}>3x</option>
                </select>
              </div>
              <button
                onClick={() => setShowEffects(!showEffects)}
                className={`px-3 py-1 rounded text-sm ${
                  showEffects 
                    ? 'bg-purple-600 text-white' 
                    : 'bg-gray-300 dark:bg-gray-600 text-gray-700 dark:text-gray-300'
                }`}
              >
                <Sparkles className="w-4 h-4 inline mr-1" />
                효과
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
                  setProducts([])
                  setTotalProduction(0)
                  setAlerts([{ id: '1', type: 'info', message: '시스템 초기화 완료', timestamp: new Date() }])
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
            <div className="w-12 h-12 bg-gradient-to-br from-orange-500 to-red-600 rounded-xl flex items-center justify-center">
              <Package className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                실시간 생산 라인 모니터
                {isRunning && overallEfficiency > 90 && (
                  <Flame className="w-6 h-6 text-orange-500 animate-pulse" />
                )}
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400">공정별 실시간 모니터링 및 분석</p>
            </div>
          </div>
        </div>

        {/* Production Line Visualization */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6 mb-6">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">생산 라인 시각화</h2>
          
          <div className="relative h-64 bg-gray-100 dark:bg-gray-700 rounded-lg overflow-hidden">
            {/* Conveyor Belt */}
            <div className="absolute bottom-0 left-0 right-0 h-20 bg-gray-300 dark:bg-gray-600">
              <div className="h-full flex items-center">
                <div className={`h-1 bg-gray-400 dark:bg-gray-500 w-full ${isRunning ? 'animate-conveyor' : ''}`}></div>
              </div>
            </div>

            {/* Stations */}
            {stations.map((station, index) => (
              <div
                key={station.id}
                className="absolute bottom-20 transform -translate-x-1/2"
                style={{ left: `${(index + 1) * 16.66}%` }}
              >
                <div className={`relative ${isRunning && station.status === 'active' ? 'animate-pulse' : ''}`}>
                  <div className={`w-16 h-24 ${getStatusBg(station.status)} rounded-t-lg flex items-center justify-center`}>
                    <span className="text-2xl">🏭</span>
                  </div>
                  <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 text-xs font-medium text-gray-700 dark:text-gray-300 whitespace-nowrap">
                    {station.name}
                  </div>
                  <div className="absolute -bottom-6 left-1/2 transform -translate-x-1/2 text-xs text-gray-500 dark:text-gray-400">
                    {station.efficiency}%
                  </div>
                  {station.items > 0 && (
                    <div className="absolute -top-2 -right-2 w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs">
                      {station.items}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Products */}
            {products.map(product => (
              <div
                key={product.id}
                className="absolute bottom-24 w-8 h-8 transform -translate-x-1/2 transition-all duration-1000"
                style={{ left: `${product.x}%` }}
              >
                <div className={`w-full h-full rounded ${
                  product.status === 'good' ? 'bg-green-500' : 'bg-red-500'
                } flex items-center justify-center animate-bounce`}>
                  📦
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Metrics Dashboard */}
          <div className="lg:col-span-2 space-y-6">
            {/* Real-time Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className={`bg-white dark:bg-gray-800 rounded-xl p-4 border ${
                isRunning && totalProduction > 100 ? 'border-green-500 animate-pulse' : 'border-gray-200 dark:border-gray-700'
              }`}>
                <div className="flex items-center gap-2 mb-2">
                  <Package className="w-5 h-5 text-blue-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">총 생산량</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {totalProduction.toLocaleString()}
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="w-5 h-5 text-green-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">생산 속도</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {productionRate}/분
                </div>
              </div>

              <div className={`bg-white dark:bg-gray-800 rounded-xl p-4 border ${
                defectRate > 3 ? 'border-red-500' : 'border-gray-200 dark:border-gray-700'
              }`}>
                <div className="flex items-center gap-2 mb-2">
                  <AlertCircle className="w-5 h-5 text-orange-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">불량률</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {defectRate.toFixed(1)}%
                </div>
              </div>

              <div className={`bg-white dark:bg-gray-800 rounded-xl p-4 border ${
                overallEfficiency > 90 ? 'border-green-500 ring-2 ring-green-500 ring-opacity-50' : 'border-gray-200 dark:border-gray-700'
              }`}>
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-5 h-5 text-purple-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">전체 효율</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {overallEfficiency}%
                </div>
              </div>
            </div>

            {/* Station Details */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">공정별 상세 정보</h3>
              
              <div className="space-y-4">
                {stations.map((station) => (
                  <div key={station.id} className="border-b border-gray-200 dark:border-gray-700 pb-4 last:border-0">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3">
                        <div className={`w-3 h-3 rounded-full ${getStatusBg(station.status)} ${
                          station.status === 'active' && isRunning ? 'animate-pulse' : ''
                        }`}></div>
                        <h4 className="font-medium text-gray-900 dark:text-white">{station.name}</h4>
                      </div>
                      <span className={`text-sm font-medium ${getStatusColor(station.status)}`}>
                        {station.status === 'active' ? '가동중' : 
                         station.status === 'warning' ? '주의' : 
                         station.status === 'error' ? '오류' : '대기'}
                      </span>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">속도</span>
                        <div className="mt-1">
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full transition-all duration-500 ${
                                station.speed > 80 ? 'bg-green-500' : 
                                station.speed > 60 ? 'bg-yellow-500' : 'bg-red-500'
                              }`}
                              style={{ width: `${station.speed}%` }}
                            ></div>
                          </div>
                          <span className="text-xs text-gray-600 dark:text-gray-400">{station.speed}%</span>
                        </div>
                      </div>
                      
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">품질</span>
                        <div className="mt-1">
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div 
                              className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                              style={{ width: `${station.quality}%` }}
                            ></div>
                          </div>
                          <span className="text-xs text-gray-600 dark:text-gray-400">{station.quality}%</span>
                        </div>
                      </div>
                      
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">효율</span>
                        <div className="mt-1">
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full transition-all duration-500 ${
                                station.efficiency > 90 ? 'bg-purple-500' : 
                                station.efficiency > 70 ? 'bg-indigo-500' : 'bg-gray-500'
                              }`}
                              style={{ width: `${station.efficiency}%` }}
                            ></div>
                          </div>
                          <span className="text-xs text-gray-600 dark:text-gray-400">{station.efficiency}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Alerts & Controls */}
          <div className="space-y-6">
            {/* Real-time Alerts */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                실시간 알림
                <span className="text-xs text-gray-500 dark:text-gray-400">({alerts.length})</span>
              </h3>
              
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {alerts.slice().reverse().map((alert) => (
                  <div 
                    key={alert.id} 
                    className={`p-3 rounded-lg text-sm ${
                      alert.type === 'error' ? 'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-300' :
                      alert.type === 'warning' ? 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-300' :
                      alert.type === 'success' ? 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-300' :
                      'bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300'
                    } animate-fade-in`}
                  >
                    <div className="flex items-start gap-2">
                      {alert.type === 'error' ? <AlertTriangle className="w-4 h-4 mt-0.5" /> :
                       alert.type === 'warning' ? <AlertCircle className="w-4 h-4 mt-0.5" /> :
                       alert.type === 'success' ? <CheckCircle className="w-4 h-4 mt-0.5" /> :
                       <Activity className="w-4 h-4 mt-0.5" />}
                      <div className="flex-1">
                        <p>{alert.message}</p>
                        <p className="text-xs opacity-70 mt-1">
                          {alert.timestamp.toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">빠른 작업</h3>
              
              <div className="space-y-3">
                <button className="w-full p-3 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors text-sm font-medium">
                  <Zap className="w-4 h-4 inline mr-2" />
                  효율성 최적화
                </button>
                
                <button className="w-full p-3 bg-orange-50 dark:bg-orange-900/20 text-orange-700 dark:text-orange-300 rounded-lg hover:bg-orange-100 dark:hover:bg-orange-900/30 transition-colors text-sm font-medium">
                  <AlertCircle className="w-4 h-4 inline mr-2" />
                  품질 검사 강화
                </button>
                
                <button className="w-full p-3 bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors text-sm font-medium">
                  <Clock className="w-4 h-4 inline mr-2" />
                  속도 조절
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes conveyor {
          0% { transform: translateX(0); }
          100% { transform: translateX(100px); }
        }
        
        .animate-conveyor {
          animation: conveyor 2s linear infinite;
        }
        
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