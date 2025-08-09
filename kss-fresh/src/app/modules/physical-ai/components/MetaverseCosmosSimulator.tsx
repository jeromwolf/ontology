'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { 
  Globe, Layers, Zap, Activity, Eye, Building, 
  Car, Lightbulb, Wind, CloudRain, Sun, Play, Pause,
  Settings, Map, Users, Battery, Wifi, BarChart3,
  RefreshCw, AlertTriangle, Check
} from 'lucide-react'

interface DigitalTwin {
  id: string
  type: 'building' | 'vehicle' | 'sensor' | 'infrastructure'
  name: string
  position: { x: number; y: number; z: number }
  status: 'active' | 'warning' | 'inactive'
  data: any
}

interface CityMetrics {
  traffic: number
  energy: number
  airQuality: number
  noise: number
  temperature: number
  population: number
}

export default function MetaverseCosmosSimulator() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [selectedView, setSelectedView] = useState<'3d' | 'map' | 'data'>('3d')
  const [selectedLayer, setSelectedLayer] = useState<'all' | 'traffic' | 'energy' | 'environment'>('all')
  const [timeScale, setTimeScale] = useState(1)
  const [currentTime, setCurrentTime] = useState(new Date())
  
  const [cityMetrics, setCityMetrics] = useState<CityMetrics>({
    traffic: 65,
    energy: 78,
    airQuality: 82,
    noise: 45,
    temperature: 22,
    population: 50000
  })

  const [digitalTwins, setDigitalTwins] = useState<DigitalTwin[]>([
    {
      id: 'building-1',
      type: 'building',
      name: '스마트 오피스 타워',
      position: { x: 100, y: 0, z: 100 },
      status: 'active',
      data: { floors: 30, occupancy: 85, energyUsage: 450 }
    },
    {
      id: 'vehicle-1',
      type: 'vehicle',
      name: '자율주행 버스 #001',
      position: { x: 200, y: 0, z: 150 },
      status: 'active',
      data: { speed: 45, passengers: 28, battery: 82 }
    },
    {
      id: 'sensor-1',
      type: 'sensor',
      name: '대기질 센서 A-12',
      position: { x: 150, y: 20, z: 200 },
      status: 'active',
      data: { pm25: 35, co2: 420, humidity: 65 }
    }
  ])

  // 3D 씬 렌더링
  const render3DScene = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Clear canvas
    ctx.fillStyle = '#1a1a1a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    
    // Grid
    ctx.strokeStyle = '#333'
    ctx.lineWidth = 1
    for (let i = 0; i <= 10; i++) {
      const x = (i / 10) * canvas.width
      const y = (i / 10) * canvas.height
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, canvas.height)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(canvas.width, y)
      ctx.stroke()
    }
    
    // Render digital twins
    digitalTwins.forEach(twin => {
      const x = (twin.position.x / 300) * canvas.width
      const y = canvas.height - (twin.position.z / 300) * canvas.height
      
      // Twin visualization
      if (twin.type === 'building') {
        const height = twin.data.floors * 2
        ctx.fillStyle = twin.status === 'active' ? '#4ade80' : '#f87171'
        ctx.fillRect(x - 20, y - height, 40, height)
        ctx.strokeStyle = '#fff'
        ctx.strokeRect(x - 20, y - height, 40, height)
      } else if (twin.type === 'vehicle') {
        ctx.beginPath()
        ctx.arc(x, y, 8, 0, Math.PI * 2)
        ctx.fillStyle = '#60a5fa'
        ctx.fill()
        ctx.strokeStyle = '#fff'
        ctx.stroke()
      } else if (twin.type === 'sensor') {
        ctx.beginPath()
        ctx.moveTo(x, y - 10)
        ctx.lineTo(x - 8, y + 5)
        ctx.lineTo(x + 8, y + 5)
        ctx.closePath()
        ctx.fillStyle = '#fbbf24'
        ctx.fill()
        ctx.strokeStyle = '#fff'
        ctx.stroke()
      }
      
      // Label
      ctx.fillStyle = '#fff'
      ctx.font = '10px Arial'
      ctx.textAlign = 'center'
      ctx.fillText(twin.name, x, y + 20)
    })
    
    // Time display
    ctx.fillStyle = '#fff'
    ctx.font = '14px Arial'
    ctx.textAlign = 'left'
    ctx.fillText(`시뮬레이션 시간: ${currentTime.toLocaleTimeString()}`, 10, 20)
  }, [digitalTwins, currentTime])

  // 시뮬레이션 업데이트
  const updateSimulation = useCallback(() => {
    // Update time
    setCurrentTime(prev => new Date(prev.getTime() + 1000 * 60 * timeScale))
    
    // Update metrics
    setCityMetrics(prev => ({
      traffic: Math.max(0, Math.min(100, prev.traffic + (Math.random() - 0.5) * 5)),
      energy: Math.max(0, Math.min(100, prev.energy + (Math.random() - 0.5) * 3)),
      airQuality: Math.max(0, Math.min(100, prev.airQuality + (Math.random() - 0.5) * 2)),
      noise: Math.max(0, Math.min(100, prev.noise + (Math.random() - 0.5) * 4)),
      temperature: Math.max(-10, Math.min(40, prev.temperature + (Math.random() - 0.5) * 0.5)),
      population: prev.population + Math.floor((Math.random() - 0.5) * 10)
    }))
    
    // Update digital twins
    setDigitalTwins(prev => prev.map(twin => {
      if (twin.type === 'vehicle') {
        return {
          ...twin,
          position: {
            x: (twin.position.x + (Math.random() - 0.5) * 10 + 300) % 300,
            y: 0,
            z: (twin.position.z + (Math.random() - 0.5) * 10 + 300) % 300
          },
          data: {
            ...twin.data,
            speed: Math.max(0, Math.min(80, twin.data.speed + (Math.random() - 0.5) * 5)),
            battery: Math.max(0, twin.data.battery - 0.1)
          }
        }
      }
      return twin
    }))
  }, [timeScale])

  // Animation loop
  useEffect(() => {
    if (isRunning) {
      const animate = () => {
        render3DScene()
        updateSimulation()
        animationRef.current = requestAnimationFrame(animate)
      }
      animate()
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning, render3DScene, updateSimulation])

  // Initial render
  useEffect(() => {
    render3DScene()
  }, [render3DScene])

  const getMetricColor = (value: number) => {
    if (value >= 80) return 'text-green-600'
    if (value >= 60) return 'text-yellow-600'
    return 'text-red-600'
  }

  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <div className="flex flex-wrap items-center gap-4">
          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
              isRunning
                ? 'bg-red-600 text-white hover:bg-red-700'
                : 'bg-green-600 text-white hover:bg-green-700'
            } transition-colors`}
          >
            {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isRunning ? '일시정지' : '시작'}
          </button>
          
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium">시간 배속:</label>
            <select
              value={timeScale}
              onChange={(e) => setTimeScale(Number(e.target.value))}
              className="px-3 py-1 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-800"
            >
              <option value={1}>1x</option>
              <option value={5}>5x</option>
              <option value={10}>10x</option>
              <option value={60}>60x</option>
            </select>
          </div>
          
          <div className="flex gap-2">
            <button
              onClick={() => setSelectedView('3d')}
              className={`px-3 py-1 rounded ${selectedView === '3d' ? 'bg-purple-600 text-white' : 'bg-gray-200 dark:bg-gray-600'}`}
            >
              <Eye className="w-4 h-4" />
            </button>
            <button
              onClick={() => setSelectedView('map')}
              className={`px-3 py-1 rounded ${selectedView === 'map' ? 'bg-purple-600 text-white' : 'bg-gray-200 dark:bg-gray-600'}`}
            >
              <Map className="w-4 h-4" />
            </button>
            <button
              onClick={() => setSelectedView('data')}
              className={`px-3 py-1 rounded ${selectedView === 'data' ? 'bg-purple-600 text-white' : 'bg-gray-200 dark:bg-gray-600'}`}
            >
              <BarChart3 className="w-4 h-4" />
            </button>
          </div>
          
          <div className="flex gap-2 ml-auto">
            <select
              value={selectedLayer}
              onChange={(e) => setSelectedLayer(e.target.value as any)}
              className="px-3 py-1 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-800"
            >
              <option value="all">모든 레이어</option>
              <option value="traffic">교통</option>
              <option value="energy">에너지</option>
              <option value="environment">환경</option>
            </select>
          </div>
        </div>
      </div>

      {/* Main View */}
      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          {selectedView === '3d' && (
            <div className="bg-black rounded-lg overflow-hidden">
              <canvas
                ref={canvasRef}
                width={800}
                height={600}
                className="w-full"
              />
            </div>
          )}
          
          {selectedView === 'map' && (
            <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-8 h-[600px] flex items-center justify-center">
              <div className="text-center text-gray-600 dark:text-gray-400">
                <Map className="w-16 h-16 mx-auto mb-4" />
                <p>2D 지도 뷰 (구현 예정)</p>
              </div>
            </div>
          )}
          
          {selectedView === 'data' && (
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 h-[600px] overflow-auto">
              <h3 className="text-lg font-semibold mb-4">실시간 데이터 스트림</h3>
              <div className="space-y-4">
                {digitalTwins.map(twin => (
                  <div key={twin.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium">{twin.name}</h4>
                      <span className={`px-2 py-1 rounded text-xs ${
                        twin.status === 'active' ? 'bg-green-100 text-green-700' :
                        twin.status === 'warning' ? 'bg-yellow-100 text-yellow-700' :
                        'bg-red-100 text-red-700'
                      }`}>
                        {twin.status}
                      </span>
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      <pre>{JSON.stringify(twin.data, null, 2)}</pre>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Metrics Panel */}
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-purple-600" />
              도시 메트릭스
            </h3>
            
            <div className="space-y-3">
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm flex items-center gap-1">
                    <Car className="w-4 h-4" /> 교통 흐름
                  </span>
                  <span className={`font-semibold ${getMetricColor(cityMetrics.traffic)}`}>
                    {cityMetrics.traffic.toFixed(0)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      cityMetrics.traffic >= 80 ? 'bg-green-500' :
                      cityMetrics.traffic >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${cityMetrics.traffic}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm flex items-center gap-1">
                    <Battery className="w-4 h-4" /> 에너지 효율
                  </span>
                  <span className={`font-semibold ${getMetricColor(cityMetrics.energy)}`}>
                    {cityMetrics.energy.toFixed(0)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      cityMetrics.energy >= 80 ? 'bg-green-500' :
                      cityMetrics.energy >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${cityMetrics.energy}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm flex items-center gap-1">
                    <Wind className="w-4 h-4" /> 대기질
                  </span>
                  <span className={`font-semibold ${getMetricColor(cityMetrics.airQuality)}`}>
                    {cityMetrics.airQuality.toFixed(0)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      cityMetrics.airQuality >= 80 ? 'bg-green-500' :
                      cityMetrics.airQuality >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${cityMetrics.airQuality}%` }}
                  />
                </div>
              </div>

              <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
                <div className="flex justify-between text-sm">
                  <span>온도</span>
                  <span className="font-semibold">{cityMetrics.temperature.toFixed(1)}°C</span>
                </div>
                <div className="flex justify-between text-sm mt-1">
                  <span>인구</span>
                  <span className="font-semibold">{cityMetrics.population.toLocaleString()}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Digital Twin Status */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Layers className="w-5 h-5 text-purple-600" />
              디지털 트윈 상태
            </h3>
            
            <div className="space-y-2">
              {digitalTwins.slice(0, 5).map(twin => (
                <div key={twin.id} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded">
                  <div className="flex items-center gap-2">
                    {twin.type === 'building' && <Building className="w-4 h-4" />}
                    {twin.type === 'vehicle' && <Car className="w-4 h-4" />}
                    {twin.type === 'sensor' && <Wifi className="w-4 h-4" />}
                    <span className="text-sm">{twin.name}</span>
                  </div>
                  {twin.status === 'active' && <Check className="w-4 h-4 text-green-600" />}
                  {twin.status === 'warning' && <AlertTriangle className="w-4 h-4 text-yellow-600" />}
                </div>
              ))}
            </div>
            
            <button className="mt-3 w-full text-center text-sm text-purple-600 hover:text-purple-700">
              전체 보기 ({digitalTwins.length})
            </button>
          </div>

          {/* Weather Simulation */}
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-gray-800 dark:to-gray-700 rounded-lg p-4">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <CloudRain className="w-5 h-5 text-blue-600" />
              날씨 시뮬레이션
            </h3>
            
            <div className="flex items-center justify-around">
              <div className="text-center">
                <Sun className="w-8 h-8 text-yellow-500 mx-auto mb-1" />
                <p className="text-xs">맑음</p>
                <p className="text-sm font-semibold">70%</p>
              </div>
              <div className="text-center">
                <CloudRain className="w-8 h-8 text-gray-500 mx-auto mb-1" />
                <p className="text-xs">비</p>
                <p className="text-sm font-semibold">20%</p>
              </div>
              <div className="text-center">
                <Wind className="w-8 h-8 text-blue-500 mx-auto mb-1" />
                <p className="text-xs">바람</p>
                <p className="text-sm font-semibold">10m/s</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Info Panel */}
      <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
          <Globe className="w-6 h-6 text-purple-600" />
          메타버스 COSMOS 시뮬레이터
        </h3>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          이 시뮬레이터는 NVIDIA Omniverse와 COSMOS 비전을 기반으로 한 도시 규모 디지털 트윈 환경입니다.
          실시간 센서 데이터, 물리 시뮬레이션, AI 예측이 통합되어 미래 도시의 운영을 최적화합니다.
        </p>
        <div className="grid md:grid-cols-3 gap-4 text-sm">
          <div>
            <h4 className="font-semibold mb-1">실시간 동기화</h4>
            <p className="text-gray-600 dark:text-gray-400">
              IoT 센서와 디지털 트윈의 밀리초 단위 동기화
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-1">물리 시뮬레이션</h4>
            <p className="text-gray-600 dark:text-gray-400">
              PhysX 기반 정확한 물리 법칙 적용
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-1">AI 예측</h4>
            <p className="text-gray-600 dark:text-gray-400">
              교통, 에너지, 환경 패턴 예측 및 최적화
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}