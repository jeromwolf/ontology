'use client'

import { useState, useEffect, useRef } from 'react'
import { Car, Navigation, AlertTriangle, Gauge, MapPin, Play, Pause, Settings2, Zap } from 'lucide-react'

interface Vehicle {
  x: number
  y: number
  angle: number
  speed: number
  targetSpeed: number
  lane: number
  type: 'ego' | 'traffic'
  color: string
}

interface Obstacle {
  x: number
  y: number
  width: number
  height: number
  type: 'static' | 'pedestrian'
}

interface PathPoint {
  x: number
  y: number
  speed: number
}

export default function AutonomousDrivingSim() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const [isRunning, setIsRunning] = useState(false)
  const [autonomyLevel, setAutonomyLevel] = useState(3) // SAE Level
  const [scenario, setScenario] = useState<'highway' | 'urban' | 'parking'>('highway')
  const [speed, setSpeed] = useState(0)
  const [steeringAngle, setSteeringAngle] = useState(0)
  const [detectedObjects, setDetectedObjects] = useState<string[]>([])
  const [plannedPath, setPlannedPath] = useState<PathPoint[]>([])
  const [safetyMetrics, setSafetyMetrics] = useState({
    ttc: Infinity, // Time to Collision
    minDistance: Infinity,
    laneKeeping: 100
  })

  const egoVehicleRef = useRef<Vehicle>({
    x: 0,
    y: 0,
    angle: 0,
    speed: 0,
    targetSpeed: 60,
    lane: 1,
    type: 'ego',
    color: '#3B82F6'
  })

  const trafficRef = useRef<Vehicle[]>([])
  const obstaclesRef = useRef<Obstacle[]>([])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Canvas 크기 조정
    const resizeCanvas = () => {
      const container = canvas.parentElement
      if (container) {
        canvas.width = container.clientWidth
        canvas.height = container.clientHeight
      }
    }
    
    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)

    // 시나리오별 초기화
    const initScenario = () => {
      const ego = egoVehicleRef.current
      trafficRef.current = []
      obstaclesRef.current = []
      
      switch (scenario) {
        case 'highway':
          ego.x = 100
          ego.y = canvas.height / 2
          ego.angle = 0
          ego.lane = 1
          ego.targetSpeed = 100
          
          // 고속도로 교통 차량
          for (let i = 0; i < 5; i++) {
            trafficRef.current.push({
              x: 300 + i * 200,
              y: canvas.height / 2 + (Math.random() - 0.5) * 150,
              angle: 0,
              speed: 80 + Math.random() * 40,
              targetSpeed: 90,
              lane: Math.floor(Math.random() * 3),
              type: 'traffic',
              color: ['#EF4444', '#10B981', '#F59E0B', '#8B5CF6'][i % 4]
            })
          }
          break
          
        case 'urban':
          ego.x = 50
          ego.y = canvas.height - 100
          ego.angle = 0
          ego.lane = 0
          ego.targetSpeed = 40
          
          // 도심 장애물 (건물, 주차된 차량)
          obstaclesRef.current = [
            { x: 200, y: 50, width: 100, height: 150, type: 'static' },
            { x: 400, y: 200, width: 80, height: 120, type: 'static' },
            { x: 300, y: canvas.height - 100, width: 60, height: 30, type: 'static' }
          ]
          
          // 보행자
          for (let i = 0; i < 3; i++) {
            obstaclesRef.current.push({
              x: 150 + i * 150,
              y: canvas.height / 2 + Math.random() * 100,
              width: 10,
              height: 10,
              type: 'pedestrian'
            })
          }
          break
          
        case 'parking':
          ego.x = 50
          ego.y = canvas.height / 2
          ego.angle = 0
          ego.lane = 0
          ego.targetSpeed = 10
          
          // 주차 공간
          for (let i = 0; i < 8; i++) {
            if (i !== 4) { // 빈 주차 공간
              obstaclesRef.current.push({
                x: 150 + i * 80,
                y: 100,
                width: 60,
                height: 100,
                type: 'static'
              })
            }
          }
          break
      }
    }

    initScenario()

    const animate = () => {
      if (!isRunning) return

      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // 배경 (도로)
      ctx.fillStyle = '#374151'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // 차선 그리기
      ctx.strokeStyle = '#FFF'
      ctx.lineWidth = 2
      ctx.setLineDash([20, 10])
      
      if (scenario === 'highway') {
        // 3차선 고속도로
        for (let i = 1; i < 3; i++) {
          ctx.beginPath()
          ctx.moveTo(0, canvas.height / 3 * i)
          ctx.lineTo(canvas.width, canvas.height / 3 * i)
          ctx.stroke()
        }
      } else {
        // 일반 도로
        ctx.beginPath()
        ctx.moveTo(0, canvas.height / 2)
        ctx.lineTo(canvas.width, canvas.height / 2)
        ctx.stroke()
      }
      ctx.setLineDash([])

      // 장애물 그리기
      obstaclesRef.current.forEach(obstacle => {
        ctx.fillStyle = obstacle.type === 'static' ? '#6B7280' : '#EF4444'
        ctx.fillRect(obstacle.x, obstacle.y, obstacle.width, obstacle.height)
        
        if (obstacle.type === 'pedestrian') {
          // 보행자 이동
          obstacle.x += Math.sin(Date.now() * 0.001) * 0.5
          obstacle.y += Math.cos(Date.now() * 0.002) * 0.3
        }
      })

      // 경로 계획 시각화
      if (plannedPath.length > 0) {
        ctx.strokeStyle = 'rgba(34, 197, 94, 0.5)'
        ctx.lineWidth = 3
        ctx.beginPath()
        ctx.moveTo(plannedPath[0].x, plannedPath[0].y)
        plannedPath.forEach(point => {
          ctx.lineTo(point.x, point.y)
        })
        ctx.stroke()

        // 경로 점 표시
        plannedPath.forEach(point => {
          ctx.fillStyle = 'rgba(34, 197, 94, 0.8)'
          ctx.beginPath()
          ctx.arc(point.x, point.y, 3, 0, Math.PI * 2)
          ctx.fill()
        })
      }

      // 교통 차량 업데이트 및 그리기
      trafficRef.current.forEach(vehicle => {
        // 간단한 차량 행동
        vehicle.x += vehicle.speed * 0.1
        
        // 차선 유지 (약간의 웨이빙)
        vehicle.y += Math.sin(vehicle.x * 0.01) * 0.5
        
        // 화면 벗어나면 재생성
        if (vehicle.x > canvas.width + 50) {
          vehicle.x = -50
          vehicle.y = canvas.height / 3 * (vehicle.lane + 0.5)
          vehicle.speed = 80 + Math.random() * 40
        }

        // 차량 그리기
        ctx.save()
        ctx.translate(vehicle.x, vehicle.y)
        ctx.rotate(vehicle.angle)
        ctx.fillStyle = vehicle.color
        ctx.fillRect(-20, -10, 40, 20)
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)'
        ctx.fillRect(-15, -7, 30, 14)
        ctx.restore()
      })

      // 자율주행 차량 업데이트
      const ego = egoVehicleRef.current
      
      // 자율주행 레벨에 따른 제어
      if (autonomyLevel >= 2) {
        // 적응형 크루즈 컨트롤
        const frontVehicle = trafficRef.current.find(v => 
          v.x > ego.x && v.x < ego.x + 200 && 
          Math.abs(v.y - ego.y) < 50
        )
        
        if (frontVehicle) {
          const distance = frontVehicle.x - ego.x
          if (distance < 100) {
            ego.targetSpeed = Math.min(frontVehicle.speed - 10, ego.targetSpeed)
          }
        } else {
          ego.targetSpeed = scenario === 'highway' ? 100 : 
                            scenario === 'urban' ? 40 : 10
        }
      }

      if (autonomyLevel >= 3) {
        // 차선 변경 결정
        const shouldChangeLane = trafficRef.current.some(v => 
          v.x > ego.x - 50 && v.x < ego.x + 100 && 
          Math.abs(v.y - ego.y) < 30
        )
        
        if (shouldChangeLane && ego.lane > 0) {
          steeringAngle = -0.1
        } else if (shouldChangeLane && ego.lane < 2) {
          steeringAngle = 0.1
        } else {
          steeringAngle = 0
        }
      }

      // 물리 업데이트
      ego.speed += (ego.targetSpeed - ego.speed) * 0.05
      ego.angle += steeringAngle * 0.1
      ego.x += Math.cos(ego.angle) * ego.speed * 0.1
      ego.y += Math.sin(ego.angle) * ego.speed * 0.1

      // 경로 계획 업데이트
      const newPath: PathPoint[] = []
      for (let i = 0; i < 10; i++) {
        newPath.push({
          x: ego.x + i * 30 * Math.cos(ego.angle),
          y: ego.y + i * 30 * Math.sin(ego.angle),
          speed: ego.targetSpeed
        })
      }
      setPlannedPath(newPath)

      // 자율주행 차량 그리기
      ctx.save()
      ctx.translate(ego.x, ego.y)
      ctx.rotate(ego.angle)
      
      // 차체
      ctx.fillStyle = ego.color
      ctx.fillRect(-25, -12, 50, 24)
      
      // 창문
      ctx.fillStyle = 'rgba(30, 64, 175, 0.8)'
      ctx.fillRect(-20, -8, 40, 16)
      
      // 센서 범위
      if (autonomyLevel >= 2) {
        ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)'
        ctx.fillStyle = 'rgba(59, 130, 246, 0.1)'
        ctx.beginPath()
        ctx.moveTo(25, 0)
        ctx.lineTo(200, -50)
        ctx.lineTo(200, 50)
        ctx.closePath()
        ctx.fill()
        ctx.stroke()
      }
      
      ctx.restore()

      // 감지된 객체 업데이트
      const detected: string[] = []
      
      // 차량 감지
      trafficRef.current.forEach((vehicle, idx) => {
        const dist = Math.sqrt((vehicle.x - ego.x) ** 2 + (vehicle.y - ego.y) ** 2)
        if (dist < 200) {
          detected.push(`차량 ${idx + 1} (${dist.toFixed(0)}m)`)
          
          // 감지 표시
          ctx.strokeStyle = '#F59E0B'
          ctx.lineWidth = 2
          ctx.strokeRect(vehicle.x - 25, vehicle.y - 15, 50, 30)
        }
      })
      
      // 장애물 감지
      obstaclesRef.current.forEach((obstacle, idx) => {
        const dist = Math.sqrt((obstacle.x - ego.x) ** 2 + (obstacle.y - ego.y) ** 2)
        if (dist < 150) {
          detected.push(`${obstacle.type === 'pedestrian' ? '보행자' : '장애물'} ${idx + 1}`)
          
          // 감지 표시
          ctx.strokeStyle = '#EF4444'
          ctx.lineWidth = 2
          ctx.strokeRect(obstacle.x - 5, obstacle.y - 5, obstacle.width + 10, obstacle.height + 10)
        }
      })
      
      setDetectedObjects(detected)

      // 안전 메트릭 계산
      let minDist = Infinity
      let ttc = Infinity
      
      trafficRef.current.forEach(vehicle => {
        const dist = Math.sqrt((vehicle.x - ego.x) ** 2 + (vehicle.y - ego.y) ** 2)
        minDist = Math.min(minDist, dist)
        
        if (vehicle.x > ego.x && vehicle.speed < ego.speed) {
          const relSpeed = ego.speed - vehicle.speed
          const timeToColl = (vehicle.x - ego.x) / (relSpeed * 0.1)
          ttc = Math.min(ttc, timeToColl)
        }
      })
      
      setSafetyMetrics({
        ttc: ttc,
        minDistance: minDist,
        laneKeeping: 100 - Math.abs(steeringAngle) * 100
      })

      setSpeed(ego.speed)
      setSteeringAngle(steeringAngle)

      animationRef.current = requestAnimationFrame(animate)
    }

    if (isRunning) {
      animate()
    }

    return () => {
      window.removeEventListener('resize', resizeCanvas)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning, autonomyLevel, scenario])

  return (
    <div className="flex flex-col h-full bg-gray-100 dark:bg-gray-900">
      {/* 헤더 */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white p-4">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Car className="w-6 h-6" />
          자율주행 시뮬레이터
        </h2>
        <p className="text-blue-100 mt-1">SAE Level별 자율주행 기능 체험</p>
      </div>

      {/* 컨트롤 바 */}
      <div className="bg-white dark:bg-gray-800 shadow-md p-4">
        <div className="flex flex-wrap items-center gap-4">
          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`px-6 py-2 rounded-lg flex items-center gap-2 font-medium ${
              isRunning 
                ? 'bg-red-500 hover:bg-red-600 text-white'
                : 'bg-green-500 hover:bg-green-600 text-white'
            }`}
          >
            {isRunning ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
            {isRunning ? '정지' : '시작'}
          </button>

          <div className="flex items-center gap-3">
            <label className="text-sm font-medium">시나리오:</label>
            <select
              value={scenario}
              onChange={(e) => setScenario(e.target.value as any)}
              className="px-3 py-1 border rounded-lg dark:bg-gray-700 dark:border-gray-600"
            >
              <option value="highway">고속도로</option>
              <option value="urban">도심</option>
              <option value="parking">주차</option>
            </select>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-sm font-medium">자율주행 레벨:</label>
            <select
              value={autonomyLevel}
              onChange={(e) => setAutonomyLevel(parseInt(e.target.value))}
              className="px-3 py-1 border rounded-lg dark:bg-gray-700 dark:border-gray-600"
            >
              <option value="0">Level 0 (수동)</option>
              <option value="1">Level 1 (운전자 보조)</option>
              <option value="2">Level 2 (부분 자동화)</option>
              <option value="3">Level 3 (조건부 자동화)</option>
              <option value="4">Level 4 (고도 자동화)</option>
            </select>
          </div>
        </div>
      </div>

      {/* 메인 컨텐츠 */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-5 gap-4 p-4">
        {/* 시뮬레이션 뷰 - 4/5 공간 */}
        <div className="lg:col-span-4 bg-gray-800 rounded-lg overflow-hidden">
          <canvas
            ref={canvasRef}
            className="w-full h-full"
          />
        </div>

        {/* 상태 패널 - 1/5 공간 */}
        <div className="space-y-4">
          {/* 차량 상태 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Gauge className="w-5 h-5" />
              차량 상태
            </h3>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>속도</span>
                <span className="font-mono font-bold">{speed.toFixed(0)} km/h</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>조향각</span>
                <span className="font-mono font-bold">{(steeringAngle * 180 / Math.PI).toFixed(1)}°</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>자율주행</span>
                <span className={`font-bold ${autonomyLevel >= 3 ? 'text-green-600' : 'text-yellow-600'}`}>
                  Level {autonomyLevel}
                </span>
              </div>
            </div>
          </div>

          {/* 안전 메트릭 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" />
              안전 지표
            </h3>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>TTC</span>
                <span className={`font-mono font-bold ${
                  safetyMetrics.ttc < 3 ? 'text-red-600' : 
                  safetyMetrics.ttc < 5 ? 'text-yellow-600' : 'text-green-600'
                }`}>
                  {safetyMetrics.ttc === Infinity ? '안전' : `${safetyMetrics.ttc.toFixed(1)}초`}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span>최소 거리</span>
                <span className={`font-mono font-bold ${
                  safetyMetrics.minDistance < 30 ? 'text-red-600' : 
                  safetyMetrics.minDistance < 50 ? 'text-yellow-600' : 'text-green-600'
                }`}>
                  {safetyMetrics.minDistance === Infinity ? '-' : `${safetyMetrics.minDistance.toFixed(0)}m`}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span>차선 유지</span>
                <span className="font-mono font-bold text-green-600">
                  {safetyMetrics.laneKeeping.toFixed(0)}%
                </span>
              </div>
            </div>
          </div>

          {/* 감지된 객체 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Navigation className="w-5 h-5" />
              감지된 객체
            </h3>
            <div className="space-y-1 max-h-32 overflow-y-auto">
              {detectedObjects.length > 0 ? (
                detectedObjects.map((obj, idx) => (
                  <div key={idx} className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                    {obj}
                  </div>
                ))
              ) : (
                <div className="text-sm text-gray-500">감지된 객체 없음</div>
              )}
            </div>
          </div>

          {/* 시스템 상태 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Settings2 className="w-5 h-5" />
              활성 기능
            </h3>
            <div className="space-y-2">
              <div className={`text-xs px-2 py-1 rounded flex items-center gap-2 ${
                autonomyLevel >= 1 ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' : 'bg-gray-100 text-gray-500'
              }`}>
                <Zap className="w-3 h-3" />
                크루즈 컨트롤
              </div>
              <div className={`text-xs px-2 py-1 rounded flex items-center gap-2 ${
                autonomyLevel >= 2 ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' : 'bg-gray-100 text-gray-500'
              }`}>
                <Zap className="w-3 h-3" />
                차선 유지
              </div>
              <div className={`text-xs px-2 py-1 rounded flex items-center gap-2 ${
                autonomyLevel >= 3 ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' : 'bg-gray-100 text-gray-500'
              }`}>
                <Zap className="w-3 h-3" />
                자동 차선 변경
              </div>
              <div className={`text-xs px-2 py-1 rounded flex items-center gap-2 ${
                autonomyLevel >= 4 ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' : 'bg-gray-100 text-gray-500'
              }`}>
                <Zap className="w-3 h-3" />
                완전 자율주행
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}