'use client'

import { useState, useEffect, useRef } from 'react'
import { Eye, Radio, Camera, Activity, Settings, Play, Pause, RotateCcw } from 'lucide-react'

interface SensorData {
  lidar: { distance: number; angle: number; confidence: number }[]
  camera: { objects: { type: string; bbox: number[]; confidence: number }[] }
  radar: { velocity: number; distance: number; angle: number }[]
}

interface FusedObject {
  id: number
  x: number
  y: number
  vx: number
  vy: number
  type: string
  confidence: number
  sensors: string[]
}

export default function SensorFusionLab() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const [isRunning, setIsRunning] = useState(false)
  const [sensorWeights, setSensorWeights] = useState({ lidar: 0.4, camera: 0.3, radar: 0.3 })
  const [noiseLevel, setNoiseLevel] = useState(0.1)
  const [fusedObjects, setFusedObjects] = useState<FusedObject[]>([])
  const [kalmanGain, setKalmanGain] = useState(0.5)
  const [showTrajectory, setShowTrajectory] = useState(true)
  const trajectoryRef = useRef<Map<number, { x: number; y: number }[]>>(new Map())

  // Canvas 기반 센서 데이터 시각화
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Canvas 크기를 화면에 맞게 조정
    const resizeCanvas = () => {
      const container = canvas.parentElement
      if (container) {
        canvas.width = container.clientWidth
        canvas.height = container.clientHeight * 0.7 // 화면의 70% 사용
      }
    }
    
    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)

    let time = 0
    const objects = [
      { id: 1, x: 100, y: 200, vx: 2, vy: 0, type: 'car', true_x: 100, true_y: 200 },
      { id: 2, x: 300, y: 150, vx: -1, vy: 1, type: 'pedestrian', true_x: 300, true_y: 150 },
      { id: 3, x: 200, y: 100, vx: 0, vy: 2, type: 'bicycle', true_x: 200, true_y: 100 }
    ]

    const animate = () => {
      if (!isRunning) return

      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // 배경 그리드
      ctx.strokeStyle = 'rgba(100, 100, 100, 0.2)'
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

      // 자율주행 차량 (중앙)
      const carX = canvas.width / 2
      const carY = canvas.height / 2
      ctx.fillStyle = '#3B82F6'
      ctx.fillRect(carX - 20, carY - 15, 40, 30)
      ctx.fillStyle = '#1E40AF'
      ctx.fillRect(carX - 15, carY - 10, 30, 20)

      // 센서 범위 시각화
      // LiDAR (360도 스캔)
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.2)'
      ctx.fillStyle = 'rgba(59, 130, 246, 0.05)'
      ctx.beginPath()
      ctx.arc(carX, carY, 250, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()

      // Camera (전방 FOV)
      ctx.strokeStyle = 'rgba(34, 197, 94, 0.2)'
      ctx.fillStyle = 'rgba(34, 197, 94, 0.05)'
      ctx.beginPath()
      ctx.moveTo(carX, carY)
      ctx.arc(carX, carY, 200, -Math.PI / 4, Math.PI / 4)
      ctx.closePath()
      ctx.fill()
      ctx.stroke()

      // Radar (전방 장거리)
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.2)'
      ctx.fillStyle = 'rgba(168, 85, 247, 0.05)'
      ctx.beginPath()
      ctx.moveTo(carX, carY)
      ctx.arc(carX, carY, 300, -Math.PI / 6, Math.PI / 6)
      ctx.closePath()
      ctx.fill()
      ctx.stroke()

      // 객체 업데이트 및 센서 데이터 생성
      const newFusedObjects: FusedObject[] = []
      
      objects.forEach(obj => {
        // 실제 위치 업데이트
        obj.true_x += obj.vx
        obj.true_y += obj.vy

        // 경계 체크
        if (obj.true_x < 50 || obj.true_x > canvas.width - 50) obj.vx *= -1
        if (obj.true_y < 50 || obj.true_y > canvas.height - 50) obj.vy *= -1

        // 센서별 측정값 (노이즈 포함)
        const sensors: string[] = []
        let measurementX = 0
        let measurementY = 0
        let totalWeight = 0

        // LiDAR 측정
        const lidarNoise = (Math.random() - 0.5) * noiseLevel * 20
        const lidarX = obj.true_x + lidarNoise
        const lidarY = obj.true_y + lidarNoise
        const lidarDist = Math.sqrt((lidarX - carX) ** 2 + (lidarY - carY) ** 2)
        
        if (lidarDist < 250) {
          sensors.push('LiDAR')
          measurementX += lidarX * sensorWeights.lidar
          measurementY += lidarY * sensorWeights.lidar
          totalWeight += sensorWeights.lidar

          // LiDAR 포인트 표시
          ctx.fillStyle = 'rgba(59, 130, 246, 0.5)'
          ctx.fillRect(lidarX - 2, lidarY - 2, 4, 4)
        }

        // Camera 측정
        const cameraNoise = (Math.random() - 0.5) * noiseLevel * 30
        const cameraX = obj.true_x + cameraNoise
        const cameraY = obj.true_y + cameraNoise
        const cameraAngle = Math.atan2(cameraY - carY, cameraX - carX)
        
        if (Math.abs(cameraAngle) < Math.PI / 4 && Math.sqrt((cameraX - carX) ** 2 + (cameraY - carY) ** 2) < 200) {
          sensors.push('Camera')
          measurementX += cameraX * sensorWeights.camera
          measurementY += cameraY * sensorWeights.camera
          totalWeight += sensorWeights.camera

          // Camera 바운딩 박스
          ctx.strokeStyle = 'rgba(34, 197, 94, 0.5)'
          ctx.strokeRect(cameraX - 15, cameraY - 15, 30, 30)
        }

        // Radar 측정
        const radarNoise = (Math.random() - 0.5) * noiseLevel * 40
        const radarX = obj.true_x + radarNoise
        const radarY = obj.true_y + radarNoise
        const radarAngle = Math.atan2(radarY - carY, radarX - carX)
        
        if (Math.abs(radarAngle) < Math.PI / 6 && Math.sqrt((radarX - carX) ** 2 + (radarY - carY) ** 2) < 300) {
          sensors.push('Radar')
          measurementX += radarX * sensorWeights.radar
          measurementY += radarY * sensorWeights.radar
          totalWeight += sensorWeights.radar

          // Radar 검출 표시
          ctx.strokeStyle = 'rgba(168, 85, 247, 0.5)'
          ctx.beginPath()
          ctx.arc(radarX, radarY, 5, 0, Math.PI * 2)
          ctx.stroke()
        }

        if (sensors.length > 0 && totalWeight > 0) {
          // 칼만 필터 적용 (간소화된 버전)
          measurementX /= totalWeight
          measurementY /= totalWeight

          // 예측값과 측정값의 가중 평균
          const fusedX = obj.x + kalmanGain * (measurementX - obj.x)
          const fusedY = obj.y + kalmanGain * (measurementY - obj.y)

          // 속도 추정
          const fusedVx = (fusedX - obj.x) * 0.1 + obj.vx * 0.9
          const fusedVy = (fusedY - obj.y) * 0.1 + obj.vy * 0.9

          obj.x = fusedX
          obj.y = fusedY

          // Trajectory 저장
          if (showTrajectory) {
            if (!trajectoryRef.current.has(obj.id)) {
              trajectoryRef.current.set(obj.id, [])
            }
            const trajectory = trajectoryRef.current.get(obj.id)!
            trajectory.push({ x: fusedX, y: fusedY })
            if (trajectory.length > 50) trajectory.shift()
          }

          newFusedObjects.push({
            id: obj.id,
            x: fusedX,
            y: fusedY,
            vx: fusedVx,
            vy: fusedVy,
            type: obj.type,
            confidence: sensors.length / 3,
            sensors
          })
        }
      })

      // Trajectory 그리기
      if (showTrajectory) {
        trajectoryRef.current.forEach((trajectory, id) => {
          if (trajectory.length > 1) {
            ctx.strokeStyle = 'rgba(251, 191, 36, 0.5)'
            ctx.lineWidth = 2
            ctx.beginPath()
            ctx.moveTo(trajectory[0].x, trajectory[0].y)
            trajectory.forEach(point => {
              ctx.lineTo(point.x, point.y)
            })
            ctx.stroke()
          }
        })
      }

      // 융합된 객체 표시
      newFusedObjects.forEach(obj => {
        // 객체 그리기
        ctx.fillStyle = obj.type === 'car' ? '#EF4444' : obj.type === 'pedestrian' ? '#10B981' : '#F59E0B'
        ctx.fillRect(obj.x - 10, obj.y - 10, 20, 20)

        // 신뢰도 표시
        ctx.fillStyle = `rgba(255, 255, 255, ${obj.confidence})`
        ctx.fillRect(obj.x - 8, obj.y - 8, 16, 16)

        // 속도 벡터
        ctx.strokeStyle = '#000'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(obj.x, obj.y)
        ctx.lineTo(obj.x + obj.vx * 10, obj.y + obj.vy * 10)
        ctx.stroke()

        // ID 및 센서 정보
        ctx.fillStyle = '#000'
        ctx.font = '12px monospace'
        ctx.fillText(`ID: ${obj.id}`, obj.x + 15, obj.y - 5)
        ctx.fillText(obj.sensors.join(', '), obj.x + 15, obj.y + 10)
      })

      setFusedObjects(newFusedObjects)
      time += 0.1
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
  }, [isRunning, sensorWeights, noiseLevel, kalmanGain, showTrajectory])

  const resetSimulation = () => {
    setIsRunning(false)
    trajectoryRef.current.clear()
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* 헤더 */}
      <div className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white p-4">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Activity className="w-6 h-6" />
          센서 퓨전 실험실
        </h2>
        <p className="text-cyan-100 mt-1">LiDAR, Camera, Radar 데이터의 실시간 융합 시뮬레이션</p>
      </div>

      {/* 메인 컨텐츠 - 화면 효율적 사용 */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-4 p-4 bg-gray-50 dark:bg-gray-900">
        {/* 시뮬레이션 캔버스 - 3/4 공간 사용 */}
        <div className="lg:col-span-3 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-white">센서 융합 시각화</h3>
            <div className="flex gap-2">
              <button
                onClick={() => setIsRunning(!isRunning)}
                className={`px-4 py-2 rounded flex items-center gap-2 ${
                  isRunning 
                    ? 'bg-red-500 hover:bg-red-600 text-white'
                    : 'bg-green-500 hover:bg-green-600 text-white'
                }`}
              >
                {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isRunning ? '일시정지' : '시작'}
              </button>
              <button
                onClick={resetSimulation}
                className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded flex items-center gap-2"
              >
                <RotateCcw className="w-4 h-4" />
                리셋
              </button>
            </div>
          </div>
          
          <canvas
            ref={canvasRef}
            className="w-full bg-gray-100 dark:bg-gray-900 rounded"
            style={{ height: 'calc(100% - 60px)' }}
          />
        </div>

        {/* 컨트롤 패널 - 1/4 공간 사용 */}
        <div className="space-y-4">
          {/* 센서 가중치 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Settings className="w-5 h-5" />
              센서 가중치
            </h3>
            
            <div className="space-y-3">
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm flex items-center gap-2">
                    <Eye className="w-4 h-4 text-blue-500" />
                    LiDAR
                  </span>
                  <span className="text-sm font-mono">{sensorWeights.lidar.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={sensorWeights.lidar}
                  onChange={(e) => setSensorWeights({...sensorWeights, lidar: parseFloat(e.target.value)})}
                  className="w-full"
                />
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm flex items-center gap-2">
                    <Camera className="w-4 h-4 text-green-500" />
                    Camera
                  </span>
                  <span className="text-sm font-mono">{sensorWeights.camera.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={sensorWeights.camera}
                  onChange={(e) => setSensorWeights({...sensorWeights, camera: parseFloat(e.target.value)})}
                  className="w-full"
                />
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm flex items-center gap-2">
                    <Radio className="w-4 h-4 text-purple-500" />
                    Radar
                  </span>
                  <span className="text-sm font-mono">{sensorWeights.radar.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={sensorWeights.radar}
                  onChange={(e) => setSensorWeights({...sensorWeights, radar: parseFloat(e.target.value)})}
                  className="w-full"
                />
              </div>
            </div>
          </div>

          {/* 필터 설정 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
            <h3 className="text-lg font-semibold mb-4">필터 파라미터</h3>
            
            <div className="space-y-3">
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm">노이즈 레벨</span>
                  <span className="text-sm font-mono">{(noiseLevel * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="0.5"
                  step="0.05"
                  value={noiseLevel}
                  onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm">칼만 이득</span>
                  <span className="text-sm font-mono">{kalmanGain.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={kalmanGain}
                  onChange={(e) => setKalmanGain(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="trajectory"
                  checked={showTrajectory}
                  onChange={(e) => setShowTrajectory(e.target.checked)}
                  className="rounded"
                />
                <label htmlFor="trajectory" className="text-sm">궤적 표시</label>
              </div>
            </div>
          </div>

          {/* 객체 정보 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
            <h3 className="text-lg font-semibold mb-4">추적 중인 객체</h3>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {fusedObjects.map(obj => (
                <div key={obj.id} className="text-xs bg-gray-100 dark:bg-gray-700 p-2 rounded">
                  <div className="font-semibold">ID: {obj.id} ({obj.type})</div>
                  <div>위치: ({obj.x.toFixed(0)}, {obj.y.toFixed(0)})</div>
                  <div>속도: ({obj.vx.toFixed(1)}, {obj.vy.toFixed(1)})</div>
                  <div>센서: {obj.sensors.join(', ')}</div>
                  <div>신뢰도: {(obj.confidence * 100).toFixed(0)}%</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}