'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { 
  Play, Pause, RotateCcw, Settings, Zap, 
  Activity, Target, Info, Download,
  Plus, Minus, Move, Circle
} from 'lucide-react'

type SimulationType = 'pendulum' | 'doublePendulum' | 'collision' | 'gravity' | 'spring'

interface PhysicsObject {
  id: string
  type: 'ball' | 'pendulum' | 'spring'
  x: number
  y: number
  vx: number
  vy: number
  ax: number
  ay: number
  mass: number
  radius: number
  color: string
  trail: { x: number; y: number }[]
}

interface PendulumState {
  angle: number
  angleVelocity: number
  angleAcceleration: number
  length: number
}

interface SimulationParams {
  gravity: number
  friction: number
  elasticity: number
  timeScale: number
  showTrails: boolean
  showVectors: boolean
  showGrid: boolean
}

export default function NewtonMechanicsLab() {
  const [simulationType, setSimulationType] = useState<SimulationType>('pendulum')
  const [isRunning, setIsRunning] = useState(false)
  const [objects, setObjects] = useState<PhysicsObject[]>([])
  const [pendulums, setPendulums] = useState<PendulumState[]>([])
  const [time, setTime] = useState(0)
  const [params, setParams] = useState<SimulationParams>({
    gravity: 9.81,
    friction: 0.01,
    elasticity: 0.8,
    timeScale: 1.0,
    showTrails: true,
    showVectors: true,
    showGrid: true
  })
  
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | null>(null)
  const lastTimeRef = useRef<number>(0)
  
  // 시뮬레이션 초기화
  const initializeSimulation = useCallback(() => {
    setTime(0)
    setObjects([])
    setPendulums([])
    
    switch (simulationType) {
      case 'pendulum':
        setPendulums([{
          angle: Math.PI / 4,
          angleVelocity: 0,
          angleAcceleration: 0,
          length: 200
        }])
        break
        
      case 'doublePendulum':
        setPendulums([
          {
            angle: Math.PI / 4,
            angleVelocity: 0,
            angleAcceleration: 0,
            length: 150
          },
          {
            angle: Math.PI / 3,
            angleVelocity: 0,
            angleAcceleration: 0,
            length: 150
          }
        ])
        break
        
      case 'collision':
        setObjects([
          {
            id: 'ball1',
            type: 'ball',
            x: 100,
            y: 300,
            vx: 150,
            vy: 0,
            ax: 0,
            ay: 0,
            mass: 20,
            radius: 20,
            color: '#3b82f6',
            trail: []
          },
          {
            id: 'ball2',
            type: 'ball',
            x: 500,
            y: 300,
            vx: -50,
            vy: 0,
            ax: 0,
            ay: 0,
            mass: 30,
            radius: 25,
            color: '#ef4444',
            trail: []
          }
        ])
        break
        
      case 'gravity':
        // 여러 물체의 자유낙하
        const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']
        const newObjects = Array.from({ length: 5 }, (_, i) => ({
          id: `ball${i}`,
          type: 'ball' as const,
          x: 100 + i * 120,
          y: 50 + Math.random() * 50,
          vx: 0,
          vy: 0,
          ax: 0,
          ay: params.gravity * 100, // 픽셀 단위로 변환
          mass: 10 + Math.random() * 30,
          radius: 10 + Math.random() * 15,
          color: colors[i],
          trail: []
        }))
        setObjects(newObjects)
        break
        
      case 'spring':
        setObjects([
          {
            id: 'mass1',
            type: 'spring',
            x: 300,
            y: 200,
            vx: 0,
            vy: 0,
            ax: 0,
            ay: 0,
            mass: 20,
            radius: 20,
            color: '#10b981',
            trail: []
          }
        ])
        break
    }
  }, [simulationType, params.gravity])
  
  // 물리 업데이트
  const updatePhysics = useCallback((deltaTime: number) => {
    const dt = deltaTime * params.timeScale
    
    if (simulationType === 'pendulum' || simulationType === 'doublePendulum') {
      setPendulums(prev => {
        const updated = [...prev]
        
        if (simulationType === 'pendulum') {
          const p = updated[0]
          // 단진자 운동방정식: θ'' = -(g/L)sin(θ) - bθ'
          p.angleAcceleration = -(params.gravity / (p.length / 100)) * Math.sin(p.angle) 
                               - params.friction * p.angleVelocity
          p.angleVelocity += p.angleAcceleration * dt
          p.angle += p.angleVelocity * dt
        } else {
          // 이중진자 (간단한 근사)
          const p1 = updated[0]
          const p2 = updated[1]
          
          // 첫 번째 진자
          p1.angleAcceleration = -(params.gravity / (p1.length / 100)) * Math.sin(p1.angle) 
                                - params.friction * p1.angleVelocity
          p1.angleVelocity += p1.angleAcceleration * dt
          p1.angle += p1.angleVelocity * dt
          
          // 두 번째 진자 (첫 번째 진자의 영향 포함)
          const coupling = 0.1 * Math.sin(p1.angle - p2.angle)
          p2.angleAcceleration = -(params.gravity / (p2.length / 100)) * Math.sin(p2.angle) 
                                - params.friction * p2.angleVelocity + coupling
          p2.angleVelocity += p2.angleAcceleration * dt
          p2.angle += p2.angleVelocity * dt
        }
        
        return updated
      })
    } else {
      setObjects(prev => {
        const updated = [...prev]
        
        // 충돌 검사
        for (let i = 0; i < updated.length; i++) {
          for (let j = i + 1; j < updated.length; j++) {
            const obj1 = updated[i]
            const obj2 = updated[j]
            
            const dx = obj2.x - obj1.x
            const dy = obj2.y - obj1.y
            const distance = Math.sqrt(dx * dx + dy * dy)
            const minDistance = obj1.radius + obj2.radius
            
            if (distance < minDistance) {
              // 충돌 발생 - 탄성 충돌 계산
              const nx = dx / distance
              const ny = dy / distance
              
              // 상대 속도
              const dvx = obj2.vx - obj1.vx
              const dvy = obj2.vy - obj1.vy
              const dvn = dvx * nx + dvy * ny
              
              // 충돌 후 속도 (운동량 보존)
              const impulse = 2 * dvn / (1/obj1.mass + 1/obj2.mass)
              
              obj1.vx += impulse * nx / obj1.mass * params.elasticity
              obj1.vy += impulse * ny / obj1.mass * params.elasticity
              obj2.vx -= impulse * nx / obj2.mass * params.elasticity
              obj2.vy -= impulse * ny / obj2.mass * params.elasticity
              
              // 겹침 방지
              const overlap = minDistance - distance
              const separationX = nx * overlap / 2
              const separationY = ny * overlap / 2
              
              obj1.x -= separationX
              obj1.y -= separationY
              obj2.x += separationX
              obj2.y += separationY
            }
          }
        }
        
        // 각 물체 업데이트
        updated.forEach(obj => {
          // 중력 적용
          if (simulationType === 'gravity') {
            obj.ay = params.gravity * 100
          }
          
          // 스프링 힘 (훅의 법칙)
          if (obj.type === 'spring') {
            const restY = 200
            const k = 0.5 // 스프링 상수
            const displacement = obj.y - restY
            obj.ay = -k * displacement / obj.mass * 100 - params.friction * obj.vy
          }
          
          // 속도 업데이트 (뉴턴 제2법칙: F = ma)
          obj.vx += obj.ax * dt
          obj.vy += obj.ay * dt
          
          // 위치 업데이트
          obj.x += obj.vx * dt
          obj.y += obj.vy * dt
          
          // 트레일 추가
          if (params.showTrails) {
            obj.trail.push({ x: obj.x, y: obj.y })
            if (obj.trail.length > 50) {
              obj.trail.shift()
            }
          }
          
          // 벽 충돌
          const canvas = canvasRef.current
          if (canvas) {
            if (obj.x - obj.radius < 0 || obj.x + obj.radius > canvas.width) {
              obj.vx *= -params.elasticity
              obj.x = obj.x - obj.radius < 0 ? obj.radius : canvas.width - obj.radius
            }
            if (obj.y - obj.radius < 0 || obj.y + obj.radius > canvas.height) {
              obj.vy *= -params.elasticity
              obj.y = obj.y - obj.radius < 0 ? obj.radius : canvas.height - obj.radius
            }
          }
        })
        
        return updated
      })
    }
    
    setTime(prev => prev + dt)
  }, [simulationType, params])
  
  // 렌더링
  const render = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // 캔버스 초기화
    ctx.fillStyle = '#1a1a1a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    
    // 그리드 그리기
    if (params.showGrid) {
      ctx.strokeStyle = '#333'
      ctx.lineWidth = 0.5
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
    }
    
    // 진자 그리기
    if (simulationType === 'pendulum' || simulationType === 'doublePendulum') {
      const centerX = canvas.width / 2
      const centerY = 100
      
      ctx.strokeStyle = '#666'
      ctx.lineWidth = 2
      
      let prevX = centerX
      let prevY = centerY
      
      pendulums.forEach((pendulum, index) => {
        const x = prevX + pendulum.length * Math.sin(pendulum.angle)
        const y = prevY + pendulum.length * Math.cos(pendulum.angle)
        
        // 줄 그리기
        ctx.beginPath()
        ctx.moveTo(prevX, prevY)
        ctx.lineTo(x, y)
        ctx.stroke()
        
        // 추 그리기
        ctx.fillStyle = index === 0 ? '#3b82f6' : '#ef4444'
        ctx.beginPath()
        ctx.arc(x, y, 20, 0, Math.PI * 2)
        ctx.fill()
        
        // 속도 벡터
        if (params.showVectors) {
          const vx = pendulum.angleVelocity * pendulum.length * Math.cos(pendulum.angle) * 20
          const vy = -pendulum.angleVelocity * pendulum.length * Math.sin(pendulum.angle) * 20
          
          ctx.strokeStyle = '#10b981'
          ctx.lineWidth = 2
          ctx.beginPath()
          ctx.moveTo(x, y)
          ctx.lineTo(x + vx, y + vy)
          ctx.stroke()
          
          // 화살표
          const angle = Math.atan2(vy, vx)
          ctx.beginPath()
          ctx.moveTo(x + vx, y + vy)
          ctx.lineTo(x + vx - 10 * Math.cos(angle - 0.5), y + vy - 10 * Math.sin(angle - 0.5))
          ctx.moveTo(x + vx, y + vy)
          ctx.lineTo(x + vx - 10 * Math.cos(angle + 0.5), y + vy - 10 * Math.sin(angle + 0.5))
          ctx.stroke()
        }
        
        // 고정점
        ctx.fillStyle = '#666'
        ctx.fillRect(prevX - 5, prevY - 5, 10, 10)
        
        prevX = x
        prevY = y
      })
    }
    
    // 물체 그리기
    objects.forEach(obj => {
      // 트레일
      if (params.showTrails && obj.trail.length > 1) {
        ctx.strokeStyle = obj.color + '40'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(obj.trail[0].x, obj.trail[0].y)
        obj.trail.forEach(point => {
          ctx.lineTo(point.x, point.y)
        })
        ctx.stroke()
      }
      
      // 스프링 그리기
      if (obj.type === 'spring') {
        const restY = 200
        const coils = 10
        const width = 30
        
        ctx.strokeStyle = '#666'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(canvas.width / 2, 50)
        
        for (let i = 0; i <= coils; i++) {
          const t = i / coils
          const y = 50 + t * (obj.y - 50)
          const x = canvas.width / 2 + (i % 2 === 0 ? -width : width)
          ctx.lineTo(x, y)
        }
        ctx.lineTo(obj.x, obj.y)
        ctx.stroke()
        
        // 고정점
        ctx.fillStyle = '#666'
        ctx.fillRect(canvas.width / 2 - 20, 40, 40, 10)
      }
      
      // 물체
      ctx.fillStyle = obj.color
      ctx.beginPath()
      ctx.arc(obj.x, obj.y, obj.radius, 0, Math.PI * 2)
      ctx.fill()
      
      // 질량 표시
      ctx.fillStyle = 'white'
      ctx.font = '12px Arial'
      ctx.textAlign = 'center'
      ctx.fillText(`${obj.mass}kg`, obj.x, obj.y + 4)
      
      // 속도 벡터
      if (params.showVectors && (obj.vx !== 0 || obj.vy !== 0)) {
        const scale = 0.3
        ctx.strokeStyle = '#10b981'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(obj.x, obj.y)
        ctx.lineTo(obj.x + obj.vx * scale, obj.y + obj.vy * scale)
        ctx.stroke()
        
        // 가속도 벡터
        if (obj.ax !== 0 || obj.ay !== 0) {
          ctx.strokeStyle = '#f59e0b'
          ctx.beginPath()
          ctx.moveTo(obj.x, obj.y)
          ctx.lineTo(obj.x + obj.ax * scale * 10, obj.y + obj.ay * scale * 10)
          ctx.stroke()
        }
      }
    })
    
    // 정보 표시
    ctx.fillStyle = 'white'
    ctx.font = '14px Arial'
    ctx.textAlign = 'left'
    ctx.fillText(`시간: ${time.toFixed(2)}s`, 10, 25)
    ctx.fillText(`중력: ${params.gravity.toFixed(1)} m/s²`, 10, 45)
    
    if (simulationType === 'pendulum' && pendulums.length > 0) {
      const pendulum = pendulums[0]
      const kineticEnergy = 0.5 * 1 * Math.pow(pendulum.angleVelocity * pendulum.length / 100, 2)
      const potentialEnergy = 1 * params.gravity * (pendulum.length / 100) * (1 - Math.cos(pendulum.angle))
      const totalEnergy = kineticEnergy + potentialEnergy
      
      ctx.fillText(`운동 에너지: ${kineticEnergy.toFixed(2)} J`, 10, 65)
      ctx.fillText(`위치 에너지: ${potentialEnergy.toFixed(2)} J`, 10, 85)
      ctx.fillText(`총 에너지: ${totalEnergy.toFixed(2)} J`, 10, 105)
    }
    
    // 범례
    if (params.showVectors) {
      ctx.font = '12px Arial'
      ctx.fillStyle = '#10b981'
      ctx.fillText('— 속도', canvas.width - 80, 25)
      ctx.fillStyle = '#f59e0b'
      ctx.fillText('— 가속도', canvas.width - 80, 45)
    }
  }, [simulationType, pendulums, objects, params, time])
  
  // 애니메이션 루프
  useEffect(() => {
    if (isRunning) {
      const animate = (currentTime: number) => {
        if (lastTimeRef.current === 0) {
          lastTimeRef.current = currentTime
        }
        
        const deltaTime = (currentTime - lastTimeRef.current) / 1000
        lastTimeRef.current = currentTime
        
        updatePhysics(deltaTime)
        render()
        
        animationRef.current = requestAnimationFrame(animate)
      }
      
      animationRef.current = requestAnimationFrame(animate)
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
        lastTimeRef.current = 0
      }
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning, updatePhysics, render])
  
  // 초기 렌더링
  useEffect(() => {
    render()
  }, [render])
  
  // 시뮬레이션 타입 변경 시 초기화
  useEffect(() => {
    initializeSimulation()
  }, [simulationType, initializeSimulation])
  
  // 마우스 인터랙션
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (simulationType === 'gravity' || simulationType === 'collision') {
      const rect = canvasRef.current?.getBoundingClientRect()
      if (!rect) return
      
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      
      const newObject: PhysicsObject = {
        id: `ball${Date.now()}`,
        type: 'ball',
        x,
        y,
        vx: 0,
        vy: 0,
        ax: 0,
        ay: simulationType === 'gravity' ? params.gravity * 100 : 0,
        mass: 20,
        radius: 20,
        color: `hsl(${Math.random() * 360}, 70%, 50%)`,
        trail: []
      }
      
      setObjects(prev => [...prev, newObject])
    }
  }
  
  return (
    <div className="space-y-6">
      {/* 툴바 */}
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <div className="flex flex-wrap items-center gap-4">
          <select
            value={simulationType}
            onChange={(e) => setSimulationType(e.target.value as SimulationType)}
            className="px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800"
          >
            <option value="pendulum">단진자</option>
            <option value="doublePendulum">이중진자</option>
            <option value="collision">충돌 시뮬레이션</option>
            <option value="gravity">중력 낙하</option>
            <option value="spring">스프링 진동</option>
          </select>
          
          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`px-4 py-2 rounded-lg text-white flex items-center gap-2 ${
              isRunning ? 'bg-orange-600 hover:bg-orange-700' : 'bg-green-600 hover:bg-green-700'
            }`}
          >
            {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isRunning ? '일시정지' : '시작'}
          </button>
          
          <button
            onClick={() => {
              setIsRunning(false)
              initializeSimulation()
            }}
            className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 flex items-center gap-2"
          >
            <RotateCcw className="w-4 h-4" />
            초기화
          </button>
          
          <div className="flex items-center gap-2 ml-auto">
            <label className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={params.showTrails}
                onChange={(e) => setParams(prev => ({ ...prev, showTrails: e.target.checked }))}
                className="rounded"
              />
              <span className="text-sm">궤적</span>
            </label>
            
            <label className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={params.showVectors}
                onChange={(e) => setParams(prev => ({ ...prev, showVectors: e.target.checked }))}
                className="rounded"
              />
              <span className="text-sm">벡터</span>
            </label>
            
            <label className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={params.showGrid}
                onChange={(e) => setParams(prev => ({ ...prev, showGrid: e.target.checked }))}
                className="rounded"
              />
              <span className="text-sm">격자</span>
            </label>
          </div>
        </div>
      </div>
      
      {/* 메인 시뮬레이션 영역 */}
      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="bg-black rounded-lg overflow-hidden">
            <canvas
              ref={canvasRef}
              width={800}
              height={600}
              onClick={handleCanvasClick}
              className="w-full cursor-crosshair"
            />
          </div>
          
          {(simulationType === 'gravity' || simulationType === 'collision') && (
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
              💡 캔버스를 클릭하여 새로운 물체를 추가할 수 있습니다.
            </p>
          )}
        </div>
        
        {/* 컨트롤 패널 */}
        <div className="space-y-4">
          {/* 물리 파라미터 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Settings className="w-5 h-5 text-purple-600" />
              물리 파라미터
            </h3>
            
            <div className="space-y-3">
              <div>
                <label className="flex items-center justify-between text-sm mb-1">
                  <span>중력 (g)</span>
                  <span>{params.gravity.toFixed(1)} m/s²</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="20"
                  step="0.1"
                  value={params.gravity}
                  onChange={(e) => setParams(prev => ({ ...prev, gravity: parseFloat(e.target.value) }))}
                  className="w-full"
                />
              </div>
              
              <div>
                <label className="flex items-center justify-between text-sm mb-1">
                  <span>마찰력</span>
                  <span>{params.friction.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="0.1"
                  step="0.001"
                  value={params.friction}
                  onChange={(e) => setParams(prev => ({ ...prev, friction: parseFloat(e.target.value) }))}
                  className="w-full"
                />
              </div>
              
              <div>
                <label className="flex items-center justify-between text-sm mb-1">
                  <span>탄성계수</span>
                  <span>{params.elasticity.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={params.elasticity}
                  onChange={(e) => setParams(prev => ({ ...prev, elasticity: parseFloat(e.target.value) }))}
                  className="w-full"
                />
              </div>
              
              <div>
                <label className="flex items-center justify-between text-sm mb-1">
                  <span>시간 배속</span>
                  <span>{params.timeScale.toFixed(1)}x</span>
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="2"
                  step="0.1"
                  value={params.timeScale}
                  onChange={(e) => setParams(prev => ({ ...prev, timeScale: parseFloat(e.target.value) }))}
                  className="w-full"
                />
              </div>
            </div>
          </div>
          
          {/* 시뮬레이션 설명 */}
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <Info className="w-5 h-5 text-purple-600" />
              시뮬레이션 설명
            </h3>
            
            <div className="text-sm space-y-2">
              {simulationType === 'pendulum' && (
                <>
                  <p>단진자의 운동을 시뮬레이션합니다.</p>
                  <p className="font-mono bg-gray-100 dark:bg-gray-800 p-2 rounded">
                    θ'' = -(g/L)sin(θ) - bθ'
                  </p>
                  <p>에너지 보존 법칙을 관찰할 수 있습니다.</p>
                </>
              )}
              
              {simulationType === 'doublePendulum' && (
                <>
                  <p>이중진자는 카오스 현상을 보여줍니다.</p>
                  <p>초기 조건의 미세한 차이가 큰 결과 차이를 만듭니다.</p>
                </>
              )}
              
              {simulationType === 'collision' && (
                <>
                  <p>탄성 충돌에서 운동량과 에너지가 보존됩니다.</p>
                  <p className="font-mono bg-gray-100 dark:bg-gray-800 p-2 rounded">
                    m₁v₁ + m₂v₂ = m₁v₁' + m₂v₂'
                  </p>
                </>
              )}
              
              {simulationType === 'gravity' && (
                <>
                  <p>중력 가속도는 질량과 무관합니다.</p>
                  <p className="font-mono bg-gray-100 dark:bg-gray-800 p-2 rounded">
                    F = mg, a = F/m = g
                  </p>
                  <p>모든 물체가 같은 가속도로 낙하합니다.</p>
                </>
              )}
              
              {simulationType === 'spring' && (
                <>
                  <p>훅의 법칙에 따른 단순조화운동입니다.</p>
                  <p className="font-mono bg-gray-100 dark:bg-gray-800 p-2 rounded">
                    F = -kx, ω = √(k/m)
                  </p>
                </>
              )}
            </div>
          </div>
          
          {/* 물리량 표시 */}
          {objects.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Activity className="w-5 h-5 text-purple-600" />
                물체 상태
              </h3>
              
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {objects.slice(0, 5).map((obj, index) => (
                  <div key={obj.id} className="text-sm border-b border-gray-200 dark:border-gray-700 pb-2">
                    <div className="flex items-center gap-2 mb-1">
                      <div 
                        className="w-4 h-4 rounded-full" 
                        style={{ backgroundColor: obj.color }}
                      />
                      <span className="font-medium">물체 {index + 1}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-1 text-xs">
                      <span>질량: {obj.mass}kg</span>
                      <span>속력: {Math.sqrt(obj.vx**2 + obj.vy**2).toFixed(1)}m/s</span>
                      <span>운동E: {(0.5 * obj.mass * (obj.vx**2 + obj.vy**2) / 10000).toFixed(2)}J</span>
                      <span>위치E: {(obj.mass * params.gravity * (600 - obj.y) / 100).toFixed(2)}J</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}