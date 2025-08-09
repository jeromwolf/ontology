'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { 
  Zap, Play, Pause, RotateCcw, Settings, 
  Box, Circle, Triangle, Waves, Wind,
  Flame, Snowflake, ArrowDown, Move3d
} from 'lucide-react'

interface PhysicsObject {
  id: string
  type: 'box' | 'sphere' | 'triangle'
  position: { x: number; y: number }
  velocity: { x: number; y: number }
  mass: number
  size: number
  color: string
  material: 'metal' | 'rubber' | 'wood' | 'glass'
  temperature: number
}

interface PhysicsSettings {
  gravity: number
  airResistance: number
  restitution: number
  friction: number
  timeScale: number
  temperature: number
  windForce: { x: number; y: number }
}

export default function OmniversePhysicsLab() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [selectedTool, setSelectedTool] = useState<'box' | 'sphere' | 'triangle'>('box')
  const [selectedMaterial, setSelectedMaterial] = useState<'metal' | 'rubber' | 'wood' | 'glass'>('metal')
  
  const [physicsSettings, setPhysicsSettings] = useState<PhysicsSettings>({
    gravity: 9.81,
    airResistance: 0.01,
    restitution: 0.8,
    friction: 0.3,
    timeScale: 1,
    temperature: 20,
    windForce: { x: 0, y: 0 }
  })

  const [objects, setObjects] = useState<PhysicsObject[]>([])
  const [selectedObject, setSelectedObject] = useState<string | null>(null)
  const [showTrails, setShowTrails] = useState(false)
  const [trails, setTrails] = useState<Map<string, Array<{x: number, y: number}>>>(new Map())

  // Material properties
  const materialProperties = {
    metal: { density: 7.8, restitution: 0.5, friction: 0.4, heatCapacity: 0.46 },
    rubber: { density: 1.5, restitution: 0.9, friction: 0.8, heatCapacity: 2.0 },
    wood: { density: 0.7, restitution: 0.3, friction: 0.5, heatCapacity: 1.7 },
    glass: { density: 2.5, restitution: 0.7, friction: 0.2, heatCapacity: 0.84 }
  }

  // Add object to scene
  const addObject = useCallback((x: number, y: number) => {
    const newObject: PhysicsObject = {
      id: `obj-${Date.now()}`,
      type: selectedTool,
      position: { x, y },
      velocity: { x: 0, y: 0 },
      mass: materialProperties[selectedMaterial].density * 10,
      size: 30,
      color: selectedMaterial === 'metal' ? '#6b7280' :
             selectedMaterial === 'rubber' ? '#1f2937' :
             selectedMaterial === 'wood' ? '#92400e' : '#60a5fa',
      material: selectedMaterial,
      temperature: physicsSettings.temperature
    }
    setObjects(prev => [...prev, newObject])
  }, [selectedTool, selectedMaterial, physicsSettings.temperature])

  // Physics update
  const updatePhysics = useCallback((deltaTime: number) => {
    setObjects(prevObjects => {
      return prevObjects.map(obj => {
        const material = materialProperties[obj.material]
        
        // Apply gravity
        const gravityForce = physicsSettings.gravity * obj.mass * deltaTime
        
        // Apply air resistance
        const dragX = -physicsSettings.airResistance * obj.velocity.x * Math.abs(obj.velocity.x)
        const dragY = -physicsSettings.airResistance * obj.velocity.y * Math.abs(obj.velocity.y)
        
        // Apply wind
        const windX = physicsSettings.windForce.x * deltaTime
        const windY = physicsSettings.windForce.y * deltaTime
        
        // Update velocity
        const newVelocityX = obj.velocity.x + (dragX + windX) * deltaTime
        const newVelocityY = obj.velocity.y + (gravityForce + dragY + windY) * deltaTime
        
        // Update position
        const newX = obj.position.x + newVelocityX * deltaTime * physicsSettings.timeScale
        const newY = obj.position.y + newVelocityY * deltaTime * physicsSettings.timeScale
        
        // Temperature effects
        const tempDiff = physicsSettings.temperature - obj.temperature
        const newTemp = obj.temperature + tempDiff * 0.01 * material.heatCapacity
        
        // Boundary collisions
        const canvas = canvasRef.current
        if (canvas) {
          let finalVelocityX = newVelocityX
          let finalVelocityY = newVelocityY
          let finalX = newX
          let finalY = newY
          
          // Floor collision
          if (newY + obj.size > canvas.height) {
            finalY = canvas.height - obj.size
            finalVelocityY = -newVelocityY * material.restitution * physicsSettings.restitution
            finalVelocityX = newVelocityX * (1 - material.friction * physicsSettings.friction)
          }
          
          // Ceiling collision
          if (newY - obj.size < 0) {
            finalY = obj.size
            finalVelocityY = -newVelocityY * material.restitution * physicsSettings.restitution
          }
          
          // Wall collisions
          if (newX - obj.size < 0 || newX + obj.size > canvas.width) {
            finalX = newX - obj.size < 0 ? obj.size : canvas.width - obj.size
            finalVelocityX = -newVelocityX * material.restitution * physicsSettings.restitution
          }
          
          return {
            ...obj,
            position: { x: finalX, y: finalY },
            velocity: { x: finalVelocityX, y: finalVelocityY },
            temperature: newTemp
          }
        }
        
        return obj
      })
    })
    
    // Update trails
    if (showTrails) {
      setTrails(prevTrails => {
        const newTrails = new Map(prevTrails)
        objects.forEach(obj => {
          const trail = newTrails.get(obj.id) || []
          trail.push({ x: obj.position.x, y: obj.position.y })
          if (trail.length > 50) trail.shift()
          newTrails.set(obj.id, trail)
        })
        return newTrails
      })
    }
  }, [objects, physicsSettings, showTrails])

  // Render scene
  const renderScene = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Clear canvas
    ctx.fillStyle = '#f3f4f6'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    
    // Draw grid
    ctx.strokeStyle = '#e5e7eb'
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
    
    // Draw trails
    if (showTrails) {
      trails.forEach((trail, objId) => {
        const obj = objects.find(o => o.id === objId)
        if (!obj) return
        
        ctx.strokeStyle = obj.color + '40'
        ctx.lineWidth = 2
        ctx.beginPath()
        trail.forEach((point, index) => {
          if (index === 0) {
            ctx.moveTo(point.x, point.y)
          } else {
            ctx.lineTo(point.x, point.y)
          }
        })
        ctx.stroke()
      })
    }
    
    // Draw objects
    objects.forEach(obj => {
      ctx.save()
      
      // Temperature visualization
      const tempColor = obj.temperature > 50 ? `rgba(239, 68, 68, ${Math.min(obj.temperature / 100, 1)})` :
                       obj.temperature < 0 ? `rgba(59, 130, 246, ${Math.min(-obj.temperature / 50, 1)})` :
                       'transparent'
      
      if (tempColor !== 'transparent') {
        ctx.shadowColor = tempColor
        ctx.shadowBlur = 20
      }
      
      ctx.fillStyle = obj.color
      ctx.strokeStyle = selectedObject === obj.id ? '#7c3aed' : '#374151'
      ctx.lineWidth = selectedObject === obj.id ? 3 : 1
      
      if (obj.type === 'box') {
        ctx.fillRect(obj.position.x - obj.size, obj.position.y - obj.size, obj.size * 2, obj.size * 2)
        ctx.strokeRect(obj.position.x - obj.size, obj.position.y - obj.size, obj.size * 2, obj.size * 2)
      } else if (obj.type === 'sphere') {
        ctx.beginPath()
        ctx.arc(obj.position.x, obj.position.y, obj.size, 0, Math.PI * 2)
        ctx.fill()
        ctx.stroke()
      } else if (obj.type === 'triangle') {
        ctx.beginPath()
        ctx.moveTo(obj.position.x, obj.position.y - obj.size)
        ctx.lineTo(obj.position.x - obj.size, obj.position.y + obj.size)
        ctx.lineTo(obj.position.x + obj.size, obj.position.y + obj.size)
        ctx.closePath()
        ctx.fill()
        ctx.stroke()
      }
      
      // Velocity vector
      if (selectedObject === obj.id) {
        ctx.strokeStyle = '#ef4444'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(obj.position.x, obj.position.y)
        ctx.lineTo(obj.position.x + obj.velocity.x * 10, obj.position.y + obj.velocity.y * 10)
        ctx.stroke()
        
        // Arrowhead
        const angle = Math.atan2(obj.velocity.y, obj.velocity.x)
        const arrowLength = 10
        ctx.beginPath()
        ctx.moveTo(
          obj.position.x + obj.velocity.x * 10,
          obj.position.y + obj.velocity.y * 10
        )
        ctx.lineTo(
          obj.position.x + obj.velocity.x * 10 - arrowLength * Math.cos(angle - Math.PI / 6),
          obj.position.y + obj.velocity.y * 10 - arrowLength * Math.sin(angle - Math.PI / 6)
        )
        ctx.moveTo(
          obj.position.x + obj.velocity.x * 10,
          obj.position.y + obj.velocity.y * 10
        )
        ctx.lineTo(
          obj.position.x + obj.velocity.x * 10 - arrowLength * Math.cos(angle + Math.PI / 6),
          obj.position.y + obj.velocity.y * 10 - arrowLength * Math.sin(angle + Math.PI / 6)
        )
        ctx.stroke()
      }
      
      ctx.restore()
    })
    
    // Draw wind indicator
    if (physicsSettings.windForce.x !== 0 || physicsSettings.windForce.y !== 0) {
      ctx.save()
      ctx.strokeStyle = '#60a5fa'
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.moveTo(50, 50)
      ctx.lineTo(50 + physicsSettings.windForce.x * 20, 50 + physicsSettings.windForce.y * 20)
      ctx.stroke()
      ctx.restore()
    }
  }, [objects, selectedObject, showTrails, trails, physicsSettings.windForce])

  // Animation loop
  useEffect(() => {
    let lastTime = performance.now()
    
    const animate = (currentTime: number) => {
      const deltaTime = (currentTime - lastTime) / 1000
      lastTime = currentTime
      
      if (isRunning) {
        updatePhysics(deltaTime)
      }
      
      renderScene()
      animationRef.current = requestAnimationFrame(animate)
    }
    
    animationRef.current = requestAnimationFrame(animate)
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning, updatePhysics, renderScene])

  // Canvas click handler
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    // Check if clicking on existing object
    const clickedObject = objects.find(obj => {
      const dist = Math.sqrt(Math.pow(x - obj.position.x, 2) + Math.pow(y - obj.position.y, 2))
      return dist < obj.size
    })
    
    if (clickedObject) {
      setSelectedObject(clickedObject.id)
    } else {
      addObject(x, y)
    }
  }

  // Reset simulation
  const resetSimulation = () => {
    setObjects([])
    setTrails(new Map())
    setSelectedObject(null)
    setIsRunning(false)
  }

  return (
    <div className="space-y-6">
      {/* Toolbar */}
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
          
          <button
            onClick={resetSimulation}
            className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 flex items-center gap-2"
          >
            <RotateCcw className="w-4 h-4" />
            초기화
          </button>
          
          <div className="flex gap-2">
            <button
              onClick={() => setSelectedTool('box')}
              className={`p-2 rounded ${selectedTool === 'box' ? 'bg-purple-600 text-white' : 'bg-gray-200 dark:bg-gray-600'}`}
            >
              <Box className="w-5 h-5" />
            </button>
            <button
              onClick={() => setSelectedTool('sphere')}
              className={`p-2 rounded ${selectedTool === 'sphere' ? 'bg-purple-600 text-white' : 'bg-gray-200 dark:bg-gray-600'}`}
            >
              <Circle className="w-5 h-5" />
            </button>
            <button
              onClick={() => setSelectedTool('triangle')}
              className={`p-2 rounded ${selectedTool === 'triangle' ? 'bg-purple-600 text-white' : 'bg-gray-200 dark:bg-gray-600'}`}
            >
              <Triangle className="w-5 h-5" />
            </button>
          </div>
          
          <select
            value={selectedMaterial}
            onChange={(e) => setSelectedMaterial(e.target.value as any)}
            className="px-3 py-2 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-800"
          >
            <option value="metal">금속</option>
            <option value="rubber">고무</option>
            <option value="wood">나무</option>
            <option value="glass">유리</option>
          </select>
          
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={showTrails}
              onChange={(e) => setShowTrails(e.target.checked)}
              className="rounded"
            />
            <span className="text-sm">궤적 표시</span>
          </label>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Canvas */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow-lg overflow-hidden">
            <canvas
              ref={canvasRef}
              width={800}
              height={600}
              onClick={handleCanvasClick}
              className="w-full cursor-crosshair"
            />
          </div>
          
          <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
            클릭하여 물체를 추가하세요. 물체를 클릭하면 선택됩니다.
          </div>
        </div>

        {/* Controls */}
        <div className="space-y-4">
          {/* Physics Settings */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Settings className="w-5 h-5 text-purple-600" />
              물리 설정
            </h3>
            
            <div className="space-y-3">
              <div>
                <label className="text-sm font-medium flex items-center gap-2">
                  <ArrowDown className="w-4 h-4" /> 중력
                </label>
                <input
                  type="range"
                  min="0"
                  max="20"
                  step="0.1"
                  value={physicsSettings.gravity}
                  onChange={(e) => setPhysicsSettings(prev => ({ ...prev, gravity: parseFloat(e.target.value) }))}
                  className="w-full"
                />
                <span className="text-xs text-gray-600">{physicsSettings.gravity.toFixed(1)} m/s²</span>
              </div>
              
              <div>
                <label className="text-sm font-medium flex items-center gap-2">
                  <Wind className="w-4 h-4" /> 공기 저항
                </label>
                <input
                  type="range"
                  min="0"
                  max="0.1"
                  step="0.001"
                  value={physicsSettings.airResistance}
                  onChange={(e) => setPhysicsSettings(prev => ({ ...prev, airResistance: parseFloat(e.target.value) }))}
                  className="w-full"
                />
                <span className="text-xs text-gray-600">{physicsSettings.airResistance.toFixed(3)}</span>
              </div>
              
              <div>
                <label className="text-sm font-medium">반발 계수</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={physicsSettings.restitution}
                  onChange={(e) => setPhysicsSettings(prev => ({ ...prev, restitution: parseFloat(e.target.value) }))}
                  className="w-full"
                />
                <span className="text-xs text-gray-600">{physicsSettings.restitution.toFixed(2)}</span>
              </div>
              
              <div>
                <label className="text-sm font-medium">마찰 계수</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={physicsSettings.friction}
                  onChange={(e) => setPhysicsSettings(prev => ({ ...prev, friction: parseFloat(e.target.value) }))}
                  className="w-full"
                />
                <span className="text-xs text-gray-600">{physicsSettings.friction.toFixed(2)}</span>
              </div>
              
              <div>
                <label className="text-sm font-medium flex items-center gap-2">
                  <Flame className="w-4 h-4" /> 온도
                </label>
                <input
                  type="range"
                  min="-20"
                  max="100"
                  value={physicsSettings.temperature}
                  onChange={(e) => setPhysicsSettings(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                  className="w-full"
                />
                <span className="text-xs text-gray-600">{physicsSettings.temperature}°C</span>
              </div>
            </div>
          </div>

          {/* Wind Control */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Wind className="w-5 h-5 text-blue-600" />
              바람 제어
            </h3>
            
            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={() => setPhysicsSettings(prev => ({ ...prev, windForce: { x: -2, y: -2 } }))}
                className="p-2 bg-gray-100 dark:bg-gray-700 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
              >
                ↖
              </button>
              <button
                onClick={() => setPhysicsSettings(prev => ({ ...prev, windForce: { x: 0, y: -2 } }))}
                className="p-2 bg-gray-100 dark:bg-gray-700 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
              >
                ↑
              </button>
              <button
                onClick={() => setPhysicsSettings(prev => ({ ...prev, windForce: { x: 2, y: -2 } }))}
                className="p-2 bg-gray-100 dark:bg-gray-700 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
              >
                ↗
              </button>
              <button
                onClick={() => setPhysicsSettings(prev => ({ ...prev, windForce: { x: -2, y: 0 } }))}
                className="p-2 bg-gray-100 dark:bg-gray-700 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
              >
                ←
              </button>
              <button
                onClick={() => setPhysicsSettings(prev => ({ ...prev, windForce: { x: 0, y: 0 } }))}
                className="p-2 bg-gray-100 dark:bg-gray-700 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
              >
                ◯
              </button>
              <button
                onClick={() => setPhysicsSettings(prev => ({ ...prev, windForce: { x: 2, y: 0 } }))}
                className="p-2 bg-gray-100 dark:bg-gray-700 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
              >
                →
              </button>
              <button
                onClick={() => setPhysicsSettings(prev => ({ ...prev, windForce: { x: -2, y: 2 } }))}
                className="p-2 bg-gray-100 dark:bg-gray-700 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
              >
                ↙
              </button>
              <button
                onClick={() => setPhysicsSettings(prev => ({ ...prev, windForce: { x: 0, y: 2 } }))}
                className="p-2 bg-gray-100 dark:bg-gray-700 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
              >
                ↓
              </button>
              <button
                onClick={() => setPhysicsSettings(prev => ({ ...prev, windForce: { x: 2, y: 2 } }))}
                className="p-2 bg-gray-100 dark:bg-gray-700 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
              >
                ↘
              </button>
            </div>
            
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              바람: ({physicsSettings.windForce.x}, {physicsSettings.windForce.y})
            </p>
          </div>

          {/* Object Info */}
          {selectedObject && (
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold mb-4">선택된 물체</h3>
              {(() => {
                const obj = objects.find(o => o.id === selectedObject)
                if (!obj) return null
                
                return (
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>종류:</span>
                      <span>{obj.type}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>재질:</span>
                      <span>{obj.material}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>질량:</span>
                      <span>{obj.mass.toFixed(1)} kg</span>
                    </div>
                    <div className="flex justify-between">
                      <span>속도:</span>
                      <span>{Math.sqrt(obj.velocity.x ** 2 + obj.velocity.y ** 2).toFixed(1)} m/s</span>
                    </div>
                    <div className="flex justify-between">
                      <span>온도:</span>
                      <span>{obj.temperature.toFixed(1)}°C</span>
                    </div>
                  </div>
                )
              })()}
            </div>
          )}
        </div>
      </div>

      {/* Info */}
      <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-3">Omniverse 물리 엔진</h3>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          이 시뮬레이터는 NVIDIA PhysX 엔진의 핵심 기능을 구현합니다. 
          다양한 재질의 물체들이 현실적인 물리 법칙에 따라 상호작용합니다.
        </p>
        <div className="grid md:grid-cols-4 gap-4 text-sm">
          <div>
            <h4 className="font-semibold mb-1">강체 동역학</h4>
            <p className="text-gray-600 dark:text-gray-400">중력, 충돌, 반발</p>
          </div>
          <div>
            <h4 className="font-semibold mb-1">재질 속성</h4>
            <p className="text-gray-600 dark:text-gray-400">밀도, 마찰, 탄성</p>
          </div>
          <div>
            <h4 className="font-semibold mb-1">환경 요소</h4>
            <p className="text-gray-600 dark:text-gray-400">바람, 온도, 공기저항</p>
          </div>
          <div>
            <h4 className="font-semibold mb-1">실시간 시각화</h4>
            <p className="text-gray-600 dark:text-gray-400">속도 벡터, 궤적</p>
          </div>
        </div>
      </div>
    </div>
  )
}