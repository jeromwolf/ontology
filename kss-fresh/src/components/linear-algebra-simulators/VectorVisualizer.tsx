'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, RotateCcw, Grid3x3, Plus, Minus, X as MultiplyIcon } from 'lucide-react'

interface Vector3D {
  x: number
  y: number
  z: number
}

export default function VectorVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [vectorA, setVectorA] = useState<Vector3D>({ x: 3, y: 2, z: 0 })
  const [vectorB, setVectorB] = useState<Vector3D>({ x: 1, y: 3, z: 0 })
  const [operation, setOperation] = useState<'add' | 'subtract' | 'scalar' | 'dot' | 'cross'>('add')
  const [scalar, setScalar] = useState<number>(2)
  const [showGrid, setShowGrid] = useState<boolean>(true)
  const [rotation, setRotation] = useState<{ x: number; y: number }>({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 })

  useEffect(() => {
    drawCanvas()
  }, [vectorA, vectorB, operation, scalar, showGrid, rotation])

  const drawCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height
    const centerX = width / 2
    const centerY = height / 2
    const scale = 40

    // Clear canvas
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, width, height)

    // Draw grid
    if (showGrid) {
      ctx.strokeStyle = '#1e293b'
      ctx.lineWidth = 1
      for (let i = -10; i <= 10; i++) {
        // Vertical lines
        ctx.beginPath()
        ctx.moveTo(centerX + i * scale, 0)
        ctx.lineTo(centerX + i * scale, height)
        ctx.stroke()
        // Horizontal lines
        ctx.beginPath()
        ctx.moveTo(0, centerY + i * scale)
        ctx.lineTo(width, centerY + i * scale)
        ctx.stroke()
      }
    }

    // Draw axes
    ctx.strokeStyle = '#475569'
    ctx.lineWidth = 2
    // X axis
    ctx.beginPath()
    ctx.moveTo(0, centerY)
    ctx.lineTo(width, centerY)
    ctx.stroke()
    // Y axis
    ctx.beginPath()
    ctx.moveTo(centerX, 0)
    ctx.lineTo(centerX, height)
    ctx.stroke()

    // Axis labels
    ctx.fillStyle = '#94a3b8'
    ctx.font = '14px Inter'
    ctx.fillText('X', width - 20, centerY - 10)
    ctx.fillText('Y', centerX + 10, 20)

    // Apply rotation (simple 2D rotation for now)
    const rotatePoint = (x: number, y: number) => {
      const cosX = Math.cos(rotation.x)
      const sinX = Math.sin(rotation.x)
      const cosY = Math.cos(rotation.y)
      const sinY = Math.sin(rotation.y)

      // Simple 2D rotation
      const newX = x * cosY - y * sinY
      const newY = x * sinY + y * cosY
      return { x: newX, y: newY }
    }

    // Draw vector function
    const drawVector = (v: Vector3D, color: string, label: string, offsetX = 0, offsetY = 0) => {
      const rotated = rotatePoint(v.x, v.y)
      const endX = centerX + rotated.x * scale + offsetX
      const endY = centerY - rotated.y * scale + offsetY
      const startX = centerX + offsetX
      const startY = centerY + offsetY

      // Vector line
      ctx.strokeStyle = color
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.moveTo(startX, startY)
      ctx.lineTo(endX, endY)
      ctx.stroke()

      // Arrowhead
      const angle = Math.atan2(endY - startY, endX - startX)
      const arrowSize = 12
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.moveTo(endX, endY)
      ctx.lineTo(
        endX - arrowSize * Math.cos(angle - Math.PI / 6),
        endY - arrowSize * Math.sin(angle - Math.PI / 6)
      )
      ctx.lineTo(
        endX - arrowSize * Math.cos(angle + Math.PI / 6),
        endY - arrowSize * Math.sin(angle + Math.PI / 6)
      )
      ctx.closePath()
      ctx.fill()

      // Label
      ctx.fillStyle = color
      ctx.font = 'bold 16px Inter'
      ctx.fillText(label, endX + 10, endY - 10)
      ctx.font = '12px monospace'
      ctx.fillText(`(${v.x.toFixed(1)}, ${v.y.toFixed(1)})`, endX + 10, endY + 10)
    }

    // Draw vectors based on operation
    drawVector(vectorA, '#3b82f6', 'A')

    if (operation === 'add') {
      drawVector(vectorB, '#10b981', 'B')
      const result = { x: vectorA.x + vectorB.x, y: vectorA.y + vectorB.y, z: vectorA.z + vectorB.z }
      drawVector(result, '#f59e0b', 'A+B')

      // Draw parallelogram
      const rotatedA = rotatePoint(vectorA.x, vectorA.y)
      const rotatedB = rotatePoint(vectorB.x, vectorB.y)
      ctx.strokeStyle = '#64748b'
      ctx.setLineDash([5, 5])
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(centerX + rotatedA.x * scale, centerY - rotatedA.y * scale)
      ctx.lineTo(centerX + (rotatedA.x + rotatedB.x) * scale, centerY - (rotatedA.y + rotatedB.y) * scale)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(centerX + rotatedB.x * scale, centerY - rotatedB.y * scale)
      ctx.lineTo(centerX + (rotatedA.x + rotatedB.x) * scale, centerY - (rotatedA.y + rotatedB.y) * scale)
      ctx.stroke()
      ctx.setLineDash([])

    } else if (operation === 'subtract') {
      drawVector(vectorB, '#10b981', 'B')
      const result = { x: vectorA.x - vectorB.x, y: vectorA.y - vectorB.y, z: vectorA.z - vectorB.z }
      drawVector(result, '#f59e0b', 'A-B')

    } else if (operation === 'scalar') {
      const result = { x: vectorA.x * scalar, y: vectorA.y * scalar, z: vectorA.z * scalar }
      drawVector(result, '#f59e0b', `${scalar}A`)

    } else if (operation === 'dot') {
      drawVector(vectorB, '#10b981', 'B')
      const dotProduct = vectorA.x * vectorB.x + vectorA.y * vectorB.y + vectorA.z * vectorB.z

      // Display dot product result
      ctx.fillStyle = '#f59e0b'
      ctx.font = 'bold 20px Inter'
      ctx.fillText(`A · B = ${dotProduct.toFixed(2)}`, 20, 40)

      // Show angle between vectors
      const magA = Math.sqrt(vectorA.x ** 2 + vectorA.y ** 2 + vectorA.z ** 2)
      const magB = Math.sqrt(vectorB.x ** 2 + vectorB.y ** 2 + vectorB.z ** 2)
      const angle = Math.acos(dotProduct / (magA * magB)) * (180 / Math.PI)
      ctx.font = '14px Inter'
      ctx.fillText(`Angle: ${angle.toFixed(1)}°`, 20, 65)

    } else if (operation === 'cross') {
      drawVector(vectorB, '#10b981', 'B')
      const result = {
        x: vectorA.y * vectorB.z - vectorA.z * vectorB.y,
        y: vectorA.z * vectorB.x - vectorA.x * vectorB.z,
        z: vectorA.x * vectorB.y - vectorA.y * vectorB.x
      }
      drawVector(result, '#f59e0b', 'A×B')

      // Display magnitude
      const mag = Math.sqrt(result.x ** 2 + result.y ** 2 + result.z ** 2)
      ctx.fillStyle = '#f59e0b'
      ctx.font = 'bold 16px Inter'
      ctx.fillText(`|A×B| = ${mag.toFixed(2)}`, 20, 40)
    }
  }

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true)
    setLastMousePos({ x: e.clientX, y: e.clientY })
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging) return

    const deltaX = e.clientX - lastMousePos.x
    const deltaY = e.clientY - lastMousePos.y

    setRotation({
      x: rotation.x + deltaY * 0.01,
      y: rotation.y + deltaX * 0.01
    })

    setLastMousePos({ x: e.clientX, y: e.clientY })
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const resetView = () => {
    setRotation({ x: 0, y: 0 })
    setVectorA({ x: 3, y: 2, z: 0 })
    setVectorB({ x: 1, y: 3, z: 0 })
    setScalar(2)
  }

  const presets = [
    { name: 'Unit Vectors', a: { x: 1, y: 0, z: 0 }, b: { x: 0, y: 1, z: 0 } },
    { name: 'Perpendicular', a: { x: 3, y: 0, z: 0 }, b: { x: 0, y: 3, z: 0 } },
    { name: 'Parallel', a: { x: 2, y: 2, z: 0 }, b: { x: 4, y: 4, z: 0 } },
    { name: 'Opposite', a: { x: 3, y: 2, z: 0 }, b: { x: -3, y: -2, z: 0 } }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">벡터 시각화 도구</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Canvas */}
          <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">시각화</h2>
              <div className="flex gap-2">
                <button
                  onClick={() => setShowGrid(!showGrid)}
                  className={`p-2 rounded-lg transition-colors ${
                    showGrid ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-300'
                  }`}
                  title="Toggle Grid"
                >
                  <Grid3x3 className="w-5 h-5" />
                </button>
                <button
                  onClick={resetView}
                  className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
                  title="Reset View"
                >
                  <RotateCcw className="w-5 h-5" />
                </button>
              </div>
            </div>

            <canvas
              ref={canvasRef}
              width={800}
              height={600}
              className="w-full border border-slate-600 rounded-lg cursor-move"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            />

            <p className="text-sm text-slate-400 mt-2">
              💡 Tip: 마우스로 드래그하여 회전할 수 있습니다
            </p>
          </div>

          {/* Controls */}
          <div className="space-y-6">
            {/* Operation Selection */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">연산 선택</h3>
              <div className="space-y-2">
                {[
                  { id: 'add', label: 'A + B (덧셈)', icon: Plus },
                  { id: 'subtract', label: 'A - B (뺄셈)', icon: Minus },
                  { id: 'scalar', label: 'kA (스칼라 곱)', icon: MultiplyIcon },
                  { id: 'dot', label: 'A · B (내적)', icon: MultiplyIcon },
                  { id: 'cross', label: 'A × B (외적)', icon: MultiplyIcon }
                ].map(({ id, label, icon: Icon }) => (
                  <button
                    key={id}
                    onClick={() => setOperation(id as any)}
                    className={`w-full flex items-center gap-2 px-4 py-3 rounded-lg transition-colors ${
                      operation === id
                        ? 'bg-blue-600 text-white'
                        : 'bg-slate-700/50 hover:bg-slate-700 text-slate-300'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span className="text-sm">{label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Vector A Controls */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 text-blue-400">벡터 A</h3>
              <div className="space-y-3">
                <div>
                  <label className="text-sm text-slate-300 mb-1 block">X: {vectorA.x.toFixed(1)}</label>
                  <input
                    type="range"
                    min="-5"
                    max="5"
                    step="0.1"
                    value={vectorA.x}
                    onChange={(e) => setVectorA({ ...vectorA, x: parseFloat(e.target.value) })}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="text-sm text-slate-300 mb-1 block">Y: {vectorA.y.toFixed(1)}</label>
                  <input
                    type="range"
                    min="-5"
                    max="5"
                    step="0.1"
                    value={vectorA.y}
                    onChange={(e) => setVectorA({ ...vectorA, y: parseFloat(e.target.value) })}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="text-sm text-slate-300 mb-1 block">Z: {vectorA.z.toFixed(1)}</label>
                  <input
                    type="range"
                    min="-5"
                    max="5"
                    step="0.1"
                    value={vectorA.z}
                    onChange={(e) => setVectorA({ ...vectorA, z: parseFloat(e.target.value) })}
                    className="w-full"
                  />
                </div>
              </div>
            </div>

            {/* Vector B Controls */}
            {operation !== 'scalar' && (
              <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
                <h3 className="text-lg font-semibold mb-4 text-green-400">벡터 B</h3>
                <div className="space-y-3">
                  <div>
                    <label className="text-sm text-slate-300 mb-1 block">X: {vectorB.x.toFixed(1)}</label>
                    <input
                      type="range"
                      min="-5"
                      max="5"
                      step="0.1"
                      value={vectorB.x}
                      onChange={(e) => setVectorB({ ...vectorB, x: parseFloat(e.target.value) })}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-slate-300 mb-1 block">Y: {vectorB.y.toFixed(1)}</label>
                    <input
                      type="range"
                      min="-5"
                      max="5"
                      step="0.1"
                      value={vectorB.y}
                      onChange={(e) => setVectorB({ ...vectorB, y: parseFloat(e.target.value) })}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-slate-300 mb-1 block">Z: {vectorB.z.toFixed(1)}</label>
                    <input
                      type="range"
                      min="-5"
                      max="5"
                      step="0.1"
                      value={vectorB.z}
                      onChange={(e) => setVectorB({ ...vectorB, z: parseFloat(e.target.value) })}
                      className="w-full"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Scalar Control */}
            {operation === 'scalar' && (
              <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
                <h3 className="text-lg font-semibold mb-4 text-yellow-400">스칼라 k</h3>
                <div>
                  <label className="text-sm text-slate-300 mb-1 block">k: {scalar.toFixed(1)}</label>
                  <input
                    type="range"
                    min="-3"
                    max="3"
                    step="0.1"
                    value={scalar}
                    onChange={(e) => setScalar(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            )}

            {/* Presets */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">프리셋</h3>
              <div className="grid grid-cols-2 gap-2">
                {presets.map((preset) => (
                  <button
                    key={preset.name}
                    onClick={() => {
                      setVectorA(preset.a)
                      setVectorB(preset.b)
                    }}
                    className="px-3 py-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors text-sm"
                  >
                    {preset.name}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
