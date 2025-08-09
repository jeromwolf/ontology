'use client'

import { useState, useRef, useEffect } from 'react'
import Link from 'next/link'
import { 
  ArrowLeft,
  Move,
  Plus,
  Minus,
  RotateCw,
  RefreshCw,
  Play,
  Pause,
  Settings,
  Grid3x3,
  Maximize2,
  Download,
  Info,
  Zap,
  Eye,
  EyeOff,
  Sliders
} from 'lucide-react'

interface Vector2D {
  x: number
  y: number
  color?: string
  label?: string
}

interface Vector3D extends Vector2D {
  z: number
}

export default function VectorVisualizerPage() {
  const [mode, setMode] = useState<'2D' | '3D'>('2D')
  const [vectors, setVectors] = useState<Vector2D[]>([
    { x: 3, y: 2, color: '#3B82F6', label: 'a' },
    { x: 1, y: 3, color: '#10B981', label: 'b' }
  ])
  const [showGrid, setShowGrid] = useState(true)
  const [showAxes, setShowAxes] = useState(true)
  const [showLabels, setShowLabels] = useState(true)
  const [showOperations, setShowOperations] = useState(true)
  const [operation, setOperation] = useState<'add' | 'subtract' | 'dot' | 'cross'>('add')
  const [scale, setScale] = useState(20)
  const [isAnimating, setIsAnimating] = useState(false)
  
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()

  // Vector operations
  const addVectors = (a: Vector2D, b: Vector2D): Vector2D => ({
    x: a.x + b.x,
    y: a.y + b.y,
    color: '#8B5CF6',
    label: 'a+b'
  })

  const subtractVectors = (a: Vector2D, b: Vector2D): Vector2D => ({
    x: a.x - b.x,
    y: a.y - b.y,
    color: '#F59E0B',
    label: 'a-b'
  })

  const dotProduct = (a: Vector2D, b: Vector2D): number => {
    return a.x * b.x + a.y * b.y
  }

  const crossProduct2D = (a: Vector2D, b: Vector2D): number => {
    return a.x * b.y - a.y * b.x
  }

  const magnitude = (v: Vector2D): number => {
    return Math.sqrt(v.x * v.x + v.y * v.y)
  }

  const normalize = (v: Vector2D): Vector2D => {
    const mag = magnitude(v)
    return mag === 0 ? { x: 0, y: 0 } : { x: v.x / mag, y: v.y / mag }
  }

  const angle = (a: Vector2D, b: Vector2D): number => {
    const dot = dotProduct(a, b)
    const magA = magnitude(a)
    const magB = magnitude(b)
    if (magA === 0 || magB === 0) return 0
    return Math.acos(dot / (magA * magB)) * (180 / Math.PI)
  }

  // Drawing functions
  const drawGrid = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    ctx.strokeStyle = '#E5E7EB'
    ctx.lineWidth = 0.5
    
    for (let x = 0; x <= width; x += scale) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.stroke()
    }
    
    for (let y = 0; y <= height; y += scale) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }
  }

  const drawAxes = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const centerX = width / 2
    const centerY = height / 2
    
    ctx.strokeStyle = '#6B7280'
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
    if (showLabels) {
      ctx.fillStyle = '#374151'
      ctx.font = '14px monospace'
      ctx.fillText('X', width - 20, centerY - 10)
      ctx.fillText('Y', centerX + 10, 20)
    }
  }

  const drawVector = (
    ctx: CanvasRenderingContext2D, 
    vector: Vector2D, 
    origin: { x: number, y: number },
    color: string,
    label?: string
  ) => {
    const endX = origin.x + vector.x * scale
    const endY = origin.y - vector.y * scale // Negative because canvas Y is inverted
    
    // Draw vector line
    ctx.strokeStyle = color
    ctx.fillStyle = color
    ctx.lineWidth = 3
    
    ctx.beginPath()
    ctx.moveTo(origin.x, origin.y)
    ctx.lineTo(endX, endY)
    ctx.stroke()
    
    // Draw arrowhead
    const angle = Math.atan2(origin.y - endY, endX - origin.x)
    const arrowLength = 10
    
    ctx.beginPath()
    ctx.moveTo(endX, endY)
    ctx.lineTo(
      endX - arrowLength * Math.cos(angle - Math.PI / 6),
      endY + arrowLength * Math.sin(angle - Math.PI / 6)
    )
    ctx.lineTo(
      endX - arrowLength * Math.cos(angle + Math.PI / 6),
      endY + arrowLength * Math.sin(angle + Math.PI / 6)
    )
    ctx.closePath()
    ctx.fill()
    
    // Draw label
    if (showLabels && label) {
      ctx.font = 'bold 16px monospace'
      ctx.fillText(label, endX + 10, endY - 10)
    }
  }

  // Main render function
  const render = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const width = canvas.width
    const height = canvas.height
    const centerX = width / 2
    const centerY = height / 2
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height)
    ctx.fillStyle = '#FFFFFF'
    ctx.fillRect(0, 0, width, height)
    
    // Draw grid
    if (showGrid) {
      drawGrid(ctx, width, height)
    }
    
    // Draw axes
    if (showAxes) {
      drawAxes(ctx, width, height)
    }
    
    // Draw vectors
    vectors.forEach((vector, idx) => {
      drawVector(ctx, vector, { x: centerX, y: centerY }, vector.color || '#3B82F6', vector.label)
    })
    
    // Draw operation result
    if (showOperations && vectors.length >= 2) {
      const a = vectors[0]
      const b = vectors[1]
      
      if (operation === 'add') {
        const result = addVectors(a, b)
        drawVector(ctx, result, { x: centerX, y: centerY }, result.color!, result.label)
        
        // Draw parallelogram
        ctx.strokeStyle = 'rgba(139, 92, 246, 0.3)'
        ctx.setLineDash([5, 5])
        ctx.beginPath()
        ctx.moveTo(centerX + a.x * scale, centerY - a.y * scale)
        ctx.lineTo(centerX + result.x * scale, centerY - result.y * scale)
        ctx.stroke()
        ctx.beginPath()
        ctx.moveTo(centerX + b.x * scale, centerY - b.y * scale)
        ctx.lineTo(centerX + result.x * scale, centerY - result.y * scale)
        ctx.stroke()
        ctx.setLineDash([])
      } else if (operation === 'subtract') {
        const result = subtractVectors(a, b)
        drawVector(ctx, result, { x: centerX, y: centerY }, result.color!, result.label)
      }
    }
  }

  // Animation loop
  const animate = () => {
    if (isAnimating) {
      // Rotate vectors
      setVectors(prev => prev.map(v => {
        const angle = 0.02
        return {
          ...v,
          x: v.x * Math.cos(angle) - v.y * Math.sin(angle),
          y: v.x * Math.sin(angle) + v.y * Math.cos(angle)
        }
      }))
      animationRef.current = requestAnimationFrame(animate)
    }
  }

  useEffect(() => {
    render()
  }, [vectors, showGrid, showAxes, showLabels, showOperations, operation, scale])

  useEffect(() => {
    if (isAnimating) {
      animationRef.current = requestAnimationFrame(animate)
    } else if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
    }
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isAnimating])

  const handleVectorChange = (index: number, axis: 'x' | 'y', value: string) => {
    const numValue = parseFloat(value) || 0
    setVectors(prev => {
      const newVectors = [...prev]
      newVectors[index] = { ...newVectors[index], [axis]: numValue }
      return newVectors
    })
  }

  const addVector = () => {
    const colors = ['#EF4444', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6', '#EC4899']
    const labels = ['c', 'd', 'e', 'f', 'g', 'h']
    const newIndex = vectors.length
    setVectors([...vectors, {
      x: Math.random() * 4 - 2,
      y: Math.random() * 4 - 2,
      color: colors[newIndex % colors.length],
      label: labels[newIndex % labels.length]
    }])
  }

  const reset = () => {
    setVectors([
      { x: 3, y: 2, color: '#3B82F6', label: 'a' },
      { x: 1, y: 3, color: '#10B981', label: 'b' }
    ])
    setScale(20)
    setIsAnimating(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <header className="sticky top-0 z-30 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/linear-algebra"
                className="flex items-center gap-2 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>돌아가기</span>
              </Link>
              <div className="h-6 w-px bg-gray-300 dark:bg-gray-700"></div>
              <h1 className="text-xl font-semibold text-gray-900 dark:text-white">
                Vector Visualizer
              </h1>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setMode(mode === '2D' ? '3D' : '2D')}
                className="px-3 py-1 bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 rounded-full text-sm font-medium"
              >
                {mode} Mode
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Canvas */}
          <div className="lg:col-span-2">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
              <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    Vector Space
                  </h3>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setShowGrid(!showGrid)}
                      className={`p-2 rounded-lg transition-colors ${
                        showGrid ? 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600' : 'text-gray-400'
                      }`}
                      title="Toggle Grid"
                    >
                      <Grid3x3 className="w-5 h-5" />
                    </button>
                    <button
                      onClick={() => setShowAxes(!showAxes)}
                      className={`p-2 rounded-lg transition-colors ${
                        showAxes ? 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600' : 'text-gray-400'
                      }`}
                      title="Toggle Axes"
                    >
                      <Maximize2 className="w-5 h-5" />
                    </button>
                    <button
                      onClick={() => setShowLabels(!showLabels)}
                      className={`p-2 rounded-lg transition-colors ${
                        showLabels ? 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600' : 'text-gray-400'
                      }`}
                      title="Toggle Labels"
                    >
                      {showLabels ? <Eye className="w-5 h-5" /> : <EyeOff className="w-5 h-5" />}
                    </button>
                    <button
                      onClick={() => setIsAnimating(!isAnimating)}
                      className={`p-2 rounded-lg transition-colors ${
                        isAnimating ? 'bg-red-100 dark:bg-red-900/30 text-red-600' : 'text-gray-400'
                      }`}
                      title="Animate"
                    >
                      {isAnimating ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                    </button>
                    <button
                      onClick={reset}
                      className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded-lg transition-colors"
                      title="Reset"
                    >
                      <RefreshCw className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              </div>
              <div className="relative bg-gray-50 dark:bg-gray-900">
                <canvas
                  ref={canvasRef}
                  width={600}
                  height={600}
                  className="w-full h-auto"
                />
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="space-y-6">
            {/* Vector Controls */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
                Vectors
              </h3>
              <div className="space-y-4">
                {vectors.map((vector, idx) => (
                  <div key={idx} className="space-y-2">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-4 h-4 rounded-full"
                        style={{ backgroundColor: vector.color }}
                      />
                      <span className="font-mono text-sm text-gray-700 dark:text-gray-300">
                        {vector.label}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="text-xs text-gray-500 dark:text-gray-400">X</label>
                        <input
                          type="number"
                          value={vector.x.toFixed(2)}
                          onChange={(e) => handleVectorChange(idx, 'x', e.target.value)}
                          step="0.1"
                          className="w-full px-2 py-1 bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded text-sm"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-gray-500 dark:text-gray-400">Y</label>
                        <input
                          type="number"
                          value={vector.y.toFixed(2)}
                          onChange={(e) => handleVectorChange(idx, 'y', e.target.value)}
                          step="0.1"
                          className="w-full px-2 py-1 bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded text-sm"
                        />
                      </div>
                    </div>
                  </div>
                ))}
                <button
                  onClick={addVector}
                  className="w-full px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                >
                  <Plus className="w-4 h-4 inline mr-2" />
                  Add Vector
                </button>
              </div>
            </div>

            {/* Operations */}
            {vectors.length >= 2 && (
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
                <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
                  Operations
                </h3>
                <div className="space-y-3">
                  <button
                    onClick={() => setOperation('add')}
                    className={`w-full px-4 py-2 rounded-lg transition-colors ${
                      operation === 'add'
                        ? 'bg-indigo-600 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    Addition (a + b)
                  </button>
                  <button
                    onClick={() => setOperation('subtract')}
                    className={`w-full px-4 py-2 rounded-lg transition-colors ${
                      operation === 'subtract'
                        ? 'bg-indigo-600 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    Subtraction (a - b)
                  </button>
                </div>
                
                <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                  <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Calculations
                  </h4>
                  <div className="space-y-2 text-sm font-mono">
                    <div className="flex justify-between">
                      <span className="text-gray-500">a · b:</span>
                      <span className="text-gray-900 dark:text-white">
                        {dotProduct(vectors[0], vectors[1]).toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">|a|:</span>
                      <span className="text-gray-900 dark:text-white">
                        {magnitude(vectors[0]).toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">|b|:</span>
                      <span className="text-gray-900 dark:text-white">
                        {magnitude(vectors[1]).toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">θ:</span>
                      <span className="text-gray-900 dark:text-white">
                        {angle(vectors[0], vectors[1]).toFixed(1)}°
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Info */}
            <div className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-indigo-600 dark:text-indigo-400 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                    사용법
                  </h4>
                  <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    <li>• 입력 필드에서 벡터 값 조정</li>
                    <li>• 연산 버튼으로 벡터 연산 시각화</li>
                    <li>• Play 버튼으로 회전 애니메이션</li>
                    <li>• 그리드와 축 표시 토글 가능</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}