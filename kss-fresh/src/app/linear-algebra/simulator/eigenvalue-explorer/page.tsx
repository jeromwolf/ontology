'use client'

import { useState, useRef, useEffect } from 'react'
import Link from 'next/link'
import { 
  ArrowLeft,
  GitBranch,
  Play,
  Pause,
  RefreshCw,
  Info,
  Settings,
  ChevronRight,
  Sparkles,
  Maximize2,
  Grid3x3,
  TrendingUp,
  Eye,
  EyeOff
} from 'lucide-react'

interface ComplexNumber {
  real: number
  imag: number
}

interface Eigenpair {
  value: ComplexNumber
  vector: number[]
}

export default function EigenvalueExplorerPage() {
  const [matrix, setMatrix] = useState<number[][]>([
    [3, 1],
    [2, 2]
  ])
  const [eigenpairs, setEigenpairs] = useState<Eigenpair[]>([])
  const [isAnimating, setIsAnimating] = useState(false)
  const [showVectors, setShowVectors] = useState(true)
  const [showTransform, setShowTransform] = useState(true)
  const [selectedVector, setSelectedVector] = useState<number[]>([1, 1])
  const [transformedVector, setTransformedVector] = useState<number[]>([4, 4])
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const animationProgress = useRef(0)

  // Calculate eigenvalues and eigenvectors (2x2 case)
  const calculateEigenvalues = (m: number[][]): Eigenpair[] => {
    if (m.length !== 2 || m[0].length !== 2) {
      console.error('Only 2x2 matrices supported')
      return []
    }

    const a = m[0][0]
    const b = m[0][1]
    const c = m[1][0]
    const d = m[1][1]

    // Characteristic equation: λ² - (a+d)λ + (ad-bc) = 0
    const trace = a + d
    const det = a * d - b * c
    const discriminant = trace * trace - 4 * det

    let eigenvalues: ComplexNumber[] = []
    
    if (discriminant >= 0) {
      const sqrt = Math.sqrt(discriminant)
      eigenvalues = [
        { real: (trace + sqrt) / 2, imag: 0 },
        { real: (trace - sqrt) / 2, imag: 0 }
      ]
    } else {
      const imagPart = Math.sqrt(-discriminant) / 2
      eigenvalues = [
        { real: trace / 2, imag: imagPart },
        { real: trace / 2, imag: -imagPart }
      ]
    }

    // Calculate eigenvectors for real eigenvalues
    const eigenpairs: Eigenpair[] = eigenvalues.map(lambda => {
      if (lambda.imag !== 0) {
        // For complex eigenvalues, return normalized random vector
        return {
          value: lambda,
          vector: [1, 0]
        }
      }

      // For real eigenvalues
      const l = lambda.real
      let v: number[]
      
      if (b !== 0) {
        v = [b, l - a]
      } else if (c !== 0) {
        v = [l - d, c]
      } else {
        v = [1, 0]
      }

      // Normalize
      const norm = Math.sqrt(v[0] * v[0] + v[1] * v[1])
      if (norm > 0) {
        v[0] /= norm
        v[1] /= norm
      }

      return {
        value: lambda,
        vector: v
      }
    })

    return eigenpairs
  }

  // Apply matrix transformation
  const applyTransform = (m: number[][], v: number[]): number[] => {
    return [
      m[0][0] * v[0] + m[0][1] * v[1],
      m[1][0] * v[0] + m[1][1] * v[1]
    ]
  }

  // Draw on canvas
  const draw = () => {
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
    ctx.clearRect(0, 0, width, height)
    
    // Draw grid
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
    
    // Draw axes
    ctx.strokeStyle = '#6B7280'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(0, centerY)
    ctx.lineTo(width, centerY)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(centerX, 0)
    ctx.lineTo(centerX, height)
    ctx.stroke()
    
    // Draw unit circle
    ctx.strokeStyle = '#CBD5E1'
    ctx.lineWidth = 1
    ctx.setLineDash([5, 5])
    ctx.beginPath()
    ctx.arc(centerX, centerY, scale, 0, 2 * Math.PI)
    ctx.stroke()
    ctx.setLineDash([])
    
    // Draw eigenvectors
    if (showVectors && eigenpairs.length > 0) {
      eigenpairs.forEach((pair, idx) => {
        if (pair.value.imag === 0) {
          const color = idx === 0 ? '#10B981' : '#F59E0B'
          drawVector(ctx, centerX, centerY, pair.vector[0] * scale * 2, -pair.vector[1] * scale * 2, color, `v${idx + 1}`)
          
          // Draw eigenvalue scaled version
          const scaledVector = [
            pair.vector[0] * pair.value.real,
            pair.vector[1] * pair.value.real
          ]
          ctx.strokeStyle = color
          ctx.lineWidth = 1
          ctx.setLineDash([3, 3])
          ctx.beginPath()
          ctx.moveTo(centerX, centerY)
          ctx.lineTo(centerX + scaledVector[0] * scale, centerY - scaledVector[1] * scale)
          ctx.stroke()
          ctx.setLineDash([])
        }
      })
    }
    
    // Draw selected vector and its transformation
    if (showTransform) {
      drawVector(ctx, centerX, centerY, selectedVector[0] * scale, -selectedVector[1] * scale, '#3B82F6', 'v')
      
      if (isAnimating) {
        const t = (Math.sin(animationProgress.current) + 1) / 2
        const interpolatedVector = [
          selectedVector[0] * (1 - t) + transformedVector[0] * t,
          selectedVector[1] * (1 - t) + transformedVector[1] * t
        ]
        drawVector(ctx, centerX, centerY, interpolatedVector[0] * scale, -interpolatedVector[1] * scale, '#8B5CF6', 'Av')
      } else {
        drawVector(ctx, centerX, centerY, transformedVector[0] * scale, -transformedVector[1] * scale, '#8B5CF6', 'Av')
      }
    }
  }

  const drawVector = (
    ctx: CanvasRenderingContext2D,
    fromX: number,
    fromY: number,
    dx: number,
    dy: number,
    color: string,
    label: string
  ) => {
    const toX = fromX + dx
    const toY = fromY + dy
    
    ctx.strokeStyle = color
    ctx.fillStyle = color
    ctx.lineWidth = 3
    
    // Draw line
    ctx.beginPath()
    ctx.moveTo(fromX, fromY)
    ctx.lineTo(toX, toY)
    ctx.stroke()
    
    // Draw arrowhead
    const angle = Math.atan2(dy, dx)
    const arrowLength = 10
    ctx.beginPath()
    ctx.moveTo(toX, toY)
    ctx.lineTo(
      toX - arrowLength * Math.cos(angle - Math.PI / 6),
      toY - arrowLength * Math.sin(angle - Math.PI / 6)
    )
    ctx.lineTo(
      toX - arrowLength * Math.cos(angle + Math.PI / 6),
      toY - arrowLength * Math.sin(angle + Math.PI / 6)
    )
    ctx.closePath()
    ctx.fill()
    
    // Draw label
    ctx.font = 'bold 14px monospace'
    ctx.fillText(label, toX + 10, toY - 10)
  }

  // Animation loop
  const animate = () => {
    if (isAnimating) {
      animationProgress.current += 0.05
      draw()
      animationRef.current = requestAnimationFrame(animate)
    }
  }

  useEffect(() => {
    const newEigenpairs = calculateEigenvalues(matrix)
    setEigenpairs(newEigenpairs)
    const newTransformed = applyTransform(matrix, selectedVector)
    setTransformedVector(newTransformed)
  }, [matrix, selectedVector])

  useEffect(() => {
    draw()
  }, [eigenpairs, showVectors, showTransform, selectedVector, transformedVector])

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

  const handleMatrixChange = (row: number, col: number, value: string) => {
    const newMatrix = [...matrix]
    newMatrix[row][col] = parseFloat(value) || 0
    setMatrix(newMatrix)
  }

  const handleVectorChange = (index: number, value: string) => {
    const newVector = [...selectedVector]
    newVector[index] = parseFloat(value) || 0
    setSelectedVector(newVector)
  }

  const generateRandomMatrix = () => {
    const newMatrix = [
      [Math.floor(Math.random() * 10) - 5, Math.floor(Math.random() * 10) - 5],
      [Math.floor(Math.random() * 10) - 5, Math.floor(Math.random() * 10) - 5]
    ]
    setMatrix(newMatrix)
  }

  const reset = () => {
    setMatrix([[3, 1], [2, 2]])
    setSelectedVector([1, 1])
    setIsAnimating(false)
    animationProgress.current = 0
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
                Eigenvalue Explorer
              </h1>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setIsAnimating(!isAnimating)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  isAnimating
                    ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400'
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
                }`}
              >
                {isAnimating ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              </button>
              <button
                onClick={reset}
                className="px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
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
                    Eigenspace Visualization
                  </h3>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setShowVectors(!showVectors)}
                      className={`p-2 rounded-lg transition-colors ${
                        showVectors 
                          ? 'bg-green-100 dark:bg-green-900/30 text-green-600'
                          : 'text-gray-400'
                      }`}
                      title="Toggle Eigenvectors"
                    >
                      <GitBranch className="w-5 h-5" />
                    </button>
                    <button
                      onClick={() => setShowTransform(!showTransform)}
                      className={`p-2 rounded-lg transition-colors ${
                        showTransform
                          ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600'
                          : 'text-gray-400'
                      }`}
                      title="Toggle Transform"
                    >
                      <Maximize2 className="w-5 h-5" />
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
            {/* Matrix Input */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Matrix A
                </h3>
                <button
                  onClick={generateRandomMatrix}
                  className="p-2 text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                  title="랜덤 생성"
                >
                  <Sparkles className="w-5 h-5" />
                </button>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="number"
                  value={matrix[0][0]}
                  onChange={(e) => handleMatrixChange(0, 0, e.target.value)}
                  className="px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded text-center"
                  step="0.1"
                />
                <input
                  type="number"
                  value={matrix[0][1]}
                  onChange={(e) => handleMatrixChange(0, 1, e.target.value)}
                  className="px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded text-center"
                  step="0.1"
                />
                <input
                  type="number"
                  value={matrix[1][0]}
                  onChange={(e) => handleMatrixChange(1, 0, e.target.value)}
                  className="px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded text-center"
                  step="0.1"
                />
                <input
                  type="number"
                  value={matrix[1][1]}
                  onChange={(e) => handleMatrixChange(1, 1, e.target.value)}
                  className="px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded text-center"
                  step="0.1"
                />
              </div>
            </div>

            {/* Eigenvalues & Eigenvectors */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
                고유값과 고유벡터
              </h3>
              <div className="space-y-4">
                {eigenpairs.map((pair, idx) => (
                  <div key={idx} className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <div className={`w-3 h-3 rounded-full ${
                        idx === 0 ? 'bg-green-500' : 'bg-orange-500'
                      }`}></div>
                      <span className="font-medium text-gray-900 dark:text-white">
                        λ{idx + 1}
                      </span>
                    </div>
                    {pair.value.imag === 0 ? (
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          고유값: {pair.value.real.toFixed(3)}
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          고유벡터: [{pair.vector[0].toFixed(3)}, {pair.vector[1].toFixed(3)}]
                        </p>
                      </div>
                    ) : (
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          고유값: {pair.value.real.toFixed(3)} ± {Math.abs(pair.value.imag).toFixed(3)}i
                        </p>
                        <p className="text-sm text-orange-600 dark:text-orange-400">
                          복소수 고유값 (회전 변환)
                        </p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Test Vector */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
                테스트 벡터
              </h3>
              <div className="grid grid-cols-2 gap-2 mb-4">
                <div>
                  <label className="text-xs text-gray-500 dark:text-gray-400">X</label>
                  <input
                    type="number"
                    value={selectedVector[0]}
                    onChange={(e) => handleVectorChange(0, e.target.value)}
                    className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded"
                    step="0.1"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-500 dark:text-gray-400">Y</label>
                  <input
                    type="number"
                    value={selectedVector[1]}
                    onChange={(e) => handleVectorChange(1, e.target.value)}
                    className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded"
                    step="0.1"
                  />
                </div>
              </div>
              <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                <p className="text-sm text-blue-600 dark:text-blue-400">
                  변환 후: [{transformedVector[0].toFixed(3)}, {transformedVector[1].toFixed(3)}]
                </p>
              </div>
            </div>

            {/* Matrix Properties */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
                행렬 특성
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">행렬식 (det)</span>
                  <span className="font-mono text-gray-900 dark:text-white">
                    {(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]).toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">대각합 (trace)</span>
                  <span className="font-mono text-gray-900 dark:text-white">
                    {(matrix[0][0] + matrix[1][1]).toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">고유값 곱</span>
                  <span className="font-mono text-gray-900 dark:text-white">
                    {eigenpairs.length > 0 && eigenpairs[0].value.imag === 0
                      ? (eigenpairs[0].value.real * (eigenpairs[1]?.value.real || 1)).toFixed(3)
                      : 'Complex'}
                  </span>
                </div>
              </div>
            </div>

            {/* Info */}
            <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                    사용법
                  </h4>
                  <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    <li>• 행렬 값을 조정하여 고유값 변화 관찰</li>
                    <li>• 녹색/주황색 벡터가 고유벡터</li>
                    <li>• 파란색 벡터(v)와 보라색 벡터(Av) 비교</li>
                    <li>• Play 버튼으로 변환 애니메이션</li>
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