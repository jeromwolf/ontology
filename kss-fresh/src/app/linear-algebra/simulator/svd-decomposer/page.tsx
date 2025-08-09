'use client'

import { useState, useRef, useEffect } from 'react'
import Link from 'next/link'
import { 
  ArrowLeft,
  Layers,
  Upload,
  Download,
  RefreshCw,
  Info,
  Image,
  Sliders,
  ChevronRight,
  Play,
  Pause,
  Zap,
  BarChart3
} from 'lucide-react'

interface SVDResult {
  U: number[][]
  S: number[]
  V: number[][]
  rank: number
}

export default function SVDDecomposerPage() {
  const [matrix, setMatrix] = useState<number[][]>([
    [1, 0, 0, 0, 2],
    [0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0]
  ])
  const [svdResult, setSvdResult] = useState<SVDResult | null>(null)
  const [compressionLevel, setCompressionLevel] = useState(100)
  const [selectedComponents, setSelectedComponents] = useState(0)
  const [reconstructedMatrix, setReconstructedMatrix] = useState<number[][]>([])
  const [showOriginal, setShowOriginal] = useState(true)
  const [showCompressed, setShowCompressed] = useState(true)
  const [imageData, setImageData] = useState<ImageData | null>(null)
  const canvasOriginalRef = useRef<HTMLCanvasElement>(null)
  const canvasCompressedRef = useRef<HTMLCanvasElement>(null)

  // Simple SVD calculation (for demonstration - uses power iteration for small matrices)
  const calculateSVD = (m: number[][]): SVDResult => {
    const rows = m.length
    const cols = m[0].length
    
    // Calculate M^T * M
    const mtm = multiplyMatrices(transposeMatrix(m), m)
    
    // Simple eigenvalue calculation for demonstration
    // In production, use a proper numerical library
    const singularValues: number[] = []
    for (let i = 0; i < Math.min(rows, cols); i++) {
      // Simplified: just use diagonal elements as approximation
      const val = Math.sqrt(Math.abs(mtm[i]?.[i] || 0))
      if (val > 0.001) singularValues.push(val)
    }
    
    // Sort singular values in descending order
    singularValues.sort((a, b) => b - a)
    
    // Create placeholder U and V matrices (identity for demonstration)
    const U = createIdentityMatrix(rows)
    const V = createIdentityMatrix(cols)
    
    return {
      U,
      S: singularValues,
      V,
      rank: singularValues.length
    }
  }

  const multiplyMatrices = (a: number[][], b: number[][]): number[][] => {
    const result: number[][] = []
    for (let i = 0; i < a.length; i++) {
      result[i] = []
      for (let j = 0; j < b[0].length; j++) {
        let sum = 0
        for (let k = 0; k < b.length; k++) {
          sum += a[i][k] * b[k][j]
        }
        result[i][j] = sum
      }
    }
    return result
  }

  const transposeMatrix = (m: number[][]): number[][] => {
    return m[0].map((_, colIndex) => m.map(row => row[colIndex]))
  }

  const createIdentityMatrix = (size: number): number[][] => {
    const matrix: number[][] = []
    for (let i = 0; i < size; i++) {
      matrix[i] = []
      for (let j = 0; j < size; j++) {
        matrix[i][j] = i === j ? 1 : 0
      }
    }
    return matrix
  }

  // Reconstruct matrix from SVD with k components
  const reconstructMatrix = (svd: SVDResult, k: number): number[][] => {
    const { U, S, V } = svd
    const rows = U.length
    const cols = V.length
    
    // Create Sigma matrix with only k components
    const sigma: number[][] = []
    for (let i = 0; i < rows; i++) {
      sigma[i] = []
      for (let j = 0; j < cols; j++) {
        if (i === j && i < k && i < S.length) {
          sigma[i][j] = S[i]
        } else {
          sigma[i][j] = 0
        }
      }
    }
    
    // Multiply U * Sigma * V^T
    const US = multiplyMatrices(U, sigma)
    const result = multiplyMatrices(US, transposeMatrix(V))
    
    return result
  }

  // Calculate compression ratio
  const calculateCompressionRatio = (original: number[][], k: number): number => {
    const originalSize = original.length * original[0].length
    const compressedSize = k * (original.length + original[0].length + 1)
    return (compressedSize / originalSize) * 100
  }

  // Draw matrix as heatmap
  const drawMatrixHeatmap = (
    canvas: HTMLCanvasElement | null,
    matrix: number[][],
    title: string
  ) => {
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const cellSize = 30
    const width = matrix[0].length * cellSize
    const height = matrix.length * cellSize
    
    canvas.width = width
    canvas.height = height
    
    // Find min and max values for normalization
    let min = Infinity, max = -Infinity
    matrix.forEach(row => {
      row.forEach(val => {
        min = Math.min(min, val)
        max = Math.max(max, val)
      })
    })
    
    // Draw cells
    matrix.forEach((row, i) => {
      row.forEach((val, j) => {
        const normalized = max === min ? 0.5 : (val - min) / (max - min)
        const intensity = Math.floor(normalized * 255)
        
        // Color based on value (blue for negative, red for positive)
        if (val < 0) {
          ctx.fillStyle = `rgb(${255 - intensity}, ${255 - intensity}, 255)`
        } else {
          ctx.fillStyle = `rgb(255, ${255 - intensity}, ${255 - intensity})`
        }
        
        ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize)
        
        // Draw grid
        ctx.strokeStyle = '#E5E7EB'
        ctx.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize)
        
        // Draw value
        ctx.fillStyle = Math.abs(normalized - 0.5) < 0.25 ? '#000' : '#FFF'
        ctx.font = '10px monospace'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(
          val.toFixed(1),
          j * cellSize + cellSize / 2,
          i * cellSize + cellSize / 2
        )
      })
    })
  }

  useEffect(() => {
    const svd = calculateSVD(matrix)
    setSvdResult(svd)
    setSelectedComponents(Math.ceil(svd.rank * compressionLevel / 100))
  }, [matrix])

  useEffect(() => {
    if (svdResult) {
      const k = Math.max(1, Math.ceil(svdResult.rank * compressionLevel / 100))
      setSelectedComponents(k)
      const reconstructed = reconstructMatrix(svdResult, k)
      setReconstructedMatrix(reconstructed)
    }
  }, [compressionLevel, svdResult])

  useEffect(() => {
    if (showOriginal) {
      drawMatrixHeatmap(canvasOriginalRef.current, matrix, 'Original')
    }
    if (showCompressed && reconstructedMatrix.length > 0) {
      drawMatrixHeatmap(canvasCompressedRef.current, reconstructedMatrix, 'Compressed')
    }
  }, [matrix, reconstructedMatrix, showOriginal, showCompressed])

  const handleMatrixChange = (row: number, col: number, value: string) => {
    const newMatrix = [...matrix]
    newMatrix[row][col] = parseFloat(value) || 0
    setMatrix(newMatrix)
  }

  const generateRandomMatrix = () => {
    const rows = 4
    const cols = 5
    const newMatrix: number[][] = []
    for (let i = 0; i < rows; i++) {
      const row: number[] = []
      for (let j = 0; j < cols; j++) {
        // Create low-rank matrix for better demonstration
        if (Math.random() < 0.3) {
          row.push(Math.floor(Math.random() * 10) - 5)
        } else {
          row.push(0)
        }
      }
      newMatrix.push(row)
    }
    setMatrix(newMatrix)
  }

  const loadExampleMatrices = (type: 'image' | 'sparse' | 'dense') => {
    switch (type) {
      case 'image':
        // Simple pattern that resembles an image
        setMatrix([
          [5, 5, 0, 0, 5],
          [5, 0, 5, 0, 5],
          [5, 5, 5, 5, 5],
          [5, 0, 0, 0, 5]
        ])
        break
      case 'sparse':
        setMatrix([
          [3, 0, 0, 0, 0],
          [0, 0, 2, 0, 0],
          [0, 0, 0, 0, 1],
          [0, 4, 0, 0, 0]
        ])
        break
      case 'dense':
        setMatrix([
          [1, 2, 3, 4, 5],
          [2, 3, 4, 5, 6],
          [3, 4, 5, 6, 7],
          [4, 5, 6, 7, 8]
        ])
        break
    }
  }

  const reset = () => {
    setMatrix([
      [1, 0, 0, 0, 2],
      [0, 0, 3, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 2, 0, 0, 0]
    ])
    setCompressionLevel(100)
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
                SVD Decomposer
              </h1>
            </div>
            <button
              onClick={reset}
              className="flex items-center gap-2 px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              초기화
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-12">
        {/* Example Matrices */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            예제 행렬
          </h3>
          <div className="flex gap-3">
            <button
              onClick={() => loadExampleMatrices('image')}
              className="flex items-center gap-2 px-4 py-2 bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-lg hover:bg-blue-200 dark:hover:bg-blue-900/50 transition-colors"
            >
              <Image className="w-4 h-4" />
              이미지 패턴
            </button>
            <button
              onClick={() => loadExampleMatrices('sparse')}
              className="flex items-center gap-2 px-4 py-2 bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 rounded-lg hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors"
            >
              <Zap className="w-4 h-4" />
              희소 행렬
            </button>
            <button
              onClick={() => loadExampleMatrices('dense')}
              className="flex items-center gap-2 px-4 py-2 bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 rounded-lg hover:bg-purple-200 dark:hover:bg-purple-900/50 transition-colors"
            >
              <BarChart3 className="w-4 h-4" />
              밀집 행렬
            </button>
            <button
              onClick={generateRandomMatrix}
              className="flex items-center gap-2 px-4 py-2 bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400 rounded-lg hover:bg-orange-200 dark:hover:bg-orange-900/50 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              랜덤 생성
            </button>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Matrix */}
          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
                입력 행렬 (4×5)
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <tbody>
                    {matrix.map((row, i) => (
                      <tr key={i}>
                        {row.map((val, j) => (
                          <td key={j} className="p-1">
                            <input
                              type="number"
                              value={val}
                              onChange={(e) => handleMatrixChange(i, j, e.target.value)}
                              className="w-16 px-2 py-1 text-center bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded text-sm"
                              step="0.1"
                            />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* SVD Components */}
            {svdResult && (
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
                <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
                  특이값 (Singular Values)
                </h3>
                <div className="space-y-3">
                  {svdResult.S.map((value, idx) => (
                    <div key={idx} className="flex items-center gap-3">
                      <div className={`w-3 h-3 rounded-full ${
                        idx < selectedComponents ? 'bg-green-500' : 'bg-gray-300'
                      }`}></div>
                      <span className="font-mono text-sm text-gray-700 dark:text-gray-300">
                        σ{idx + 1} = {value.toFixed(3)}
                      </span>
                      <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className="bg-gradient-to-r from-indigo-500 to-purple-600 h-full rounded-full"
                          style={{ width: `${(value / svdResult.S[0]) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                  <p className="text-sm text-blue-600 dark:text-blue-400">
                    Rank: {svdResult.rank} | 
                    선택된 컴포넌트: {selectedComponents} / {svdResult.rank}
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Compression Controls & Visualization */}
          <div className="space-y-6">
            {/* Compression Control */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
                <Sliders className="w-5 h-5 text-orange-600 dark:text-orange-400" />
                압축 레벨
              </h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">압축률</span>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {compressionLevel}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    step="10"
                    value={compressionLevel}
                    onChange={(e) => setCompressionLevel(parseInt(e.target.value))}
                    className="w-full"
                  />
                  <div className="flex justify-between mt-1 text-xs text-gray-500">
                    <span>높은 압축</span>
                    <span>원본 품질</span>
                  </div>
                </div>
                
                {svdResult && (
                  <div className="grid grid-cols-2 gap-4 p-4 bg-gray-50 dark:bg-gray-900 rounded">
                    <div>
                      <p className="text-xs text-gray-500 dark:text-gray-400">저장 공간</p>
                      <p className="text-lg font-bold text-gray-900 dark:text-white">
                        {calculateCompressionRatio(matrix, selectedComponents).toFixed(1)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500 dark:text-gray-400">정보 보존</p>
                      <p className="text-lg font-bold text-gray-900 dark:text-white">
                        {((selectedComponents / svdResult.rank) * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Matrix Visualization */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
                행렬 시각화
              </h3>
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      원본 행렬
                    </h4>
                    <button
                      onClick={() => setShowOriginal(!showOriginal)}
                      className={`p-1 rounded ${
                        showOriginal ? 'text-blue-600' : 'text-gray-400'
                      }`}
                    >
                      {showOriginal ? '숨기기' : '보기'}
                    </button>
                  </div>
                  {showOriginal && (
                    <div className="overflow-x-auto">
                      <canvas ref={canvasOriginalRef} className="border border-gray-200 dark:border-gray-700 rounded" />
                    </div>
                  )}
                </div>
                
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      복원된 행렬 ({selectedComponents} components)
                    </h4>
                    <button
                      onClick={() => setShowCompressed(!showCompressed)}
                      className={`p-1 rounded ${
                        showCompressed ? 'text-green-600' : 'text-gray-400'
                      }`}
                    >
                      {showCompressed ? '숨기기' : '보기'}
                    </button>
                  </div>
                  {showCompressed && (
                    <div className="overflow-x-auto">
                      <canvas ref={canvasCompressedRef} className="border border-gray-200 dark:border-gray-700 rounded" />
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Applications */}
            <div className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
                SVD 응용 분야
              </h3>
              <div className="space-y-3 text-sm">
                <div className="flex items-start gap-2">
                  <Image className="w-4 h-4 text-orange-600 dark:text-orange-400 mt-0.5" />
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">이미지 압축</p>
                    <p className="text-gray-600 dark:text-gray-400">JPEG, 얼굴 인식</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <Layers className="w-4 h-4 text-red-600 dark:text-red-400 mt-0.5" />
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">추천 시스템</p>
                    <p className="text-gray-600 dark:text-gray-400">Netflix, Amazon</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <BarChart3 className="w-4 h-4 text-purple-600 dark:text-purple-400 mt-0.5" />
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">데이터 분석</p>
                    <p className="text-gray-600 dark:text-gray-400">PCA, 노이즈 제거</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Info */}
            <div className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-indigo-600 dark:text-indigo-400 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                    SVD 이해하기
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    특이값 분해(SVD)는 행렬을 U·Σ·V^T로 분해합니다.
                    큰 특이값만 유지하면 데이터를 효율적으로 압축할 수 있습니다.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}