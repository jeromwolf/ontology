'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  ArrowLeft,
  Grid3x3,
  Plus,
  Minus,
  X,
  Calculator,
  RefreshCw,
  Info,
  Copy,
  Download,
  Settings,
  ChevronRight,
  Sparkles,
  RotateCw,
  ArrowRightLeft
} from 'lucide-react'

type Matrix = number[][]

export default function MatrixCalculatorPage() {
  const [matrixA, setMatrixA] = useState<Matrix>([
    [1, 2],
    [3, 4]
  ])
  const [matrixB, setMatrixB] = useState<Matrix>([
    [5, 6],
    [7, 8]
  ])
  const [result, setResult] = useState<Matrix | null>(null)
  const [operation, setOperation] = useState<'add' | 'subtract' | 'multiply' | 'transpose' | 'inverse' | 'determinant'>('multiply')
  const [matrixSize, setMatrixSize] = useState({ rows: 2, cols: 2 })
  const [selectedMatrix, setSelectedMatrix] = useState<'A' | 'B'>('A')
  const [showSteps, setShowSteps] = useState(false)

  // Matrix operations
  const addMatrices = (a: Matrix, b: Matrix): Matrix => {
    if (a.length !== b.length || a[0].length !== b[0].length) {
      throw new Error('행렬 크기가 일치하지 않습니다')
    }
    return a.map((row, i) => row.map((val, j) => val + b[i][j]))
  }

  const subtractMatrices = (a: Matrix, b: Matrix): Matrix => {
    if (a.length !== b.length || a[0].length !== b[0].length) {
      throw new Error('행렬 크기가 일치하지 않습니다')
    }
    return a.map((row, i) => row.map((val, j) => val - b[i][j]))
  }

  const multiplyMatrices = (a: Matrix, b: Matrix): Matrix => {
    if (a[0].length !== b.length) {
      throw new Error('첫 번째 행렬의 열 수와 두 번째 행렬의 행 수가 일치해야 합니다')
    }
    
    const result: Matrix = Array(a.length).fill(null).map(() => Array(b[0].length).fill(0))
    
    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < b[0].length; j++) {
        for (let k = 0; k < b.length; k++) {
          result[i][j] += a[i][k] * b[k][j]
        }
      }
    }
    
    return result
  }

  const transposeMatrix = (matrix: Matrix): Matrix => {
    return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]))
  }

  const calculateDeterminant = (matrix: Matrix): number => {
    if (matrix.length !== matrix[0].length) {
      throw new Error('정사각 행렬만 행렬식을 계산할 수 있습니다')
    }
    
    if (matrix.length === 1) return matrix[0][0]
    if (matrix.length === 2) {
      return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    }
    
    let det = 0
    for (let j = 0; j < matrix[0].length; j++) {
      const minor = matrix.slice(1).map(row => [...row.slice(0, j), ...row.slice(j + 1)])
      det += (j % 2 === 0 ? 1 : -1) * matrix[0][j] * calculateDeterminant(minor)
    }
    return det
  }

  const calculateInverse = (matrix: Matrix): Matrix => {
    if (matrix.length !== matrix[0].length) {
      throw new Error('정사각 행렬만 역행렬을 계산할 수 있습니다')
    }
    
    const det = calculateDeterminant(matrix)
    if (Math.abs(det) < 0.0001) {
      throw new Error('행렬식이 0이므로 역행렬이 존재하지 않습니다')
    }
    
    if (matrix.length === 2) {
      return [
        [matrix[1][1] / det, -matrix[0][1] / det],
        [-matrix[1][0] / det, matrix[0][0] / det]
      ]
    }
    
    // For larger matrices, use Gaussian elimination (simplified)
    throw new Error('3x3 이상 행렬의 역행렬 계산은 현재 지원되지 않습니다')
  }

  const performOperation = () => {
    try {
      let newResult: Matrix | null = null
      
      switch (operation) {
        case 'add':
          newResult = addMatrices(matrixA, matrixB)
          break
        case 'subtract':
          newResult = subtractMatrices(matrixA, matrixB)
          break
        case 'multiply':
          newResult = multiplyMatrices(matrixA, matrixB)
          break
        case 'transpose':
          newResult = transposeMatrix(selectedMatrix === 'A' ? matrixA : matrixB)
          break
        case 'inverse':
          newResult = calculateInverse(selectedMatrix === 'A' ? matrixA : matrixB)
          break
        case 'determinant':
          const det = calculateDeterminant(selectedMatrix === 'A' ? matrixA : matrixB)
          newResult = [[det]]
          break
      }
      
      setResult(newResult)
    } catch (error: any) {
      alert(error.message)
    }
  }

  const handleMatrixChange = (
    matrix: 'A' | 'B',
    row: number,
    col: number,
    value: string
  ) => {
    const numValue = parseFloat(value) || 0
    
    if (matrix === 'A') {
      const newMatrix = [...matrixA]
      newMatrix[row][col] = numValue
      setMatrixA(newMatrix)
    } else {
      const newMatrix = [...matrixB]
      newMatrix[row][col] = numValue
      setMatrixB(newMatrix)
    }
  }

  const resizeMatrix = (matrix: 'A' | 'B', rows: number, cols: number) => {
    const currentMatrix = matrix === 'A' ? matrixA : matrixB
    const newMatrix: Matrix = []
    
    for (let i = 0; i < rows; i++) {
      const row: number[] = []
      for (let j = 0; j < cols; j++) {
        row.push(currentMatrix[i]?.[j] ?? 0)
      }
      newMatrix.push(row)
    }
    
    if (matrix === 'A') {
      setMatrixA(newMatrix)
    } else {
      setMatrixB(newMatrix)
    }
  }

  const resetMatrices = () => {
    setMatrixA([[1, 2], [3, 4]])
    setMatrixB([[5, 6], [7, 8]])
    setResult(null)
    setMatrixSize({ rows: 2, cols: 2 })
  }

  const generateRandomMatrix = (matrix: 'A' | 'B') => {
    const rows = matrix === 'A' ? matrixA.length : matrixB.length
    const cols = matrix === 'A' ? matrixA[0].length : matrixB[0].length
    const newMatrix: Matrix = []
    
    for (let i = 0; i < rows; i++) {
      const row: number[] = []
      for (let j = 0; j < cols; j++) {
        row.push(Math.floor(Math.random() * 10) - 5)
      }
      newMatrix.push(row)
    }
    
    if (matrix === 'A') {
      setMatrixA(newMatrix)
    } else {
      setMatrixB(newMatrix)
    }
  }

  const MatrixDisplay = ({ matrix, label, editable = false, onChange }: {
    matrix: Matrix,
    label: string,
    editable?: boolean,
    onChange?: (row: number, col: number, value: string) => void
  }) => (
    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-semibold text-gray-900 dark:text-white">{label}</h4>
        {editable && (
          <div className="flex gap-2">
            <button
              onClick={() => generateRandomMatrix(label === 'Matrix A' ? 'A' : 'B')}
              className="p-1.5 text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
              title="랜덤 생성"
            >
              <Sparkles className="w-4 h-4" />
            </button>
            <button
              onClick={() => resizeMatrix(
                label === 'Matrix A' ? 'A' : 'B',
                matrix.length + 1,
                matrix[0].length
              )}
              className="p-1.5 text-gray-400 hover:text-green-600 dark:hover:text-green-400 transition-colors"
              title="행 추가"
            >
              <Plus className="w-4 h-4" />
            </button>
            {matrix.length > 1 && (
              <button
                onClick={() => resizeMatrix(
                  label === 'Matrix A' ? 'A' : 'B',
                  matrix.length - 1,
                  matrix[0].length
                )}
                className="p-1.5 text-gray-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
                title="행 제거"
              >
                <Minus className="w-4 h-4" />
              </button>
            )}
          </div>
        )}
      </div>
      <div className="inline-block">
        <div className="flex items-center gap-2">
          <div className="text-4xl text-gray-400">[</div>
          <div className="grid gap-2" style={{ 
            gridTemplateColumns: `repeat(${matrix[0]?.length || 1}, 1fr)` 
          }}>
            {matrix.map((row, i) => (
              row.map((val, j) => (
                <div key={`${i}-${j}`}>
                  {editable ? (
                    <input
                      type="number"
                      value={val}
                      onChange={(e) => onChange?.(i, j, e.target.value)}
                      className="w-16 px-2 py-1 text-center bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded text-sm"
                      step="0.1"
                    />
                  ) : (
                    <div className="w-16 px-2 py-1 text-center bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded text-sm">
                      {typeof val === 'number' ? val.toFixed(2) : val}
                    </div>
                  )}
                </div>
              ))
            ))}
          </div>
          <div className="text-4xl text-gray-400">]</div>
        </div>
      </div>
    </div>
  )

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
                Matrix Calculator
              </h1>
            </div>
            <button
              onClick={resetMatrices}
              className="flex items-center gap-2 px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              초기화
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-12">
        {/* Operation Selector */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            연산 선택
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            <button
              onClick={() => setOperation('add')}
              className={`px-4 py-3 rounded-lg font-medium transition-all ${
                operation === 'add'
                  ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              <Plus className="w-5 h-5 mx-auto mb-1" />
              덧셈
            </button>
            <button
              onClick={() => setOperation('subtract')}
              className={`px-4 py-3 rounded-lg font-medium transition-all ${
                operation === 'subtract'
                  ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              <Minus className="w-5 h-5 mx-auto mb-1" />
              뺄셈
            </button>
            <button
              onClick={() => setOperation('multiply')}
              className={`px-4 py-3 rounded-lg font-medium transition-all ${
                operation === 'multiply'
                  ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              <X className="w-5 h-5 mx-auto mb-1" />
              곱셈
            </button>
            <button
              onClick={() => setOperation('transpose')}
              className={`px-4 py-3 rounded-lg font-medium transition-all ${
                operation === 'transpose'
                  ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              <ArrowRightLeft className="w-5 h-5 mx-auto mb-1" />
              전치
            </button>
            <button
              onClick={() => setOperation('inverse')}
              className={`px-4 py-3 rounded-lg font-medium transition-all ${
                operation === 'inverse'
                  ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              <RotateCw className="w-5 h-5 mx-auto mb-1" />
              역행렬
            </button>
            <button
              onClick={() => setOperation('determinant')}
              className={`px-4 py-3 rounded-lg font-medium transition-all ${
                operation === 'determinant'
                  ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              <Calculator className="w-5 h-5 mx-auto mb-1" />
              행렬식
            </button>
          </div>
          
          {(operation === 'transpose' || operation === 'inverse' || operation === 'determinant') && (
            <div className="mt-4 p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg">
              <p className="text-sm text-indigo-600 dark:text-indigo-400">
                단항 연산: 
                <button
                  onClick={() => setSelectedMatrix('A')}
                  className={`mx-2 px-3 py-1 rounded ${
                    selectedMatrix === 'A'
                      ? 'bg-indigo-600 text-white'
                      : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  Matrix A
                </button>
                <button
                  onClick={() => setSelectedMatrix('B')}
                  className={`px-3 py-1 rounded ${
                    selectedMatrix === 'B'
                      ? 'bg-indigo-600 text-white'
                      : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  Matrix B
                </button>
                선택
              </p>
            </div>
          )}
        </div>

        {/* Matrices Input/Display */}
        <div className="grid lg:grid-cols-3 gap-8 mb-8">
          {/* Matrix A */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <MatrixDisplay
              matrix={matrixA}
              label="Matrix A"
              editable={true}
              onChange={(row, col, value) => handleMatrixChange('A', row, col, value)}
            />
          </div>

          {/* Operation Symbol */}
          <div className="flex items-center justify-center">
            <div className="text-4xl font-bold text-gray-400">
              {operation === 'add' && '+'}
              {operation === 'subtract' && '−'}
              {operation === 'multiply' && '×'}
              {operation === 'transpose' && 'T'}
              {operation === 'inverse' && '⁻¹'}
              {operation === 'determinant' && 'det'}
            </div>
          </div>

          {/* Matrix B */}
          {(operation === 'add' || operation === 'subtract' || operation === 'multiply') && (
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <MatrixDisplay
                matrix={matrixB}
                label="Matrix B"
                editable={true}
                onChange={(row, col, value) => handleMatrixChange('B', row, col, value)}
              />
            </div>
          )}
        </div>

        {/* Calculate Button */}
        <div className="text-center mb-8">
          <button
            onClick={performOperation}
            className="px-8 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl font-semibold hover:shadow-xl hover:scale-105 transition-all"
          >
            계산하기
          </button>
        </div>

        {/* Result */}
        {result && (
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              결과
            </h3>
            <MatrixDisplay
              matrix={result}
              label="Result Matrix"
              editable={false}
            />
            
            {operation === 'determinant' && (
              <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <p className="text-blue-600 dark:text-blue-400">
                  행렬식 값: <span className="font-bold text-2xl">{result[0][0].toFixed(4)}</span>
                </p>
              </div>
            )}
          </div>
        )}

        {/* Info Section */}
        <div className="mt-8 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <div className="flex items-start gap-3">
            <Info className="w-5 h-5 text-indigo-600 dark:text-indigo-400 mt-0.5" />
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                사용법
              </h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• 행렬 값을 직접 입력하거나 랜덤 생성 버튼을 사용하세요</li>
                <li>• +/- 버튼으로 행렬 크기를 조절할 수 있습니다</li>
                <li>• 원하는 연산을 선택하고 계산하기 버튼을 누르세요</li>
                <li>• 전치, 역행렬, 행렬식은 단일 행렬에 대한 연산입니다</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}