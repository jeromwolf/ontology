'use client'

import React, { useState } from 'react'
import { Calculator, Plus, Minus, X as MultiplyIcon, Grid3x3, RotateCcw } from 'lucide-react'

type Matrix = number[][]

export default function MatrixCalculator() {
  const [matrixA, setMatrixA] = useState<Matrix>([
    [2, 1, 3],
    [0, -1, 4],
    [1, 2, -2]
  ])
  const [matrixB, setMatrixB] = useState<Matrix>([
    [1, 0, 2],
    [3, -1, 1],
    [0, 2, -1]
  ])
  const [scalar, setScalar] = useState<number>(2)
  const [operation, setOperation] = useState<'add' | 'subtract' | 'multiply' | 'scalar' | 'transpose' | 'determinant' | 'inverse' | 'eigenvalue'>('add')
  const [result, setResult] = useState<Matrix | null>(null)
  const [scalarResult, setScalarResult] = useState<number | null>(null)
  const [eigenvalues, setEigenvalues] = useState<number[] | null>(null)

  const updateMatrixA = (row: number, col: number, value: string) => {
    const newMatrix = matrixA.map((r, i) =>
      i === row ? r.map((c, j) => (j === col ? parseFloat(value) || 0 : c)) : r
    )
    setMatrixA(newMatrix)
  }

  const updateMatrixB = (row: number, col: number, value: string) => {
    const newMatrix = matrixB.map((r, i) =>
      i === row ? r.map((c, j) => (j === col ? parseFloat(value) || 0 : c)) : r
    )
    setMatrixB(newMatrix)
  }

  const addMatrices = (a: Matrix, b: Matrix): Matrix => {
    return a.map((row, i) => row.map((val, j) => val + b[i][j]))
  }

  const subtractMatrices = (a: Matrix, b: Matrix): Matrix => {
    return a.map((row, i) => row.map((val, j) => val - b[i][j]))
  }

  const multiplyMatrices = (a: Matrix, b: Matrix): Matrix => {
    const result: Matrix = []
    for (let i = 0; i < a.length; i++) {
      result[i] = []
      for (let j = 0; j < b[0].length; j++) {
        let sum = 0
        for (let k = 0; k < a[0].length; k++) {
          sum += a[i][k] * b[k][j]
        }
        result[i][j] = sum
      }
    }
    return result
  }

  const scalarMultiply = (a: Matrix, s: number): Matrix => {
    return a.map(row => row.map(val => val * s))
  }

  const transpose = (a: Matrix): Matrix => {
    return a[0].map((_, colIndex) => a.map(row => row[colIndex]))
  }

  const determinant = (a: Matrix): number => {
    const n = a.length
    if (n === 1) return a[0][0]
    if (n === 2) return a[0][0] * a[1][1] - a[0][1] * a[1][0]

    let det = 0
    for (let j = 0; j < n; j++) {
      const minor = a.slice(1).map(row => row.filter((_, colIndex) => colIndex !== j))
      det += (j % 2 === 0 ? 1 : -1) * a[0][j] * determinant(minor)
    }
    return det
  }

  const inverse = (a: Matrix): Matrix | null => {
    const det = determinant(a)
    if (Math.abs(det) < 1e-10) return null // Singular matrix

    if (a.length === 2) {
      return [
        [a[1][1] / det, -a[0][1] / det],
        [-a[1][0] / det, a[0][0] / det]
      ]
    }

    // For 3x3 and larger, use Gauss-Jordan elimination
    const n = a.length
    const augmented = a.map((row, i) => [...row, ...Array(n).fill(0).map((_, j) => (i === j ? 1 : 0))])

    // Forward elimination
    for (let i = 0; i < n; i++) {
      let maxRow = i
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
          maxRow = k
        }
      }
      ;[augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]]

      for (let k = i + 1; k < n; k++) {
        const factor = augmented[k][i] / augmented[i][i]
        for (let j = i; j < 2 * n; j++) {
          augmented[k][j] -= factor * augmented[i][j]
        }
      }
    }

    // Back substitution
    for (let i = n - 1; i >= 0; i--) {
      for (let k = i - 1; k >= 0; k--) {
        const factor = augmented[k][i] / augmented[i][i]
        for (let j = 0; j < 2 * n; j++) {
          augmented[k][j] -= factor * augmented[i][j]
        }
      }
      const pivot = augmented[i][i]
      for (let j = 0; j < 2 * n; j++) {
        augmented[i][j] /= pivot
      }
    }

    return augmented.map(row => row.slice(n))
  }

  const calculateEigenvalues = (a: Matrix): number[] => {
    // For 2x2 matrices only (simplified)
    if (a.length === 2) {
      const trace = a[0][0] + a[1][1]
      const det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
      const discriminant = trace * trace - 4 * det
      if (discriminant >= 0) {
        const sqrt = Math.sqrt(discriminant)
        return [(trace + sqrt) / 2, (trace - sqrt) / 2]
      }
    }
    return [] // Complex eigenvalues or larger matrices
  }

  const calculate = () => {
    setResult(null)
    setScalarResult(null)
    setEigenvalues(null)

    try {
      switch (operation) {
        case 'add':
          setResult(addMatrices(matrixA, matrixB))
          break
        case 'subtract':
          setResult(subtractMatrices(matrixA, matrixB))
          break
        case 'multiply':
          setResult(multiplyMatrices(matrixA, matrixB))
          break
        case 'scalar':
          setResult(scalarMultiply(matrixA, scalar))
          break
        case 'transpose':
          setResult(transpose(matrixA))
          break
        case 'determinant':
          setScalarResult(determinant(matrixA))
          break
        case 'inverse':
          const inv = inverse(matrixA)
          if (inv) {
            setResult(inv)
          } else {
            alert('행렬이 역행렬을 가지지 않습니다 (det = 0)')
          }
          break
        case 'eigenvalue':
          const eigenvals = calculateEigenvalues(matrixA)
          if (eigenvals.length > 0) {
            setEigenvalues(eigenvals)
          } else {
            alert('2×2 행렬만 지원됩니다')
          }
          break
      }
    } catch (error) {
      alert('계산 오류가 발생했습니다')
    }
  }

  const renderMatrix = (matrix: Matrix, label: string, color: string, editable = false, isB = false) => {
    return (
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <h3 className={`text-lg font-semibold mb-4 ${color}`}>{label}</h3>
        <div className="inline-block">
          <div className="flex items-center gap-2">
            <div className="text-3xl text-slate-600">[</div>
            <div className="space-y-2">
              {matrix.map((row, i) => (
                <div key={i} className="flex gap-2">
                  {row.map((val, j) => (
                    <input
                      key={`${i}-${j}`}
                      type="number"
                      value={val}
                      onChange={(e) =>
                        editable && (isB ? updateMatrixB(i, j, e.target.value) : updateMatrixA(i, j, e.target.value))
                      }
                      readOnly={!editable}
                      className={`w-16 px-2 py-2 text-center rounded ${
                        editable
                          ? 'bg-slate-700 border border-slate-600'
                          : 'bg-slate-900/50 border border-slate-800'
                      } text-white font-mono`}
                      step="0.1"
                    />
                  ))}
                </div>
              ))}
            </div>
            <div className="text-3xl text-slate-600">]</div>
          </div>
        </div>
        <p className="text-xs text-slate-400 mt-2">
          크기: {matrix.length}×{matrix[0].length}
        </p>
      </div>
    )
  }

  const resetMatrices = () => {
    setMatrixA([
      [2, 1, 3],
      [0, -1, 4],
      [1, 2, -2]
    ])
    setMatrixB([
      [1, 0, 2],
      [3, -1, 1],
      [0, 2, -1]
    ])
    setResult(null)
    setScalarResult(null)
    setEigenvalues(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">행렬 계산기</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Matrix A */}
          <div>{renderMatrix(matrixA, '행렬 A', 'text-blue-400', true, false)}</div>

          {/* Matrix B */}
          {(operation === 'add' || operation === 'subtract' || operation === 'multiply') &&
            renderMatrix(matrixB, '행렬 B', 'text-green-400', true, true)}

          {/* Controls */}
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4">연산 선택</h3>
            <div className="space-y-2 mb-6">
              {[
                { id: 'add', label: 'A + B', icon: Plus },
                { id: 'subtract', label: 'A - B', icon: Minus },
                { id: 'multiply', label: 'A × B', icon: MultiplyIcon },
                { id: 'scalar', label: 'kA', icon: MultiplyIcon },
                { id: 'transpose', label: 'A^T', icon: Grid3x3 },
                { id: 'determinant', label: 'det(A)', icon: Calculator },
                { id: 'inverse', label: 'A^(-1)', icon: Calculator },
                { id: 'eigenvalue', label: '고유값', icon: Calculator }
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

            {operation === 'scalar' && (
              <div className="mb-6">
                <label className="text-sm text-slate-300 mb-2 block">스칼라 k: {scalar}</label>
                <input
                  type="range"
                  min="-5"
                  max="5"
                  step="0.5"
                  value={scalar}
                  onChange={(e) => setScalar(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            )}

            <div className="space-y-2">
              <button
                onClick={calculate}
                className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
              >
                <Calculator className="w-5 h-5" />
                <span>계산하기</span>
              </button>
              <button
                onClick={resetMatrices}
                className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
              >
                <RotateCcw className="w-5 h-5" />
                <span>초기화</span>
              </button>
            </div>
          </div>
        </div>

        {/* Result */}
        {result && (
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-yellow-400">결과</h3>
            {renderMatrix(result, '', '', false)}
          </div>
        )}

        {scalarResult !== null && (
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-yellow-400">결과</h3>
            <div className="text-4xl font-mono font-bold text-center py-8">
              {scalarResult.toFixed(4)}
            </div>
          </div>
        )}

        {eigenvalues && (
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-yellow-400">고유값</h3>
            <div className="space-y-2">
              {eigenvalues.map((val, i) => (
                <div key={i} className="text-2xl font-mono">
                  λ<sub>{i + 1}</sub> = {val.toFixed(4)}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
