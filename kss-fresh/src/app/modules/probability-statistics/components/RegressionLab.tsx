'use client'

import { useState, useEffect, useRef } from 'react'
import { TrendingUp, RefreshCw, Download, Plus, Trash2 } from 'lucide-react'

interface DataPoint {
  x: number
  y: number
}

interface RegressionResult {
  slope: number
  intercept: number
  rSquared: number
  correlation: number
  residuals: number[]
  predictions: number[]
}

export default function RegressionLab() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([])
  const [newX, setNewX] = useState('')
  const [newY, setNewY] = useState('')
  const [regressionResult, setRegressionResult] = useState<RegressionResult | null>(null)
  const [showResiduals, setShowResiduals] = useState(true)
  const [showConfidenceInterval, setShowConfidenceInterval] = useState(false)
  const [polynomialDegree, setPolynomialDegree] = useState(1)
  const [regressionType, setRegressionType] = useState<'linear' | 'polynomial' | 'exponential'>('linear')

  // 샘플 데이터 생성
  const generateSampleData = () => {
    const sampleData: DataPoint[] = []
    const n = 20
    const slope = 2 + Math.random() * 3
    const intercept = 10 + Math.random() * 20
    
    for (let i = 0; i < n; i++) {
      const x = i * 5 + Math.random() * 5
      const noise = (Math.random() - 0.5) * 20
      const y = slope * x + intercept + noise
      sampleData.push({ x, y })
    }
    
    setDataPoints(sampleData)
  }

  // 데이터 포인트 추가
  const addDataPoint = () => {
    const x = parseFloat(newX)
    const y = parseFloat(newY)
    
    if (!isNaN(x) && !isNaN(y)) {
      setDataPoints([...dataPoints, { x, y }])
      setNewX('')
      setNewY('')
    }
  }

  // 데이터 포인트 삭제
  const removeDataPoint = (index: number) => {
    setDataPoints(dataPoints.filter((_, i) => i !== index))
  }

  // 선형 회귀 계산
  const calculateLinearRegression = (): RegressionResult | null => {
    if (dataPoints.length < 2) return null
    
    const n = dataPoints.length
    const sumX = dataPoints.reduce((sum, p) => sum + p.x, 0)
    const sumY = dataPoints.reduce((sum, p) => sum + p.y, 0)
    const sumXY = dataPoints.reduce((sum, p) => sum + p.x * p.y, 0)
    const sumX2 = dataPoints.reduce((sum, p) => sum + p.x * p.x, 0)
    const sumY2 = dataPoints.reduce((sum, p) => sum + p.y * p.y, 0)
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
    const intercept = (sumY - slope * sumX) / n
    
    // 예측값과 잔차
    const predictions = dataPoints.map(p => slope * p.x + intercept)
    const residuals = dataPoints.map((p, i) => p.y - predictions[i])
    
    // R² 계산
    const meanY = sumY / n
    const ssTot = dataPoints.reduce((sum, p) => sum + Math.pow(p.y - meanY, 2), 0)
    const ssRes = residuals.reduce((sum, r) => sum + r * r, 0)
    const rSquared = 1 - (ssRes / ssTot)
    
    // 상관계수
    const correlation = (n * sumXY - sumX * sumY) / 
      Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))
    
    return {
      slope,
      intercept,
      rSquared,
      correlation,
      residuals,
      predictions
    }
  }

  // 회귀 분석 실행
  useEffect(() => {
    if (dataPoints.length >= 2) {
      const result = calculateLinearRegression()
      setRegressionResult(result)
    } else {
      setRegressionResult(null)
    }
  }, [dataPoints, regressionType, polynomialDegree])

  // 그래프 그리기
  useEffect(() => {
    if (!canvasRef.current) return
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Canvas 크기 설정
    canvas.width = canvas.offsetWidth * 2
    canvas.height = canvas.offsetHeight * 2
    ctx.scale(2, 2)
    
    const width = canvas.offsetWidth
    const height = canvas.offsetHeight
    const padding = 40
    
    // 배경
    ctx.fillStyle = '#f9fafb'
    ctx.fillRect(0, 0, width, height)
    
    // 축 그리기
    ctx.strokeStyle = '#374151'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(padding, padding)
    ctx.lineTo(padding, height - padding)
    ctx.lineTo(width - padding, height - padding)
    ctx.stroke()
    
    if (dataPoints.length === 0) return
    
    // 데이터 범위 계산
    const xValues = dataPoints.map(p => p.x)
    const yValues = dataPoints.map(p => p.y)
    const minX = Math.min(...xValues) - 10
    const maxX = Math.max(...xValues) + 10
    const minY = Math.min(...yValues) - 10
    const maxY = Math.max(...yValues) + 10
    
    const xScale = (width - 2 * padding) / (maxX - minX)
    const yScale = (height - 2 * padding) / (maxY - minY)
    
    // 그리드 그리기
    ctx.strokeStyle = '#e5e7eb'
    ctx.lineWidth = 0.5
    for (let i = 0; i <= 10; i++) {
      const x = padding + (i / 10) * (width - 2 * padding)
      const y = padding + (i / 10) * (height - 2 * padding)
      
      ctx.beginPath()
      ctx.moveTo(x, padding)
      ctx.lineTo(x, height - padding)
      ctx.stroke()
      
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(width - padding, y)
      ctx.stroke()
    }
    
    // 회귀선 그리기
    if (regressionResult) {
      ctx.strokeStyle = '#dc2626'
      ctx.lineWidth = 2
      ctx.beginPath()
      
      const startX = minX
      const endX = maxX
      const startY = regressionResult.slope * startX + regressionResult.intercept
      const endY = regressionResult.slope * endX + regressionResult.intercept
      
      ctx.moveTo(
        padding + (startX - minX) * xScale,
        height - padding - (startY - minY) * yScale
      )
      ctx.lineTo(
        padding + (endX - minX) * xScale,
        height - padding - (endY - minY) * yScale
      )
      ctx.stroke()
      
      // 잔차 그리기
      if (showResiduals) {
        ctx.strokeStyle = '#9ca3af'
        ctx.lineWidth = 1
        ctx.setLineDash([5, 5])
        
        dataPoints.forEach((point, i) => {
          const predY = regressionResult.predictions[i]
          ctx.beginPath()
          ctx.moveTo(
            padding + (point.x - minX) * xScale,
            height - padding - (point.y - minY) * yScale
          )
          ctx.lineTo(
            padding + (point.x - minX) * xScale,
            height - padding - (predY - minY) * yScale
          )
          ctx.stroke()
        })
        
        ctx.setLineDash([])
      }
    }
    
    // 데이터 포인트 그리기
    dataPoints.forEach(point => {
      ctx.fillStyle = '#3b82f6'
      ctx.beginPath()
      ctx.arc(
        padding + (point.x - minX) * xScale,
        height - padding - (point.y - minY) * yScale,
        4,
        0,
        2 * Math.PI
      )
      ctx.fill()
    })
    
    // 축 라벨
    ctx.fillStyle = '#374151'
    ctx.font = '12px sans-serif'
    ctx.textAlign = 'center'
    
    // X축 라벨
    for (let i = 0; i <= 5; i++) {
      const value = minX + (i / 5) * (maxX - minX)
      const x = padding + (i / 5) * (width - 2 * padding)
      ctx.fillText(value.toFixed(1), x, height - padding + 20)
    }
    
    // Y축 라벨
    ctx.textAlign = 'right'
    for (let i = 0; i <= 5; i++) {
      const value = minY + (i / 5) * (maxY - minY)
      const y = height - padding - (i / 5) * (height - 2 * padding)
      ctx.fillText(value.toFixed(1), padding - 10, y + 5)
    }
  }, [dataPoints, regressionResult, showResiduals, showConfidenceInterval])

  // 초기 샘플 데이터
  useEffect(() => {
    generateSampleData()
  }, [])

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <TrendingUp className="w-6 h-6 text-purple-600" />
          회귀분석 연구실
        </h2>
        <button
          onClick={generateSampleData}
          className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg flex items-center gap-2 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          샘플 데이터
        </button>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* 그래프 영역 */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700">
            <canvas
              ref={canvasRef}
              className="w-full h-96"
              style={{ width: '100%', height: '384px' }}
            />
          </div>

          {/* 회귀 결과 */}
          {regressionResult && (
            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                <div className="text-sm text-gray-600 dark:text-gray-400">기울기</div>
                <div className="font-semibold">{regressionResult.slope.toFixed(3)}</div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                <div className="text-sm text-gray-600 dark:text-gray-400">절편</div>
                <div className="font-semibold">{regressionResult.intercept.toFixed(3)}</div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                <div className="text-sm text-gray-600 dark:text-gray-400">R²</div>
                <div className="font-semibold">{regressionResult.rSquared.toFixed(4)}</div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                <div className="text-sm text-gray-600 dark:text-gray-400">상관계수</div>
                <div className="font-semibold">{regressionResult.correlation.toFixed(4)}</div>
              </div>
            </div>
          )}
        </div>

        {/* 설정 패널 */}
        <div className="space-y-4">
          {/* 데이터 입력 */}
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold mb-3">데이터 추가</h3>
            <div className="space-y-2">
              <input
                type="number"
                placeholder="X 값"
                value={newX}
                onChange={(e) => setNewX(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg dark:bg-gray-700"
              />
              <input
                type="number"
                placeholder="Y 값"
                value={newY}
                onChange={(e) => setNewY(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg dark:bg-gray-700"
              />
              <button
                onClick={addDataPoint}
                className="w-full py-2 px-4 bg-purple-600 hover:bg-purple-700 text-white rounded-lg flex items-center justify-center gap-2 transition-colors"
              >
                <Plus className="w-4 h-4" />
                추가
              </button>
            </div>
          </div>

          {/* 옵션 */}
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold mb-3">표시 옵션</h3>
            <div className="space-y-2">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showResiduals}
                  onChange={(e) => setShowResiduals(e.target.checked)}
                  className="rounded"
                />
                <span className="text-sm">잔차 표시</span>
              </label>
            </div>
          </div>

          {/* 데이터 목록 */}
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold mb-3">데이터 포인트 ({dataPoints.length}개)</h3>
            <div className="max-h-64 overflow-y-auto space-y-1">
              {dataPoints.map((point, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded">
                  <span className="text-sm font-mono">
                    ({point.x.toFixed(2)}, {point.y.toFixed(2)})
                  </span>
                  <button
                    onClick={() => removeDataPoint(index)}
                    className="text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* 회귀 방정식 */}
          {regressionResult && (
            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
              <h3 className="font-semibold mb-2">회귀 방정식</h3>
              <p className="font-mono text-sm">
                y = {regressionResult.slope.toFixed(3)}x + {regressionResult.intercept.toFixed(3)}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                {regressionResult.rSquared > 0.8 
                  ? '매우 강한 선형 관계'
                  : regressionResult.rSquared > 0.5
                  ? '중간 정도의 선형 관계'
                  : '약한 선형 관계'}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}