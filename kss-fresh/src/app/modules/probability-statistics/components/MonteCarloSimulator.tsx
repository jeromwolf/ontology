'use client'

import { useState, useEffect, useRef } from 'react'
import { Brain, Play, Pause, RefreshCw, Zap } from 'lucide-react'

interface SimulationResult {
  piEstimate: number
  areaEstimate: number
  optionPrice: number
  integralEstimate: number
}

export default function MonteCarloSimulator() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [selectedSimulation, setSelectedSimulation] = useState('pi')
  const [numSamples, setNumSamples] = useState(1000)
  const [isRunning, setIsRunning] = useState(false)
  const [currentSamples, setCurrentSamples] = useState(0)
  const [results, setResults] = useState<Partial<SimulationResult>>({})
  
  // 시뮬레이션별 파라미터
  const [stockPrice, setStockPrice] = useState(100)
  const [strikePrice, setStrikePrice] = useState(105)
  const [volatility, setVolatility] = useState(0.2)
  const [riskFreeRate, setRiskFreeRate] = useState(0.05)
  const [timeToExpiry, setTimeToExpiry] = useState(1)
  
  // 적분 함수 선택
  const [integralFunction, setIntegralFunction] = useState('x^2')
  const [integralBounds, setIntegralBounds] = useState({ a: 0, b: 1 })

  // 파이(π) 추정 시뮬레이션
  const estimatePi = (samples: number) => {
    let insideCircle = 0
    const points: { x: number; y: number; inside: boolean }[] = []
    
    for (let i = 0; i < samples; i++) {
      const x = Math.random() * 2 - 1
      const y = Math.random() * 2 - 1
      const distance = x * x + y * y
      
      if (distance <= 1) {
        insideCircle++
      }
      
      // 시각화를 위해 일부 포인트만 저장
      if (i < 1000) {
        points.push({ x, y, inside: distance <= 1 })
      }
    }
    
    const piEstimate = (insideCircle / samples) * 4
    setResults(prev => ({ ...prev, piEstimate }))
    
    // 시각화
    if (canvasRef.current) {
      drawPiEstimation(points)
    }
  }

  // 파이 추정 시각화
  const drawPiEstimation = (points: { x: number; y: number; inside: boolean }[]) => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    canvas.width = canvas.offsetWidth * 2
    canvas.height = canvas.offsetHeight * 2
    ctx.scale(2, 2)
    
    const size = Math.min(canvas.offsetWidth, canvas.offsetHeight)
    const center = size / 2
    const radius = size * 0.4
    
    // 배경
    ctx.fillStyle = '#f9fafb'
    ctx.fillRect(0, 0, size, size)
    
    // 사각형
    ctx.strokeStyle = '#374151'
    ctx.lineWidth = 2
    ctx.strokeRect(center - radius, center - radius, radius * 2, radius * 2)
    
    // 원
    ctx.beginPath()
    ctx.arc(center, center, radius, 0, 2 * Math.PI)
    ctx.stroke()
    
    // 포인트
    points.forEach(point => {
      ctx.fillStyle = point.inside ? '#3b82f6' : '#ef4444'
      ctx.beginPath()
      ctx.arc(
        center + point.x * radius,
        center + point.y * radius,
        2,
        0,
        2 * Math.PI
      )
      ctx.fill()
    })
  }

  // 면적 계산 시뮬레이션
  const estimateArea = (samples: number) => {
    let underCurve = 0
    const points: { x: number; y: number; under: boolean }[] = []
    
    for (let i = 0; i < samples; i++) {
      const x = Math.random()
      const y = Math.random()
      // f(x) = x^2 + 0.5*sin(10*x) 함수 아래 면적
      const fx = x * x + 0.5 * Math.sin(10 * x)
      const under = y <= fx
      
      if (under) {
        underCurve++
      }
      
      if (i < 1000) {
        points.push({ x, y, under })
      }
    }
    
    const areaEstimate = underCurve / samples
    setResults(prev => ({ ...prev, areaEstimate }))
    
    if (canvasRef.current) {
      drawAreaEstimation(points)
    }
  }

  // 면적 추정 시각화
  const drawAreaEstimation = (points: { x: number; y: number; under: boolean }[]) => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    canvas.width = canvas.offsetWidth * 2
    canvas.height = canvas.offsetHeight * 2
    ctx.scale(2, 2)
    
    const width = canvas.offsetWidth
    const height = canvas.offsetHeight
    const padding = 20
    
    // 배경
    ctx.fillStyle = '#f9fafb'
    ctx.fillRect(0, 0, width, height)
    
    // 함수 그리기
    ctx.strokeStyle = '#374151'
    ctx.lineWidth = 2
    ctx.beginPath()
    for (let i = 0; i <= 100; i++) {
      const x = i / 100
      const y = x * x + 0.5 * Math.sin(10 * x)
      const px = padding + x * (width - 2 * padding)
      const py = height - padding - y * (height - 2 * padding)
      
      if (i === 0) {
        ctx.moveTo(px, py)
      } else {
        ctx.lineTo(px, py)
      }
    }
    ctx.stroke()
    
    // 포인트
    points.forEach(point => {
      ctx.fillStyle = point.under ? '#3b82f6' : '#ef4444'
      ctx.beginPath()
      ctx.arc(
        padding + point.x * (width - 2 * padding),
        height - padding - point.y * (height - 2 * padding),
        1.5,
        0,
        2 * Math.PI
      )
      ctx.fill()
    })
  }

  // 블랙-숄즈 옵션 가격 계산
  const estimateOptionPrice = (samples: number) => {
    let payoffSum = 0
    
    for (let i = 0; i < samples; i++) {
      // 기하 브라운 운동 시뮬레이션
      const z = Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random())
      const ST = stockPrice * Math.exp(
        (riskFreeRate - 0.5 * volatility * volatility) * timeToExpiry + 
        volatility * Math.sqrt(timeToExpiry) * z
      )
      
      // 콜 옵션 페이오프
      const payoff = Math.max(ST - strikePrice, 0)
      payoffSum += payoff
    }
    
    const optionPrice = Math.exp(-riskFreeRate * timeToExpiry) * (payoffSum / samples)
    setResults(prev => ({ ...prev, optionPrice }))
  }

  // 적분 계산
  const estimateIntegral = (samples: number) => {
    let sum = 0
    const { a, b } = integralBounds
    
    for (let i = 0; i < samples; i++) {
      const x = a + Math.random() * (b - a)
      let fx = 0
      
      switch (integralFunction) {
        case 'x^2':
          fx = x * x
          break
        case 'sin(x)':
          fx = Math.sin(x)
          break
        case 'e^x':
          fx = Math.exp(x)
          break
        case '1/(1+x^2)':
          fx = 1 / (1 + x * x)
          break
      }
      
      sum += fx
    }
    
    const integralEstimate = (b - a) * sum / samples
    setResults(prev => ({ ...prev, integralEstimate }))
  }

  // 시뮬레이션 실행
  const runSimulation = () => {
    setIsRunning(true)
    setCurrentSamples(0)
    setResults({})
    
    let completed = 0
    const batchSize = 100
    
    const interval = setInterval(() => {
      const batch = Math.min(batchSize, numSamples - completed)
      
      switch (selectedSimulation) {
        case 'pi':
          estimatePi(completed + batch)
          break
        case 'area':
          estimateArea(completed + batch)
          break
        case 'option':
          estimateOptionPrice(completed + batch)
          break
        case 'integral':
          estimateIntegral(completed + batch)
          break
      }
      
      completed += batch
      setCurrentSamples(completed)
      
      if (completed >= numSamples) {
        clearInterval(interval)
        setIsRunning(false)
      }
    }, 50)
  }

  // 시뮬레이션 중지
  const stopSimulation = () => {
    setIsRunning(false)
  }

  // 리셋
  const resetSimulation = () => {
    setIsRunning(false)
    setCurrentSamples(0)
    setResults({})
    
    const canvas = canvasRef.current
    if (canvas) {
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height)
      }
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Brain className="w-6 h-6 text-purple-600" />
          몬테카를로 시뮬레이션
        </h2>
        <div className="flex gap-2">
          {isRunning ? (
            <button
              onClick={stopSimulation}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg flex items-center gap-2 transition-colors"
            >
              <Pause className="w-4 h-4" />
              정지
            </button>
          ) : (
            <button
              onClick={runSimulation}
              className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg flex items-center gap-2 transition-colors"
            >
              <Play className="w-4 h-4" />
              시작
            </button>
          )}
          <button
            onClick={resetSimulation}
            className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg flex items-center gap-2 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            리셋
          </button>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* 시각화 영역 */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700">
            {(selectedSimulation === 'pi' || selectedSimulation === 'area') ? (
              <canvas
                ref={canvasRef}
                className="w-full aspect-square max-h-96 mx-auto"
                style={{ maxWidth: '384px' }}
              />
            ) : (
              <div className="h-96 flex items-center justify-center">
                <div className="text-center">
                  <Zap className="w-16 h-16 text-purple-300 dark:text-purple-700 mx-auto mb-4" />
                  <p className="text-gray-600 dark:text-gray-400">
                    {selectedSimulation === 'option' 
                      ? '옵션 가격 계산 중...' 
                      : '적분 값 계산 중...'}
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* 진행 상황 */}
          <div className="mt-4 bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium">진행률</span>
              <span className="text-sm">{currentSamples.toLocaleString()} / {numSamples.toLocaleString()}</span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
              <div
                className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${(currentSamples / numSamples) * 100}%` }}
              />
            </div>
          </div>

          {/* 결과 표시 */}
          <div className="mt-4 space-y-4">
            {selectedSimulation === 'pi' && results.piEstimate && (
              <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">π (파이) 추정값</h4>
                <div className="text-2xl font-mono">{results.piEstimate.toFixed(6)}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  실제값: 3.141592... | 오차: {Math.abs(Math.PI - results.piEstimate).toFixed(6)}
                </div>
              </div>
            )}

            {selectedSimulation === 'area' && results.areaEstimate && (
              <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">면적 추정값</h4>
                <div className="text-2xl font-mono">{results.areaEstimate.toFixed(6)}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  함수: f(x) = x² + 0.5sin(10x), 0 ≤ x ≤ 1
                </div>
              </div>
            )}

            {selectedSimulation === 'option' && results.optionPrice && (
              <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">콜 옵션 가격</h4>
                <div className="text-2xl font-mono">${results.optionPrice.toFixed(2)}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  블랙-숄즈 모델 기반 몬테카를로 시뮬레이션
                </div>
              </div>
            )}

            {selectedSimulation === 'integral' && results.integralEstimate && (
              <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">적분 추정값</h4>
                <div className="text-2xl font-mono">{results.integralEstimate.toFixed(6)}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  ∫{integralBounds.a}^{integralBounds.b} {integralFunction} dx
                </div>
              </div>
            )}
          </div>
        </div>

        {/* 설정 패널 */}
        <div className="space-y-4">
          {/* 시뮬레이션 선택 */}
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <label className="block text-sm font-medium mb-2">시뮬레이션 유형</label>
            <select
              value={selectedSimulation}
              onChange={(e) => {
                setSelectedSimulation(e.target.value)
                resetSimulation()
              }}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg dark:bg-gray-700"
              disabled={isRunning}
            >
              <option value="pi">π (파이) 추정</option>
              <option value="area">곡선 아래 면적</option>
              <option value="option">옵션 가격 계산</option>
              <option value="integral">적분 계산</option>
            </select>
          </div>

          {/* 샘플 수 설정 */}
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <label className="block text-sm font-medium mb-2">샘플 수</label>
            <input
              type="range"
              min="100"
              max="100000"
              step="100"
              value={numSamples}
              onChange={(e) => setNumSamples(Number(e.target.value))}
              className="w-full"
              disabled={isRunning}
            />
            <div className="text-center text-sm mt-1">{numSamples.toLocaleString()}</div>
          </div>

          {/* 옵션 가격 파라미터 */}
          {selectedSimulation === 'option' && (
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700 space-y-3">
              <h3 className="font-semibold">옵션 파라미터</h3>
              <div>
                <label className="block text-sm mb-1">현재 주가</label>
                <input
                  type="number"
                  value={stockPrice}
                  onChange={(e) => setStockPrice(Number(e.target.value))}
                  className="w-full px-3 py-1 border border-gray-300 dark:border-gray-600 rounded dark:bg-gray-700"
                  disabled={isRunning}
                />
              </div>
              <div>
                <label className="block text-sm mb-1">행사가격</label>
                <input
                  type="number"
                  value={strikePrice}
                  onChange={(e) => setStrikePrice(Number(e.target.value))}
                  className="w-full px-3 py-1 border border-gray-300 dark:border-gray-600 rounded dark:bg-gray-700"
                  disabled={isRunning}
                />
              </div>
              <div>
                <label className="block text-sm mb-1">변동성 (σ)</label>
                <input
                  type="number"
                  step="0.01"
                  value={volatility}
                  onChange={(e) => setVolatility(Number(e.target.value))}
                  className="w-full px-3 py-1 border border-gray-300 dark:border-gray-600 rounded dark:bg-gray-700"
                  disabled={isRunning}
                />
              </div>
            </div>
          )}

          {/* 적분 파라미터 */}
          {selectedSimulation === 'integral' && (
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700 space-y-3">
              <h3 className="font-semibold">적분 설정</h3>
              <div>
                <label className="block text-sm mb-1">함수</label>
                <select
                  value={integralFunction}
                  onChange={(e) => setIntegralFunction(e.target.value)}
                  className="w-full px-3 py-1 border border-gray-300 dark:border-gray-600 rounded dark:bg-gray-700"
                  disabled={isRunning}
                >
                  <option value="x^2">x²</option>
                  <option value="sin(x)">sin(x)</option>
                  <option value="e^x">e^x</option>
                  <option value="1/(1+x^2)">1/(1+x²)</option>
                </select>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-sm mb-1">하한 (a)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={integralBounds.a}
                    onChange={(e) => setIntegralBounds(prev => ({ ...prev, a: Number(e.target.value) }))}
                    className="w-full px-3 py-1 border border-gray-300 dark:border-gray-600 rounded dark:bg-gray-700"
                    disabled={isRunning}
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">상한 (b)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={integralBounds.b}
                    onChange={(e) => setIntegralBounds(prev => ({ ...prev, b: Number(e.target.value) }))}
                    className="w-full px-3 py-1 border border-gray-300 dark:border-gray-600 rounded dark:bg-gray-700"
                    disabled={isRunning}
                  />
                </div>
              </div>
            </div>
          )}

          {/* 몬테카를로 설명 */}
          <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
            <h4 className="font-semibold mb-2">몬테카를로 방법이란?</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              무작위 샘플링을 사용하여 수치적 결과를 얻는 계산 알고리즘입니다. 
              복잡한 적분, 최적화, 시뮬레이션 문제를 해결하는 데 사용됩니다.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}