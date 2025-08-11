'use client'

import { useState, useEffect, useRef } from 'react'
import { TestTube, TrendingUp, AlertCircle, CheckCircle, BarChart3, Users, Target, Calculator } from 'lucide-react'

interface Variant {
  name: string
  visitors: number
  conversions: number
  conversionRate: number
  color: string
}

interface TestResult {
  winner: string | null
  confidence: number
  pValue: number
  uplift: number
  sampleSizeReached: boolean
  significanceReached: boolean
}

export default function ABTestSimulator() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [testType, setTestType] = useState<'conversion' | 'revenue' | 'engagement'>('conversion')
  const [sampleSize, setSampleSize] = useState(1000)
  const [confidenceLevel, setConfidenceLevel] = useState(95)
  const [minimumDetectableEffect, setMinimumDetectableEffect] = useState(5)
  
  const [variantA, setVariantA] = useState<Variant>({
    name: 'Control (A)',
    visitors: 0,
    conversions: 0,
    conversionRate: 15,
    color: '#3b82f6'
  })
  
  const [variantB, setVariantB] = useState<Variant>({
    name: 'Variant (B)',
    visitors: 0,
    conversions: 0,
    conversionRate: 17,
    color: '#10b981'
  })
  
  const [testResult, setTestResult] = useState<TestResult>({
    winner: null,
    confidence: 0,
    pValue: 1,
    uplift: 0,
    sampleSizeReached: false,
    significanceReached: false
  })
  
  const [dailyData, setDailyData] = useState<{
    day: number
    aRate: number
    bRate: number
  }[]>([])
  
  // Z-score 계산
  const calculateZScore = (p1: number, p2: number, n1: number, n2: number): number => {
    if (n1 === 0 || n2 === 0) return 0
    
    const pooledP = ((p1 * n1) + (p2 * n2)) / (n1 + n2)
    const standardError = Math.sqrt(pooledP * (1 - pooledP) * (1/n1 + 1/n2))
    
    if (standardError === 0) return 0
    
    return (p2 - p1) / standardError
  }
  
  // P-value 계산 (정규분포 근사)
  const calculatePValue = (zScore: number): number => {
    const absZ = Math.abs(zScore)
    
    // 간단한 정규분포 CDF 근사
    const t = 1 / (1 + 0.2316419 * absZ)
    const d = 0.3989423 * Math.exp(-absZ * absZ / 2)
    const probability = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    
    return 2 * probability // 양측 검정
  }
  
  // 필요한 샘플 크기 계산
  const calculateRequiredSampleSize = (): number => {
    const alpha = (100 - confidenceLevel) / 100
    const beta = 0.2 // 80% power
    const p1 = variantA.conversionRate / 100
    const p2 = p1 * (1 + minimumDetectableEffect / 100)
    
    const zAlpha = 1.96 // 95% 신뢰수준
    const zBeta = 0.84 // 80% 검정력
    
    const pooledP = (p1 + p2) / 2
    const n = Math.ceil(
      2 * pooledP * (1 - pooledP) * Math.pow(zAlpha + zBeta, 2) / Math.pow(p2 - p1, 2)
    )
    
    return n
  }
  
  // 시뮬레이션 단계
  const simulateStep = () => {
    if (variantA.visitors >= sampleSize && variantB.visitors >= sampleSize) {
      setIsRunning(false)
      return
    }
    
    // 방문자 추가 (동시에 50명씩)
    const newVisitors = 50
    
    // A 그룹
    const aVisitors = Math.min(newVisitors, sampleSize - variantA.visitors)
    const aConversions = Array(aVisitors).fill(0).filter(() => 
      Math.random() < (variantA.conversionRate / 100)
    ).length
    
    // B 그룹
    const bVisitors = Math.min(newVisitors, sampleSize - variantB.visitors)
    const bConversions = Array(bVisitors).fill(0).filter(() => 
      Math.random() < (variantB.conversionRate / 100)
    ).length
    
    const newA = {
      ...variantA,
      visitors: variantA.visitors + aVisitors,
      conversions: variantA.conversions + aConversions
    }
    
    const newB = {
      ...variantB,
      visitors: variantB.visitors + bVisitors,
      conversions: variantB.conversions + bConversions
    }
    
    setVariantA(newA)
    setVariantB(newB)
    
    // 결과 계산
    const rateA = newA.visitors > 0 ? newA.conversions / newA.visitors : 0
    const rateB = newB.visitors > 0 ? newB.conversions / newB.visitors : 0
    
    const zScore = calculateZScore(rateA, rateB, newA.visitors, newB.visitors)
    const pValue = calculatePValue(zScore)
    const confidence = (1 - pValue) * 100
    const uplift = rateA > 0 ? ((rateB - rateA) / rateA) * 100 : 0
    
    const requiredSampleSize = calculateRequiredSampleSize()
    const sampleSizeReached = newA.visitors >= requiredSampleSize && newB.visitors >= requiredSampleSize
    const significanceReached = pValue < (1 - confidenceLevel / 100)
    
    setTestResult({
      winner: significanceReached ? (rateB > rateA ? 'B' : 'A') : null,
      confidence,
      pValue,
      uplift,
      sampleSizeReached,
      significanceReached
    })
    
    // 일별 데이터 추가
    if ((newA.visitors + newB.visitors) % 100 === 0) {
      setDailyData(prev => [...prev, {
        day: prev.length + 1,
        aRate: rateA * 100,
        bRate: rateB * 100
      }])
    }
  }
  
  // 차트 그리기
  const drawChart = () => {
    const canvas = canvasRef.current
    if (!canvas || dailyData.length === 0) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const width = 600
    const height = 300
    const padding = 40
    
    ctx.clearRect(0, 0, width, height)
    
    // 배경 그리드
    ctx.strokeStyle = '#e5e7eb'
    ctx.lineWidth = 1
    
    // Y축 그리드와 레이블
    const maxRate = Math.max(
      ...dailyData.flatMap(d => [d.aRate, d.bRate]),
      variantA.conversionRate,
      variantB.conversionRate
    ) * 1.2
    
    for (let i = 0; i <= 5; i++) {
      const y = height - padding - (i / 5) * (height - 2 * padding)
      
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(width - padding, y)
      ctx.stroke()
      
      ctx.fillStyle = '#666'
      ctx.font = '10px sans-serif'
      ctx.textAlign = 'right'
      ctx.fillText(`${(maxRate * i / 5).toFixed(1)}%`, padding - 5, y + 3)
    }
    
    // 데이터 포인트 그리기
    if (dailyData.length > 1) {
      const xStep = (width - 2 * padding) / (dailyData.length - 1)
      
      // A 변형
      ctx.strokeStyle = variantA.color
      ctx.lineWidth = 2
      ctx.beginPath()
      
      dailyData.forEach((point, i) => {
        const x = padding + i * xStep
        const y = height - padding - (point.aRate / maxRate) * (height - 2 * padding)
        
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()
      
      // B 변형
      ctx.strokeStyle = variantB.color
      ctx.beginPath()
      
      dailyData.forEach((point, i) => {
        const x = padding + i * xStep
        const y = height - padding - (point.bRate / maxRate) * (height - 2 * padding)
        
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()
      
      // 신뢰구간 영역 (간단한 시각화)
      if (testResult.confidence > 0) {
        const lastPoint = dailyData[dailyData.length - 1]
        const avgRate = (lastPoint.aRate + lastPoint.bRate) / 2
        const errorMargin = 1.96 * Math.sqrt(avgRate * (100 - avgRate) / 100) // 간단한 근사
        
        ctx.fillStyle = 'rgba(99, 102, 241, 0.1)'
        ctx.fillRect(
          padding,
          height - padding - ((avgRate + errorMargin) / maxRate) * (height - 2 * padding),
          width - 2 * padding,
          (2 * errorMargin / maxRate) * (height - 2 * padding)
        )
      }
    }
    
    // 범례
    ctx.fillStyle = variantA.color
    ctx.fillRect(width - 150, 20, 20, 3)
    ctx.fillStyle = '#333'
    ctx.font = '12px sans-serif'
    ctx.textAlign = 'left'
    ctx.fillText(variantA.name, width - 125, 25)
    
    ctx.fillStyle = variantB.color
    ctx.fillRect(width - 150, 40, 20, 3)
    ctx.fillStyle = '#333'
    ctx.fillText(variantB.name, width - 125, 45)
  }
  
  // 파워 분석 차트
  const drawPowerAnalysis = () => {
    const canvas = document.getElementById('power-canvas') as HTMLCanvasElement
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    ctx.clearRect(0, 0, 300, 200)
    
    // 정규분포 곡선 그리기
    const mean = 0
    const stdDev = 1
    
    // Null hypothesis (H0)
    ctx.strokeStyle = '#9ca3af'
    ctx.lineWidth = 2
    ctx.beginPath()
    
    for (let x = -4; x <= 4; x += 0.1) {
      const y = (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * x * x)
      const canvasX = 150 + x * 30
      const canvasY = 180 - y * 150
      
      if (x === -4) ctx.moveTo(canvasX, canvasY)
      else ctx.lineTo(canvasX, canvasY)
    }
    ctx.stroke()
    
    // Alternative hypothesis (H1)
    const effect = minimumDetectableEffect / 20 // 스케일 조정
    ctx.strokeStyle = '#3b82f6'
    ctx.beginPath()
    
    for (let x = -4; x <= 4; x += 0.1) {
      const y = (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * Math.pow(x - effect, 2))
      const canvasX = 150 + x * 30
      const canvasY = 180 - y * 150
      
      if (x === -4) ctx.moveTo(canvasX, canvasY)
      else ctx.lineTo(canvasX, canvasY)
    }
    ctx.stroke()
    
    // 유의수준 영역
    const criticalValue = 1.96
    ctx.fillStyle = 'rgba(239, 68, 68, 0.2)'
    ctx.fillRect(150 + criticalValue * 30, 20, 100, 160)
    ctx.fillRect(150 - criticalValue * 30 - 100, 20, 100, 160)
    
    // 레이블
    ctx.fillStyle = '#666'
    ctx.font = '10px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('H₀', 150, 195)
    ctx.fillText('H₁', 150 + effect * 30, 195)
  }
  
  useEffect(() => {
    if (isRunning) {
      const timer = setTimeout(simulateStep, 100)
      return () => clearTimeout(timer)
    }
  }, [isRunning, variantA, variantB])
  
  useEffect(() => {
    drawChart()
  }, [dailyData])
  
  useEffect(() => {
    drawPowerAnalysis()
  }, [minimumDetectableEffect])
  
  const reset = () => {
    setVariantA({...variantA, visitors: 0, conversions: 0})
    setVariantB({...variantB, visitors: 0, conversions: 0})
    setDailyData([])
    setTestResult({
      winner: null,
      confidence: 0,
      pValue: 1,
      uplift: 0,
      sampleSizeReached: false,
      significanceReached: false
    })
    setIsRunning(false)
  }
  
  return (
    <div className="w-full max-w-7xl mx-auto">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-6">A/B 테스트 시뮬레이터</h2>
        
        <div className="grid lg:grid-cols-3 gap-6">
          {/* 메인 영역 */}
          <div className="lg:col-span-2 space-y-6">
            {/* 변형 카드 */}
            <div className="grid md:grid-cols-2 gap-4">
              {/* Control (A) */}
              <div className={`bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-xl p-6 border-2 ${
                testResult.winner === 'A' ? 'border-blue-500' : 'border-transparent'
              }`}>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold flex items-center gap-2">
                    <div className="w-4 h-4 bg-blue-500 rounded" />
                    {variantA.name}
                  </h3>
                  {testResult.winner === 'A' && (
                    <CheckCircle className="w-5 h-5 text-green-500" />
                  )}
                </div>
                
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">방문자</span>
                    <span className="font-semibold">{variantA.visitors.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">전환</span>
                    <span className="font-semibold">{variantA.conversions.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">전환율</span>
                    <span className="font-semibold text-blue-600">
                      {variantA.visitors > 0 
                        ? `${(variantA.conversions / variantA.visitors * 100).toFixed(2)}%`
                        : '0%'}
                    </span>
                  </div>
                  
                  <div className="pt-3 border-t border-blue-200 dark:border-blue-700">
                    <label className="text-sm font-medium">기본 전환율 설정</label>
                    <input
                      type="range"
                      min="1"
                      max="30"
                      value={variantA.conversionRate}
                      onChange={(e) => setVariantA({...variantA, conversionRate: parseInt(e.target.value)})}
                      disabled={isRunning}
                      className="w-full mt-1"
                    />
                    <span className="text-xs text-gray-500">{variantA.conversionRate}%</span>
                  </div>
                </div>
              </div>
              
              {/* Variant (B) */}
              <div className={`bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-xl p-6 border-2 ${
                testResult.winner === 'B' ? 'border-green-500' : 'border-transparent'
              }`}>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold flex items-center gap-2">
                    <div className="w-4 h-4 bg-green-500 rounded" />
                    {variantB.name}
                  </h3>
                  {testResult.winner === 'B' && (
                    <CheckCircle className="w-5 h-5 text-green-500" />
                  )}
                </div>
                
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">방문자</span>
                    <span className="font-semibold">{variantB.visitors.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">전환</span>
                    <span className="font-semibold">{variantB.conversions.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">전환율</span>
                    <span className="font-semibold text-green-600">
                      {variantB.visitors > 0 
                        ? `${(variantB.conversions / variantB.visitors * 100).toFixed(2)}%`
                        : '0%'}
                    </span>
                  </div>
                  
                  <div className="pt-3 border-t border-green-200 dark:border-green-700">
                    <label className="text-sm font-medium">기본 전환율 설정</label>
                    <input
                      type="range"
                      min="1"
                      max="30"
                      value={variantB.conversionRate}
                      onChange={(e) => setVariantB({...variantB, conversionRate: parseInt(e.target.value)})}
                      disabled={isRunning}
                      className="w-full mt-1"
                    />
                    <span className="text-xs text-gray-500">{variantB.conversionRate}%</span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* 차트 */}
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <h3 className="font-semibold mb-3">전환율 추이</h3>
              <canvas
                ref={canvasRef}
                width={600}
                height={300}
                className="w-full border border-gray-300 dark:border-gray-600 rounded"
              />
            </div>
            
            {/* 결과 패널 */}
            <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                테스트 결과
              </h3>
              
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">통계적 유의성</p>
                  <div className="flex items-center gap-2">
                    {testResult.significanceReached ? (
                      <>
                        <CheckCircle className="w-5 h-5 text-green-500" />
                        <span className="font-semibold text-green-600">달성</span>
                      </>
                    ) : (
                      <>
                        <AlertCircle className="w-5 h-5 text-yellow-500" />
                        <span className="font-semibold text-yellow-600">미달성</span>
                      </>
                    )}
                  </div>
                </div>
                
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">신뢰도</p>
                  <p className="font-semibold text-lg">{testResult.confidence.toFixed(1)}%</p>
                </div>
                
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">P-value</p>
                  <p className="font-semibold text-lg">{testResult.pValue.toFixed(4)}</p>
                </div>
                
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">상승률</p>
                  <p className={`font-semibold text-lg ${
                    testResult.uplift > 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {testResult.uplift > 0 ? '+' : ''}{testResult.uplift.toFixed(1)}%
                  </p>
                </div>
              </div>
              
              {testResult.winner && (
                <div className="mt-4 p-3 bg-white dark:bg-gray-800 rounded-lg">
                  <p className="text-center">
                    <span className="font-semibold text-lg">
                      변형 {testResult.winner}가 승리했습니다!
                    </span>
                  </p>
                </div>
              )}
            </div>
            
            {/* 컨트롤 버튼 */}
            <div className="flex gap-2">
              <button
                onClick={() => setIsRunning(!isRunning)}
                disabled={variantA.visitors >= sampleSize && variantB.visitors >= sampleSize}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                  isRunning
                    ? 'bg-red-500 text-white hover:bg-red-600'
                    : 'bg-green-500 text-white hover:bg-green-600'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                <TestTube className="w-4 h-4" />
                {isRunning ? '테스트 중지' : '테스트 시작'}
              </button>
              
              <button
                onClick={reset}
                className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded-lg font-medium hover:bg-gray-600 transition-colors"
              >
                초기화
              </button>
            </div>
          </div>
          
          {/* 설정 패널 */}
          <div className="space-y-6">
            {/* 테스트 설정 */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Target className="w-5 h-5" />
                테스트 설정
              </h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">
                    샘플 크기 (그룹당): {sampleSize.toLocaleString()}
                  </label>
                  <input
                    type="range"
                    min="100"
                    max="10000"
                    step="100"
                    value={sampleSize}
                    onChange={(e) => setSampleSize(parseInt(e.target.value))}
                    disabled={isRunning}
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">
                    신뢰수준: {confidenceLevel}%
                  </label>
                  <input
                    type="range"
                    min="90"
                    max="99"
                    value={confidenceLevel}
                    onChange={(e) => setConfidenceLevel(parseInt(e.target.value))}
                    disabled={isRunning}
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">
                    최소 감지 효과: {minimumDetectableEffect}%
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="20"
                    value={minimumDetectableEffect}
                    onChange={(e) => setMinimumDetectableEffect(parseInt(e.target.value))}
                    disabled={isRunning}
                    className="w-full"
                  />
                </div>
              </div>
            </div>
            
            {/* 샘플 크기 계산 */}
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <Calculator className="w-4 h-4" />
                필요 샘플 크기
              </h4>
              <p className="text-2xl font-bold text-blue-600">
                {calculateRequiredSampleSize().toLocaleString()}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                그룹당 필요한 최소 방문자 수
              </p>
            </div>
            
            {/* 파워 분석 */}
            <div>
              <h3 className="text-lg font-semibold mb-3">통계적 검정력</h3>
              <canvas
                id="power-canvas"
                width={300}
                height={200}
                className="w-full border border-gray-300 dark:border-gray-600 rounded"
              />
              <div className="mt-2 text-xs text-gray-500">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-gray-400 rounded" />
                  <span>귀무가설 (H₀)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-blue-500 rounded" />
                  <span>대립가설 (H₁)</span>
                </div>
              </div>
            </div>
            
            {/* 진행 상황 */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                진행 상황
              </h3>
              
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>A 그룹</span>
                    <span>{((variantA.visitors / sampleSize) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className="h-full bg-blue-500 rounded-full transition-all duration-300"
                      style={{ width: `${Math.min((variantA.visitors / sampleSize) * 100, 100)}%` }}
                    />
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>B 그룹</span>
                    <span>{((variantB.visitors / sampleSize) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className="h-full bg-green-500 rounded-full transition-all duration-300"
                      style={{ width: `${Math.min((variantB.visitors / sampleSize) * 100, 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}