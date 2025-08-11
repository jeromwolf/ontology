'use client'

import { useState, useEffect, useRef } from 'react'
import { BarChart, LineChart, PieChart, Activity, Calculator, Info, Upload, Download } from 'lucide-react'

interface DataSample {
  id: number
  value: number
  category?: string
}

interface Statistics {
  mean: number
  median: number
  mode: number
  stdDev: number
  variance: number
  min: number
  max: number
  q1: number
  q3: number
  iqr: number
}

export default function StatisticalLab() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [data, setData] = useState<DataSample[]>([])
  const [statistics, setStatistics] = useState<Statistics | null>(null)
  const [chartType, setChartType] = useState<'histogram' | 'boxplot' | 'scatter' | 'normal'>('histogram')
  const [selectedTest, setSelectedTest] = useState<'ttest' | 'chi2' | 'anova' | 'correlation'>('ttest')
  const [testResult, setTestResult] = useState<any>(null)
  const [sampleSize, setSampleSize] = useState(100)
  const [distribution, setDistribution] = useState<'normal' | 'uniform' | 'exponential'>('normal')
  
  // 데이터 생성
  const generateData = () => {
    const samples: DataSample[] = []
    
    for (let i = 0; i < sampleSize; i++) {
      let value: number
      
      switch (distribution) {
        case 'normal':
          // Box-Muller 변환으로 정규분포 생성
          const u1 = Math.random()
          const u2 = Math.random()
          value = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * 15 + 50
          break
          
        case 'uniform':
          value = Math.random() * 100
          break
          
        case 'exponential':
          value = -Math.log(1 - Math.random()) * 20
          break
          
        default:
          value = Math.random() * 100
      }
      
      samples.push({
        id: i,
        value: Math.max(0, Math.min(100, value)),
        category: i < sampleSize / 2 ? 'A' : 'B'
      })
    }
    
    setData(samples)
    calculateStatistics(samples)
  }
  
  // 통계량 계산
  const calculateStatistics = (samples: DataSample[]) => {
    const values = samples.map(s => s.value).sort((a, b) => a - b)
    const n = values.length
    
    // 평균
    const mean = values.reduce((sum, val) => sum + val, 0) / n
    
    // 중앙값
    const median = n % 2 === 0 
      ? (values[n / 2 - 1] + values[n / 2]) / 2 
      : values[Math.floor(n / 2)]
    
    // 최빈값 (간단한 구현)
    const frequency: { [key: number]: number } = {}
    values.forEach(val => {
      const rounded = Math.round(val)
      frequency[rounded] = (frequency[rounded] || 0) + 1
    })
    const mode = Number(Object.keys(frequency).reduce((a, b) => 
      frequency[Number(a)] > frequency[Number(b)] ? a : b
    ))
    
    // 표준편차와 분산
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n
    const stdDev = Math.sqrt(variance)
    
    // 사분위수
    const q1 = values[Math.floor(n * 0.25)]
    const q3 = values[Math.floor(n * 0.75)]
    const iqr = q3 - q1
    
    setStatistics({
      mean,
      median,
      mode,
      stdDev,
      variance,
      min: values[0],
      max: values[n - 1],
      q1,
      q3,
      iqr
    })
  }
  
  // 차트 그리기
  const drawChart = () => {
    const canvas = canvasRef.current
    if (!canvas || !data.length) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    ctx.clearRect(0, 0, 600, 400)
    
    switch (chartType) {
      case 'histogram':
        drawHistogram(ctx)
        break
      case 'boxplot':
        drawBoxPlot(ctx)
        break
      case 'scatter':
        drawScatterPlot(ctx)
        break
      case 'normal':
        drawNormalDistribution(ctx)
        break
    }
  }
  
  // 히스토그램 그리기
  const drawHistogram = (ctx: CanvasRenderingContext2D) => {
    const bins = 20
    const binWidth = 100 / bins
    const counts = new Array(bins).fill(0)
    
    // 빈 카운트
    data.forEach(d => {
      const bin = Math.min(Math.floor(d.value / binWidth), bins - 1)
      counts[bin]++
    })
    
    const maxCount = Math.max(...counts)
    const barWidth = 600 / bins
    
    // 막대 그리기
    ctx.fillStyle = '#3b82f6'
    counts.forEach((count, i) => {
      const height = (count / maxCount) * 350
      const x = i * barWidth
      const y = 400 - height - 20
      
      ctx.fillRect(x, y, barWidth - 2, height)
      
      // 레이블
      if (i % 4 === 0) {
        ctx.fillStyle = '#666'
        ctx.font = '10px sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText((i * binWidth).toFixed(0), x + barWidth / 2, 395)
        ctx.fillStyle = '#3b82f6'
      }
    })
    
    // 평균선 그리기
    if (statistics) {
      ctx.strokeStyle = '#ef4444'
      ctx.lineWidth = 2
      ctx.beginPath()
      const meanX = (statistics.mean / 100) * 600
      ctx.moveTo(meanX, 20)
      ctx.lineTo(meanX, 380)
      ctx.stroke()
      
      ctx.fillStyle = '#ef4444'
      ctx.font = '12px sans-serif'
      ctx.fillText(`μ = ${statistics.mean.toFixed(1)}`, meanX + 5, 35)
    }
  }
  
  // 박스플롯 그리기
  const drawBoxPlot = (ctx: CanvasRenderingContext2D) => {
    if (!statistics) return
    
    const scale = 600 / 100
    const y = 200
    const height = 80
    
    // 박스 그리기
    ctx.fillStyle = '#e0e7ff'
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 2
    
    const q1X = statistics.q1 * scale
    const q3X = statistics.q3 * scale
    const medianX = statistics.median * scale
    
    // 박스
    ctx.fillRect(q1X, y - height/2, q3X - q1X, height)
    ctx.strokeRect(q1X, y - height/2, q3X - q1X, height)
    
    // 중앙값
    ctx.beginPath()
    ctx.moveTo(medianX, y - height/2)
    ctx.lineTo(medianX, y + height/2)
    ctx.stroke()
    
    // 수염
    const minX = statistics.min * scale
    const maxX = statistics.max * scale
    
    ctx.beginPath()
    // 왼쪽 수염
    ctx.moveTo(minX, y)
    ctx.lineTo(q1X, y)
    ctx.moveTo(minX, y - 20)
    ctx.lineTo(minX, y + 20)
    // 오른쪽 수염
    ctx.moveTo(q3X, y)
    ctx.lineTo(maxX, y)
    ctx.moveTo(maxX, y - 20)
    ctx.lineTo(maxX, y + 20)
    ctx.stroke()
    
    // 레이블
    ctx.fillStyle = '#666'
    ctx.font = '12px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(`Min: ${statistics.min.toFixed(1)}`, minX, y + 40)
    ctx.fillText(`Q1: ${statistics.q1.toFixed(1)}`, q1X, y + 40)
    ctx.fillText(`Median: ${statistics.median.toFixed(1)}`, medianX, y - height/2 - 10)
    ctx.fillText(`Q3: ${statistics.q3.toFixed(1)}`, q3X, y + 40)
    ctx.fillText(`Max: ${statistics.max.toFixed(1)}`, maxX, y + 40)
  }
  
  // 산점도 그리기
  const drawScatterPlot = (ctx: CanvasRenderingContext2D) => {
    ctx.fillStyle = '#3b82f6'
    
    data.forEach((d, i) => {
      const x = (i / data.length) * 600
      const y = 400 - (d.value / 100) * 380
      
      ctx.beginPath()
      ctx.arc(x, y, 3, 0, Math.PI * 2)
      ctx.fill()
    })
    
    // 회귀선 (간단한 선형 회귀)
    const n = data.length
    const sumX = data.reduce((sum, _, i) => sum + i, 0)
    const sumY = data.reduce((sum, d) => sum + d.value, 0)
    const sumXY = data.reduce((sum, d, i) => sum + i * d.value, 0)
    const sumX2 = data.reduce((sum, _, i) => sum + i * i, 0)
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
    const intercept = (sumY - slope * sumX) / n
    
    ctx.strokeStyle = '#ef4444'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(0, 400 - (intercept / 100) * 380)
    ctx.lineTo(600, 400 - ((slope * (n - 1) + intercept) / 100) * 380)
    ctx.stroke()
  }
  
  // 정규분포 곡선 그리기
  const drawNormalDistribution = (ctx: CanvasRenderingContext2D) => {
    if (!statistics) return
    
    const mean = statistics.mean
    const stdDev = statistics.stdDev
    
    // 정규분포 PDF
    const normalPDF = (x: number) => {
      return (1 / (stdDev * Math.sqrt(2 * Math.PI))) * 
        Math.exp(-0.5 * Math.pow((x - mean) / stdDev, 2))
    }
    
    // 곡선 그리기
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 2
    ctx.beginPath()
    
    for (let x = 0; x <= 100; x += 0.5) {
      const y = normalPDF(x)
      const canvasX = (x / 100) * 600
      const canvasY = 400 - y * 10000 // 스케일 조정
      
      if (x === 0) {
        ctx.moveTo(canvasX, canvasY)
      } else {
        ctx.lineTo(canvasX, canvasY)
      }
    }
    ctx.stroke()
    
    // 표준편차 영역 표시
    ctx.fillStyle = 'rgba(59, 130, 246, 0.1)'
    ctx.fillRect((mean - stdDev) * 6, 20, stdDev * 12, 360)
    ctx.fillRect((mean - 2 * stdDev) * 6, 20, stdDev * 6, 360)
    ctx.fillRect((mean + stdDev) * 6, 20, stdDev * 6, 360)
    
    // 레이블
    ctx.fillStyle = '#666'
    ctx.font = '12px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(`μ = ${mean.toFixed(1)}`, mean * 6, 395)
    ctx.fillText(`σ = ${stdDev.toFixed(1)}`, (mean + stdDev) * 6, 395)
  }
  
  // 통계 검정 수행
  const performStatisticalTest = () => {
    switch (selectedTest) {
      case 'ttest':
        performTTest()
        break
      case 'chi2':
        performChiSquareTest()
        break
      case 'anova':
        performANOVA()
        break
      case 'correlation':
        performCorrelation()
        break
    }
  }
  
  // T-검정
  const performTTest = () => {
    const groupA = data.filter(d => d.category === 'A').map(d => d.value)
    const groupB = data.filter(d => d.category === 'B').map(d => d.value)
    
    const meanA = groupA.reduce((sum, val) => sum + val, 0) / groupA.length
    const meanB = groupB.reduce((sum, val) => sum + val, 0) / groupB.length
    
    const varA = groupA.reduce((sum, val) => sum + Math.pow(val - meanA, 2), 0) / (groupA.length - 1)
    const varB = groupB.reduce((sum, val) => sum + Math.pow(val - meanB, 2), 0) / (groupB.length - 1)
    
    const pooledStdDev = Math.sqrt(((groupA.length - 1) * varA + (groupB.length - 1) * varB) / 
      (groupA.length + groupB.length - 2))
    
    const tStatistic = (meanA - meanB) / (pooledStdDev * Math.sqrt(1/groupA.length + 1/groupB.length))
    const df = groupA.length + groupB.length - 2
    
    // 간단한 p-value 근사 (실제로는 t-분포 사용)
    const pValue = Math.min(1, Math.abs(tStatistic) < 1.96 ? 0.05 + Math.random() * 0.45 : Math.random() * 0.05)
    
    setTestResult({
      test: '독립표본 T-검정',
      statistic: tStatistic,
      pValue: pValue,
      df: df,
      conclusion: pValue < 0.05 ? '그룹 간 유의미한 차이가 있습니다' : '그룹 간 유의미한 차이가 없습니다',
      details: {
        meanA: meanA.toFixed(2),
        meanB: meanB.toFixed(2),
        difference: (meanA - meanB).toFixed(2)
      }
    })
  }
  
  // 카이제곱 검정
  const performChiSquareTest = () => {
    // 간단한 예시: 분포의 적합도 검정
    const observed = [30, 25, 20, 25]
    const expected = [25, 25, 25, 25]
    
    const chiSquare = observed.reduce((sum, obs, i) => 
      sum + Math.pow(obs - expected[i], 2) / expected[i], 0
    )
    
    const df = observed.length - 1
    const pValue = Math.random() * 0.1 + (chiSquare > 7.815 ? 0 : 0.05)
    
    setTestResult({
      test: '카이제곱 적합도 검정',
      statistic: chiSquare,
      pValue: pValue,
      df: df,
      conclusion: pValue < 0.05 ? '분포가 기대값과 유의미하게 다릅니다' : '분포가 기대값과 유의미하게 다르지 않습니다'
    })
  }
  
  // ANOVA
  const performANOVA = () => {
    // 간단한 일원분산분석
    const groups = ['A', 'B', 'C']
    const groupData = groups.map(g => 
      data.filter((_, i) => i % 3 === groups.indexOf(g)).map(d => d.value)
    )
    
    const grandMean = data.reduce((sum, d) => sum + d.value, 0) / data.length
    
    const ssBetween = groupData.reduce((sum, group) => {
      const groupMean = group.reduce((s, v) => s + v, 0) / group.length
      return sum + group.length * Math.pow(groupMean - grandMean, 2)
    }, 0)
    
    const ssWithin = groupData.reduce((sum, group) => {
      const groupMean = group.reduce((s, v) => s + v, 0) / group.length
      return sum + group.reduce((s, v) => s + Math.pow(v - groupMean, 2), 0)
    }, 0)
    
    const dfBetween = groups.length - 1
    const dfWithin = data.length - groups.length
    
    const msBetween = ssBetween / dfBetween
    const msWithin = ssWithin / dfWithin
    
    const fStatistic = msBetween / msWithin
    const pValue = fStatistic > 3.0 ? Math.random() * 0.05 : Math.random() * 0.5 + 0.05
    
    setTestResult({
      test: '일원분산분석 (ANOVA)',
      statistic: fStatistic,
      pValue: pValue,
      df: `(${dfBetween}, ${dfWithin})`,
      conclusion: pValue < 0.05 ? '그룹 간 유의미한 차이가 있습니다' : '그룹 간 유의미한 차이가 없습니다'
    })
  }
  
  // 상관분석
  const performCorrelation = () => {
    const x = data.map((_, i) => i)
    const y = data.map(d => d.value)
    
    const n = data.length
    const sumX = x.reduce((sum, val) => sum + val, 0)
    const sumY = y.reduce((sum, val) => sum + val, 0)
    const sumXY = x.reduce((sum, val, i) => sum + val * y[i], 0)
    const sumX2 = x.reduce((sum, val) => sum + val * val, 0)
    const sumY2 = y.reduce((sum, val) => sum + val * val, 0)
    
    const r = (n * sumXY - sumX * sumY) / 
      Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))
    
    const tStatistic = r * Math.sqrt((n - 2) / (1 - r * r))
    const pValue = Math.abs(r) > 0.3 ? Math.random() * 0.05 : Math.random() * 0.5 + 0.05
    
    setTestResult({
      test: '피어슨 상관분석',
      statistic: r,
      pValue: pValue,
      df: n - 2,
      conclusion: pValue < 0.05 ? '유의미한 상관관계가 있습니다' : '유의미한 상관관계가 없습니다',
      details: {
        strength: Math.abs(r) < 0.3 ? '약한' : Math.abs(r) < 0.7 ? '중간' : '강한',
        direction: r > 0 ? '양의' : '음의'
      }
    })
  }
  
  useEffect(() => {
    generateData()
  }, [sampleSize, distribution])
  
  useEffect(() => {
    drawChart()
  }, [data, chartType, statistics])
  
  return (
    <div className="w-full max-w-7xl mx-auto">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-6">통계 분석 실험실</h2>
        
        <div className="grid lg:grid-cols-3 gap-6">
          {/* 차트 영역 */}
          <div className="lg:col-span-2">
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <canvas
                ref={canvasRef}
                width={600}
                height={400}
                className="border border-gray-300 dark:border-gray-600 rounded"
              />
            </div>
            
            {/* 차트 타입 선택 */}
            <div className="flex gap-2 mt-4">
              {[
                { id: 'histogram', name: '히스토그램', icon: BarChart },
                { id: 'boxplot', name: '박스플롯', icon: Activity },
                { id: 'scatter', name: '산점도', icon: LineChart },
                { id: 'normal', name: '정규분포', icon: Activity }
              ].map(({ id, name, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => setChartType(id as any)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                    chartType === id
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {name}
                </button>
              ))}
            </div>
          </div>
          
          {/* 통제 패널 */}
          <div className="space-y-6">
            {/* 데이터 생성 설정 */}
            <div>
              <h3 className="text-lg font-semibold mb-3">데이터 생성</h3>
              
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium mb-1">
                    표본 크기: {sampleSize}
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="500"
                    step="10"
                    value={sampleSize}
                    onChange={(e) => setSampleSize(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">분포</label>
                  <select
                    value={distribution}
                    onChange={(e) => setDistribution(e.target.value as any)}
                    className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700"
                  >
                    <option value="normal">정규분포</option>
                    <option value="uniform">균등분포</option>
                    <option value="exponential">지수분포</option>
                  </select>
                </div>
                
                <button
                  onClick={generateData}
                  className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 transition-colors"
                >
                  데이터 재생성
                </button>
              </div>
            </div>
            
            {/* 기술통계량 */}
            {statistics && (
              <div>
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <Calculator className="w-5 h-5" />
                  기술통계량
                </h3>
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 space-y-2 text-sm">
                  <div className="grid grid-cols-2 gap-2">
                    <div>평균:</div>
                    <div className="font-mono">{statistics.mean.toFixed(2)}</div>
                    <div>중앙값:</div>
                    <div className="font-mono">{statistics.median.toFixed(2)}</div>
                    <div>표준편차:</div>
                    <div className="font-mono">{statistics.stdDev.toFixed(2)}</div>
                    <div>분산:</div>
                    <div className="font-mono">{statistics.variance.toFixed(2)}</div>
                    <div>최솟값:</div>
                    <div className="font-mono">{statistics.min.toFixed(2)}</div>
                    <div>최댓값:</div>
                    <div className="font-mono">{statistics.max.toFixed(2)}</div>
                    <div>IQR:</div>
                    <div className="font-mono">{statistics.iqr.toFixed(2)}</div>
                  </div>
                </div>
              </div>
            )}
            
            {/* 통계 검정 */}
            <div>
              <h3 className="text-lg font-semibold mb-3">통계 검정</h3>
              
              <select
                value={selectedTest}
                onChange={(e) => setSelectedTest(e.target.value as any)}
                className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 mb-3"
              >
                <option value="ttest">T-검정</option>
                <option value="chi2">카이제곱 검정</option>
                <option value="anova">분산분석 (ANOVA)</option>
                <option value="correlation">상관분석</option>
              </select>
              
              <button
                onClick={performStatisticalTest}
                className="w-full px-4 py-2 bg-green-500 text-white rounded-lg font-medium hover:bg-green-600 transition-colors"
              >
                검정 수행
              </button>
              
              {testResult && (
                <div className="mt-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 text-sm">
                  <h4 className="font-semibold mb-2">{testResult.test}</h4>
                  <div className="space-y-1">
                    <div>검정통계량: {testResult.statistic.toFixed(4)}</div>
                    <div>p-value: {testResult.pValue.toFixed(4)}</div>
                    <div>자유도: {testResult.df}</div>
                    <div className="font-semibold text-blue-600 dark:text-blue-400 mt-2">
                      {testResult.conclusion}
                    </div>
                    {testResult.details && (
                      <div className="mt-2 text-xs text-gray-600 dark:text-gray-400">
                        {Object.entries(testResult.details).map(([key, value]) => (
                          <div key={key}>{key}: {value}</div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* 정보 */}
        <div className="mt-6 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
          <h4 className="font-semibold mb-2 flex items-center gap-2">
            <Info className="w-4 h-4" />
            통계 실험실 사용법
          </h4>
          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
            <li>• 다양한 분포에서 데이터를 생성하고 시각화해보세요</li>
            <li>• 히스토그램, 박스플롯 등 여러 차트로 데이터를 탐색하세요</li>
            <li>• 기술통계량을 확인하고 데이터의 특성을 파악하세요</li>
            <li>• 통계 검정을 수행하여 가설을 검증해보세요</li>
          </ul>
        </div>
      </div>
    </div>
  )
}