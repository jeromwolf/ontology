'use client'

import { useState, useEffect, useRef } from 'react'
import { TrendingUp, Calendar, Activity, Settings, Play, Pause, RotateCcw, Download } from 'lucide-react'

interface TimePoint {
  time: number
  value: number
  forecast?: number
  lowerBound?: number
  upperBound?: number
}

interface ModelConfig {
  model: 'arima' | 'exponential' | 'prophet' | 'lstm'
  seasonality: boolean
  trend: 'none' | 'linear' | 'polynomial'
  forecastHorizon: number
  confidenceInterval: number
}

export default function TimeSeriesForecaster() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [data, setData] = useState<TimePoint[]>([])
  const [isForecasting, setIsForecasting] = useState(false)
  const [dataType, setDataType] = useState<'sales' | 'stock' | 'weather' | 'energy'>('sales')
  const [noise, setNoise] = useState(0.1)
  const [seasonalStrength, setSeasonalStrength] = useState(0.5)
  
  const [config, setConfig] = useState<ModelConfig>({
    model: 'arima',
    seasonality: true,
    trend: 'linear',
    forecastHorizon: 30,
    confidenceInterval: 95
  })
  
  const [metrics, setMetrics] = useState({
    mae: 0,
    rmse: 0,
    mape: 0
  })
  
  // 시계열 데이터 생성
  const generateTimeSeries = () => {
    const points: TimePoint[] = []
    const numPoints = 100
    
    for (let i = 0; i < numPoints; i++) {
      let value = 0
      const t = i / 10
      
      switch (dataType) {
        case 'sales':
          // 판매 데이터: 상승 추세 + 주간 계절성
          value = 50 + t * 2 + 
            (config.seasonality ? Math.sin(t * Math.PI * 2 / 7) * 20 * seasonalStrength : 0) +
            (Math.random() - 0.5) * 20 * noise
          break
          
        case 'stock':
          // 주식 데이터: 랜덤 워크 + 변동성
          if (i === 0) {
            value = 100
          } else {
            const lastValue = points[i - 1].value
            const change = (Math.random() - 0.5) * 5
            value = lastValue + change + 
              (config.trend === 'linear' ? 0.1 : 0) +
              (Math.random() - 0.5) * 10 * noise
          }
          break
          
        case 'weather':
          // 날씨 데이터: 연간 계절성
          value = 20 + Math.sin(t * Math.PI * 2 / 365) * 15 +
            (config.seasonality ? Math.sin(t * Math.PI * 2 / 7) * 5 * seasonalStrength : 0) +
            (Math.random() - 0.5) * 10 * noise
          break
          
        case 'energy':
          // 에너지 소비: 일일 패턴 + 주간 패턴
          const hourOfDay = (i % 24) / 24
          const dayOfWeek = Math.floor(i / 24) % 7
          
          value = 100 + 
            Math.sin(hourOfDay * Math.PI * 2 - Math.PI / 2) * 30 +
            (dayOfWeek === 0 || dayOfWeek === 6 ? -20 : 0) +
            (Math.random() - 0.5) * 20 * noise
          break
      }
      
      points.push({ time: i, value: Math.max(0, value) })
    }
    
    setData(points)
    setMetrics({ mae: 0, rmse: 0, mape: 0 })
  }
  
  // 예측 수행
  const performForecast = () => {
    const historicalData = data.filter(d => !d.forecast)
    const lastTime = historicalData[historicalData.length - 1].time
    const newData = [...historicalData]
    
    // 간단한 예측 모델들
    switch (config.model) {
      case 'arima':
        // 간단한 ARIMA(1,1,1) 모델
        const arimaForecast = forecastARIMA(historicalData)
        newData.push(...arimaForecast)
        break
        
      case 'exponential':
        // 지수평활법
        const expForecast = forecastExponentialSmoothing(historicalData)
        newData.push(...expForecast)
        break
        
      case 'prophet':
        // Prophet 스타일 (추세 + 계절성)
        const prophetForecast = forecastProphet(historicalData)
        newData.push(...prophetForecast)
        break
        
      case 'lstm':
        // LSTM 시뮬레이션
        const lstmForecast = forecastLSTM(historicalData)
        newData.push(...lstmForecast)
        break
    }
    
    setData(newData)
    calculateMetrics(newData)
  }
  
  // ARIMA 예측
  const forecastARIMA = (historical: TimePoint[]): TimePoint[] => {
    const forecast: TimePoint[] = []
    const n = historical.length
    
    // 차분 계산
    const diff = historical.slice(1).map((d, i) => d.value - historical[i].value)
    const avgDiff = diff.reduce((sum, d) => sum + d, 0) / diff.length
    
    // MA 계산
    const ma = historical.slice(-5).reduce((sum, d) => sum + d.value, 0) / 5
    
    let lastValue = historical[n - 1].value
    
    for (let i = 0; i < config.forecastHorizon; i++) {
      const trend = config.trend === 'linear' ? avgDiff : 0
      const seasonal = config.seasonality ? 
        Math.sin((historical[n - 1].time + i + 1) * Math.PI * 2 / 7) * 10 : 0
      
      const forecastValue = lastValue + trend + seasonal + (Math.random() - 0.5) * 5
      const std = 10 + i * 0.5 // 불확실성 증가
      
      forecast.push({
        time: historical[n - 1].time + i + 1,
        value: historical[n - 1].value, // 원본값은 유지
        forecast: forecastValue,
        lowerBound: forecastValue - std * 1.96,
        upperBound: forecastValue + std * 1.96
      })
      
      lastValue = forecastValue
    }
    
    return forecast
  }
  
  // 지수평활법
  const forecastExponentialSmoothing = (historical: TimePoint[]): TimePoint[] => {
    const forecast: TimePoint[] = []
    const alpha = 0.3 // 평활 파라미터
    const beta = 0.1  // 추세 파라미터
    
    let level = historical[historical.length - 1].value
    let trend = 0
    
    // 추세 추정
    if (config.trend !== 'none') {
      const recentTrend = historical.slice(-10).map((d, i, arr) => 
        i > 0 ? d.value - arr[i - 1].value : 0
      ).slice(1)
      trend = recentTrend.reduce((sum, t) => sum + t, 0) / recentTrend.length
    }
    
    for (let i = 0; i < config.forecastHorizon; i++) {
      const seasonal = config.seasonality ? 
        Math.sin((historical[historical.length - 1].time + i + 1) * Math.PI * 2 / 7) * 10 : 0
      
      const forecastValue = level + trend * (i + 1) + seasonal
      const std = 8 + i * 0.3
      
      forecast.push({
        time: historical[historical.length - 1].time + i + 1,
        value: historical[historical.length - 1].value,
        forecast: forecastValue,
        lowerBound: forecastValue - std * 1.96,
        upperBound: forecastValue + std * 1.96
      })
    }
    
    return forecast
  }
  
  // Prophet 스타일 예측
  const forecastProphet = (historical: TimePoint[]): TimePoint[] => {
    const forecast: TimePoint[] = []
    const n = historical.length
    
    // 선형 추세 계산
    const times = historical.map(d => d.time)
    const values = historical.map(d => d.value)
    
    const avgTime = times.reduce((sum, t) => sum + t, 0) / n
    const avgValue = values.reduce((sum, v) => sum + v, 0) / n
    
    let slope = 0
    let numerator = 0
    let denominator = 0
    
    for (let i = 0; i < n; i++) {
      numerator += (times[i] - avgTime) * (values[i] - avgValue)
      denominator += Math.pow(times[i] - avgTime, 2)
    }
    
    if (denominator !== 0) {
      slope = numerator / denominator
    }
    
    const intercept = avgValue - slope * avgTime
    
    // 계절성 컴포넌트
    const seasonalPattern = Array(7).fill(0)
    if (config.seasonality) {
      for (let i = 0; i < n; i++) {
        const dayOfWeek = i % 7
        const detrended = values[i] - (slope * times[i] + intercept)
        seasonalPattern[dayOfWeek] += detrended
      }
      
      for (let i = 0; i < 7; i++) {
        seasonalPattern[i] /= Math.floor(n / 7)
      }
    }
    
    // 예측 생성
    for (let i = 0; i < config.forecastHorizon; i++) {
      const futureTime = historical[n - 1].time + i + 1
      const trend = slope * futureTime + intercept
      const seasonal = config.seasonality ? seasonalPattern[futureTime % 7] : 0
      
      const forecastValue = trend + seasonal
      const std = 12 + i * 0.4
      
      forecast.push({
        time: futureTime,
        value: historical[n - 1].value,
        forecast: forecastValue,
        lowerBound: forecastValue - std * 1.96,
        upperBound: forecastValue + std * 1.96
      })
    }
    
    return forecast
  }
  
  // LSTM 시뮬레이션
  const forecastLSTM = (historical: TimePoint[]): TimePoint[] => {
    const forecast: TimePoint[] = []
    const windowSize = 10
    
    // 최근 패턴 학습
    const recentValues = historical.slice(-windowSize).map(d => d.value)
    const patterns: number[] = []
    
    for (let i = 1; i < recentValues.length; i++) {
      patterns.push(recentValues[i] - recentValues[i - 1])
    }
    
    const avgPattern = patterns.reduce((sum, p) => sum + p, 0) / patterns.length
    const patternStd = Math.sqrt(
      patterns.reduce((sum, p) => sum + Math.pow(p - avgPattern, 2), 0) / patterns.length
    )
    
    let lastValue = historical[historical.length - 1].value
    
    for (let i = 0; i < config.forecastHorizon; i++) {
      // LSTM은 복잡한 패턴을 학습하므로 약간의 비선형성 추가
      const momentum = i > 0 ? 
        (forecast[i - 1].forecast! - lastValue) * 0.5 : 0
      
      const change = avgPattern + momentum + 
        (Math.random() - 0.5) * patternStd * 2
      
      const forecastValue = lastValue + change
      const std = 10 + i * 0.6 // LSTM은 불확실성이 더 빠르게 증가
      
      forecast.push({
        time: historical[historical.length - 1].time + i + 1,
        value: historical[historical.length - 1].value,
        forecast: forecastValue,
        lowerBound: forecastValue - std * 1.96,
        upperBound: forecastValue + std * 1.96
      })
      
      lastValue = forecastValue
    }
    
    return forecast
  }
  
  // 평가 지표 계산
  const calculateMetrics = (fullData: TimePoint[]) => {
    // 실제로는 테스트 데이터로 평가해야 하지만, 여기서는 시뮬레이션
    const forecastData = fullData.filter(d => d.forecast !== undefined)
    
    if (forecastData.length > 0) {
      // 가상의 실제값 생성 (노이즈 추가)
      const actualValues = forecastData.map(d => 
        d.forecast! + (Math.random() - 0.5) * 10
      )
      
      let sumAE = 0
      let sumSE = 0
      let sumAPE = 0
      
      forecastData.forEach((d, i) => {
        const error = Math.abs(actualValues[i] - d.forecast!)
        sumAE += error
        sumSE += error * error
        sumAPE += error / Math.abs(actualValues[i])
      })
      
      const n = forecastData.length
      
      setMetrics({
        mae: sumAE / n,
        rmse: Math.sqrt(sumSE / n),
        mape: (sumAPE / n) * 100
      })
    }
  }
  
  // 차트 그리기
  const drawChart = () => {
    const canvas = canvasRef.current
    if (!canvas || data.length === 0) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const width = 800
    const height = 400
    const padding = 40
    
    ctx.clearRect(0, 0, width, height)
    
    // 데이터 범위 계산
    const allValues = data.flatMap(d => [
      d.value, 
      d.forecast, 
      d.lowerBound, 
      d.upperBound
    ].filter(v => v !== undefined)) as number[]
    
    const minValue = Math.min(...allValues) * 0.9
    const maxValue = Math.max(...allValues) * 1.1
    const timeRange = data[data.length - 1].time - data[0].time
    
    // 스케일 함수
    const scaleX = (time: number) => 
      padding + ((time - data[0].time) / timeRange) * (width - 2 * padding)
    
    const scaleY = (value: number) => 
      height - padding - ((value - minValue) / (maxValue - minValue)) * (height - 2 * padding)
    
    // 그리드 그리기
    ctx.strokeStyle = '#e5e7eb'
    ctx.lineWidth = 1
    
    // Y축 그리드
    for (let i = 0; i <= 5; i++) {
      const y = padding + i * (height - 2 * padding) / 5
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(width - padding, y)
      ctx.stroke()
      
      // Y축 레이블
      const value = maxValue - (i / 5) * (maxValue - minValue)
      ctx.fillStyle = '#666'
      ctx.font = '10px sans-serif'
      ctx.textAlign = 'right'
      ctx.fillText(value.toFixed(1), padding - 5, y + 3)
    }
    
    // 예측 구간 배경
    const forecastStart = data.findIndex(d => d.forecast !== undefined)
    if (forecastStart !== -1) {
      ctx.fillStyle = 'rgba(59, 130, 246, 0.05)'
      ctx.fillRect(
        scaleX(data[forecastStart].time),
        padding,
        width - padding - scaleX(data[forecastStart].time),
        height - 2 * padding
      )
    }
    
    // 신뢰 구간 그리기
    const confidenceData = data.filter(d => d.lowerBound !== undefined && d.upperBound !== undefined)
    if (confidenceData.length > 0) {
      ctx.fillStyle = 'rgba(59, 130, 246, 0.2)'
      ctx.beginPath()
      
      // 상한선
      confidenceData.forEach((d, i) => {
        const x = scaleX(d.time)
        const y = scaleY(d.upperBound!)
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      
      // 하한선 (역순)
      confidenceData.reverse().forEach(d => {
        ctx.lineTo(scaleX(d.time), scaleY(d.lowerBound!))
      })
      
      ctx.closePath()
      ctx.fill()
    }
    
    // 실제 데이터 그리기
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 2
    ctx.beginPath()
    
    const historicalData = data.filter(d => d.forecast === undefined)
    historicalData.forEach((d, i) => {
      const x = scaleX(d.time)
      const y = scaleY(d.value)
      
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()
    
    // 예측 데이터 그리기
    const forecastData = data.filter(d => d.forecast !== undefined)
    if (forecastData.length > 0) {
      ctx.strokeStyle = '#ef4444'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      
      // 연결점
      if (historicalData.length > 0) {
        const lastHistorical = historicalData[historicalData.length - 1]
        ctx.moveTo(scaleX(lastHistorical.time), scaleY(lastHistorical.value))
        ctx.lineTo(scaleX(forecastData[0].time), scaleY(forecastData[0].forecast!))
      }
      
      forecastData.forEach((d, i) => {
        const x = scaleX(d.time)
        const y = scaleY(d.forecast!)
        
        if (i === 0 && historicalData.length === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()
      ctx.setLineDash([])
    }
    
    // 범례
    const legendY = 20
    
    ctx.fillStyle = '#3b82f6'
    ctx.fillRect(width - 150, legendY, 20, 3)
    ctx.fillStyle = '#333'
    ctx.font = '12px sans-serif'
    ctx.textAlign = 'left'
    ctx.fillText('실제값', width - 125, legendY + 5)
    
    if (forecastData.length > 0) {
      ctx.strokeStyle = '#ef4444'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      ctx.moveTo(width - 150, legendY + 20)
      ctx.lineTo(width - 130, legendY + 20)
      ctx.stroke()
      ctx.setLineDash([])
      
      ctx.fillStyle = '#333'
      ctx.fillText('예측값', width - 125, legendY + 25)
    }
  }
  
  useEffect(() => {
    generateTimeSeries()
  }, [dataType, noise, seasonalStrength, config.seasonality])
  
  useEffect(() => {
    drawChart()
  }, [data])
  
  return (
    <div className="w-full max-w-7xl mx-auto">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-6">시계열 예측기</h2>
        
        <div className="grid lg:grid-cols-3 gap-6">
          {/* 차트 영역 */}
          <div className="lg:col-span-2">
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <canvas
                ref={canvasRef}
                width={800}
                height={400}
                className="w-full border border-gray-300 dark:border-gray-600 rounded"
              />
            </div>
            
            {/* 컨트롤 버튼 */}
            <div className="flex gap-2 mt-4">
              <button
                onClick={() => {
                  setIsForecasting(true)
                  performForecast()
                  setTimeout(() => setIsForecasting(false), 1000)
                }}
                disabled={isForecasting}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                  isForecasting
                    ? 'bg-gray-400 text-white cursor-not-allowed'
                    : 'bg-blue-500 text-white hover:bg-blue-600'
                }`}
              >
                <TrendingUp className="w-4 h-4" />
                예측 실행
              </button>
              
              <button
                onClick={generateTimeSeries}
                className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded-lg font-medium hover:bg-gray-600 transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                데이터 재생성
              </button>
              
              <button
                onClick={() => {
                  const csvData = data.map(d => 
                    `${d.time},${d.value}${d.forecast ? `,${d.forecast}` : ''}`
                  ).join('\n')
                  
                  const blob = new Blob([`time,value,forecast\n${csvData}`], { type: 'text/csv' })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `timeseries-${Date.now()}.csv`
                  a.click()
                  URL.revokeObjectURL(url)
                }}
                className="flex items-center gap-2 px-4 py-2 bg-purple-500 text-white rounded-lg font-medium hover:bg-purple-600 transition-colors"
              >
                <Download className="w-4 h-4" />
                CSV 다운로드
              </button>
            </div>
            
            {/* 평가 지표 */}
            {metrics.mae > 0 && (
              <div className="mt-4 bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-semibold mb-2">예측 성능 지표</h4>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">MAE:</span>
                    <span className="ml-2 font-mono">{metrics.mae.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">RMSE:</span>
                    <span className="ml-2 font-mono">{metrics.rmse.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">MAPE:</span>
                    <span className="ml-2 font-mono">{metrics.mape.toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* 설정 패널 */}
          <div className="space-y-6">
            {/* 데이터 타입 */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Calendar className="w-5 h-5" />
                시계열 타입
              </h3>
              <select
                value={dataType}
                onChange={(e) => setDataType(e.target.value as any)}
                className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700"
              >
                <option value="sales">판매 데이터</option>
                <option value="stock">주식 가격</option>
                <option value="weather">날씨 온도</option>
                <option value="energy">에너지 소비</option>
              </select>
            </div>
            
            {/* 모델 설정 */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Activity className="w-5 h-5" />
                예측 모델
              </h3>
              <select
                value={config.model}
                onChange={(e) => setConfig({...config, model: e.target.value as any})}
                className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700"
              >
                <option value="arima">ARIMA</option>
                <option value="exponential">지수평활법</option>
                <option value="prophet">Prophet</option>
                <option value="lstm">LSTM (신경망)</option>
              </select>
              
              <div className="mt-3 space-y-3">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.seasonality}
                    onChange={(e) => setConfig({...config, seasonality: e.target.checked})}
                    className="rounded"
                  />
                  <span className="text-sm">계절성 포함</span>
                </label>
                
                <div>
                  <label className="block text-sm font-medium mb-1">
                    추세 타입
                  </label>
                  <select
                    value={config.trend}
                    onChange={(e) => setConfig({...config, trend: e.target.value as any})}
                    className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-sm"
                  >
                    <option value="none">없음</option>
                    <option value="linear">선형</option>
                    <option value="polynomial">다항식</option>
                  </select>
                </div>
              </div>
            </div>
            
            {/* 예측 설정 */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Settings className="w-5 h-5" />
                예측 설정
              </h3>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium mb-1">
                    예측 기간: {config.forecastHorizon}
                  </label>
                  <input
                    type="range"
                    min="5"
                    max="50"
                    step="5"
                    value={config.forecastHorizon}
                    onChange={(e) => setConfig({...config, forecastHorizon: parseInt(e.target.value)})}
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">
                    신뢰구간: {config.confidenceInterval}%
                  </label>
                  <input
                    type="range"
                    min="80"
                    max="99"
                    value={config.confidenceInterval}
                    onChange={(e) => setConfig({...config, confidenceInterval: parseInt(e.target.value)})}
                    className="w-full"
                  />
                </div>
              </div>
            </div>
            
            {/* 데이터 특성 */}
            <div>
              <h3 className="text-lg font-semibold mb-3">데이터 특성</h3>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium mb-1">
                    노이즈 수준: {(noise * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={noise}
                    onChange={(e) => setNoise(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
                
                {config.seasonality && (
                  <div>
                    <label className="block text-sm font-medium mb-1">
                      계절성 강도: {(seasonalStrength * 100).toFixed(0)}%
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={seasonalStrength}
                      onChange={(e) => setSeasonalStrength(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}