'use client'

import { useState, useEffect, useRef } from 'react'
import { 
  TrendingUp, Calendar, Activity, Settings, Play, Pause, 
  RotateCcw, Download, Brain, BarChart3, Info, ChevronRight,
  Clock, Zap, AlertCircle, Check, X
} from 'lucide-react'

interface TimePoint {
  date: Date
  value: number
  forecast?: number
  lowerBound?: number
  upperBound?: number
}

interface ModelResult {
  name: string
  mae: number
  rmse: number
  mape: number
  trainTime: number
  forecast: TimePoint[]
}

interface DatasetOption {
  name: string
  description: string
  frequency: 'daily' | 'weekly' | 'monthly'
  hasSeasonality: boolean
  hasTrend: boolean
  length: number
}

const DATASETS: DatasetOption[] = [
  {
    name: '전자상거래 매출',
    description: '일별 온라인 쇼핑몰 매출 데이터',
    frequency: 'daily',
    hasSeasonality: true,
    hasTrend: true,
    length: 365
  },
  {
    name: '주가 데이터',
    description: 'KOSPI 일별 종가',
    frequency: 'daily',
    hasSeasonality: false,
    hasTrend: true,
    length: 250
  },
  {
    name: '전력 수요',
    description: '시간별 전력 사용량',
    frequency: 'daily',
    hasSeasonality: true,
    hasTrend: false,
    length: 180
  },
  {
    name: '웹사이트 트래픽',
    description: '일별 방문자 수',
    frequency: 'daily',
    hasSeasonality: true,
    hasTrend: true,
    length: 90
  }
]

export default function TimeSeriesForecasterPyCaret() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [selectedDataset, setSelectedDataset] = useState<DatasetOption>(DATASETS[0])
  const [timeSeriesData, setTimeSeriesData] = useState<TimePoint[]>([])
  const [isTraining, setIsTraining] = useState(false)
  const [modelResults, setModelResults] = useState<ModelResult[]>([])
  const [selectedModel, setSelectedModel] = useState<ModelResult | null>(null)
  const [forecastHorizon, setForecastHorizon] = useState(30)
  const [showDecomposition, setShowDecomposition] = useState(false)
  const [currentStep, setCurrentStep] = useState<'data' | 'setup' | 'compare' | 'forecast'>('data')

  // 시계열 데이터 생성
  const generateTimeSeries = (dataset: DatasetOption) => {
    const data: TimePoint[] = []
    const startDate = new Date()
    startDate.setDate(startDate.getDate() - dataset.length)

    for (let i = 0; i < dataset.length; i++) {
      const date = new Date(startDate)
      date.setDate(date.getDate() + i)
      
      let value = 1000
      
      // 트렌드
      if (dataset.hasTrend) {
        value += i * 2
      }
      
      // 계절성
      if (dataset.hasSeasonality) {
        // 주간 패턴
        const dayOfWeek = date.getDay()
        value += Math.sin(dayOfWeek * Math.PI / 3.5) * 100
        
        // 월간 패턴
        const dayOfMonth = date.getDate()
        value += Math.sin(dayOfMonth * Math.PI / 15) * 50
      }
      
      // 노이즈
      value += (Math.random() - 0.5) * 100
      
      // 특별 이벤트 (랜덤)
      if (Math.random() > 0.95) {
        value *= 1.5
      }
      
      data.push({
        date,
        value: Math.max(0, Math.round(value))
      })
    }
    
    return data
  }

  // PyCaret 시계열 모델 학습 시뮬레이션
  const trainModels = async () => {
    setIsTraining(true)
    setModelResults([])
    setCurrentStep('compare')
    
    const models = [
      { name: 'ARIMA', complexity: 'medium' },
      { name: 'Prophet', complexity: 'high' },
      { name: 'Exponential Smoothing', complexity: 'low' },
      { name: 'LSTM', complexity: 'very-high' },
      { name: 'Random Forest', complexity: 'high' },
      { name: 'XGBoost', complexity: 'high' }
    ]
    
    for (const model of models) {
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // 모델별 성능 시뮬레이션
      const baseError = model.complexity === 'low' ? 15 : 
                       model.complexity === 'medium' ? 10 : 
                       model.complexity === 'high' ? 8 : 5
      
      const mae = baseError + Math.random() * 5
      const rmse = mae * 1.2 + Math.random() * 3
      const mape = mae / 10 + Math.random() * 2
      
      // 예측 생성
      const lastDate = timeSeriesData[timeSeriesData.length - 1].date
      const lastValue = timeSeriesData[timeSeriesData.length - 1].value
      const forecast: TimePoint[] = []
      
      for (let i = 1; i <= forecastHorizon; i++) {
        const forecastDate = new Date(lastDate)
        forecastDate.setDate(forecastDate.getDate() + i)
        
        // 모델별 예측 패턴
        let forecastValue = lastValue
        
        if (model.name === 'Prophet' && selectedDataset.hasSeasonality) {
          const dayOfWeek = forecastDate.getDay()
          forecastValue += Math.sin(dayOfWeek * Math.PI / 3.5) * 50
        }
        
        if (selectedDataset.hasTrend) {
          forecastValue += i * 2
        }
        
        // 불확실성 증가
        const uncertainty = 10 + i * 2
        forecastValue += (Math.random() - 0.5) * uncertainty
        
        forecast.push({
          date: forecastDate,
          value: 0,
          forecast: Math.max(0, Math.round(forecastValue)),
          lowerBound: Math.max(0, Math.round(forecastValue - uncertainty * 1.96)),
          upperBound: Math.round(forecastValue + uncertainty * 1.96)
        })
      }
      
      setModelResults(prev => [...prev, {
        name: model.name,
        mae: mae,
        rmse: rmse,
        mape: mape,
        trainTime: 0.5 + Math.random() * 3,
        forecast: forecast
      }])
    }
    
    setIsTraining(false)
  }

  // 시각화
  useEffect(() => {
    if (!canvasRef.current || timeSeriesData.length === 0) return
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')!
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width
    canvas.height = rect.height
    
    // Clear
    ctx.fillStyle = '#f3f4f6'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    
    const padding = 40
    const chartWidth = canvas.width - padding * 2
    const chartHeight = canvas.height - padding * 2
    
    // 데이터 범위 계산
    let allValues = timeSeriesData.map(d => d.value)
    if (selectedModel) {
      allValues = allValues.concat(
        selectedModel.forecast.map(d => d.forecast!),
        selectedModel.forecast.map(d => d.upperBound!)
      )
    }
    
    const minValue = Math.min(...allValues)
    const maxValue = Math.max(...allValues)
    const valueRange = maxValue - minValue
    
    // 축 그리기
    ctx.strokeStyle = '#6b7280'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(padding, padding)
    ctx.lineTo(padding, padding + chartHeight)
    ctx.lineTo(padding + chartWidth, padding + chartHeight)
    ctx.stroke()
    
    // 격자
    ctx.strokeStyle = '#e5e7eb'
    for (let i = 0; i <= 5; i++) {
      const y = padding + (chartHeight / 5) * i
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(padding + chartWidth, y)
      ctx.stroke()
      
      // Y축 레이블
      ctx.fillStyle = '#6b7280'
      ctx.font = '12px sans-serif'
      ctx.textAlign = 'right'
      const value = maxValue - (valueRange / 5) * i
      ctx.fillText(value.toFixed(0), padding - 10, y + 4)
    }
    
    // 실제 데이터 그리기
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 2
    ctx.beginPath()
    
    timeSeriesData.forEach((point, i) => {
      const x = padding + (chartWidth / (timeSeriesData.length + forecastHorizon - 1)) * i
      const y = padding + chartHeight - ((point.value - minValue) / valueRange) * chartHeight
      
      if (i === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })
    ctx.stroke()
    
    // 예측 그리기
    if (selectedModel) {
      // 예측 구간
      ctx.fillStyle = 'rgba(239, 68, 68, 0.1)'
      ctx.beginPath()
      
      selectedModel.forecast.forEach((point, i) => {
        const x = padding + (chartWidth / (timeSeriesData.length + forecastHorizon - 1)) * (timeSeriesData.length + i)
        const yUpper = padding + chartHeight - ((point.upperBound! - minValue) / valueRange) * chartHeight
        const yLower = padding + chartHeight - ((point.lowerBound! - minValue) / valueRange) * chartHeight
        
        if (i === 0) {
          ctx.moveTo(x, yUpper)
        } else {
          ctx.lineTo(x, yUpper)
        }
        
        if (i === selectedModel.forecast.length - 1) {
          for (let j = selectedModel.forecast.length - 1; j >= 0; j--) {
            const x2 = padding + (chartWidth / (timeSeriesData.length + forecastHorizon - 1)) * (timeSeriesData.length + j)
            const y2Lower = padding + chartHeight - ((selectedModel.forecast[j].lowerBound! - minValue) / valueRange) * chartHeight
            ctx.lineTo(x2, y2Lower)
          }
        }
      })
      ctx.closePath()
      ctx.fill()
      
      // 예측선
      ctx.strokeStyle = '#ef4444'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      
      // 연결점
      const lastX = padding + (chartWidth / (timeSeriesData.length + forecastHorizon - 1)) * (timeSeriesData.length - 1)
      const lastY = padding + chartHeight - ((timeSeriesData[timeSeriesData.length - 1].value - minValue) / valueRange) * chartHeight
      ctx.moveTo(lastX, lastY)
      
      selectedModel.forecast.forEach((point, i) => {
        const x = padding + (chartWidth / (timeSeriesData.length + forecastHorizon - 1)) * (timeSeriesData.length + i)
        const y = padding + chartHeight - ((point.forecast! - minValue) / valueRange) * chartHeight
        ctx.lineTo(x, y)
      })
      ctx.stroke()
      ctx.setLineDash([])
    }
    
  }, [timeSeriesData, selectedModel, forecastHorizon])

  // 데이터 다운로드
  const downloadData = () => {
    if (!selectedModel) return
    
    let csv = 'Date,Actual,Forecast,Lower Bound,Upper Bound\n'
    
    // 실제 데이터
    timeSeriesData.forEach(point => {
      csv += `${point.date.toISOString().split('T')[0]},${point.value},,,\n`
    })
    
    // 예측 데이터
    selectedModel.forecast.forEach(point => {
      csv += `${point.date.toISOString().split('T')[0]},,${point.forecast},${point.lowerBound},${point.upperBound}\n`
    })
    
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `timeseries_forecast_${selectedModel.name}_${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* 헤더 */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl p-6">
        <h2 className="text-2xl font-bold mb-2">시계열 예측 with PyCaret</h2>
        <p className="text-blue-100">
          PyCaret의 시계열 예측 기능으로 여러 모델을 자동으로 비교하고 최적의 예측을 생성합니다
        </p>
      </div>

      {/* 진행 단계 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-4">
        <div className="flex items-center justify-between">
          {['data', 'setup', 'compare', 'forecast'].map((step, idx) => (
            <div key={step} className="flex items-center">
              <div className={`flex items-center justify-center w-10 h-10 rounded-full ${
                currentStep === step 
                  ? 'bg-blue-500 text-white' 
                  : idx < ['data', 'setup', 'compare', 'forecast'].indexOf(currentStep)
                    ? 'bg-green-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-500'
              }`}>
                {idx < ['data', 'setup', 'compare', 'forecast'].indexOf(currentStep) 
                  ? <Check className="w-5 h-5" />
                  : idx + 1
                }
              </div>
              <div className="ml-2">
                <div className="text-sm font-medium">
                  {step === 'data' && '데이터 선택'}
                  {step === 'setup' && '설정'}
                  {step === 'compare' && '모델 비교'}
                  {step === 'forecast' && '예측 결과'}
                </div>
              </div>
              {idx < 3 && <ChevronRight className="w-5 h-5 mx-4 text-gray-400" />}
            </div>
          ))}
        </div>
      </div>

      {/* Step 1: 데이터 선택 */}
      {currentStep === 'data' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">데이터셋 선택</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {DATASETS.map((dataset) => (
              <button
                key={dataset.name}
                onClick={() => {
                  setSelectedDataset(dataset)
                  setTimeSeriesData(generateTimeSeries(dataset))
                  setCurrentStep('setup')
                }}
                className={`p-4 rounded-lg border-2 transition-all text-left ${
                  selectedDataset.name === dataset.name
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                }`}
              >
                <h4 className="font-medium mb-1">{dataset.name}</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  {dataset.description}
                </p>
                <div className="flex items-center gap-4 text-xs text-gray-500">
                  <span>{dataset.length}일 데이터</span>
                  {dataset.hasSeasonality && (
                    <span className="px-2 py-0.5 bg-green-100 dark:bg-green-900/30 rounded">
                      계절성
                    </span>
                  )}
                  {dataset.hasTrend && (
                    <span className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900/30 rounded">
                      트렌드
                    </span>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Step 2: 설정 */}
      {currentStep === 'setup' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">PyCaret 시계열 설정</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium mb-2">예측 기간 (일)</label>
              <input
                type="number"
                value={forecastHorizon}
                onChange={(e) => setForecastHorizon(parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg"
                min="7"
                max="90"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">교차 검증 폴드</label>
              <select className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700">
                <option>3</option>
                <option>5</option>
                <option>10</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">계절 주기</label>
              <select className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700">
                <option>자동 감지</option>
                <option>7 (주간)</option>
                <option>30 (월간)</option>
                <option>365 (연간)</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">변환 방법</label>
              <select className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700">
                <option>없음</option>
                <option>Box-Cox</option>
                <option>Log</option>
                <option>Differencing</option>
              </select>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
            <div className="flex items-start gap-2">
              <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5" />
              <div className="text-sm text-yellow-700 dark:text-yellow-300">
                <p className="font-medium">자동 특성 엔지니어링</p>
                <p>PyCaret은 자동으로 날짜 특성(요일, 월, 분기 등)을 생성하고 lag 특성을 추가합니다.</p>
              </div>
            </div>
          </div>
          
          <button
            onClick={trainModels}
            className="mt-6 w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
          >
            <Brain className="w-5 h-5" />
            모델 학습 시작
          </button>
        </div>
      )}

      {/* Step 3: 모델 비교 */}
      {currentStep === 'compare' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Zap className="w-5 h-5" />
            모델 성능 비교
          </h3>

          {isTraining && (
            <div className="mb-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <div className="flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-500 border-t-transparent"></div>
                <span className="text-blue-700 dark:text-blue-300">
                  모델 학습 중... ({modelResults.length}/6)
                </span>
              </div>
            </div>
          )}

          {modelResults.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <th className="text-left py-3 px-4">모델</th>
                    <th className="text-center py-3 px-4">MAE</th>
                    <th className="text-center py-3 px-4">RMSE</th>
                    <th className="text-center py-3 px-4">MAPE (%)</th>
                    <th className="text-center py-3 px-4">학습 시간</th>
                    <th className="text-center py-3 px-4">선택</th>
                  </tr>
                </thead>
                <tbody>
                  {modelResults.map((model, idx) => {
                    const isBest = modelResults.reduce((best, m) => 
                      m.mae < best.mae ? m : best
                    ).name === model.name
                    
                    return (
                      <tr 
                        key={idx} 
                        className={`border-b border-gray-100 dark:border-gray-700 ${
                          isBest ? 'bg-green-50 dark:bg-green-900/20' : ''
                        }`}
                      >
                        <td className="py-3 px-4 font-medium">
                          {model.name}
                          {isBest && (
                            <span className="ml-2 text-xs bg-green-500 text-white px-2 py-0.5 rounded">
                              Best
                            </span>
                          )}
                        </td>
                        <td className="text-center py-3 px-4">{model.mae.toFixed(2)}</td>
                        <td className="text-center py-3 px-4">{model.rmse.toFixed(2)}</td>
                        <td className="text-center py-3 px-4">{model.mape.toFixed(1)}</td>
                        <td className="text-center py-3 px-4">{model.trainTime.toFixed(1)}s</td>
                        <td className="text-center py-3 px-4">
                          <button
                            onClick={() => {
                              setSelectedModel(model)
                              setCurrentStep('forecast')
                            }}
                            className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 transition-colors"
                          >
                            선택
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Step 4: 예측 결과 */}
      {currentStep === 'forecast' && selectedModel && (
        <>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">
                {selectedModel.name} 예측 결과
              </h3>
              <button
                onClick={downloadData}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                CSV 다운로드
              </button>
            </div>
            
            <canvas
              ref={canvasRef}
              className="w-full h-96 rounded-lg bg-gray-50"
            />
            
            <div className="mt-4 grid grid-cols-3 gap-4 text-center">
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="text-2xl font-bold text-blue-600">{selectedModel.mae.toFixed(2)}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">MAE</div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="text-2xl font-bold text-green-600">{selectedModel.rmse.toFixed(2)}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">RMSE</div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="text-2xl font-bold text-purple-600">{selectedModel.mape.toFixed(1)}%</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">MAPE</div>
              </div>
            </div>
          </div>

          {/* 시계열 분해 */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4">시계열 분해</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-medium mb-2">트렌드</h4>
                <div className="h-32 flex items-center justify-center">
                  <TrendingUp className="w-16 h-16 text-gray-300" />
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  {selectedDataset.hasTrend ? '상승 트렌드 감지됨' : '뚜렷한 트렌드 없음'}
                </p>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-medium mb-2">계절성</h4>
                <div className="h-32 flex items-center justify-center">
                  <Activity className="w-16 h-16 text-gray-300" />
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  {selectedDataset.hasSeasonality ? '주기적 패턴 발견' : '계절성 없음'}
                </p>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-medium mb-2">잔차</h4>
                <div className="h-32 flex items-center justify-center">
                  <BarChart3 className="w-16 h-16 text-gray-300" />
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  랜덤 노이즈 수준
                </p>
              </div>
            </div>
          </div>
        </>
      )}

      {/* 정보 패널 */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-600 mt-0.5" />
          <div className="text-sm text-blue-700 dark:text-blue-300">
            <p className="font-medium mb-1">PyCaret 시계열 예측</p>
            <p>
              이 시뮬레이터는 PyCaret의 시계열 예측 기능을 시뮬레이션합니다.
              실제 PyCaret은 더 많은 모델과 고급 기능을 제공합니다.
            </p>
            <div className="mt-2 flex flex-wrap gap-2">
              <span className="px-2 py-1 bg-blue-100 dark:bg-blue-800 rounded text-xs">
                자동 특성 엔지니어링
              </span>
              <span className="px-2 py-1 bg-blue-100 dark:bg-blue-800 rounded text-xs">
                교차 검증
              </span>
              <span className="px-2 py-1 bg-blue-100 dark:bg-blue-800 rounded text-xs">
                하이퍼파라미터 튜닝
              </span>
              <span className="px-2 py-1 bg-blue-100 dark:bg-blue-800 rounded text-xs">
                앙상블 예측
              </span>
            </div>
            <div className="mt-3">
              <p className="font-medium mb-1">지원 모델:</p>
              <ul className="list-disc list-inside space-y-1 ml-2">
                <li>ARIMA - 전통적 시계열 모델</li>
                <li>Prophet - Facebook의 예측 라이브러리</li>
                <li>Exponential Smoothing - 지수 평활법</li>
                <li>LSTM - 딥러닝 기반 예측</li>
                <li>Random Forest/XGBoost - ML 기반 예측</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}