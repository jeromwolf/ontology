'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'
import { ArrowLeft, Play, Pause, RotateCcw, Activity, TrendingUp, AlertTriangle, Brain, Gauge, Thermometer, Download, Upload, Shield, Sparkles, Zap } from 'lucide-react'

interface BearingData {
  timestamp: number
  vibrationX: number
  vibrationY: number
  vibrationZ: number
  temperature: number
  rpm: number
  acousticEmission: number
  oilPressure: number
}

interface PredictionModel {
  id: string
  name: string
  accuracy: number
  features: string[]
  algorithm: string
  color: string
}

interface HealthIndicator {
  name: string
  value: number
  trend: 'up' | 'down' | 'stable'
  threshold: number
  unit: string
}

export default function BearingFailurePredictionPage() {
  const searchParams = useSearchParams()
  const backUrl = searchParams.get('from') || '/modules/smart-factory'
  const [isRunning, setIsRunning] = useState(false)
  const [selectedModel, setSelectedModel] = useState('lstm')
  const [dataSource, setDataSource] = useState<'normal' | 'degrading' | 'critical'>('normal')
  const [showFrequencyAnalysis, setShowFrequencyAnalysis] = useState(true)
  const [rul, setRul] = useState(180) // Remaining Useful Life in days
  const [healthScore, setHealthScore] = useState(95)
  const [failureProbability, setFailureProbability] = useState(5)
  
  const timeSeriesCanvasRef = useRef<HTMLCanvasElement>(null)
  const frequencyCanvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  
  const [bearingData, setBearingData] = useState<BearingData[]>([])
  const [predictions, setPredictions] = useState<{time: number, probability: number}[]>([])

  const models: PredictionModel[] = [
    {
      id: 'lstm',
      name: 'LSTM Neural Network',
      accuracy: 94.5,
      features: ['진동', '온도', 'RPM', '음향'],
      algorithm: 'Long Short-Term Memory',
      color: '#3B82F6'
    },
    {
      id: 'random-forest',
      name: 'Random Forest',
      accuracy: 91.2,
      features: ['진동', '온도', '오일압력'],
      algorithm: 'Ensemble Learning',
      color: '#10B981'
    },
    {
      id: 'svm',
      name: 'Support Vector Machine',
      accuracy: 88.7,
      features: ['주파수 스펙트럼', '온도'],
      algorithm: 'Pattern Recognition',
      color: '#F59E0B'
    },
    {
      id: 'physics-ml',
      name: 'Physics-Informed ML',
      accuracy: 96.3,
      features: ['모든 센서', '물리 모델'],
      algorithm: 'Hybrid Physics + ML',
      color: '#8B5CF6'
    }
  ]

  const healthIndicators: HealthIndicator[] = [
    { name: 'RMS 진동', value: dataSource === 'normal' ? 2.5 : dataSource === 'degrading' ? 5.8 : 12.3, trend: dataSource === 'normal' ? 'stable' : 'up', threshold: 7.0, unit: 'mm/s' },
    { name: '온도', value: dataSource === 'normal' ? 45 : dataSource === 'degrading' ? 68 : 85, trend: dataSource === 'normal' ? 'stable' : 'up', threshold: 70, unit: '°C' },
    { name: '고조파 비율', value: dataSource === 'normal' ? 0.15 : dataSource === 'degrading' ? 0.42 : 0.78, trend: dataSource === 'normal' ? 'stable' : 'up', threshold: 0.5, unit: 'ratio' },
    { name: '크레스트 팩터', value: dataSource === 'normal' ? 3.2 : dataSource === 'degrading' ? 5.1 : 8.5, trend: dataSource === 'normal' ? 'stable' : 'up', threshold: 6.0, unit: '' }
  ]

  // 시계열 데이터 시각화
  useEffect(() => {
    if (!timeSeriesCanvasRef.current) return
    const canvas = timeSeriesCanvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = canvas.offsetWidth
    canvas.height = 250

    let dataPoints: number[] = new Array(100).fill(0)
    let predictionLine: number[] = new Array(100).fill(0)

    const drawChart = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // 배경 그리드
      ctx.strokeStyle = '#E5E7EB'
      ctx.lineWidth = 1
      for (let i = 0; i <= 5; i++) {
        const y = (canvas.height / 5) * i
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(canvas.width, y)
        ctx.stroke()
      }

      if (isRunning) {
        // 새 데이터 생성
        let baseValue = 0
        let noise = 0
        
        switch (dataSource) {
          case 'normal':
            baseValue = 30
            noise = Math.random() * 10 - 5
            break
          case 'degrading':
            baseValue = 50 + Math.sin(Date.now() / 1000) * 10
            noise = Math.random() * 20 - 10
            break
          case 'critical':
            baseValue = 80 + Math.sin(Date.now() / 500) * 20
            noise = Math.random() * 30 - 15
            break
        }
        
        dataPoints.push(baseValue + noise)
        dataPoints.shift()

        // 예측 라인 업데이트
        const model = models.find(m => m.id === selectedModel)
        if (model) {
          const prediction = baseValue + (model.accuracy / 100) * 10
          predictionLine.push(prediction)
          predictionLine.shift()
        }
      }

      // 실제 데이터 그리기
      ctx.strokeStyle = '#3B82F6'
      ctx.lineWidth = 2
      ctx.beginPath()
      dataPoints.forEach((value, index) => {
        const x = (canvas.width / (dataPoints.length - 1)) * index
        const y = canvas.height - (value / 100) * canvas.height
        if (index === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()

      // 예측 라인 그리기
      ctx.strokeStyle = '#10B981'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      predictionLine.forEach((value, index) => {
        const x = (canvas.width / (predictionLine.length - 1)) * index
        const y = canvas.height - (value / 100) * canvas.height
        if (index === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()
      ctx.setLineDash([])

      // 임계값 라인
      if (dataSource !== 'normal') {
        ctx.strokeStyle = '#EF4444'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(0, canvas.height * 0.3)
        ctx.lineTo(canvas.width, canvas.height * 0.3)
        ctx.stroke()
        
        ctx.fillStyle = '#EF4444'
        ctx.font = '12px Arial'
        ctx.fillText('위험 임계값', 10, canvas.height * 0.3 - 5)
      }

      // 범례
      ctx.fillStyle = '#3B82F6'
      ctx.fillRect(10, 10, 20, 3)
      ctx.fillStyle = '#000'
      ctx.font = '12px Arial'
      ctx.fillText('실제 진동', 35, 15)
      
      ctx.fillStyle = '#10B981'
      ctx.fillRect(10, 25, 20, 3)
      ctx.fillText('AI 예측', 35, 30)
    }

    const interval = setInterval(drawChart, 100)
    return () => clearInterval(interval)
  }, [isRunning, dataSource, selectedModel])

  // 주파수 분석 시각화
  useEffect(() => {
    if (!frequencyCanvasRef.current || !showFrequencyAnalysis) return
    const canvas = frequencyCanvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = canvas.offsetWidth
    canvas.height = 200

    const drawFrequencySpectrum = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // FFT 시뮬레이션
      const frequencies = 50
      const barWidth = canvas.width / frequencies
      
      for (let i = 0; i < frequencies; i++) {
        let amplitude = 0
        
        if (dataSource === 'normal') {
          amplitude = Math.exp(-i / 10) * 50 + Math.random() * 10
        } else if (dataSource === 'degrading') {
          // 특정 주파수에서 피크
          if (i === 12 || i === 24 || i === 36) {
            amplitude = 80 + Math.random() * 20
          } else {
            amplitude = Math.exp(-i / 15) * 40 + Math.random() * 20
          }
        } else {
          // 많은 하모닉스
          if (i % 6 === 0) {
            amplitude = 90 + Math.random() * 10
          } else {
            amplitude = Math.exp(-i / 20) * 60 + Math.random() * 30
          }
        }

        const height = (amplitude / 100) * canvas.height
        const hue = amplitude > 70 ? 0 : amplitude > 40 ? 60 : 120
        
        ctx.fillStyle = `hsl(${hue}, 70%, 50%)`
        ctx.fillRect(i * barWidth, canvas.height - height, barWidth - 1, height)
      }

      // 축 라벨
      ctx.fillStyle = '#000'
      ctx.font = '12px Arial'
      ctx.fillText('0 Hz', 5, canvas.height - 5)
      ctx.fillText('500 Hz', canvas.width / 2 - 20, canvas.height - 5)
      ctx.fillText('1 kHz', canvas.width - 35, canvas.height - 5)
    }

    const interval = setInterval(drawFrequencySpectrum, 200)
    return () => clearInterval(interval)
  }, [showFrequencyAnalysis, dataSource])

  // RUL 및 건강도 업데이트
  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(() => {
      if (dataSource === 'degrading') {
        setRul(prev => Math.max(0, prev - 0.5))
        setHealthScore(prev => Math.max(0, prev - 0.3))
        setFailureProbability(prev => Math.min(100, prev + 0.5))
      } else if (dataSource === 'critical') {
        setRul(prev => Math.max(0, prev - 2))
        setHealthScore(prev => Math.max(0, prev - 1))
        setFailureProbability(prev => Math.min(100, prev + 2))
      } else {
        // Normal - 천천히 회복
        setRul(prev => Math.min(365, prev + 0.1))
        setHealthScore(prev => Math.min(100, prev + 0.1))
        setFailureProbability(prev => Math.max(0, prev - 0.1))
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [isRunning, dataSource])

  const getHealthColor = (score: number) => {
    if (score > 80) return 'text-green-500'
    if (score > 50) return 'text-yellow-500'
    return 'text-red-500'
  }

  const getRULColor = (days: number) => {
    if (days > 90) return 'bg-green-500'
    if (days > 30) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Link 
                href={backUrl}
                className="flex items-center gap-2 text-amber-600 dark:text-amber-400 hover:text-amber-700 dark:hover:text-amber-300"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>학습 페이지로 돌아가기</span>
              </Link>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowFrequencyAnalysis(!showFrequencyAnalysis)}
                className={`px-3 py-1 rounded text-sm ${
                  showFrequencyAnalysis 
                    ? 'bg-purple-600 text-white' 
                    : 'bg-gray-300 dark:bg-gray-600 text-gray-700 dark:text-gray-300'
                }`}
              >
                <Activity className="w-4 h-4 inline mr-1" />
                주파수 분석
              </button>
              <button
                onClick={() => setIsRunning(!isRunning)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium ${
                  isRunning 
                    ? 'bg-red-600 text-white hover:bg-red-700' 
                    : 'bg-green-600 text-white hover:bg-green-700'
                }`}
              >
                {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isRunning ? '분석 중지' : '분석 시작'}
              </button>
              <button
                onClick={() => {
                  setIsRunning(false)
                  setRul(180)
                  setHealthScore(95)
                  setFailureProbability(5)
                  setBearingData([])
                  setPredictions([])
                }}
                className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
              >
                <RotateCcw className="w-4 h-4" />
                리셋
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Title */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                베어링 고장 예측 AI Lab
                <Sparkles className="w-6 h-6 text-yellow-500" />
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400">진동 데이터 기반 RUL 예측 및 고장 진단</p>
            </div>
          </div>
        </div>

        {/* Main Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className={`bg-white dark:bg-gray-800 rounded-xl p-4 border ${
            rul < 30 ? 'border-red-500 ring-2 ring-red-500 ring-opacity-50 animate-pulse' : 'border-gray-200 dark:border-gray-700'
          }`}>
            <div className="flex items-center gap-2 mb-2">
              <Shield className="w-5 h-5 text-blue-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">잔여 수명 (RUL)</span>
            </div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {Math.round(rul)} 일
            </div>
            <div className={`w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full mt-2`}>
              <div className={`h-2 rounded-full ${getRULColor(rul)}`} style={{ width: `${(rul / 365) * 100}%` }}></div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 mb-2">
              <Gauge className="w-5 h-5 text-green-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">건강도 점수</span>
            </div>
            <div className={`text-2xl font-bold ${getHealthColor(healthScore)}`}>
              {healthScore.toFixed(1)}%
            </div>
          </div>

          <div className={`bg-white dark:bg-gray-800 rounded-xl p-4 border ${
            failureProbability > 70 ? 'border-red-500 animate-pulse' : 'border-gray-200 dark:border-gray-700'
          }`}>
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="w-5 h-5 text-orange-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">고장 확률</span>
            </div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {failureProbability.toFixed(1)}%
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 mb-2">
              <Brain className="w-5 h-5 text-purple-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">모델 정확도</span>
            </div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {models.find(m => m.id === selectedModel)?.accuracy}%
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Charts */}
          <div className="lg:col-span-2 space-y-6">
            {/* Time Series */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">실시간 진동 데이터</h2>
              <canvas 
                ref={timeSeriesCanvasRef}
                className="w-full"
                style={{ height: '250px' }}
              />
            </div>

            {/* Frequency Analysis */}
            {showFrequencyAnalysis && (
              <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">주파수 스펙트럼 분석</h2>
                <canvas 
                  ref={frequencyCanvasRef}
                  className="w-full"
                  style={{ height: '200px' }}
                />
              </div>
            )}

            {/* Health Indicators */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">건강 지표</h2>
              <div className="grid grid-cols-2 gap-4">
                {healthIndicators.map((indicator, index) => (
                  <div key={index} className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{indicator.name}</span>
                      <TrendingUp className={`w-4 h-4 ${
                        indicator.trend === 'up' ? 'text-red-500' : 
                        indicator.trend === 'down' ? 'text-green-500' : 
                        'text-gray-500'
                      }`} />
                    </div>
                    <div className="text-lg font-bold text-gray-900 dark:text-white">
                      {indicator.value} {indicator.unit}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      임계값: {indicator.threshold} {indicator.unit}
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2 mt-2">
                      <div 
                        className={`h-2 rounded-full ${
                          indicator.value > indicator.threshold ? 'bg-red-500' : 'bg-green-500'
                        }`}
                        style={{ width: `${Math.min(100, (indicator.value / (indicator.threshold * 1.5)) * 100)}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Control Panel */}
          <div className="space-y-6">
            {/* Data Source */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">데이터 소스</h3>
              
              <div className="space-y-3">
                <button
                  onClick={() => setDataSource('normal')}
                  className={`w-full p-3 rounded-lg border text-left ${
                    dataSource === 'normal'
                      ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <div className="font-medium text-gray-900 dark:text-white">정상 베어링</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">건강한 상태의 베어링</div>
                </button>
                
                <button
                  onClick={() => setDataSource('degrading')}
                  className={`w-full p-3 rounded-lg border text-left ${
                    dataSource === 'degrading'
                      ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <div className="font-medium text-gray-900 dark:text-white">열화 진행 중</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">성능 저하가 시작됨</div>
                </button>
                
                <button
                  onClick={() => setDataSource('critical')}
                  className={`w-full p-3 rounded-lg border text-left ${
                    dataSource === 'critical'
                      ? 'border-red-500 bg-red-50 dark:bg-red-900/20 animate-pulse'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <div className="font-medium text-gray-900 dark:text-white">임박한 고장</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">즉시 교체 필요</div>
                </button>
              </div>
            </div>

            {/* AI Models */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">AI 모델 선택</h3>
              
              <div className="space-y-3">
                {models.map((model) => (
                  <button
                    key={model.id}
                    onClick={() => setSelectedModel(model.id)}
                    className={`w-full p-3 rounded-lg border text-left ${
                      selectedModel === model.id
                        ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                        : 'border-gray-200 dark:border-gray-700'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium text-gray-900 dark:text-white">{model.name}</div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">{model.algorithm}</div>
                      </div>
                      <div className="text-sm font-medium text-gray-900 dark:text-white">
                        {model.accuracy}%
                      </div>
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {model.features.map((feature, idx) => (
                        <span key={idx} className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded">
                          {feature}
                        </span>
                      ))}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Actions */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">빠른 작업</h3>
              
              <div className="space-y-3">
                <button className="w-full p-3 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors text-sm font-medium">
                  <Upload className="w-4 h-4 inline mr-2" />
                  실제 데이터 업로드
                </button>
                
                <button className="w-full p-3 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded-lg hover:bg-green-100 dark:hover:bg-green-900/30 transition-colors text-sm font-medium">
                  <Download className="w-4 h-4 inline mr-2" />
                  분석 결과 다운로드
                </button>
                
                <button className="w-full p-3 bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors text-sm font-medium">
                  <Zap className="w-4 h-4 inline mr-2" />
                  모델 재학습
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}