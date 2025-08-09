'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import { ArrowLeft, Play, Pause, RotateCcw, Camera, AlertTriangle, CheckCircle, XCircle, Eye, Upload, Settings, BarChart3, Sparkles, Zap, Target, Brain, Shield } from 'lucide-react'

interface DefectType {
  id: string
  name: string
  color: string
  count: number
  severity: 'low' | 'medium' | 'high' | 'critical'
}

interface InspectionResult {
  id: string
  timestamp: number
  image: string
  result: 'pass' | 'fail'
  defects: DefectType[]
  confidence: number
  processingTime: number
  aiModel: string
}

interface QualityMetrics {
  totalInspected: number
  passRate: number
  defectRate: number
  avgConfidence: number
  avgProcessingTime: number
  criticalDefects: number
}

interface DetectionBox {
  x: number
  y: number
  width: number
  height: number
  label: string
  confidence: number
  color: string
}

export default function EnhancedQualityVisionPage() {
  const [isRunning, setIsRunning] = useState(false)
  const [selectedModel, setSelectedModel] = useState('yolo-v8')
  const [sensitivity, setSensitivity] = useState(0.7)
  const [inspectionSpeed, setInspectionSpeed] = useState(1)
  const [showAIOverlay, setShowAIOverlay] = useState(true)
  const [inspectionResults, setInspectionResults] = useState<InspectionResult[]>([])
  const [currentImage, setCurrentImage] = useState<string | null>(null)
  const [detectionBoxes, setDetectionBoxes] = useState<DetectionBox[]>([])
  const [scanEffect, setScanEffect] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageCanvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  
  const [metrics, setMetrics] = useState<QualityMetrics>({
    totalInspected: 0,
    passRate: 95.8,
    defectRate: 4.2,
    avgConfidence: 92.5,
    avgProcessingTime: 150,
    criticalDefects: 0
  })

  const defectTypes: DefectType[] = [
    { id: 'scratch', name: '스크래치', color: 'bg-red-500', count: 0, severity: 'high' },
    { id: 'dent', name: '찌그러짐', color: 'bg-orange-500', count: 0, severity: 'medium' },
    { id: 'discoloration', name: '변색', color: 'bg-yellow-500', count: 0, severity: 'low' },
    { id: 'crack', name: '균열', color: 'bg-purple-500', count: 0, severity: 'critical' },
    { id: 'contamination', name: '오염', color: 'bg-blue-500', count: 0, severity: 'medium' }
  ]

  const aiModels = [
    { id: 'yolo-v8', name: 'YOLO v8 Ultra', accuracy: 98.5, speed: 'fast' },
    { id: 'detectron2', name: 'Detectron2 R-CNN', accuracy: 99.2, speed: 'medium' },
    { id: 'custom-cnn', name: 'Custom CNN Pro', accuracy: 97.8, speed: 'very fast' }
  ]

  // 파티클 효과
  useEffect(() => {
    if (!canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight

    let particles: Array<{x: number, y: number, vx: number, vy: number, life: number, color: string}> = []
    let scanLine = 0

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // 스캔 효과
      if (isRunning && scanEffect) {
        scanLine = (scanLine + 2) % canvas.height
        ctx.strokeStyle = '#00FF00'
        ctx.lineWidth = 2
        ctx.globalAlpha = 0.5
        ctx.beginPath()
        ctx.moveTo(0, scanLine)
        ctx.lineTo(canvas.width, scanLine)
        ctx.stroke()
        
        // 스캔라인 글로우 효과
        ctx.strokeStyle = '#00FF00'
        ctx.lineWidth = 10
        ctx.globalAlpha = 0.1
        ctx.beginPath()
        ctx.moveTo(0, scanLine)
        ctx.lineTo(canvas.width, scanLine)
        ctx.stroke()
      }

      // 파티클 업데이트
      particles = particles.filter(particle => {
        particle.x += particle.vx
        particle.y += particle.vy
        particle.life -= 0.02

        if (particle.life <= 0) return false

        ctx.globalAlpha = particle.life
        ctx.fillStyle = particle.color
        ctx.beginPath()
        ctx.arc(particle.x, particle.y, 2, 0, Math.PI * 2)
        ctx.fill()

        return true
      })

      // 검사 중일 때 파티클 생성
      if (isRunning && Math.random() < 0.1) {
        for (let i = 0; i < 3; i++) {
          particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 2,
            vy: (Math.random() - 0.5) * 2,
            life: 1,
            color: Math.random() > 0.5 ? '#00FF00' : '#0099FF'
          })
        }
      }

      animationRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning, scanEffect])

  // AI 검사 시뮬레이션
  useEffect(() => {
    let interval: NodeJS.Timeout
    
    if (isRunning) {
      interval = setInterval(() => {
        setScanEffect(true)
        setTimeout(() => setScanEffect(false), 1000)

        // 가상 제품 이미지 생성
        const productTypes = ['PCB', 'Metal Part', 'Plastic Component', 'Assembly']
        const productType = productTypes[Math.floor(Math.random() * productTypes.length)]
        
        // 불량 검출 확률
        const hasDefect = Math.random() < 0.05
        const detectedDefects: DefectType[] = []
        const boxes: DetectionBox[] = []

        if (hasDefect) {
          // 랜덤하게 1-3개의 불량 생성
          const numDefects = Math.floor(Math.random() * 3) + 1
          for (let i = 0; i < numDefects; i++) {
            const defect = defectTypes[Math.floor(Math.random() * defectTypes.length)]
            detectedDefects.push({ ...defect, count: 1 })
            
            // 검출 박스 생성
            boxes.push({
              x: Math.random() * 300 + 50,
              y: Math.random() * 200 + 50,
              width: Math.random() * 50 + 30,
              height: Math.random() * 50 + 30,
              label: defect.name,
              confidence: Math.random() * 0.2 + 0.8,
              color: defect.severity === 'critical' ? '#FF0000' : 
                     defect.severity === 'high' ? '#FF6600' :
                     defect.severity === 'medium' ? '#FFAA00' : '#FFFF00'
            })
          }
        }

        setDetectionBoxes(boxes)

        const result: InspectionResult = {
          id: Date.now().toString(),
          timestamp: Date.now(),
          image: productType,
          result: hasDefect ? 'fail' : 'pass',
          defects: detectedDefects,
          confidence: Math.random() * 0.1 + 0.9,
          processingTime: Math.random() * 50 + 100,
          aiModel: selectedModel
        }

        setInspectionResults(prev => [result, ...prev.slice(0, 9)])
        
        // 메트릭 업데이트
        setMetrics(prev => {
          const newTotal = prev.totalInspected + 1
          const passCount = inspectionResults.filter(r => r.result === 'pass').length + (hasDefect ? 0 : 1)
          const newPassRate = (passCount / newTotal) * 100
          const avgProcessing = inspectionResults.reduce((acc, r) => acc + r.processingTime, 0) / Math.max(inspectionResults.length, 1)
          
          return {
            totalInspected: newTotal,
            passRate: Math.round(newPassRate * 10) / 10,
            defectRate: Math.round((100 - newPassRate) * 10) / 10,
            avgConfidence: Math.round((result.confidence * 100 + prev.avgConfidence * (newTotal - 1)) / newTotal * 10) / 10,
            avgProcessingTime: Math.round(avgProcessing),
            criticalDefects: prev.criticalDefects + (detectedDefects.some(d => d.severity === 'critical') ? 1 : 0)
          }
        })
      }, 2000 / inspectionSpeed)
    }
    
    return () => clearInterval(interval)
  }, [isRunning, selectedModel, inspectionSpeed, inspectionResults])

  // 이미지 캔버스에 검출 박스 그리기
  useEffect(() => {
    if (!imageCanvasRef.current || !showAIOverlay) return
    const canvas = imageCanvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = 400
    canvas.height = 300

    // 배경
    ctx.fillStyle = '#1a1a1a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // 그리드
    ctx.strokeStyle = '#333'
    ctx.lineWidth = 1
    for (let i = 0; i < canvas.width; i += 20) {
      ctx.beginPath()
      ctx.moveTo(i, 0)
      ctx.lineTo(i, canvas.height)
      ctx.stroke()
    }
    for (let i = 0; i < canvas.height; i += 20) {
      ctx.beginPath()
      ctx.moveTo(0, i)
      ctx.lineTo(canvas.width, i)
      ctx.stroke()
    }

    // 제품 시뮬레이션
    ctx.fillStyle = '#444'
    ctx.fillRect(100, 75, 200, 150)
    ctx.fillStyle = '#666'
    ctx.fillRect(120, 95, 160, 110)

    // 검출 박스 그리기
    detectionBoxes.forEach(box => {
      ctx.strokeStyle = box.color
      ctx.lineWidth = 2
      ctx.strokeRect(box.x, box.y, box.width, box.height)
      
      // 라벨
      ctx.fillStyle = box.color
      ctx.fillRect(box.x, box.y - 20, box.width, 20)
      ctx.fillStyle = '#000'
      ctx.font = '12px Arial'
      ctx.fillText(`${box.label} ${Math.round(box.confidence * 100)}%`, box.x + 2, box.y - 5)
    })

    // AI 분석 오버레이
    if (scanEffect) {
      ctx.strokeStyle = '#00FF00'
      ctx.lineWidth = 1
      ctx.setLineDash([5, 5])
      ctx.strokeRect(10, 10, canvas.width - 20, canvas.height - 20)
      ctx.setLineDash([])
    }
  }, [detectionBoxes, showAIOverlay, scanEffect])

  const getSpeedBadge = (speed: string) => {
    const colors = {
      'very fast': 'bg-green-500',
      'fast': 'bg-blue-500',
      'medium': 'bg-yellow-500',
      'slow': 'bg-red-500'
    }
    return colors[speed as keyof typeof colors] || 'bg-gray-500'
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 relative overflow-hidden">
      {/* 파티클 효과 캔버스 */}
      <canvas 
        ref={canvasRef}
        className="fixed inset-0 pointer-events-none z-30"
        style={{ width: '100%', height: '100%' }}
      />

      {/* AI 분석 중 효과 */}
      {isRunning && scanEffect && (
        <div className="fixed inset-0 pointer-events-none z-40">
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-green-500/10 to-transparent animate-pulse"></div>
        </div>
      )}

      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 relative z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Link 
                href="/modules/smart-factory"
                className="flex items-center gap-2 text-amber-600 dark:text-amber-400 hover:text-amber-700 dark:hover:text-amber-300"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>Smart Factory로 돌아가기</span>
              </Link>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-600 dark:text-gray-400">속도:</label>
                <select 
                  value={inspectionSpeed} 
                  onChange={(e) => setInspectionSpeed(Number(e.target.value))}
                  className="px-2 py-1 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-sm"
                >
                  <option value={0.5}>0.5x</option>
                  <option value={1}>1x</option>
                  <option value={2}>2x</option>
                  <option value={3}>3x</option>
                </select>
              </div>
              <button
                onClick={() => setShowAIOverlay(!showAIOverlay)}
                className={`px-3 py-1 rounded text-sm flex items-center gap-1 ${
                  showAIOverlay 
                    ? 'bg-green-600 text-white' 
                    : 'bg-gray-300 dark:bg-gray-600 text-gray-700 dark:text-gray-300'
                }`}
              >
                <Brain className="w-4 h-4" />
                AI 오버레이
              </button>
              <button
                onClick={() => setIsRunning(!isRunning)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium ${
                  isRunning 
                    ? 'bg-red-600 text-white hover:bg-red-700 animate-pulse' 
                    : 'bg-green-600 text-white hover:bg-green-700'
                }`}
              >
                {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isRunning ? '검사 중지' : '검사 시작'}
              </button>
              <button
                onClick={() => {
                  setIsRunning(false)
                  setInspectionResults([])
                  setDetectionBoxes([])
                  setMetrics({
                    totalInspected: 0,
                    passRate: 95.8,
                    defectRate: 4.2,
                    avgConfidence: 92.5,
                    avgProcessingTime: 150,
                    criticalDefects: 0
                  })
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

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        {/* Title */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl flex items-center justify-center animate-pulse">
              <Eye className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                AI 품질 검사 비전 시스템
                <Sparkles className="w-6 h-6 text-yellow-500" />
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400">딥러닝 기반 실시간 불량 검출 시스템</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* AI Vision Display */}
          <div className="lg:col-span-2">
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6 mb-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                  실시간 비전 검사
                  {isRunning && <Camera className="w-5 h-5 text-green-500 animate-pulse" />}
                </h2>
                <div className="flex items-center gap-2">
                  <Target className="w-5 h-5 text-blue-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    감도: {Math.round(sensitivity * 100)}%
                  </span>
                </div>
              </div>

              {/* Vision Display */}
              <div className="relative bg-black rounded-lg overflow-hidden mb-4">
                <canvas 
                  ref={imageCanvasRef}
                  className="w-full"
                  style={{ height: '300px' }}
                />
                {scanEffect && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-green-500 font-mono text-lg animate-pulse">
                      AI ANALYZING...
                    </div>
                  </div>
                )}
              </div>

              {/* Detection Results */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                {defectTypes.map(defect => (
                  <div key={defect.id} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <div className={`w-3 h-3 rounded-full ${defect.color}`}></div>
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        {defect.name}
                      </span>
                    </div>
                    <div className="text-lg font-bold text-gray-900 dark:text-white">
                      {inspectionResults.reduce((acc, r) => 
                        acc + r.defects.filter(d => d.id === defect.id).length, 0
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Metrics Dashboard */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
              <div className={`bg-white dark:bg-gray-800 rounded-xl p-4 border ${
                metrics.passRate > 95 ? 'border-green-500 ring-2 ring-green-500 ring-opacity-50' : 'border-gray-200 dark:border-gray-700'
              }`}>
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">합격률</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metrics.passRate.toFixed(1)}%
                </div>
              </div>

              <div className={`bg-white dark:bg-gray-800 rounded-xl p-4 border ${
                metrics.criticalDefects > 0 ? 'border-red-500 animate-pulse' : 'border-gray-200 dark:border-gray-700'
              }`}>
                <div className="flex items-center gap-2 mb-2">
                  <AlertTriangle className="w-5 h-5 text-red-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">치명적 결함</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metrics.criticalDefects}
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="w-5 h-5 text-yellow-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">처리 속도</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metrics.avgProcessingTime}ms
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-2 mb-2">
                  <BarChart3 className="w-5 h-5 text-blue-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">총 검사</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metrics.totalInspected.toLocaleString()}
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-2 mb-2">
                  <Brain className="w-5 h-5 text-purple-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">AI 신뢰도</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metrics.avgConfidence}%
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-2 mb-2">
                  <XCircle className="w-5 h-5 text-orange-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">불량률</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metrics.defectRate.toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Recent Inspections */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">최근 검사 결과</h3>
              
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {inspectionResults.map((result) => (
                  <div 
                    key={result.id} 
                    className={`flex items-center justify-between p-3 rounded-lg ${
                      result.result === 'pass' 
                        ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800' 
                        : 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800'
                    } animate-fade-in`}
                  >
                    <div className="flex items-center gap-3">
                      {result.result === 'pass' ? (
                        <CheckCircle className="w-5 h-5 text-green-500" />
                      ) : (
                        <XCircle className="w-5 h-5 text-red-500" />
                      )}
                      <div>
                        <div className="font-medium text-gray-900 dark:text-white">
                          {result.image}
                        </div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">
                          {result.defects.length > 0 
                            ? `검출: ${result.defects.map(d => d.name).join(', ')}`
                            : '이상 없음'}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-medium text-gray-900 dark:text-white">
                        {result.confidence.toFixed(1)}%
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        {result.processingTime}ms
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Control Panel */}
          <div className="space-y-6">
            {/* AI Model Selection */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                AI 모델 선택
                <Brain className="w-5 h-5 text-purple-500" />
              </h3>
              
              <div className="space-y-3">
                {aiModels.map((model) => (
                  <button
                    key={model.id}
                    onClick={() => setSelectedModel(model.id)}
                    className={`w-full p-3 rounded-lg border text-left transition-all ${
                      selectedModel === model.id
                        ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20 ring-2 ring-purple-500 ring-opacity-50'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium text-gray-900 dark:text-white">
                          {model.name}
                        </div>
                        <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                          정확도: {model.accuracy}%
                        </div>
                      </div>
                      <div className={`px-2 py-1 rounded text-xs font-medium text-white ${getSpeedBadge(model.speed)}`}>
                        {model.speed}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Sensitivity Control */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">검출 감도</h3>
              
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">낮음</span>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {Math.round(sensitivity * 100)}%
                    </span>
                    <span className="text-sm text-gray-600 dark:text-gray-400">높음</span>
                  </div>
                  <input 
                    type="range" 
                    min="0" 
                    max="1" 
                    step="0.05"
                    value={sensitivity}
                    onChange={(e) => setSensitivity(Number(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                  />
                </div>
                
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  💡 높은 감도는 더 많은 불량을 검출하지만 오탐률도 증가합니다
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">빠른 작업</h3>
              
              <div className="space-y-3">
                <button className="w-full p-3 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors text-sm font-medium">
                  <Upload className="w-4 h-4 inline mr-2" />
                  학습 데이터 업로드
                </button>
                
                <button className="w-full p-3 bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors text-sm font-medium">
                  <Settings className="w-4 h-4 inline mr-2" />
                  모델 재학습
                </button>
                
                <button className="w-full p-3 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded-lg hover:bg-green-100 dark:hover:bg-green-900/30 transition-colors text-sm font-medium">
                  <Shield className="w-4 h-4 inline mr-2" />
                  품질 기준 설정
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fade-in {
          0% { opacity: 0; transform: translateY(-10px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        
        .animate-fade-in {
          animation: fade-in 0.3s ease-out;
        }
      `}</style>
    </div>
  )
}