'use client'

import { useState, useRef, useEffect } from 'react'
import Link from 'next/link'
import { 
  ArrowLeft,
  Upload,
  Scan,
  Brain,
  Activity,
  Layers,
  Grid3x3,
  Zap,
  Download,
  RefreshCw,
  Play,
  Pause,
  ChevronRight,
  AlertCircle,
  CheckCircle,
  Settings,
  Eye,
  EyeOff,
  Loader2,
  BarChart3,
  Target,
  Cpu
} from 'lucide-react'

interface ModelConfig {
  name: string
  accuracy: number
  speed: number
  params: string
  architecture: string
}

const models: ModelConfig[] = [
  { name: 'ResNet-50', accuracy: 96.2, speed: 45, params: '25.6M', architecture: 'CNN' },
  { name: 'DenseNet-121', accuracy: 97.1, speed: 32, params: '8.1M', architecture: 'CNN' },
  { name: 'EfficientNet-B7', accuracy: 98.1, speed: 78, params: '66M', architecture: 'CNN' },
  { name: 'Vision Transformer', accuracy: 98.3, speed: 112, params: '86M', architecture: 'Transformer' },
  { name: 'Swin Transformer', accuracy: 98.5, speed: 95, params: '88M', architecture: 'Transformer' }
]

const diseases = [
  { name: '정상', probability: 0, color: 'bg-green-500' },
  { name: '폐렴', probability: 0, color: 'bg-red-500' },
  { name: '결핵', probability: 0, color: 'bg-orange-500' },
  { name: '폐부종', probability: 0, color: 'bg-blue-500' },
  { name: '기흉', probability: 0, color: 'bg-purple-500' },
  { name: '무기폐', probability: 0, color: 'bg-yellow-500' },
  { name: '심비대', probability: 0, color: 'bg-pink-500' },
  { name: '종괴', probability: 0, color: 'bg-indigo-500' }
]

export default function MedicalImagingAdvancedPage() {
  const [selectedModel, setSelectedModel] = useState(models[0])
  const [isProcessing, setIsProcessing] = useState(false)
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [analysisResults, setAnalysisResults] = useState<typeof diseases | null>(null)
  const [showHeatmap, setShowHeatmap] = useState(false)
  const [showSegmentation, setShowSegmentation] = useState(false)
  const [confidence, setConfidence] = useState(0)
  const [processingStage, setProcessingStage] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const heatmapCanvasRef = useRef<HTMLCanvasElement>(null)
  const segmentationCanvasRef = useRef<HTMLCanvasElement>(null)

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string)
        setAnalysisResults(null)
        setShowHeatmap(false)
        setShowSegmentation(false)
      }
      reader.readAsDataURL(file)
    }
  }

  const processImage = async () => {
    if (!selectedImage) return
    
    setIsProcessing(true)
    setProcessingStage('전처리 중...')
    
    // 단계별 처리 시뮬레이션
    const stages = [
      '전처리 중...',
      'DICOM 메타데이터 추출...',
      '정규화 처리...',
      '모델 로딩...',
      '특징 추출...',
      '분류 진행...',
      '후처리 중...'
    ]
    
    for (const stage of stages) {
      setProcessingStage(stage)
      await new Promise(resolve => setTimeout(resolve, 500))
    }
    
    // 랜덤 결과 생성
    const results = diseases.map(disease => ({
      ...disease,
      probability: disease.name === '정상' 
        ? Math.random() * 30 + 60  // 정상일 확률 높게
        : Math.random() * 20
    }))
    
    // 정규화
    const total = results.reduce((sum, d) => sum + d.probability, 0)
    const normalizedResults = results.map(d => ({
      ...d,
      probability: (d.probability / total) * 100
    }))
    
    setAnalysisResults(normalizedResults.sort((a, b) => b.probability - a.probability))
    setConfidence(85 + Math.random() * 10)
    
    setIsProcessing(false)
    setProcessingStage('')
  }

  const generateHeatmap = () => {
    if (!canvasRef.current || !heatmapCanvasRef.current || !selectedImage) return
    
    const canvas = heatmapCanvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.src = selectedImage
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)
      
      // Grad-CAM 시뮬레이션
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
      const data = imageData.data
      
      // 중심부에 높은 활성화 영역 생성
      const centerX = canvas.width / 2
      const centerY = canvas.height / 2
      const radius = Math.min(canvas.width, canvas.height) / 3
      
      for (let i = 0; i < data.length; i += 4) {
        const x = (i / 4) % canvas.width
        const y = Math.floor((i / 4) / canvas.width)
        
        const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2)
        const intensity = Math.max(0, 1 - distance / radius)
        
        if (intensity > 0.3) {
          // 빨간색 오버레이
          data[i] = Math.min(255, data[i] + intensity * 200) // Red
          data[i + 1] = Math.max(0, data[i + 1] - intensity * 50) // Green
          data[i + 2] = Math.max(0, data[i + 2] - intensity * 50) // Blue
          data[i + 3] = Math.min(255, 180 + intensity * 75) // Alpha
        }
      }
      
      ctx.putImageData(imageData, 0, 0)
      setShowHeatmap(true)
      setShowSegmentation(false)
    }
  }

  const generateSegmentation = () => {
    if (!canvasRef.current || !segmentationCanvasRef.current || !selectedImage) return
    
    const canvas = segmentationCanvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.src = selectedImage
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height
      
      // U-Net 세그멘테이션 시뮬레이션
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      
      // 폐 영역 마스크 (간단한 타원형)
      ctx.globalCompositeOperation = 'destination-out'
      ctx.beginPath()
      ctx.ellipse(canvas.width * 0.3, canvas.height * 0.5, 80, 120, 0, 0, 2 * Math.PI)
      ctx.fill()
      ctx.beginPath()
      ctx.ellipse(canvas.width * 0.7, canvas.height * 0.5, 80, 120, 0, 0, 2 * Math.PI)
      ctx.fill()
      
      // 컬러 오버레이
      ctx.globalCompositeOperation = 'source-over'
      ctx.fillStyle = 'rgba(0, 255, 0, 0.3)'
      ctx.beginPath()
      ctx.ellipse(canvas.width * 0.3, canvas.height * 0.5, 80, 120, 0, 0, 2 * Math.PI)
      ctx.fill()
      ctx.fillStyle = 'rgba(0, 200, 255, 0.3)'
      ctx.beginPath()
      ctx.ellipse(canvas.width * 0.7, canvas.height * 0.5, 80, 120, 0, 0, 2 * Math.PI)
      ctx.fill()
      
      setShowSegmentation(true)
      setShowHeatmap(false)
    }
  }

  const reset = () => {
    setSelectedImage(null)
    setAnalysisResults(null)
    setShowHeatmap(false)
    setShowSegmentation(false)
    setConfidence(0)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <header className="sticky top-0 z-30 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/medical-ai"
                className="flex items-center gap-2 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>돌아가기</span>
              </Link>
              <div className="h-6 w-px bg-gray-300 dark:bg-gray-700"></div>
              <h1 className="text-xl font-semibold text-gray-900 dark:text-white">
                Advanced Medical Imaging Analysis
              </h1>
            </div>
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 rounded-full text-sm font-medium">
                심화 시뮬레이터
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Left Panel - Image & Controls */}
          <div className="lg:col-span-2 space-y-6">
            {/* Model Selection */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
                <Cpu className="w-5 h-5 text-purple-500" />
                AI 모델 선택
              </h3>
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                {models.map(model => (
                  <button
                    key={model.name}
                    onClick={() => setSelectedModel(model)}
                    className={`p-3 rounded-lg border-2 transition-all ${
                      selectedModel.name === model.name
                        ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                    }`}
                  >
                    <div className="text-sm font-medium text-gray-900 dark:text-white">
                      {model.name}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {model.architecture}
                    </div>
                    <div className="flex justify-between mt-2 text-xs">
                      <span className="text-green-600 dark:text-green-400">
                        {model.accuracy}%
                      </span>
                      <span className="text-blue-600 dark:text-blue-400">
                        {model.speed}ms
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Image Upload & Display */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
                의료 영상
              </h3>
              
              {!selectedImage ? (
                <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-12">
                  <div className="text-center">
                    <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                      의료 영상 업로드
                    </h3>
                    <p className="text-gray-500 dark:text-gray-400 mb-6">
                      X-Ray, CT, MRI 지원 (DICOM, JPG, PNG)
                    </p>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      className="hidden"
                    />
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-medium hover:shadow-lg transition-all"
                    >
                      파일 선택
                    </button>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative bg-gray-900 rounded-lg overflow-hidden">
                    <canvas ref={canvasRef} className="hidden" />
                    
                    {/* Original Image */}
                    <img 
                      src={selectedImage} 
                      alt="Medical Image" 
                      className={`w-full h-auto ${(showHeatmap || showSegmentation) ? 'opacity-50' : ''}`}
                    />
                    
                    {/* Heatmap Overlay */}
                    {showHeatmap && (
                      <canvas 
                        ref={heatmapCanvasRef} 
                        className="absolute inset-0 w-full h-full"
                        style={{ mixBlendMode: 'multiply' }}
                      />
                    )}
                    
                    {/* Segmentation Overlay */}
                    {showSegmentation && (
                      <canvas 
                        ref={segmentationCanvasRef} 
                        className="absolute inset-0 w-full h-full"
                      />
                    )}
                    
                    {/* Processing Overlay */}
                    {isProcessing && (
                      <div className="absolute inset-0 bg-black/70 flex flex-col items-center justify-center">
                        <Loader2 className="w-12 h-12 text-white animate-spin mb-4" />
                        <p className="text-white font-medium">{processingStage}</p>
                      </div>
                    )}
                  </div>
                  
                  {/* Control Buttons */}
                  <div className="flex gap-2">
                    <button
                      onClick={processImage}
                      disabled={isProcessing}
                      className="flex-1 px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-medium hover:shadow-lg transition-all disabled:opacity-50"
                    >
                      <Brain className="w-5 h-5 inline mr-2" />
                      AI 분석 시작
                    </button>
                    {analysisResults && (
                      <>
                        <button
                          onClick={generateHeatmap}
                          className={`px-4 py-2 rounded-lg font-medium transition-all ${
                            showHeatmap 
                              ? 'bg-red-600 text-white' 
                              : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                          }`}
                        >
                          <Activity className="w-5 h-5" />
                        </button>
                        <button
                          onClick={generateSegmentation}
                          className={`px-4 py-2 rounded-lg font-medium transition-all ${
                            showSegmentation 
                              ? 'bg-blue-600 text-white' 
                              : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                          }`}
                        >
                          <Layers className="w-5 h-5" />
                        </button>
                      </>
                    )}
                    <button
                      onClick={reset}
                      className="px-4 py-2 bg-gray-600 text-white rounded-lg font-medium hover:shadow-lg transition-all"
                    >
                      <RefreshCw className="w-5 h-5" />
                    </button>
                  </div>
                  
                  {/* Visualization Options */}
                  {analysisResults && (
                    <div className="flex gap-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                      <div className="flex items-center gap-2">
                        <Eye className="w-4 h-4 text-gray-500" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">시각화:</span>
                      </div>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={showHeatmap}
                          onChange={(e) => {
                            if (e.target.checked) generateHeatmap()
                            else setShowHeatmap(false)
                          }}
                          className="rounded"
                        />
                        <span className="text-sm text-gray-700 dark:text-gray-300">Grad-CAM</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={showSegmentation}
                          onChange={(e) => {
                            if (e.target.checked) generateSegmentation()
                            else setShowSegmentation(false)
                          }}
                          className="rounded"
                        />
                        <span className="text-sm text-gray-700 dark:text-gray-300">Segmentation</span>
                      </label>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Right Panel - Results */}
          <div className="space-y-6">
            {/* Model Info */}
            <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
                현재 모델: {selectedModel.name}
              </h3>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">아키텍처</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {selectedModel.architecture}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">정확도</span>
                  <span className="font-medium text-green-600 dark:text-green-400">
                    {selectedModel.accuracy}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">처리 속도</span>
                  <span className="font-medium text-blue-600 dark:text-blue-400">
                    {selectedModel.speed}ms
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">파라미터</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {selectedModel.params}
                  </span>
                </div>
              </div>
            </div>

            {/* Analysis Results */}
            {analysisResults && (
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
                <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
                  분석 결과
                </h3>
                
                {/* Confidence Score */}
                <div className="mb-6">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
                      전체 신뢰도
                    </span>
                    <span className="text-2xl font-bold text-gray-900 dark:text-white">
                      {confidence.toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className={`h-full transition-all duration-1000 ${
                        confidence > 90 
                          ? 'bg-gradient-to-r from-green-500 to-emerald-600'
                          : confidence > 75
                          ? 'bg-gradient-to-r from-yellow-500 to-orange-600'
                          : 'bg-gradient-to-r from-red-500 to-pink-600'
                      }`}
                      style={{ width: `${confidence}%` }}
                    />
                  </div>
                </div>
                
                {/* Disease Probabilities */}
                <div className="space-y-3">
                  <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
                    질병별 확률
                  </h4>
                  {analysisResults.map((disease, idx) => (
                    <div key={idx} className="space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                          {disease.name}
                        </span>
                        <span className="text-sm font-bold text-gray-900 dark:text-white">
                          {disease.probability.toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                        <div 
                          className={`h-full transition-all duration-1000 ${disease.color}`}
                          style={{ width: `${disease.probability}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
                
                {/* Top Finding */}
                {analysisResults[0].probability > 30 && (
                  <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                    <div className="flex items-start gap-3">
                      <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
                      <div>
                        <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">
                          주요 소견
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {analysisResults[0].name}의 가능성이 가장 높습니다.
                          전문의 확인이 필요합니다.
                        </p>
                      </div>
                    </div>
                  </div>
                )}
                
                {/* Export Results */}
                <button className="w-full mt-4 px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-medium hover:bg-gray-200 dark:hover:bg-gray-600 transition-all">
                  <Download className="w-5 h-5 inline mr-2" />
                  상세 리포트 다운로드
                </button>
              </div>
            )}

            {/* Feature Importance */}
            {analysisResults && (
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
                <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-blue-500" />
                  주요 특징 영역
                </h3>
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <div className="w-4 h-4 bg-red-500 rounded"></div>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      높은 활성화 (병변 의심)
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-4 h-4 bg-yellow-500 rounded"></div>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      중간 활성화 (관찰 필요)
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-4 h-4 bg-green-500 rounded"></div>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      낮은 활성화 (정상)
                    </span>
                  </div>
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-500 mt-4">
                  * Grad-CAM을 통해 모델이 주목한 영역을 시각화합니다
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Bottom Section - Technical Details */}
        <div className="mt-8 grid md:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h4 className="font-semibold mb-3 text-gray-900 dark:text-white flex items-center gap-2">
              <Grid3x3 className="w-5 h-5 text-green-500" />
              전처리 파이프라인
            </h4>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                DICOM to Array 변환
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                Windowing (Level/Width)
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                CLAHE 히스토그램 균등화
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                Normalization (0-1)
              </li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h4 className="font-semibold mb-3 text-gray-900 dark:text-white flex items-center gap-2">
              <Zap className="w-5 h-5 text-yellow-500" />
              데이터 증강
            </h4>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-gray-400" />
                Random Rotation (±15°)
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-gray-400" />
                Random Zoom (0.9-1.1x)
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-gray-400" />
                Horizontal Flip (p=0.5)
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-gray-400" />
                Brightness Adjustment
              </li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h4 className="font-semibold mb-3 text-gray-900 dark:text-white flex items-center gap-2">
              <Target className="w-5 h-5 text-red-500" />
              평가 메트릭
            </h4>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li className="flex justify-between">
                <span>AUC-ROC</span>
                <span className="font-mono">0.984</span>
              </li>
              <li className="flex justify-between">
                <span>Sensitivity</span>
                <span className="font-mono">94.2%</span>
              </li>
              <li className="flex justify-between">
                <span>Specificity</span>
                <span className="font-mono">96.8%</span>
              </li>
              <li className="flex justify-between">
                <span>F1-Score</span>
                <span className="font-mono">0.951</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}