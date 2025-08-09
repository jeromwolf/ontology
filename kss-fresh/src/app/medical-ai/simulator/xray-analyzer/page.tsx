'use client'

import { useState, useRef, useEffect } from 'react'
import Link from 'next/link'
import { 
  ArrowLeft,
  Upload,
  Scan,
  AlertCircle,
  CheckCircle,
  XCircle,
  Download,
  RefreshCw,
  Zap,
  Brain,
  Activity,
  Info,
  Camera,
  FileImage,
  Loader2
} from 'lucide-react'

interface AnalysisResult {
  confidence: number
  findings: string[]
  recommendations: string[]
  heatmapData?: number[][]
}

const sampleXRays = [
  {
    id: 'normal',
    name: '정상 흉부',
    description: '이상 소견 없음',
    image: '/api/placeholder/400/400',
    result: {
      confidence: 98,
      findings: ['폐야 깨끗함', '심장 크기 정상', '횡격막 정상'],
      recommendations: ['정기 검진 권장']
    }
  },
  {
    id: 'pneumonia',
    name: '폐렴 의심',
    description: '우하엽 침윤',
    image: '/api/placeholder/400/400',
    result: {
      confidence: 92,
      findings: ['우하엽 폐렴 소견', '침윤 음영 관찰', '늑막 삼출 의심'],
      recommendations: ['항생제 치료 고려', '추적 X-ray 촬영', '혈액 검사 권장']
    }
  },
  {
    id: 'tuberculosis',
    name: '결핵 의심',
    description: '상엽 공동성 병변',
    image: '/api/placeholder/400/400',
    result: {
      confidence: 88,
      findings: ['우상엽 공동성 병변', '주변 침윤', '림프절 비대'],
      recommendations: ['결핵 검사 시행', '격리 고려', '접촉자 검진']
    }
  },
  {
    id: 'covid',
    name: 'COVID-19 의심',
    description: '양측 간질성 폐렴',
    image: '/api/placeholder/400/400',
    result: {
      confidence: 85,
      findings: ['양측 말초 간질성 음영', 'Ground-glass opacity', '하엽 우세'],
      recommendations: ['COVID-19 PCR 검사', '산소포화도 모니터링', '격리 치료']
    }
  }
]

export default function XRayAnalyzerPage() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [selectedSample, setSelectedSample] = useState<typeof sampleXRays[0] | null>(null)
  const [showHeatmap, setShowHeatmap] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string)
        setSelectedSample(null)
        setAnalysisResult(null)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleSampleSelect = (sample: typeof sampleXRays[0]) => {
    setSelectedSample(sample)
    setSelectedImage(sample.image)
    setAnalysisResult(null)
  }

  const analyzeImage = async () => {
    if (!selectedImage) return
    
    setIsAnalyzing(true)
    setShowHeatmap(false)
    
    // Simulate AI analysis
    await new Promise(resolve => setTimeout(resolve, 3000))
    
    if (selectedSample) {
      setAnalysisResult(selectedSample.result)
    } else {
      // Generic result for uploaded images
      setAnalysisResult({
        confidence: Math.floor(Math.random() * 20) + 80,
        findings: [
          '분석 완료',
          '추가 검토 필요',
          '전문의 확인 권장'
        ],
        recommendations: [
          '임상 정보와 대조 필요',
          '필요시 CT 촬영 고려'
        ]
      })
    }
    
    setIsAnalyzing(false)
  }

  const generateHeatmap = () => {
    if (!canvasRef.current || !selectedImage) return
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.src = selectedImage
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)
      
      // Apply heatmap overlay
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
      const data = imageData.data
      
      for (let i = 0; i < data.length; i += 4) {
        const x = (i / 4) % canvas.width
        const y = Math.floor((i / 4) / canvas.width)
        
        // Simple heatmap effect (customize based on actual analysis)
        const intensity = Math.sin(x * 0.01) * Math.cos(y * 0.01) * 128 + 128
        
        if (intensity > 150) {
          data[i] = Math.min(255, data[i] + intensity) // Red
          data[i + 1] = Math.max(0, data[i + 1] - intensity / 2) // Green
          data[i + 2] = Math.max(0, data[i + 2] - intensity / 2) // Blue
        }
      }
      
      ctx.putImageData(imageData, 0, 0)
      setShowHeatmap(true)
    }
  }

  const reset = () => {
    setSelectedImage(null)
    setSelectedSample(null)
    setAnalysisResult(null)
    setShowHeatmap(false)
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
                X-Ray Analyzer
              </h1>
            </div>
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-full text-sm font-medium">
                시뮬레이터
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Panel - Image Upload/Display */}
          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
                X-Ray 영상
              </h2>
              
              {!selectedImage ? (
                <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-12">
                  <div className="text-center">
                    <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                      X-Ray 영상 업로드
                    </h3>
                    <p className="text-gray-500 dark:text-gray-400 mb-6">
                      DICOM, JPG, PNG 형식 지원
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
                      className="px-6 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-lg font-medium hover:shadow-lg transition-all"
                    >
                      <FileImage className="w-5 h-5 inline mr-2" />
                      파일 선택
                    </button>
                  </div>
                </div>
              ) : (
                <div className="relative">
                  <div className="relative bg-gray-900 rounded-lg overflow-hidden">
                    {showHeatmap ? (
                      <canvas ref={canvasRef} className="w-full h-auto" />
                    ) : (
                      <img 
                        src={selectedImage} 
                        alt="X-Ray" 
                        className="w-full h-auto"
                      />
                    )}
                    {isAnalyzing && (
                      <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                        <div className="text-center">
                          <Loader2 className="w-12 h-12 text-white animate-spin mx-auto mb-4" />
                          <p className="text-white font-medium">AI 분석 중...</p>
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="flex gap-2 mt-4">
                    <button
                      onClick={analyzeImage}
                      disabled={isAnalyzing}
                      className="flex-1 px-4 py-2 bg-gradient-to-r from-red-600 to-pink-600 text-white rounded-lg font-medium hover:shadow-lg transition-all disabled:opacity-50"
                    >
                      <Scan className="w-5 h-5 inline mr-2" />
                      분석 시작
                    </button>
                    {analysisResult && (
                      <button
                        onClick={generateHeatmap}
                        className="px-4 py-2 bg-purple-600 text-white rounded-lg font-medium hover:shadow-lg transition-all"
                      >
                        <Activity className="w-5 h-5 inline mr-2" />
                        Heatmap
                      </button>
                    )}
                    <button
                      onClick={reset}
                      className="px-4 py-2 bg-gray-600 text-white rounded-lg font-medium hover:shadow-lg transition-all"
                    >
                      <RefreshCw className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Sample Images */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
                샘플 X-Ray 영상
              </h3>
              <div className="grid grid-cols-2 gap-3">
                {sampleXRays.map(sample => (
                  <button
                    key={sample.id}
                    onClick={() => handleSampleSelect(sample)}
                    className={`p-3 rounded-lg border-2 transition-all ${
                      selectedSample?.id === sample.id
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                    }`}
                  >
                    <Camera className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                    <div className="text-sm font-medium text-gray-900 dark:text-white">
                      {sample.name}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {sample.description}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right Panel - Analysis Results */}
          <div className="space-y-6">
            {/* AI Model Info */}
            <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl p-6">
              <div className="flex items-start gap-4">
                <Brain className="w-8 h-8 text-blue-600 dark:text-blue-400" />
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                    AI 모델 정보
                  </h3>
                  <div className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                    <p>• 모델: ChestX-Ray14 CNN</p>
                    <p>• 학습 데이터: 112,120 X-Ray 영상</p>
                    <p>• 정확도: 95.2% (AUC 0.984)</p>
                    <p>• 검출 가능 질환: 14종</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Analysis Results */}
            {analysisResult && (
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
                <h3 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">
                  분석 결과
                </h3>

                {/* Confidence Score */}
                <div className="mb-6">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
                      신뢰도
                    </span>
                    <span className="text-2xl font-bold text-gray-900 dark:text-white">
                      {analysisResult.confidence}%
                    </span>
                  </div>
                  <div className="w-full h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className={`h-full transition-all duration-1000 ${
                        analysisResult.confidence > 90 
                          ? 'bg-gradient-to-r from-green-500 to-emerald-600'
                          : analysisResult.confidence > 70
                          ? 'bg-gradient-to-r from-yellow-500 to-orange-600'
                          : 'bg-gradient-to-r from-red-500 to-pink-600'
                      }`}
                      style={{ width: `${analysisResult.confidence}%` }}
                    />
                  </div>
                </div>

                {/* Findings */}
                <div className="mb-6">
                  <h4 className="text-lg font-semibold mb-3 text-gray-900 dark:text-white">
                    주요 소견
                  </h4>
                  <div className="space-y-2">
                    {analysisResult.findings.map((finding, idx) => (
                      <div key={idx} className="flex items-start gap-3 p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                        {finding.includes('정상') || finding.includes('깨끗') ? (
                          <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                        ) : finding.includes('의심') || finding.includes('소견') ? (
                          <AlertCircle className="w-5 h-5 text-yellow-500 mt-0.5" />
                        ) : (
                          <Info className="w-5 h-5 text-blue-500 mt-0.5" />
                        )}
                        <span className="text-gray-700 dark:text-gray-300">{finding}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Recommendations */}
                <div>
                  <h4 className="text-lg font-semibold mb-3 text-gray-900 dark:text-white">
                    권장 사항
                  </h4>
                  <div className="space-y-2">
                    {analysisResult.recommendations.map((rec, idx) => (
                      <div key={idx} className="flex items-start gap-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                        <Zap className="w-5 h-5 text-blue-500 mt-0.5" />
                        <span className="text-gray-700 dark:text-gray-300">{rec}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Export Button */}
                <button className="w-full mt-6 px-4 py-3 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-medium hover:bg-gray-200 dark:hover:bg-gray-600 transition-all">
                  <Download className="w-5 h-5 inline mr-2" />
                  분석 결과 다운로드
                </button>
              </div>
            )}

            {/* Disclaimer */}
            <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
                <div className="text-sm text-yellow-800 dark:text-yellow-300">
                  <p className="font-semibold mb-1">주의사항</p>
                  <p>
                    이 시뮬레이터는 교육 목적으로 제작되었습니다. 
                    실제 의료 진단에는 사용할 수 없으며, 반드시 전문 의료진의 판단이 필요합니다.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}