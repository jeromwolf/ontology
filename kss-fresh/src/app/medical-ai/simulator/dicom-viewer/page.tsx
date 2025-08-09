'use client'

import { useState, useRef, useEffect } from 'react'
import Link from 'next/link'
import { 
  ArrowLeft,
  Upload,
  Scan,
  ZoomIn,
  ZoomOut,
  Move,
  Maximize2,
  RotateCw,
  Ruler,
  Circle,
  Square,
  Download,
  Info,
  Layers,
  Settings,
  ChevronLeft,
  ChevronRight,
  Grid3x3,
  Contrast,
  Sun,
  Moon,
  RefreshCw,
  Play,
  Pause,
  SkipForward,
  SkipBack
} from 'lucide-react'

interface DicomMetadata {
  patientName: string
  patientID: string
  studyDate: string
  modality: string
  institution: string
  bodyPart: string
  sliceThickness: string
  pixelSpacing: string
  windowCenter: number
  windowWidth: number
}

interface MeasurementTool {
  type: 'ruler' | 'circle' | 'rectangle' | 'angle'
  active: boolean
}

export default function DicomViewerPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [currentSlice, setCurrentSlice] = useState(0)
  const [totalSlices, setTotalSlices] = useState(10)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [windowLevel, setWindowLevel] = useState(40)
  const [windowWidth, setWindowWidth] = useState(400)
  const [brightness, setBrightness] = useState(100)
  const [contrast, setContrast] = useState(100)
  const [isPlaying, setIsPlaying] = useState(false)
  const [activeTool, setActiveTool] = useState<string>('move')
  const [measurements, setMeasurements] = useState<any[]>([])
  const [showMetadata, setShowMetadata] = useState(true)
  const [showMeasurements, setShowMeasurements] = useState(true)
  
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // 샘플 메타데이터
  const metadata: DicomMetadata = {
    patientName: 'John Doe',
    patientID: 'P123456',
    studyDate: '2024-01-15',
    modality: 'CT',
    institution: 'Seoul Medical Center',
    bodyPart: 'CHEST',
    sliceThickness: '5.0 mm',
    pixelSpacing: '0.7 mm',
    windowCenter: 40,
    windowWidth: 400
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      // DICOM 파일 파싱 시뮬레이션
      setTotalSlices(Math.floor(Math.random() * 50) + 20)
      setCurrentSlice(0)
    }
  }

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev * 1.2, 5))
  }

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev / 1.2, 0.5))
  }

  const handleReset = () => {
    setZoom(1)
    setPan({ x: 0, y: 0 })
    setBrightness(100)
    setContrast(100)
    setWindowLevel(40)
    setWindowWidth(400)
  }

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying)
  }

  // 자동 재생
  useEffect(() => {
    if (isPlaying && selectedFile) {
      const interval = setInterval(() => {
        setCurrentSlice(prev => (prev + 1) % totalSlices)
      }, 100)
      return () => clearInterval(interval)
    }
  }, [isPlaying, selectedFile, totalSlices])

  // Canvas 렌더링
  useEffect(() => {
    if (canvasRef.current && selectedFile) {
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      // 샘플 이미지 렌더링 (실제로는 DICOM 데이터)
      canvas.width = 512
      canvas.height = 512
      
      // 배경
      ctx.fillStyle = '#000'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      
      // 샘플 의료 영상 시뮬레이션
      const centerX = canvas.width / 2
      const centerY = canvas.height / 2
      
      // 윈도우/레벨 적용 시뮬레이션
      ctx.filter = `brightness(${brightness}%) contrast(${contrast}%)`
      
      // 폐 영역 시뮬레이션
      ctx.fillStyle = `rgba(${windowLevel}, ${windowLevel}, ${windowLevel}, 0.8)`
      ctx.beginPath()
      ctx.ellipse(centerX - 100, centerY, 80, 120, 0, 0, 2 * Math.PI)
      ctx.fill()
      ctx.beginPath()
      ctx.ellipse(centerX + 100, centerY, 80, 120, 0, 0, 2 * Math.PI)
      ctx.fill()
      
      // 심장 영역
      ctx.fillStyle = `rgba(${windowLevel + 20}, ${windowLevel + 20}, ${windowLevel + 20}, 0.9)`
      ctx.beginPath()
      ctx.ellipse(centerX, centerY + 20, 60, 70, 0, 0, 2 * Math.PI)
      ctx.fill()
      
      // 척추
      ctx.fillStyle = `rgba(${windowLevel + 40}, ${windowLevel + 40}, ${windowLevel + 40}, 1)`
      ctx.fillRect(centerX - 20, centerY - 150, 40, 300)
      
      // 슬라이스 번호 표시
      ctx.filter = 'none'
      ctx.fillStyle = '#0f0'
      ctx.font = '14px monospace'
      ctx.fillText(`Slice: ${currentSlice + 1}/${totalSlices}`, 10, 20)
      
      // 측정 도구 그리기
      if (measurements.length > 0) {
        ctx.strokeStyle = '#0ff'
        ctx.lineWidth = 2
        measurements.forEach(m => {
          if (m.type === 'ruler') {
            ctx.beginPath()
            ctx.moveTo(m.start.x, m.start.y)
            ctx.lineTo(m.end.x, m.end.y)
            ctx.stroke()
            
            const distance = Math.sqrt(
              Math.pow(m.end.x - m.start.x, 2) + 
              Math.pow(m.end.y - m.start.y, 2)
            ) * 0.7 // 픽셀 간격 적용
            ctx.fillText(`${distance.toFixed(1)} mm`, m.end.x + 5, m.end.y)
          }
        })
      }
    }
  }, [selectedFile, currentSlice, windowLevel, windowWidth, brightness, contrast, measurements])

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="sticky top-0 z-30 bg-gray-900/90 backdrop-blur-xl border-b border-gray-800">
        <div className="max-w-full px-6 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/medical-ai"
                className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>돌아가기</span>
              </Link>
              <div className="h-6 w-px bg-gray-700"></div>
              <h1 className="text-xl font-semibold">
                DICOM Viewer
              </h1>
            </div>
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 bg-cyan-900/30 text-cyan-400 rounded-full text-sm font-medium">
                의료 영상 뷰어
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-60px)]">
        {/* Left Sidebar - Tools */}
        <div className="w-16 bg-gray-800 border-r border-gray-700 flex flex-col items-center py-4 gap-2">
          <button
            onClick={() => setActiveTool('move')}
            className={`p-3 rounded-lg transition-all ${
              activeTool === 'move' ? 'bg-cyan-600' : 'hover:bg-gray-700'
            }`}
            title="이동"
          >
            <Move className="w-5 h-5" />
          </button>
          <button
            onClick={() => setActiveTool('zoom')}
            className={`p-3 rounded-lg transition-all ${
              activeTool === 'zoom' ? 'bg-cyan-600' : 'hover:bg-gray-700'
            }`}
            title="확대/축소"
          >
            <ZoomIn className="w-5 h-5" />
          </button>
          <button
            onClick={() => setActiveTool('ruler')}
            className={`p-3 rounded-lg transition-all ${
              activeTool === 'ruler' ? 'bg-cyan-600' : 'hover:bg-gray-700'
            }`}
            title="거리 측정"
          >
            <Ruler className="w-5 h-5" />
          </button>
          <button
            onClick={() => setActiveTool('circle')}
            className={`p-3 rounded-lg transition-all ${
              activeTool === 'circle' ? 'bg-cyan-600' : 'hover:bg-gray-700'
            }`}
            title="원형 ROI"
          >
            <Circle className="w-5 h-5" />
          </button>
          <button
            onClick={() => setActiveTool('rectangle')}
            className={`p-3 rounded-lg transition-all ${
              activeTool === 'rectangle' ? 'bg-cyan-600' : 'hover:bg-gray-700'
            }`}
            title="사각형 ROI"
          >
            <Square className="w-5 h-5" />
          </button>
          <div className="h-px w-10 bg-gray-700 my-2"></div>
          <button
            onClick={handleReset}
            className="p-3 rounded-lg hover:bg-gray-700 transition-all"
            title="초기화"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
          <button
            className="p-3 rounded-lg hover:bg-gray-700 transition-all"
            title="다운로드"
          >
            <Download className="w-5 h-5" />
          </button>
        </div>

        {/* Main Viewer */}
        <div className="flex-1 flex">
          <div className="flex-1 relative bg-black" ref={containerRef}>
            {!selectedFile ? (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <Upload className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-300 mb-2">
                    DICOM 파일 업로드
                  </h3>
                  <p className="text-gray-500 mb-6">
                    .dcm, .dicom 파일 지원
                  </p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".dcm,.dicom,image/*"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="px-6 py-3 bg-cyan-600 text-white rounded-lg font-medium hover:bg-cyan-700 transition-all"
                  >
                    파일 선택
                  </button>
                  <button
                    onClick={() => {
                      // 샘플 데이터 로드
                      setSelectedFile(new File([], 'sample.dcm'))
                      setTotalSlices(30)
                      setCurrentSlice(0)
                    }}
                    className="ml-3 px-6 py-3 bg-gray-700 text-white rounded-lg font-medium hover:bg-gray-600 transition-all"
                  >
                    샘플 데이터
                  </button>
                </div>
              </div>
            ) : (
              <>
                {/* Canvas */}
                <div className="absolute inset-0 flex items-center justify-center overflow-hidden">
                  <canvas
                    ref={canvasRef}
                    className="max-w-full max-h-full"
                    style={{
                      transform: `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)`,
                      cursor: activeTool === 'move' ? 'move' : 'crosshair'
                    }}
                  />
                </div>

                {/* Top Controls */}
                <div className="absolute top-4 left-4 right-4 flex justify-between">
                  <div className="bg-gray-900/80 backdrop-blur rounded-lg px-4 py-2">
                    <span className="text-cyan-400 font-mono text-sm">
                      {metadata.patientName} | {metadata.patientID}
                    </span>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={handleZoomIn}
                      className="bg-gray-900/80 backdrop-blur p-2 rounded-lg hover:bg-gray-800/80"
                    >
                      <ZoomIn className="w-5 h-5" />
                    </button>
                    <button
                      onClick={handleZoomOut}
                      className="bg-gray-900/80 backdrop-blur p-2 rounded-lg hover:bg-gray-800/80"
                    >
                      <ZoomOut className="w-5 h-5" />
                    </button>
                    <button
                      className="bg-gray-900/80 backdrop-blur p-2 rounded-lg hover:bg-gray-800/80"
                    >
                      <Maximize2 className="w-5 h-5" />
                    </button>
                  </div>
                </div>

                {/* Bottom Controls - Slice Navigation */}
                <div className="absolute bottom-4 left-4 right-4">
                  <div className="bg-gray-900/90 backdrop-blur rounded-lg p-4">
                    <div className="flex items-center gap-4">
                      <button
                        onClick={() => setCurrentSlice(Math.max(0, currentSlice - 1))}
                        className="p-2 bg-gray-800 rounded hover:bg-gray-700"
                      >
                        <SkipBack className="w-4 h-4" />
                      </button>
                      <button
                        onClick={handlePlayPause}
                        className="p-2 bg-cyan-600 rounded hover:bg-cyan-700"
                      >
                        {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                      </button>
                      <button
                        onClick={() => setCurrentSlice(Math.min(totalSlices - 1, currentSlice + 1))}
                        className="p-2 bg-gray-800 rounded hover:bg-gray-700"
                      >
                        <SkipForward className="w-4 h-4" />
                      </button>
                      <div className="flex-1">
                        <input
                          type="range"
                          min="0"
                          max={totalSlices - 1}
                          value={currentSlice}
                          onChange={(e) => setCurrentSlice(parseInt(e.target.value))}
                          className="w-full"
                        />
                        <div className="flex justify-between text-xs text-gray-400 mt-1">
                          <span>1</span>
                          <span className="text-cyan-400 font-mono">
                            Slice {currentSlice + 1} / {totalSlices}
                          </span>
                          <span>{totalSlices}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Right Sidebar - Metadata & Controls */}
          {selectedFile && (
            <div className="w-80 bg-gray-800 border-l border-gray-700 overflow-y-auto">
              {/* Metadata */}
              <div className="p-4 border-b border-gray-700">
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <Info className="w-5 h-5 text-cyan-400" />
                  DICOM 정보
                </h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Study Date</span>
                    <span>{metadata.studyDate}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Modality</span>
                    <span className="text-cyan-400">{metadata.modality}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Institution</span>
                    <span>{metadata.institution}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Body Part</span>
                    <span>{metadata.bodyPart}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Slice Thickness</span>
                    <span>{metadata.sliceThickness}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Pixel Spacing</span>
                    <span>{metadata.pixelSpacing}</span>
                  </div>
                </div>
              </div>

              {/* Window/Level Controls */}
              <div className="p-4 border-b border-gray-700">
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <Contrast className="w-5 h-5 text-cyan-400" />
                  Window/Level
                </h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">Window Center</span>
                      <span>{windowLevel} HU</span>
                    </div>
                    <input
                      type="range"
                      min="-1000"
                      max="1000"
                      value={windowLevel}
                      onChange={(e) => setWindowLevel(parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">Window Width</span>
                      <span>{windowWidth} HU</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="4000"
                      value={windowWidth}
                      onChange={(e) => setWindowWidth(parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  
                  {/* Presets */}
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => { setWindowLevel(40); setWindowWidth(400) }}
                      className="px-3 py-2 bg-gray-700 rounded text-sm hover:bg-gray-600"
                    >
                      Lung
                    </button>
                    <button
                      onClick={() => { setWindowLevel(40); setWindowWidth(80) }}
                      className="px-3 py-2 bg-gray-700 rounded text-sm hover:bg-gray-600"
                    >
                      Brain
                    </button>
                    <button
                      onClick={() => { setWindowLevel(300); setWindowWidth(1500) }}
                      className="px-3 py-2 bg-gray-700 rounded text-sm hover:bg-gray-600"
                    >
                      Bone
                    </button>
                    <button
                      onClick={() => { setWindowLevel(50); setWindowWidth(350) }}
                      className="px-3 py-2 bg-gray-700 rounded text-sm hover:bg-gray-600"
                    >
                      Soft Tissue
                    </button>
                  </div>
                </div>
              </div>

              {/* Image Adjustments */}
              <div className="p-4 border-b border-gray-700">
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <Sun className="w-5 h-5 text-cyan-400" />
                  이미지 조정
                </h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">밝기</span>
                      <span>{brightness}%</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="200"
                      value={brightness}
                      onChange={(e) => setBrightness(parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">대비</span>
                      <span>{contrast}%</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="200"
                      value={contrast}
                      onChange={(e) => setContrast(parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </div>
              </div>

              {/* Measurements */}
              {showMeasurements && measurements.length > 0 && (
                <div className="p-4">
                  <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                    <Ruler className="w-5 h-5 text-cyan-400" />
                    측정값
                  </h3>
                  <div className="space-y-2 text-sm">
                    {measurements.map((m, idx) => (
                      <div key={idx} className="flex justify-between p-2 bg-gray-700 rounded">
                        <span>{m.type}</span>
                        <span className="text-cyan-400">{m.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}