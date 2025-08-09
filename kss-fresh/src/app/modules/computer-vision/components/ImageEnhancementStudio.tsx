'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { 
  Upload, Download, RotateCcw, Sliders, 
  Sun, Contrast, Palette, Droplets, Image,
  Maximize2, Minimize2, Eye, EyeOff, Layers,
  Zap, Sparkles, Filter, Grid3X3, Move
} from 'lucide-react'

interface FilterSettings {
  brightness: number
  contrast: number
  saturation: number
  blur: number
  sharpen: number
  grayscale: number
  sepia: number
  hueRotate: number
  noise: number
  denoise: number
}

interface HistogramData {
  red: number[]
  green: number[]
  blue: number[]
  luminance: number[]
}

type PresetFilter = 'none' | 'vintage' | 'noir' | 'vivid' | 'cool' | 'warm' | 'dramatic' | 'soft'

export default function ImageEnhancementStudio() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [processedImage, setProcessedImage] = useState<string | null>(null)
  const [presetFilter, setPresetFilter] = useState<PresetFilter>('none')
  const [showOriginal, setShowOriginal] = useState(true)
  const [showProcessed, setShowProcessed] = useState(true)
  const [compareMode, setCompareMode] = useState<'side-by-side' | 'overlay' | 'split'>('side-by-side')
  const [isProcessing, setIsProcessing] = useState(false)
  const [histogram, setHistogram] = useState<HistogramData | null>(null)
  
  const [filters, setFilters] = useState<FilterSettings>({
    brightness: 0,
    contrast: 0,
    saturation: 0,
    blur: 0,
    sharpen: 0,
    grayscale: 0,
    sepia: 0,
    hueRotate: 0,
    noise: 0,
    denoise: 0
  })

  const fileInputRef = useRef<HTMLInputElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const originalCanvasRef = useRef<HTMLCanvasElement>(null)
  const histogramCanvasRef = useRef<HTMLCanvasElement>(null)

  // 프리셋 필터 정의
  const presetConfigs: Record<PresetFilter, Partial<FilterSettings>> = {
    none: {},
    vintage: { sepia: 30, contrast: 10, saturation: -20, brightness: 5 },
    noir: { grayscale: 100, contrast: 30, brightness: -10 },
    vivid: { saturation: 50, contrast: 20, sharpen: 30 },
    cool: { hueRotate: 180, saturation: -10, brightness: 5 },
    warm: { hueRotate: 30, saturation: 20, brightness: 10 },
    dramatic: { contrast: 40, saturation: 30, sharpen: 20, brightness: -5 },
    soft: { blur: 2, brightness: 10, contrast: -10 }
  }

  // 이미지 업로드 처리
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader()
      reader.onload = (event) => {
        setSelectedImage(event.target?.result as string)
        resetFilters()
      }
      reader.readAsDataURL(file)
    }
  }

  // 드래그 앤 드롭
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader()
      reader.onload = (event) => {
        setSelectedImage(event.target?.result as string)
        resetFilters()
      }
      reader.readAsDataURL(file)
    }
  }

  // 필터 리셋
  const resetFilters = () => {
    setFilters({
      brightness: 0,
      contrast: 0,
      saturation: 0,
      blur: 0,
      sharpen: 0,
      grayscale: 0,
      sepia: 0,
      hueRotate: 0,
      noise: 0,
      denoise: 0
    })
    setPresetFilter('none')
  }

  // 프리셋 적용
  const applyPreset = (preset: PresetFilter) => {
    setPresetFilter(preset)
    if (preset === 'none') {
      resetFilters()
    } else {
      setFilters(prev => ({
        ...prev,
        ...presetConfigs[preset]
      }))
    }
  }

  // 이미지 처리
  const processImage = useCallback(() => {
    if (!selectedImage || !canvasRef.current || !originalCanvasRef.current) return
    
    setIsProcessing(true)
    
    const img = new window.Image()
    img.onload = () => {
      const canvas = canvasRef.current!
      const originalCanvas = originalCanvasRef.current!
      const ctx = canvas.getContext('2d')!
      const originalCtx = originalCanvas.getContext('2d')!
      
      // 캔버스 크기 설정
      canvas.width = img.width
      canvas.height = img.height
      originalCanvas.width = img.width
      originalCanvas.height = img.height
      
      // 원본 이미지 그리기
      originalCtx.drawImage(img, 0, 0)
      ctx.drawImage(img, 0, 0)
      
      // 이미지 데이터 가져오기
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
      const data = imageData.data
      
      // 필터 적용
      applyFilters(data, canvas.width, canvas.height)
      
      // 처리된 이미지 데이터 적용
      ctx.putImageData(imageData, 0, 0)
      
      // 히스토그램 계산
      calculateHistogram(data, canvas.width, canvas.height)
      
      // 처리된 이미지 저장
      setProcessedImage(canvas.toDataURL())
      setIsProcessing(false)
    }
    img.src = selectedImage
  }, [selectedImage, filters])

  // 필터 적용 함수
  const applyFilters = (data: Uint8ClampedArray, width: number, height: number) => {
    const tempData = new Uint8ClampedArray(data)
    
    // 각 픽셀에 대해 필터 적용
    for (let i = 0; i < data.length; i += 4) {
      let r = data[i]
      let g = data[i + 1]
      let b = data[i + 2]
      
      // 밝기 조정
      r = Math.min(255, Math.max(0, r + filters.brightness * 2.55))
      g = Math.min(255, Math.max(0, g + filters.brightness * 2.55))
      b = Math.min(255, Math.max(0, b + filters.brightness * 2.55))
      
      // 대비 조정
      const factor = (259 * (filters.contrast + 255)) / (255 * (259 - filters.contrast))
      r = Math.min(255, Math.max(0, factor * (r - 128) + 128))
      g = Math.min(255, Math.max(0, factor * (g - 128) + 128))
      b = Math.min(255, Math.max(0, factor * (b - 128) + 128))
      
      // 채도 조정
      const gray = 0.299 * r + 0.587 * g + 0.114 * b
      r = Math.min(255, Math.max(0, gray + (r - gray) * (1 + filters.saturation / 100)))
      g = Math.min(255, Math.max(0, gray + (g - gray) * (1 + filters.saturation / 100)))
      b = Math.min(255, Math.max(0, gray + (b - gray) * (1 + filters.saturation / 100)))
      
      // 그레이스케일
      if (filters.grayscale > 0) {
        const grayValue = 0.299 * r + 0.587 * g + 0.114 * b
        r = r + (grayValue - r) * (filters.grayscale / 100)
        g = g + (grayValue - g) * (filters.grayscale / 100)
        b = b + (grayValue - b) * (filters.grayscale / 100)
      }
      
      // 세피아
      if (filters.sepia > 0) {
        const tr = 0.393 * r + 0.769 * g + 0.189 * b
        const tg = 0.349 * r + 0.686 * g + 0.168 * b
        const tb = 0.272 * r + 0.534 * g + 0.131 * b
        const amount = filters.sepia / 100
        r = r + (tr - r) * amount
        g = g + (tg - g) * amount
        b = b + (tb - b) * amount
      }
      
      // 색조 회전
      if (filters.hueRotate !== 0) {
        const hsl = rgbToHsl(r, g, b)
        hsl[0] = (hsl[0] + filters.hueRotate) % 360
        const rgb = hslToRgb(hsl[0], hsl[1], hsl[2])
        r = rgb[0]
        g = rgb[1]
        b = rgb[2]
      }
      
      // 노이즈 추가
      if (filters.noise > 0) {
        const noiseAmount = filters.noise * 2.55
        r = Math.min(255, Math.max(0, r + (Math.random() - 0.5) * noiseAmount))
        g = Math.min(255, Math.max(0, g + (Math.random() - 0.5) * noiseAmount))
        b = Math.min(255, Math.max(0, b + (Math.random() - 0.5) * noiseAmount))
      }
      
      data[i] = r
      data[i + 1] = g
      data[i + 2] = b
    }
    
    // 블러 효과
    if (filters.blur > 0) {
      applyBoxBlur(data, width, height, Math.round(filters.blur))
    }
    
    // 샤프닝 효과
    if (filters.sharpen > 0) {
      applySharpen(data, width, height, filters.sharpen / 100)
    }
    
    // 노이즈 제거
    if (filters.denoise > 0) {
      applyDenoise(data, width, height, filters.denoise / 100)
    }
  }

  // 박스 블러 구현
  const applyBoxBlur = (data: Uint8ClampedArray, width: number, height: number, radius: number) => {
    const output = new Uint8ClampedArray(data)
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let r = 0, g = 0, b = 0, count = 0
        
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const nx = x + dx
            const ny = y + dy
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              const idx = (ny * width + nx) * 4
              r += output[idx]
              g += output[idx + 1]
              b += output[idx + 2]
              count++
            }
          }
        }
        
        const idx = (y * width + x) * 4
        data[idx] = r / count
        data[idx + 1] = g / count
        data[idx + 2] = b / count
      }
    }
  }

  // 샤프닝 구현
  const applySharpen = (data: Uint8ClampedArray, width: number, height: number, amount: number) => {
    const kernel = [
      0, -1, 0,
      -1, 5, -1,
      0, -1, 0
    ]
    
    const output = new Uint8ClampedArray(data)
    
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let r = 0, g = 0, b = 0
        
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const idx = ((y + ky) * width + (x + kx)) * 4
            const weight = kernel[(ky + 1) * 3 + (kx + 1)]
            r += output[idx] * weight
            g += output[idx + 1] * weight
            b += output[idx + 2] * weight
          }
        }
        
        const idx = (y * width + x) * 4
        data[idx] = Math.min(255, Math.max(0, output[idx] + (r - output[idx]) * amount))
        data[idx + 1] = Math.min(255, Math.max(0, output[idx + 1] + (g - output[idx + 1]) * amount))
        data[idx + 2] = Math.min(255, Math.max(0, output[idx + 2] + (b - output[idx + 2]) * amount))
      }
    }
  }

  // 노이즈 제거 (미디언 필터)
  const applyDenoise = (data: Uint8ClampedArray, width: number, height: number, amount: number) => {
    const output = new Uint8ClampedArray(data)
    const radius = 1
    
    for (let y = radius; y < height - radius; y++) {
      for (let x = radius; x < width - radius; x++) {
        const rValues: number[] = []
        const gValues: number[] = []
        const bValues: number[] = []
        
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const idx = ((y + dy) * width + (x + dx)) * 4
            rValues.push(output[idx])
            gValues.push(output[idx + 1])
            bValues.push(output[idx + 2])
          }
        }
        
        rValues.sort((a, b) => a - b)
        gValues.sort((a, b) => a - b)
        bValues.sort((a, b) => a - b)
        
        const mid = Math.floor(rValues.length / 2)
        const idx = (y * width + x) * 4
        
        data[idx] = output[idx] + (rValues[mid] - output[idx]) * amount
        data[idx + 1] = output[idx + 1] + (gValues[mid] - output[idx + 1]) * amount
        data[idx + 2] = output[idx + 2] + (bValues[mid] - output[idx + 2]) * amount
      }
    }
  }

  // RGB to HSL 변환
  const rgbToHsl = (r: number, g: number, b: number): [number, number, number] => {
    r /= 255
    g /= 255
    b /= 255
    
    const max = Math.max(r, g, b)
    const min = Math.min(r, g, b)
    let h = 0, s = 0
    const l = (max + min) / 2
    
    if (max !== min) {
      const d = max - min
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
      
      switch (max) {
        case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break
        case g: h = ((b - r) / d + 2) / 6; break
        case b: h = ((r - g) / d + 4) / 6; break
      }
    }
    
    return [h * 360, s, l]
  }

  // HSL to RGB 변환
  const hslToRgb = (h: number, s: number, l: number): [number, number, number] => {
    h /= 360
    
    let r, g, b
    
    if (s === 0) {
      r = g = b = l
    } else {
      const hue2rgb = (p: number, q: number, t: number) => {
        if (t < 0) t += 1
        if (t > 1) t -= 1
        if (t < 1/6) return p + (q - p) * 6 * t
        if (t < 1/2) return q
        if (t < 2/3) return p + (q - p) * (2/3 - t) * 6
        return p
      }
      
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s
      const p = 2 * l - q
      r = hue2rgb(p, q, h + 1/3)
      g = hue2rgb(p, q, h)
      b = hue2rgb(p, q, h - 1/3)
    }
    
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]
  }

  // 히스토그램 계산
  const calculateHistogram = (data: Uint8ClampedArray, width: number, height: number) => {
    const histogram: HistogramData = {
      red: new Array(256).fill(0),
      green: new Array(256).fill(0),
      blue: new Array(256).fill(0),
      luminance: new Array(256).fill(0)
    }
    
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i]
      const g = data[i + 1]
      const b = data[i + 2]
      const l = Math.round(0.299 * r + 0.587 * g + 0.114 * b)
      
      histogram.red[r]++
      histogram.green[g]++
      histogram.blue[b]++
      histogram.luminance[l]++
    }
    
    setHistogram(histogram)
    drawHistogram(histogram)
  }

  // 히스토그램 그리기
  const drawHistogram = (histogram: HistogramData) => {
    const canvas = histogramCanvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')!
    canvas.width = 256
    canvas.height = 100
    
    ctx.fillStyle = '#1f2937'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    
    // 최대값 찾기
    const maxValue = Math.max(
      ...histogram.luminance,
      ...histogram.red,
      ...histogram.green,
      ...histogram.blue
    )
    
    // 각 채널 그리기
    const channels = [
      { data: histogram.luminance, color: 'rgba(255, 255, 255, 0.8)' },
      { data: histogram.red, color: 'rgba(255, 0, 0, 0.5)' },
      { data: histogram.green, color: 'rgba(0, 255, 0, 0.5)' },
      { data: histogram.blue, color: 'rgba(0, 0, 255, 0.5)' }
    ]
    
    channels.forEach(channel => {
      ctx.strokeStyle = channel.color
      ctx.beginPath()
      
      for (let i = 0; i < 256; i++) {
        const height = (channel.data[i] / maxValue) * canvas.height
        if (i === 0) {
          ctx.moveTo(i, canvas.height - height)
        } else {
          ctx.lineTo(i, canvas.height - height)
        }
      }
      
      ctx.stroke()
    })
  }

  // 이미지 다운로드
  const downloadImage = () => {
    if (!processedImage) return
    
    const a = document.createElement('a')
    a.href = processedImage
    a.download = 'enhanced_image.png'
    a.click()
  }

  // 이미지 처리 효과
  useEffect(() => {
    if (selectedImage) {
      processImage()
    }
  }, [selectedImage, filters, processImage])

  return (
    <div className="space-y-6">
      {/* 툴바 */}
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <div className="flex flex-wrap items-center gap-4">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 flex items-center gap-2"
          >
            <Upload className="w-4 h-4" />
            이미지 업로드
          </button>
          
          <button
            onClick={resetFilters}
            className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 flex items-center gap-2"
            disabled={!selectedImage}
          >
            <RotateCcw className="w-4 h-4" />
            초기화
          </button>
          
          <button
            onClick={downloadImage}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2"
            disabled={!processedImage}
          >
            <Download className="w-4 h-4" />
            다운로드
          </button>
          
          <div className="flex gap-2 ml-auto">
            <button
              onClick={() => setShowOriginal(!showOriginal)}
              className={`p-2 rounded ${showOriginal ? 'bg-teal-600 text-white' : 'bg-gray-200 dark:bg-gray-600'}`}
              title="원본 표시"
            >
              {showOriginal ? <Eye className="w-5 h-5" /> : <EyeOff className="w-5 h-5" />}
            </button>
            
            <select
              value={compareMode}
              onChange={(e) => setCompareMode(e.target.value as any)}
              className="px-3 py-2 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-800"
            >
              <option value="side-by-side">나란히 보기</option>
              <option value="overlay">오버레이</option>
              <option value="split">분할 보기</option>
            </select>
          </div>
        </div>
      </div>

      {/* 메인 컨텐츠 */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* 이미지 영역 */}
        <div className="lg:col-span-2">
          <div
            className="bg-gray-100 dark:bg-gray-900 rounded-lg overflow-hidden relative min-h-[400px] flex items-center justify-center"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            {!selectedImage ? (
              <div className="text-center text-gray-500 p-8">
                <Image className="w-16 h-16 mx-auto mb-4" />
                <p className="text-lg mb-2">이미지를 드래그하여 놓거나</p>
                <p>클릭하여 업로드하세요</p>
              </div>
            ) : (
              <div className={`w-full ${compareMode === 'side-by-side' ? 'flex' : 'relative'}`}>
                {showOriginal && (
                  <div className={compareMode === 'side-by-side' ? 'w-1/2' : 'absolute inset-0'}>
                    <canvas
                      ref={originalCanvasRef}
                      className="max-w-full h-auto"
                      style={{ opacity: compareMode === 'overlay' ? 0.5 : 1 }}
                    />
                    {compareMode === 'side-by-side' && (
                      <div className="text-center text-sm mt-2 text-gray-600">원본</div>
                    )}
                  </div>
                )}
                
                <div className={compareMode === 'side-by-side' ? 'w-1/2' : 'relative'}>
                  <canvas
                    ref={canvasRef}
                    className="max-w-full h-auto"
                  />
                  {compareMode === 'side-by-side' && (
                    <div className="text-center text-sm mt-2 text-gray-600">처리됨</div>
                  )}
                </div>
              </div>
            )}
            
            {isProcessing && (
              <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                <div className="text-white">처리 중...</div>
              </div>
            )}
          </div>

          {/* 히스토그램 */}
          {histogram && (
            <div className="mt-4 bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold mb-3">히스토그램</h3>
              <canvas
                ref={histogramCanvasRef}
                className="w-full"
                style={{ height: '100px' }}
              />
            </div>
          )}
        </div>

        {/* 컨트롤 패널 */}
        <div className="space-y-4">
          {/* 프리셋 필터 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-teal-600" />
              프리셋 필터
            </h3>
            
            <div className="grid grid-cols-2 gap-2">
              {(['none', 'vintage', 'noir', 'vivid', 'cool', 'warm', 'dramatic', 'soft'] as PresetFilter[]).map(preset => (
                <button
                  key={preset}
                  onClick={() => applyPreset(preset)}
                  className={`px-3 py-2 rounded text-sm ${
                    presetFilter === preset
                      ? 'bg-teal-600 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  {preset === 'none' ? '없음' : preset.charAt(0).toUpperCase() + preset.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* 기본 조정 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Sliders className="w-5 h-5 text-teal-600" />
              기본 조정
            </h3>
            
            <div className="space-y-3">
              <div>
                <label className="flex items-center justify-between text-sm mb-1">
                  <span className="flex items-center gap-1">
                    <Sun className="w-4 h-4" /> 밝기
                  </span>
                  <span>{filters.brightness}</span>
                </label>
                <input
                  type="range"
                  min="-100"
                  max="100"
                  value={filters.brightness}
                  onChange={(e) => setFilters(prev => ({ ...prev, brightness: Number(e.target.value) }))}
                  className="w-full"
                />
              </div>
              
              <div>
                <label className="flex items-center justify-between text-sm mb-1">
                  <span className="flex items-center gap-1">
                    <Contrast className="w-4 h-4" /> 대비
                  </span>
                  <span>{filters.contrast}</span>
                </label>
                <input
                  type="range"
                  min="-100"
                  max="100"
                  value={filters.contrast}
                  onChange={(e) => setFilters(prev => ({ ...prev, contrast: Number(e.target.value) }))}
                  className="w-full"
                />
              </div>
              
              <div>
                <label className="flex items-center justify-between text-sm mb-1">
                  <span className="flex items-center gap-1">
                    <Palette className="w-4 h-4" /> 채도
                  </span>
                  <span>{filters.saturation}</span>
                </label>
                <input
                  type="range"
                  min="-100"
                  max="100"
                  value={filters.saturation}
                  onChange={(e) => setFilters(prev => ({ ...prev, saturation: Number(e.target.value) }))}
                  className="w-full"
                />
              </div>
            </div>
          </div>

          {/* 효과 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Filter className="w-5 h-5 text-teal-600" />
              효과
            </h3>
            
            <div className="space-y-3">
              <div>
                <label className="flex items-center justify-between text-sm mb-1">
                  <span>흑백</span>
                  <span>{filters.grayscale}%</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={filters.grayscale}
                  onChange={(e) => setFilters(prev => ({ ...prev, grayscale: Number(e.target.value) }))}
                  className="w-full"
                />
              </div>
              
              <div>
                <label className="flex items-center justify-between text-sm mb-1">
                  <span>세피아</span>
                  <span>{filters.sepia}%</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={filters.sepia}
                  onChange={(e) => setFilters(prev => ({ ...prev, sepia: Number(e.target.value) }))}
                  className="w-full"
                />
              </div>
              
              <div>
                <label className="flex items-center justify-between text-sm mb-1">
                  <span>색조 회전</span>
                  <span>{filters.hueRotate}°</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="360"
                  value={filters.hueRotate}
                  onChange={(e) => setFilters(prev => ({ ...prev, hueRotate: Number(e.target.value) }))}
                  className="w-full"
                />
              </div>
              
              <div>
                <label className="flex items-center justify-between text-sm mb-1">
                  <span>블러</span>
                  <span>{filters.blur}</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="10"
                  value={filters.blur}
                  onChange={(e) => setFilters(prev => ({ ...prev, blur: Number(e.target.value) }))}
                  className="w-full"
                />
              </div>
              
              <div>
                <label className="flex items-center justify-between text-sm mb-1">
                  <span>샤프닝</span>
                  <span>{filters.sharpen}%</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={filters.sharpen}
                  onChange={(e) => setFilters(prev => ({ ...prev, sharpen: Number(e.target.value) }))}
                  className="w-full"
                />
              </div>
            </div>
          </div>

          {/* 노이즈 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Grid3X3 className="w-5 h-5 text-teal-600" />
              노이즈
            </h3>
            
            <div className="space-y-3">
              <div>
                <label className="flex items-center justify-between text-sm mb-1">
                  <span>노이즈 추가</span>
                  <span>{filters.noise}%</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={filters.noise}
                  onChange={(e) => setFilters(prev => ({ ...prev, noise: Number(e.target.value) }))}
                  className="w-full"
                />
              </div>
              
              <div>
                <label className="flex items-center justify-between text-sm mb-1">
                  <span>노이즈 제거</span>
                  <span>{filters.denoise}%</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={filters.denoise}
                  onChange={(e) => setFilters(prev => ({ ...prev, denoise: Number(e.target.value) }))}
                  className="w-full"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleImageUpload}
        className="hidden"
      />
    </div>
  )
}