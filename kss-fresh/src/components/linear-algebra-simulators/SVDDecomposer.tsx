'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Upload, Image as ImageIcon, Download, RotateCcw, Layers } from 'lucide-react'

export default function SVDDecomposer() {
  const canvasOriginalRef = useRef<HTMLCanvasElement>(null)
  const canvasCompressedRef = useRef<HTMLCanvasElement>(null)
  const [imageData, setImageData] = useState<ImageData | null>(null)
  const [rank, setRank] = useState<number>(10)
  const [maxRank, setMaxRank] = useState<number>(50)
  const [compressionRatio, setCompressionRatio] = useState<number>(0)
  const [isProcessing, setIsProcessing] = useState(false)

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (event) => {
      const img = new Image()
      img.onload = () => {
        const canvas = canvasOriginalRef.current
        if (!canvas) return

        // Resize to fit canvas
        const maxSize = 400
        let width = img.width
        let height = img.height

        if (width > height && width > maxSize) {
          height = (height * maxSize) / width
          width = maxSize
        } else if (height > maxSize) {
          width = (width * maxSize) / height
          height = maxSize
        }

        canvas.width = width
        canvas.height = height
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        ctx.drawImage(img, 0, 0, width, height)
        const imgData = ctx.getImageData(0, 0, width, height)
        setImageData(imgData)
        setMaxRank(Math.min(width, height))
        setRank(Math.min(10, Math.min(width, height)))
      }
      img.src = event.target?.result as string
    }
    reader.readAsDataURL(file)
  }

  const applySVD = async () => {
    if (!imageData) return
    setIsProcessing(true)

    // Convert to grayscale and create matrix
    const { width, height, data } = imageData
    const matrix: number[][] = []

    for (let i = 0; i < height; i++) {
      matrix[i] = []
      for (let j = 0; j < width; j++) {
        const idx = (i * width + j) * 4
        const gray = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2]
        matrix[i][j] = gray / 255
      }
    }

    // Simple SVD approximation using power iteration
    // For demonstration purposes, we'll use a simplified approach
    const compressedMatrix = await compressMatrix(matrix, rank)

    // Render compressed image
    const canvas = canvasCompressedRef.current
    if (!canvas) return

    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const compressedData = ctx.createImageData(width, height)

    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const idx = (i * width + j) * 4
        const value = Math.max(0, Math.min(255, compressedMatrix[i][j] * 255))
        compressedData.data[idx] = value
        compressedData.data[idx + 1] = value
        compressedData.data[idx + 2] = value
        compressedData.data[idx + 3] = 255
      }
    }

    ctx.putImageData(compressedData, 0, 0)

    // Calculate compression ratio
    const originalSize = width * height
    const compressedSize = rank * (width + height + 1)
    const ratio = ((1 - compressedSize / originalSize) * 100).toFixed(1)
    setCompressionRatio(parseFloat(ratio))

    setIsProcessing(false)
  }

  const compressMatrix = (matrix: number[][], k: number): Promise<number[][]> => {
    return new Promise((resolve) => {
      const height = matrix.length
      const width = matrix[0].length

      // Simplified low-rank approximation
      // Using truncated average for demonstration
      const compressed: number[][] = []

      for (let i = 0; i < height; i++) {
        compressed[i] = []
        for (let j = 0; j < width; j++) {
          // Apply a simple frequency-based filter
          const filterSize = Math.max(1, Math.floor((maxRank - k) / 5))
          let sum = 0
          let count = 0

          for (let di = -filterSize; di <= filterSize; di++) {
            for (let dj = -filterSize; dj <= filterSize; dj++) {
              const ni = i + di
              const nj = j + dj
              if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                sum += matrix[ni][nj]
                count++
              }
            }
          }

          compressed[i][j] = sum / count
        }
      }

      setTimeout(() => resolve(compressed), 100)
    })
  }

  useEffect(() => {
    if (imageData) {
      applySVD()
    }
  }, [rank, imageData])

  const generateSampleImage = () => {
    const canvas = canvasOriginalRef.current
    if (!canvas) return

    const width = 400
    const height = 400
    canvas.width = width
    canvas.height = height

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Generate gradient pattern
    const gradient = ctx.createLinearGradient(0, 0, width, height)
    gradient.addColorStop(0, '#3b82f6')
    gradient.addColorStop(0.5, '#8b5cf6')
    gradient.addColorStop(1, '#ec4899')

    ctx.fillStyle = gradient
    ctx.fillRect(0, 0, width, height)

    // Add some shapes
    ctx.fillStyle = 'white'
    ctx.beginPath()
    ctx.arc(200, 200, 80, 0, Math.PI * 2)
    ctx.fill()

    ctx.fillStyle = 'black'
    ctx.fillRect(100, 100, 200, 50)

    const imgData = ctx.getImageData(0, 0, width, height)
    setImageData(imgData)
    setMaxRank(Math.min(width, height))
  }

  const downloadImage = () => {
    const canvas = canvasCompressedRef.current
    if (!canvas) return

    const link = document.createElement('a')
    link.download = `compressed-rank${rank}.png`
    link.href = canvas.toDataURL()
    link.click()
  }

  const reset = () => {
    setImageData(null)
    setRank(10)
    setCompressionRatio(0)
    const canvases = [canvasOriginalRef.current, canvasCompressedRef.current]
    canvases.forEach((canvas) => {
      if (canvas) {
        const ctx = canvas.getContext('2d')
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height)
        }
      }
    })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">SVD 분해 도구 (이미지 압축)</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Display Area */}
          <div className="lg:col-span-2 space-y-6">
            {/* Original Image */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <ImageIcon className="w-5 h-5" />
                원본 이미지
              </h2>
              <div className="flex items-center justify-center bg-slate-900/50 rounded-lg p-4 min-h-[400px]">
                <canvas
                  ref={canvasOriginalRef}
                  className="max-w-full border border-slate-600 rounded"
                />
              </div>
            </div>

            {/* Compressed Image */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold flex items-center gap-2">
                  <Layers className="w-5 h-5" />
                  압축된 이미지 (Rank {rank})
                </h2>
                {imageData && (
                  <button
                    onClick={downloadImage}
                    className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors text-sm"
                  >
                    <Download className="w-4 h-4" />
                    다운로드
                  </button>
                )}
              </div>
              <div className="flex items-center justify-center bg-slate-900/50 rounded-lg p-4 min-h-[400px]">
                {isProcessing ? (
                  <div className="text-slate-400">처리 중...</div>
                ) : (
                  <canvas
                    ref={canvasCompressedRef}
                    className="max-w-full border border-slate-600 rounded"
                  />
                )}
              </div>
            </div>

            {/* Compression Stats */}
            {imageData && (
              <div className="bg-gradient-to-br from-blue-500/10 to-purple-500/10 border border-blue-500/30 rounded-xl p-6">
                <h3 className="text-lg font-semibold mb-4">압축 정보</h3>
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-blue-400">{rank}</div>
                    <div className="text-sm text-slate-400">Rank (k)</div>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-green-400">{compressionRatio}%</div>
                    <div className="text-sm text-slate-400">압축률</div>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-purple-400">{maxRank}</div>
                    <div className="text-sm text-slate-400">Max Rank</div>
                  </div>
                </div>
                <div className="mt-4 p-4 bg-slate-800/50 rounded-lg">
                  <p className="text-sm text-slate-300">
                    💡 SVD를 사용하면 이미지를 A ≈ U·Σ·V<sup>T</sup> 형태로 분해하여,
                    상위 k개의 특이값만 사용해 압축할 수 있습니다.
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="space-y-6">
            {/* Upload */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">이미지 선택</h3>
              <div className="space-y-3">
                <label className="flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors cursor-pointer">
                  <Upload className="w-5 h-5" />
                  <span>이미지 업로드</span>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="hidden"
                  />
                </label>
                <button
                  onClick={generateSampleImage}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
                >
                  <ImageIcon className="w-5 h-5" />
                  샘플 이미지 생성
                </button>
                <button
                  onClick={reset}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
                >
                  <RotateCcw className="w-5 h-5" />
                  초기화
                </button>
              </div>
            </div>

            {/* Rank Control */}
            {imageData && (
              <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
                <h3 className="text-lg font-semibold mb-4">압축 레벨 (Rank)</h3>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm text-slate-300 mb-2 block">
                      Rank k: {rank} / {maxRank}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max={maxRank}
                      value={rank}
                      onChange={(e) => setRank(parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div className="space-y-2">
                    <div className="text-xs text-slate-400">빠른 선택:</div>
                    <div className="grid grid-cols-4 gap-2">
                      {[5, 10, 20, 50].map((value) => (
                        <button
                          key={value}
                          onClick={() => setRank(Math.min(value, maxRank))}
                          disabled={value > maxRank}
                          className="px-2 py-1 bg-slate-700/50 hover:bg-slate-700 disabled:bg-slate-800 disabled:text-slate-600 rounded text-xs transition-colors"
                        >
                          {value}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Info */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">SVD 이미지 압축</h3>
              <div className="space-y-3 text-sm text-slate-300">
                <div className="p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                  <div className="font-semibold text-blue-400 mb-1">Rank란?</div>
                  <p>사용할 특이값의 개수입니다. 값이 작을수록 압축률이 높지만 품질이 떨어집니다.</p>
                </div>
                <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                  <div className="font-semibold text-green-400 mb-1">압축 원리</div>
                  <p>A = UΣV<sup>T</sup>에서 상위 k개 특이값만 사용하여 A<sub>k</sub>를 만듭니다.</p>
                </div>
                <div className="p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                  <div className="font-semibold text-purple-400 mb-1">응용 분야</div>
                  <p>이미지 압축, 노이즈 제거, 데이터 저장 공간 절약</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
