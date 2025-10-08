'use client'

import { useState, useEffect, useRef } from 'react'
import { Sparkles, RefreshCw, Shuffle, TrendingUp } from 'lucide-react'

interface LatentVector {
  values: number[]
}

interface GeneratedImage {
  id: number
  latentVector: number[]
  data: ImageData | null
}

export default function GANGenerator() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartCanvasRef = useRef<HTMLCanvasElement>(null)

  const [latentDim, setLatentDim] = useState(64)
  const [currentLatent, setCurrentLatent] = useState<number[]>([])
  const [generatedImages, setGeneratedImages] = useState<GeneratedImage[]>([])
  const [isGenerating, setIsGenerating] = useState(false)
  const [epoch, setEpoch] = useState(0)
  const [generatorLoss, setGeneratorLoss] = useState<number[]>([])
  const [discriminatorLoss, setDiscriminatorLoss] = useState<number[]>([])
  const [interpolationProgress, setInterpolationProgress] = useState(0)
  const [selectedImage1, setSelectedImage1] = useState<number | null>(null)
  const [selectedImage2, setSelectedImage2] = useState<number | null>(null)

  // Initialize random latent vector
  const generateRandomLatent = (): number[] => {
    return Array.from({ length: latentDim }, () => (Math.random() - 0.5) * 2)
  }

  // Generate initial latent vector
  useEffect(() => {
    setCurrentLatent(generateRandomLatent())
  }, [latentDim])

  // Simulate GAN training progress
  useEffect(() => {
    if (epoch > 0) {
      // Simulate loss convergence
      const gLoss = 2 * Math.exp(-epoch / 50) + 0.5 + Math.random() * 0.3
      const dLoss = 1.5 * Math.exp(-epoch / 40) + 0.3 + Math.random() * 0.2

      setGeneratorLoss(prev => [...prev.slice(-99), gLoss])
      setDiscriminatorLoss(prev => [...prev.slice(-99), dLoss])
    }
  }, [epoch])

  // Generate image from latent vector using Canvas
  const generateImageFromLatent = (latent: number[], canvas: HTMLCanvasElement): ImageData | null => {
    const ctx = canvas.getContext('2d')
    if (!ctx) return null

    const size = 128
    const imageData = ctx.createImageData(size, size)

    // Use latent vector to generate procedural patterns
    // This is a simplified simulation - real GANs would use neural networks
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const idx = (y * size + x) * 4

        // Use first few latent dimensions to influence the pattern
        const freq1 = latent[0] * 0.1 || 0.05
        const freq2 = latent[1] * 0.1 || 0.03
        const phase1 = latent[2] * Math.PI || 0
        const phase2 = latent[3] * Math.PI || 0

        // Create interesting patterns using sine waves and latent features
        const pattern1 = Math.sin(x * freq1 + phase1) * Math.cos(y * freq2 + phase2)
        const pattern2 = Math.cos(x * freq2) * Math.sin(y * freq1)
        const pattern3 = Math.sin((x + y) * freq1 * 0.5)

        // Combine patterns with more latent dimensions
        const r = ((pattern1 + 1) * 0.5 * 255) * (latent[4] || 0.5)
        const g = ((pattern2 + 1) * 0.5 * 255) * (latent[5] || 0.5)
        const b = ((pattern3 + 1) * 0.5 * 255) * (latent[6] || 0.5)

        imageData.data[idx] = Math.max(0, Math.min(255, r))
        imageData.data[idx + 1] = Math.max(0, Math.min(255, g))
        imageData.data[idx + 2] = Math.max(0, Math.min(255, b))
        imageData.data[idx + 3] = 255
      }
    }

    return imageData
  }

  // Generate new image
  const handleGenerate = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    setIsGenerating(true)
    const newLatent = generateRandomLatent()
    setCurrentLatent(newLatent)

    setTimeout(() => {
      const imageData = generateImageFromLatent(newLatent, canvas)
      const ctx = canvas.getContext('2d')
      if (ctx && imageData) {
        ctx.putImageData(imageData, 0, 0)
      }

      // Add to gallery
      setGeneratedImages(prev => [
        { id: Date.now(), latentVector: newLatent, data: imageData },
        ...prev.slice(0, 7)
      ])

      setEpoch(prev => prev + 1)
      setIsGenerating(false)
    }, 500)
  }

  // Interpolate between two latent vectors
  const interpolateLatent = (lat1: number[], lat2: number[], alpha: number): number[] => {
    return lat1.map((val, idx) => val * (1 - alpha) + lat2[idx] * alpha)
  }

  // Handle interpolation
  useEffect(() => {
    if (selectedImage1 !== null && selectedImage2 !== null) {
      const img1 = generatedImages.find(img => img.id === selectedImage1)
      const img2 = generatedImages.find(img => img.id === selectedImage2)

      if (img1 && img2) {
        const interpolatedLatent = interpolateLatent(
          img1.latentVector,
          img2.latentVector,
          interpolationProgress
        )
        setCurrentLatent(interpolatedLatent)

        // Generate interpolated image
        const canvas = canvasRef.current
        if (canvas) {
          const imageData = generateImageFromLatent(interpolatedLatent, canvas)
          const ctx = canvas.getContext('2d')
          if (ctx && imageData) {
            ctx.putImageData(imageData, 0, 0)
          }
        }
      }
    }
  }, [interpolationProgress, selectedImage1, selectedImage2])

  // Draw loss curves
  useEffect(() => {
    const canvas = chartCanvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.clearRect(0, 0, width, height)

    // Draw grid
    ctx.strokeStyle = '#e5e7eb'
    ctx.lineWidth = 1
    for (let i = 0; i <= 5; i++) {
      const y = (height / 5) * i
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    // Draw Generator Loss
    if (generatorLoss.length > 1) {
      ctx.strokeStyle = '#8b5cf6'
      ctx.lineWidth = 2
      ctx.beginPath()
      generatorLoss.forEach((loss, idx) => {
        const x = (width / 100) * idx
        const y = height - (loss / 3) * height
        if (idx === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })
      ctx.stroke()
    }

    // Draw Discriminator Loss
    if (discriminatorLoss.length > 1) {
      ctx.strokeStyle = '#ec4899'
      ctx.lineWidth = 2
      ctx.beginPath()
      discriminatorLoss.forEach((loss, idx) => {
        const x = (width / 100) * idx
        const y = height - (loss / 3) * height
        if (idx === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })
      ctx.stroke()
    }
  }, [generatorLoss, discriminatorLoss])

  // Generate initial image
  useEffect(() => {
    if (currentLatent.length > 0) {
      const canvas = canvasRef.current
      if (canvas) {
        const imageData = generateImageFromLatent(currentLatent, canvas)
        const ctx = canvas.getContext('2d')
        if (ctx && imageData) {
          ctx.putImageData(imageData, 0, 0)
        }
      }
    }
  }, [])

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">
            Latent Dimension: {latentDim}
          </h3>
          <input
            type="range"
            min="8"
            max="128"
            step="8"
            value={latentDim}
            onChange={(e) => setLatentDim(parseInt(e.target.value))}
            className="w-full mb-2"
          />
          <div className="text-xs text-gray-600 dark:text-gray-400">
            잠재 공간(Latent Space)의 차원 수
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">
            Training Progress
          </h3>
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <div className="text-2xl font-bold text-violet-600 dark:text-violet-400">
                Epoch {epoch}
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400">
                Generated: {generatedImages.length} images
              </div>
            </div>
            <button
              onClick={handleGenerate}
              disabled={isGenerating}
              className="px-4 py-2 bg-violet-500 hover:bg-violet-600 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
            >
              {isGenerating ? (
                <>
                  <RefreshCw className="animate-spin" size={16} />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles size={16} />
                  Generate
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Generated Image */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Sparkles className="text-violet-500" size={20} />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Generated Image
              </h3>
            </div>
            <button
              onClick={() => setCurrentLatent(generateRandomLatent())}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              title="Random Latent Vector"
            >
              <Shuffle size={16} />
            </button>
          </div>

          <div className="border-4 border-violet-200 dark:border-violet-800 rounded-lg overflow-hidden bg-gray-100 dark:bg-gray-900">
            <canvas
              ref={canvasRef}
              width={128}
              height={128}
              className="w-full h-auto"
              style={{ imageRendering: 'pixelated' }}
            />
          </div>

          <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
            <strong>Latent Vector</strong> (first 8 dims):
            <div className="mt-2 font-mono text-xs bg-gray-50 dark:bg-gray-900 p-2 rounded">
              [{currentLatent.slice(0, 8).map(v => v.toFixed(2)).join(', ')}...]
            </div>
          </div>
        </div>

        {/* Loss Curves */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="text-pink-500" size={20} />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Training Loss
            </h3>
          </div>

          <div className="border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden bg-white dark:bg-gray-900">
            <canvas
              ref={chartCanvasRef}
              width={400}
              height={200}
              className="w-full h-auto"
            />
          </div>

          <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-violet-500 rounded"></div>
              <span className="text-gray-600 dark:text-gray-400">
                Generator: {generatorLoss[generatorLoss.length - 1]?.toFixed(3) || 'N/A'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-pink-500 rounded"></div>
              <span className="text-gray-600 dark:text-gray-400">
                Discriminator: {discriminatorLoss[discriminatorLoss.length - 1]?.toFixed(3) || 'N/A'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Image Gallery */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Generated Gallery
        </h3>

        <div className="grid grid-cols-4 md:grid-cols-8 gap-3">
          {generatedImages.map((img) => (
            <div
              key={img.id}
              className={`aspect-square rounded-lg overflow-hidden cursor-pointer border-2 transition-all ${
                selectedImage1 === img.id
                  ? 'border-violet-500 ring-2 ring-violet-300'
                  : selectedImage2 === img.id
                  ? 'border-pink-500 ring-2 ring-pink-300'
                  : 'border-gray-300 dark:border-gray-600 hover:border-violet-400'
              }`}
              onClick={() => {
                if (selectedImage1 === null) {
                  setSelectedImage1(img.id)
                } else if (selectedImage2 === null && img.id !== selectedImage1) {
                  setSelectedImage2(img.id)
                } else {
                  setSelectedImage1(img.id)
                  setSelectedImage2(null)
                }
              }}
            >
              <canvas
                width={128}
                height={128}
                className="w-full h-full"
                style={{ imageRendering: 'pixelated' }}
                ref={(canvas) => {
                  if (canvas && img.data) {
                    const ctx = canvas.getContext('2d')
                    if (ctx) {
                      ctx.putImageData(img.data, 0, 0)
                    }
                  }
                }}
              />
            </div>
          ))}
        </div>

        {generatedImages.length === 0 && (
          <div className="text-center py-12 text-gray-500 dark:text-gray-400">
            Generate images to see them in the gallery
          </div>
        )}
      </div>

      {/* Latent Space Interpolation */}
      {selectedImage1 !== null && selectedImage2 !== null && (
        <div className="bg-gradient-to-br from-violet-50 to-pink-50 dark:from-violet-900/20 dark:to-pink-900/20 rounded-xl p-6 border border-violet-200 dark:border-violet-800">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Latent Space Interpolation
          </h3>

          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
                <span>Image 1 (Violet)</span>
                <span>{(interpolationProgress * 100).toFixed(0)}%</span>
                <span>Image 2 (Pink)</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={interpolationProgress}
                onChange={(e) => setInterpolationProgress(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div className="text-sm text-gray-600 dark:text-gray-400">
              두 이미지의 잠재 벡터 사이를 보간하여 부드러운 전환을 생성합니다.
              GAN의 잠재 공간이 얼마나 연속적인지 확인할 수 있습니다.
            </div>
          </div>
        </div>
      )}

      {/* Info */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-300 mb-3">
          💡 GAN (Generative Adversarial Network) 이해하기
        </h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm text-blue-800 dark:text-blue-200">
          <div>
            <strong>Generator:</strong> 잠재 벡터에서 이미지를 생성하는 신경망
          </div>
          <div>
            <strong>Discriminator:</strong> 이미지가 진짜인지 가짜인지 판별
          </div>
          <div>
            <strong>Latent Space:</strong> 생성 가능한 모든 이미지를 표현하는 공간
          </div>
          <div>
            <strong>Interpolation:</strong> 잠재 공간에서의 부드러운 이동
          </div>
        </div>
      </div>
    </div>
  )
}
