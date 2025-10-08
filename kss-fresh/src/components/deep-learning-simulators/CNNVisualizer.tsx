'use client'

import { useState, useEffect, useRef } from 'react'
import { Image as ImageIcon, Layers, Upload } from 'lucide-react'

type FilterType = 'edge-horizontal' | 'edge-vertical' | 'blur' | 'sharpen' | 'emboss'

export default function CNNVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const outputCanvasRef = useRef<HTMLCanvasElement>(null)

  const [selectedFilter, setSelectedFilter] = useState<FilterType>('edge-horizontal')
  const [currentLayer, setCurrentLayer] = useState(1)
  const [imageData, setImageData] = useState<ImageData | null>(null)

  // Predefined filters
  const filters: Record<FilterType, number[][]> = {
    'edge-horizontal': [
      [-1, -1, -1],
      [0, 0, 0],
      [1, 1, 1]
    ],
    'edge-vertical': [
      [-1, 0, 1],
      [-1, 0, 1],
      [-1, 0, 1]
    ],
    'blur': [
      [1/9, 1/9, 1/9],
      [1/9, 1/9, 1/9],
      [1/9, 1/9, 1/9]
    ],
    'sharpen': [
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0]
    ],
    'emboss': [
      [-2, -1, 0],
      [-1, 1, 1],
      [0, 1, 2]
    ]
  }

  // Generate a simple image (checkboard pattern)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const size = 200

    // Draw a simple pattern
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, size, size)

    // Draw some shapes
    ctx.fillStyle = '#000000'

    // Horizontal lines
    for (let i = 0; i < 5; i++) {
      ctx.fillRect(0, i * 40, size, 10)
    }

    // Vertical lines
    for (let i = 0; i < 5; i++) {
      ctx.fillRect(i * 40, 0, 10, size)
    }

    // Circle
    ctx.beginPath()
    ctx.arc(100, 100, 30, 0, Math.PI * 2)
    ctx.fill()

    const imgData = ctx.getImageData(0, 0, size, size)
    setImageData(imgData)
  }, [])

  // Apply convolution filter
  useEffect(() => {
    if (!imageData) return

    const outputCanvas = outputCanvasRef.current
    if (!outputCanvas) return

    const ctx = outputCanvas.getContext('2d')
    if (!ctx) return

    const width = imageData.width
    const height = imageData.height
    const filter = filters[selectedFilter]

    const outputData = ctx.createImageData(width, height)

    // Apply convolution
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let r = 0, g = 0, b = 0

        for (let fy = 0; fy < 3; fy++) {
          for (let fx = 0; fx < 3; fx++) {
            const px = x + fx - 1
            const py = y + fy - 1
            const pidx = (py * width + px) * 4

            const filterValue = filter[fy][fx]
            r += imageData.data[pidx] * filterValue
            g += imageData.data[pidx + 1] * filterValue
            b += imageData.data[pidx + 2] * filterValue
          }
        }

        const idx = (y * width + x) * 4
        outputData.data[idx] = Math.min(255, Math.max(0, r + 128))
        outputData.data[idx + 1] = Math.min(255, Math.max(0, g + 128))
        outputData.data[idx + 2] = Math.min(255, Math.max(0, b + 128))
        outputData.data[idx + 3] = 255
      }
    }

    ctx.putImageData(outputData, 0, 0)
  }, [imageData, selectedFilter])

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
            <Layers size={16} />
            CNN Layer: {currentLayer}
          </h3>
          <input
            type="range"
            min="1"
            max="5"
            value={currentLayer}
            onChange={(e) => setCurrentLayer(parseInt(e.target.value))}
            className="w-full mb-2"
          />
          <div className="text-xs text-gray-600 dark:text-gray-400">
            레이어가 깊어질수록 더 복잡한 특징을 추출합니다
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">
            Filter Type
          </h3>
          <select
            value={selectedFilter}
            onChange={(e) => setSelectedFilter(e.target.value as FilterType)}
            className="w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg text-sm"
          >
            <option value="edge-horizontal">Horizontal Edge Detector</option>
            <option value="edge-vertical">Vertical Edge Detector</option>
            <option value="blur">Blur (Average Pooling)</option>
            <option value="sharpen">Sharpen</option>
            <option value="emboss">Emboss</option>
          </select>
        </div>
      </div>

      {/* Main Visualization */}
      <div className="grid md:grid-cols-3 gap-6">
        {/* Input Image */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2 mb-4">
            <ImageIcon className="text-blue-500" size={20} />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Input Image
            </h3>
          </div>

          <div className="border-4 border-blue-200 dark:border-blue-800 rounded-lg overflow-hidden">
            <canvas
              ref={canvasRef}
              width={200}
              height={200}
              className="w-full h-auto"
            />
          </div>

          <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
            Shape: 200 × 200 × 1
          </div>
        </div>

        {/* Filter Visualization */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2 mb-4">
            <Layers className="text-purple-500" size={20} />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Convolution Filter
            </h3>
          </div>

          {/* Filter Matrix */}
          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 border border-purple-200 dark:border-purple-800">
            <div className="grid grid-cols-3 gap-2 mb-4">
              {filters[selectedFilter].flat().map((value, idx) => (
                <div
                  key={idx}
                  className="aspect-square flex items-center justify-center bg-white dark:bg-gray-800 rounded-lg border-2 border-purple-300 dark:border-purple-600 font-mono font-bold text-lg"
                  style={{
                    backgroundColor: value > 0
                      ? `rgba(139, 92, 246, ${Math.abs(value)})`
                      : value < 0
                      ? `rgba(239, 68, 68, ${Math.abs(value)})`
                      : '#ffffff'
                  }}
                >
                  {value.toFixed(1)}
                </div>
              ))}
            </div>

            <div className="text-xs text-center text-gray-600 dark:text-gray-400">
              3×3 Kernel
            </div>
          </div>

          <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
            Filter Size: 3 × 3
          </div>
        </div>

        {/* Feature Map */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2 mb-4">
            <Layers className="text-green-500" size={20} />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Feature Map
            </h3>
          </div>

          <div className="border-4 border-green-200 dark:border-green-800 rounded-lg overflow-hidden">
            <canvas
              ref={outputCanvasRef}
              width={200}
              height={200}
              className="w-full h-auto"
            />
          </div>

          <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
            Activated: ReLU
          </div>
        </div>
      </div>

      {/* CNN Architecture Overview */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
          <Layers className="text-violet-500" size={24} />
          CNN Architecture Flow
        </h3>

        <div className="flex items-center justify-between">
          {/* Input */}
          <div className="flex flex-col items-center">
            <div className="w-20 h-20 bg-gradient-to-br from-blue-400 to-blue-600 rounded-lg shadow-lg flex items-center justify-center text-white font-bold">
              Input
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              200×200×1
            </div>
          </div>

          <div className="flex-1 h-px bg-gray-300 dark:bg-gray-600 mx-2" />

          {/* Conv1 */}
          <div className="flex flex-col items-center">
            <div className="w-20 h-20 bg-gradient-to-br from-purple-400 to-purple-600 rounded-lg shadow-lg flex items-center justify-center text-white text-xs font-bold text-center">
              Conv<br/>3×3
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              200×200×32
            </div>
          </div>

          <div className="flex-1 h-px bg-gray-300 dark:bg-gray-600 mx-2" />

          {/* Pool1 */}
          <div className="flex flex-col items-center">
            <div className="w-16 h-16 bg-gradient-to-br from-pink-400 to-pink-600 rounded-lg shadow-lg flex items-center justify-center text-white text-xs font-bold text-center">
              MaxPool<br/>2×2
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              100×100×32
            </div>
          </div>

          <div className="flex-1 h-px bg-gray-300 dark:bg-gray-600 mx-2" />

          {/* Conv2 */}
          <div className="flex flex-col items-center">
            <div className="w-20 h-20 bg-gradient-to-br from-violet-400 to-violet-600 rounded-lg shadow-lg flex items-center justify-center text-white text-xs font-bold text-center">
              Conv<br/>3×3
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              100×100×64
            </div>
          </div>

          <div className="flex-1 h-px bg-gray-300 dark:bg-gray-600 mx-2" />

          {/* Output */}
          <div className="flex flex-col items-center">
            <div className="w-20 h-20 bg-gradient-to-br from-green-400 to-green-600 rounded-lg shadow-lg flex items-center justify-center text-white font-bold">
              Output
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              Features
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="grid md:grid-cols-3 gap-4 mt-8 text-sm">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
            <strong className="text-blue-700 dark:text-blue-300">Input Layer:</strong>
            <div className="text-gray-600 dark:text-gray-400 mt-1">
              원본 이미지 데이터
            </div>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3">
            <strong className="text-purple-700 dark:text-purple-300">Convolution:</strong>
            <div className="text-gray-600 dark:text-gray-400 mt-1">
              필터를 이용해 특징 추출
            </div>
          </div>
          <div className="bg-pink-50 dark:bg-pink-900/20 rounded-lg p-3">
            <strong className="text-pink-700 dark:text-pink-300">Pooling:</strong>
            <div className="text-gray-600 dark:text-gray-400 mt-1">
              차원 축소 및 불변성 확보
            </div>
          </div>
        </div>
      </div>

      {/* Info */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-300 mb-3">
          💡 CNN Filter 이해하기
        </h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm text-blue-800 dark:text-blue-200">
          <div>
            <strong>Edge Detection:</strong> 가장자리를 강조하여 객체의 윤곽을 찾아냄
          </div>
          <div>
            <strong>Blur:</strong> 이미지를 부드럽게 만들어 노이즈 제거
          </div>
          <div>
            <strong>Sharpen:</strong> 이미지를 선명하게 만들어 디테일 강조
          </div>
          <div>
            <strong>Emboss:</strong> 3D 입체 효과를 만들어 질감 표현
          </div>
        </div>
      </div>
    </div>
  )
}
