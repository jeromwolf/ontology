'use client'

import { useState, useEffect, useRef } from 'react'
import { Play, Pause, RotateCcw, Settings } from 'lucide-react'

type ActivationFunction = 'relu' | 'sigmoid' | 'tanh'
type DatasetType = 'xor' | 'circle' | 'spiral' | 'linear'

interface Point {
  x: number
  y: number
  label: number
}

export default function NeuralNetworkPlayground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Network Architecture
  const [hiddenLayers, setHiddenLayers] = useState<number[]>([4, 4])
  const [activation, setActivation] = useState<ActivationFunction>('relu')

  // Training Settings
  const [learningRate, setLearningRate] = useState(0.03)
  const [batchSize, setBatchSize] = useState(10)
  const [dataset, setDataset] = useState<DatasetType>('xor')

  // Training State
  const [isTraining, setIsTraining] = useState(false)
  const [epoch, setEpoch] = useState(0)
  const [loss, setLoss] = useState(0)
  const [accuracy, setAccuracy] = useState(0)

  // Data
  const [dataPoints, setDataPoints] = useState<Point[]>([])

  // Generate dataset
  useEffect(() => {
    const points: Point[] = []
    const numPoints = 200

    for (let i = 0; i < numPoints; i++) {
      const x = Math.random() * 2 - 1 // -1 to 1
      const y = Math.random() * 2 - 1
      let label = 0

      switch (dataset) {
        case 'xor':
          label = (x * y > 0) ? 1 : 0
          break
        case 'circle':
          label = (x * x + y * y < 0.5) ? 1 : 0
          break
        case 'spiral':
          const angle = Math.atan2(y, x) + Math.PI
          const radius = Math.sqrt(x * x + y * y)
          label = (Math.sin(angle * 2 + radius * 5) > 0) ? 1 : 0
          break
        case 'linear':
          label = (x + y > 0) ? 1 : 0
          break
      }

      points.push({ x, y, label })
    }

    setDataPoints(points)
    setEpoch(0)
    setLoss(1.0)
    setAccuracy(50)
  }, [dataset])

  // Draw canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    // Clear canvas
    ctx.fillStyle = '#1f2937'
    ctx.fillRect(0, 0, width, height)

    // Draw grid
    ctx.strokeStyle = '#374151'
    ctx.lineWidth = 1
    for (let i = 0; i <= 10; i++) {
      ctx.beginPath()
      ctx.moveTo((i / 10) * width, 0)
      ctx.lineTo((i / 10) * width, height)
      ctx.stroke()

      ctx.beginPath()
      ctx.moveTo(0, (i / 10) * height)
      ctx.lineTo(width, (i / 10) * height)
      ctx.stroke()
    }

    // Draw decision boundary (simplified visualization)
    if (epoch > 0) {
      const gridSize = 50
      for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
          const x = (i / gridSize) * 2 - 1
          const y = (j / gridSize) * 2 - 1

          // Simplified prediction (this would use actual network in real impl)
          let prediction = 0
          switch (dataset) {
            case 'xor':
              prediction = (x * y > 0) ? 1 : 0
              break
            case 'circle':
              prediction = (x * x + y * y < 0.5) ? 1 : 0
              break
            case 'spiral':
              const angle = Math.atan2(y, x) + Math.PI
              const radius = Math.sqrt(x * x + y * y)
              prediction = (Math.sin(angle * 2 + radius * 5) > 0) ? 1 : 0
              break
            case 'linear':
              prediction = (x + y > 0) ? 1 : 0
              break
          }

          const opacity = 0.1 + (epoch / 100) * 0.3
          ctx.fillStyle = prediction === 1
            ? `rgba(59, 130, 246, ${opacity})`
            : `rgba(239, 68, 68, ${opacity})`

          const px = ((x + 1) / 2) * width
          const py = ((y + 1) / 2) * height
          ctx.fillRect(px, py, width / gridSize, height / gridSize)
        }
      }
    }

    // Draw data points
    dataPoints.forEach(point => {
      const px = ((point.x + 1) / 2) * width
      const py = ((point.y + 1) / 2) * height

      ctx.fillStyle = point.label === 1 ? '#3b82f6' : '#ef4444'
      ctx.beginPath()
      ctx.arc(px, py, 4, 0, Math.PI * 2)
      ctx.fill()

      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 1
      ctx.stroke()
    })
  }, [dataPoints, epoch, dataset])

  // Training simulation
  useEffect(() => {
    if (!isTraining) return

    const interval = setInterval(() => {
      setEpoch(prev => {
        const newEpoch = prev + 1

        // Simulate loss decrease
        setLoss(Math.max(0.1, 1.0 * Math.exp(-newEpoch / 50)))

        // Simulate accuracy increase
        setAccuracy(Math.min(95, 50 + (newEpoch / 100) * 45))

        if (newEpoch >= 100) {
          setIsTraining(false)
        }

        return newEpoch
      })
    }, 100)

    return () => clearInterval(interval)
  }, [isTraining])

  const handleReset = () => {
    setIsTraining(false)
    setEpoch(0)
    setLoss(1.0)
    setAccuracy(50)
  }

  const addHiddenLayer = () => {
    if (hiddenLayers.length < 5) {
      setHiddenLayers([...hiddenLayers, 4])
    }
  }

  const removeHiddenLayer = () => {
    if (hiddenLayers.length > 1) {
      setHiddenLayers(hiddenLayers.slice(0, -1))
    }
  }

  const updateLayerSize = (index: number, size: number) => {
    const newLayers = [...hiddenLayers]
    newLayers[index] = Math.max(1, Math.min(8, size))
    setHiddenLayers(newLayers)
  }

  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Dataset Selection */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">
            데이터셋
          </h3>
          <select
            value={dataset}
            onChange={(e) => setDataset(e.target.value as DatasetType)}
            className="w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg text-sm"
            disabled={isTraining}
          >
            <option value="xor">XOR Problem</option>
            <option value="circle">Circle</option>
            <option value="spiral">Spiral</option>
            <option value="linear">Linear</option>
          </select>
        </div>

        {/* Activation Function */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">
            활성화 함수
          </h3>
          <select
            value={activation}
            onChange={(e) => setActivation(e.target.value as ActivationFunction)}
            className="w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg text-sm"
            disabled={isTraining}
          >
            <option value="relu">ReLU</option>
            <option value="sigmoid">Sigmoid</option>
            <option value="tanh">Tanh</option>
          </select>
        </div>

        {/* Learning Rate */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">
            학습률: {learningRate.toFixed(3)}
          </h3>
          <input
            type="range"
            min="0.001"
            max="0.1"
            step="0.001"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            className="w-full"
            disabled={isTraining}
          />
        </div>

        {/* Batch Size */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">
            배치 크기: {batchSize}
          </h3>
          <input
            type="range"
            min="1"
            max="50"
            step="1"
            value={batchSize}
            onChange={(e) => setBatchSize(parseInt(e.target.value))}
            className="w-full"
            disabled={isTraining}
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Network Architecture */}
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
              <Settings size={20} className="text-violet-500" />
              네트워크 구조
            </h3>

            {/* Input Layer */}
            <div className="mb-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <div className="text-sm font-semibold text-blue-700 dark:text-blue-300 mb-2">
                입력층
              </div>
              <div className="flex gap-2">
                {[1, 2].map(i => (
                  <div key={i} className="w-8 h-8 rounded-full bg-blue-500 dark:bg-blue-600" />
                ))}
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                2 neurons (x, y)
              </div>
            </div>

            {/* Hidden Layers */}
            <div className="space-y-3 mb-4">
              {hiddenLayers.map((size, index) => (
                <div key={index} className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <div className="text-sm font-semibold text-purple-700 dark:text-purple-300 mb-2">
                    은닉층 {index + 1}
                  </div>
                  <div className="flex flex-wrap gap-2 mb-2">
                    {Array.from({ length: size }).map((_, i) => (
                      <div key={i} className="w-6 h-6 rounded-full bg-purple-500 dark:bg-purple-600" />
                    ))}
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="8"
                    value={size}
                    onChange={(e) => updateLayerSize(index, parseInt(e.target.value))}
                    className="w-full"
                    disabled={isTraining}
                  />
                  <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    {size} neurons
                  </div>
                </div>
              ))}
            </div>

            {/* Add/Remove Layer Buttons */}
            <div className="flex gap-2 mb-4">
              <button
                onClick={addHiddenLayer}
                disabled={isTraining || hiddenLayers.length >= 5}
                className="flex-1 px-3 py-2 bg-violet-500 hover:bg-violet-600 disabled:bg-gray-400 text-white rounded-lg text-sm font-medium transition-colors"
              >
                레이어 추가
              </button>
              <button
                onClick={removeHiddenLayer}
                disabled={isTraining || hiddenLayers.length <= 1}
                className="flex-1 px-3 py-2 bg-gray-500 hover:bg-gray-600 disabled:bg-gray-400 text-white rounded-lg text-sm font-medium transition-colors"
              >
                레이어 제거
              </button>
            </div>

            {/* Output Layer */}
            <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <div className="text-sm font-semibold text-green-700 dark:text-green-300 mb-2">
                출력층
              </div>
              <div className="flex gap-2">
                <div className="w-8 h-8 rounded-full bg-green-500 dark:bg-green-600" />
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                1 neuron (classification)
              </div>
            </div>
          </div>

          {/* Training Metrics */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              학습 지표
            </h3>

            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-600 dark:text-gray-400">Epoch</span>
                  <span className="font-semibold text-gray-900 dark:text-white">{epoch}</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-violet-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${(epoch / 100) * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-600 dark:text-gray-400">Loss</span>
                  <span className="font-semibold text-gray-900 dark:text-white">{loss.toFixed(3)}</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-red-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${loss * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-600 dark:text-gray-400">Accuracy</span>
                  <span className="font-semibold text-gray-900 dark:text-white">{accuracy.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-green-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${accuracy}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Visualization Canvas */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Decision Boundary Visualization
              </h3>

              {/* Training Controls */}
              <div className="flex gap-2">
                <button
                  onClick={() => setIsTraining(!isTraining)}
                  className="flex items-center gap-2 px-4 py-2 bg-violet-500 hover:bg-violet-600 text-white rounded-lg font-medium transition-colors"
                >
                  {isTraining ? (
                    <>
                      <Pause size={16} />
                      일시정지
                    </>
                  ) : (
                    <>
                      <Play size={16} />
                      학습 시작
                    </>
                  )}
                </button>

                <button
                  onClick={handleReset}
                  className="flex items-center gap-2 px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg font-medium transition-colors"
                >
                  <RotateCcw size={16} />
                  초기화
                </button>
              </div>
            </div>

            <div className="relative">
              <canvas
                ref={canvasRef}
                width={600}
                height={600}
                className="w-full h-auto border border-gray-300 dark:border-gray-600 rounded-lg"
              />

              {/* Legend */}
              <div className="absolute top-4 right-4 bg-white/90 dark:bg-gray-800/90 backdrop-blur-sm rounded-lg p-3 text-sm">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-4 h-4 rounded-full bg-blue-500 border border-white" />
                  <span className="text-gray-700 dark:text-gray-300">Class 1</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-red-500 border border-white" />
                  <span className="text-gray-700 dark:text-gray-300">Class 0</span>
                </div>
              </div>
            </div>

            <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
              💡 <strong>팁:</strong> 네트워크 구조를 조정하고 학습 버튼을 클릭해보세요.
              Decision boundary가 데이터를 어떻게 분류하는지 실시간으로 확인할 수 있습니다.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
