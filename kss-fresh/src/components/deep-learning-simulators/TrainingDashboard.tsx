'use client'

import { useState, useEffect, useRef } from 'react'
import { Play, Pause, Square, TrendingUp, Activity, Layers, Clock } from 'lucide-react'

interface TrainingMetrics {
  epoch: number
  trainLoss: number
  trainAcc: number
  valLoss: number
  valAcc: number
  learningRate: number
  batchTime: number
}

interface LayerGradients {
  layer: string
  gradientNorm: number
}

export default function TrainingDashboard() {
  const lossCanvasRef = useRef<HTMLCanvasElement>(null)
  const accCanvasRef = useRef<HTMLCanvasElement>(null)
  const gradientCanvasRef = useRef<HTMLCanvasElement>(null)

  const [isTraining, setIsTraining] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [currentEpoch, setCurrentEpoch] = useState(0)
  const [maxEpochs, setMaxEpochs] = useState(100)
  const [batchSize, setBatchSize] = useState(32)
  const [learningRate, setLearningRate] = useState(0.001)
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([])
  const [layerGradients, setLayerGradients] = useState<LayerGradients[]>([])
  const [trainingSpeed, setTrainingSpeed] = useState(1)

  // Simulate training progress
  useEffect(() => {
    if (!isTraining || isPaused) return

    const interval = setInterval(() => {
      setCurrentEpoch(prev => {
        if (prev >= maxEpochs) {
          setIsTraining(false)
          return prev
        }

        const newEpoch = prev + 1

        // Simulate loss convergence
        const trainLoss = 2.5 * Math.exp(-newEpoch / 20) + 0.1 + Math.random() * 0.05
        const valLoss = 2.5 * Math.exp(-newEpoch / 20) + 0.15 + Math.random() * 0.1

        // Simulate accuracy improvement
        const trainAcc = 100 * (1 - Math.exp(-newEpoch / 15)) * (0.95 + Math.random() * 0.05)
        const valAcc = 100 * (1 - Math.exp(-newEpoch / 15)) * (0.92 + Math.random() * 0.05)

        // Learning rate decay
        const currentLR = learningRate * Math.pow(0.95, Math.floor(newEpoch / 10))

        // Batch time simulation
        const batchTime = 50 + Math.random() * 20

        const newMetrics: TrainingMetrics = {
          epoch: newEpoch,
          trainLoss,
          trainAcc,
          valLoss,
          valAcc,
          learningRate: currentLR,
          batchTime
        }

        setMetrics(prev => [...prev, newMetrics])

        // Simulate gradient norms
        setLayerGradients([
          { layer: 'conv1', gradientNorm: 0.5 + Math.random() * 0.5 },
          { layer: 'conv2', gradientNorm: 0.4 + Math.random() * 0.6 },
          { layer: 'conv3', gradientNorm: 0.3 + Math.random() * 0.7 },
          { layer: 'fc1', gradientNorm: 0.2 + Math.random() * 0.8 },
          { layer: 'fc2', gradientNorm: 0.1 + Math.random() * 0.9 },
          { layer: 'output', gradientNorm: 0.05 + Math.random() * 0.5 }
        ])

        return newEpoch
      })
    }, 1000 / trainingSpeed)

    return () => clearInterval(interval)
  }, [isTraining, isPaused, maxEpochs, learningRate, trainingSpeed])

  // Draw loss chart
  useEffect(() => {
    const canvas = lossCanvasRef.current
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

    if (metrics.length < 2) return

    const maxLoss = Math.max(...metrics.map(m => Math.max(m.trainLoss, m.valLoss)))

    // Draw Train Loss
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 2
    ctx.beginPath()
    metrics.forEach((metric, idx) => {
      const x = (width / maxEpochs) * metric.epoch
      const y = height - (metric.trainLoss / maxLoss) * height * 0.9
      if (idx === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })
    ctx.stroke()

    // Draw Val Loss
    ctx.strokeStyle = '#ef4444'
    ctx.lineWidth = 2
    ctx.beginPath()
    metrics.forEach((metric, idx) => {
      const x = (width / maxEpochs) * metric.epoch
      const y = height - (metric.valLoss / maxLoss) * height * 0.9
      if (idx === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })
    ctx.stroke()
  }, [metrics, maxEpochs])

  // Draw accuracy chart
  useEffect(() => {
    const canvas = accCanvasRef.current
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

    if (metrics.length < 2) return

    // Draw Train Accuracy
    ctx.strokeStyle = '#10b981'
    ctx.lineWidth = 2
    ctx.beginPath()
    metrics.forEach((metric, idx) => {
      const x = (width / maxEpochs) * metric.epoch
      const y = height - (metric.trainAcc / 100) * height
      if (idx === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })
    ctx.stroke()

    // Draw Val Accuracy
    ctx.strokeStyle = '#f59e0b'
    ctx.lineWidth = 2
    ctx.beginPath()
    metrics.forEach((metric, idx) => {
      const x = (width / maxEpochs) * metric.epoch
      const y = height - (metric.valAcc / 100) * height
      if (idx === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })
    ctx.stroke()
  }, [metrics, maxEpochs])

  // Draw gradient flow
  useEffect(() => {
    const canvas = gradientCanvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.clearRect(0, 0, width, height)

    if (layerGradients.length === 0) return

    const barWidth = width / layerGradients.length
    const maxGradient = Math.max(...layerGradients.map(g => g.gradientNorm))

    layerGradients.forEach((grad, idx) => {
      const x = idx * barWidth
      const barHeight = (grad.gradientNorm / maxGradient) * height * 0.9

      // Gradient color based on magnitude
      const intensity = Math.floor((grad.gradientNorm / maxGradient) * 255)
      ctx.fillStyle = `rgb(${intensity}, ${255 - intensity}, 128)`
      ctx.fillRect(x + 5, height - barHeight, barWidth - 10, barHeight)

      // Layer label
      ctx.fillStyle = '#6b7280'
      ctx.font = '10px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(grad.layer, x + barWidth / 2, height - 5)
    })
  }, [layerGradients])

  const handleStart = () => {
    if (currentEpoch >= maxEpochs) {
      // Reset training
      setCurrentEpoch(0)
      setMetrics([])
      setLayerGradients([])
    }
    setIsTraining(true)
    setIsPaused(false)
  }

  const handlePause = () => {
    setIsPaused(!isPaused)
  }

  const handleStop = () => {
    setIsTraining(false)
    setIsPaused(false)
    setCurrentEpoch(0)
    setMetrics([])
    setLayerGradients([])
  }

  const currentMetrics = metrics[metrics.length - 1]

  return (
    <div className="space-y-6">
      {/* Training Controls */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white flex items-center gap-2">
            <Activity className="text-blue-500" size={24} />
            Training Controls
          </h3>

          <div className="flex items-center gap-2">
            <button
              onClick={handleStart}
              disabled={isTraining && !isPaused}
              className="px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
            >
              <Play size={16} />
              {currentEpoch >= maxEpochs ? 'Restart' : 'Start'}
            </button>
            <button
              onClick={handlePause}
              disabled={!isTraining}
              className="px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
            >
              <Pause size={16} />
              {isPaused ? 'Resume' : 'Pause'}
            </button>
            <button
              onClick={handleStop}
              disabled={!isTraining && currentEpoch === 0}
              className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
            >
              <Square size={16} />
              Stop
            </button>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              Max Epochs: {maxEpochs}
            </div>
            <input
              type="range"
              min="10"
              max="200"
              value={maxEpochs}
              onChange={(e) => setMaxEpochs(parseInt(e.target.value))}
              disabled={isTraining}
              className="w-full"
            />
          </div>

          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              Batch Size: {batchSize}
            </div>
            <input
              type="range"
              min="8"
              max="128"
              step="8"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
              disabled={isTraining}
              className="w-full"
            />
          </div>

          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              Training Speed: {trainingSpeed}x
            </div>
            <input
              type="range"
              min="0.5"
              max="5"
              step="0.5"
              value={trainingSpeed}
              onChange={(e) => setTrainingSpeed(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* Progress Overview */}
      <div className="grid md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl p-6 text-white">
          <div className="flex items-center gap-2 mb-2">
            <Clock size={20} />
            <div className="text-sm opacity-90">Epoch</div>
          </div>
          <div className="text-3xl font-bold">{currentEpoch} / {maxEpochs}</div>
          <div className="mt-2 bg-white/20 rounded-full h-2">
            <div
              className="bg-white rounded-full h-2 transition-all"
              style={{ width: `${(currentEpoch / maxEpochs) * 100}%` }}
            />
          </div>
        </div>

        <div className="bg-gradient-to-br from-red-500 to-red-600 rounded-xl p-6 text-white">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp size={20} />
            <div className="text-sm opacity-90">Val Loss</div>
          </div>
          <div className="text-3xl font-bold">
            {currentMetrics?.valLoss.toFixed(4) || '-.----'}
          </div>
          <div className="text-sm mt-2 opacity-90">
            Train: {currentMetrics?.trainLoss.toFixed(4) || '-.----'}
          </div>
        </div>

        <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-xl p-6 text-white">
          <div className="flex items-center gap-2 mb-2">
            <Activity size={20} />
            <div className="text-sm opacity-90">Val Accuracy</div>
          </div>
          <div className="text-3xl font-bold">
            {currentMetrics?.valAcc.toFixed(2) || '--.-'}%
          </div>
          <div className="text-sm mt-2 opacity-90">
            Train: {currentMetrics?.trainAcc.toFixed(2) || '--.-'}%
          </div>
        </div>

        <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl p-6 text-white">
          <div className="flex items-center gap-2 mb-2">
            <Layers size={20} />
            <div className="text-sm opacity-90">Learning Rate</div>
          </div>
          <div className="text-3xl font-bold">
            {currentMetrics?.learningRate.toExponential(2) || '-.--e-0'}
          </div>
          <div className="text-sm mt-2 opacity-90">
            Batch: {batchSize}
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Loss Chart */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <TrendingUp className="text-red-500" size={20} />
            Loss Curves
          </h3>

          <div className="border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden bg-white dark:bg-gray-900">
            <canvas
              ref={lossCanvasRef}
              width={400}
              height={200}
              className="w-full h-auto"
            />
          </div>

          <div className="mt-4 flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-blue-500 rounded"></div>
              <span className="text-gray-600 dark:text-gray-400">Train Loss</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-red-500 rounded"></div>
              <span className="text-gray-600 dark:text-gray-400">Val Loss</span>
            </div>
          </div>
        </div>

        {/* Accuracy Chart */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <Activity className="text-green-500" size={20} />
            Accuracy Curves
          </h3>

          <div className="border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden bg-white dark:bg-gray-900">
            <canvas
              ref={accCanvasRef}
              width={400}
              height={200}
              className="w-full h-auto"
            />
          </div>

          <div className="mt-4 flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-green-500 rounded"></div>
              <span className="text-gray-600 dark:text-gray-400">Train Acc</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-yellow-500 rounded"></div>
              <span className="text-gray-600 dark:text-gray-400">Val Acc</span>
            </div>
          </div>
        </div>
      </div>

      {/* Gradient Flow */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Layers className="text-purple-500" size={20} />
          Gradient Flow
        </h3>

        <div className="border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden bg-white dark:bg-gray-900">
          <canvas
            ref={gradientCanvasRef}
            width={600}
            height={150}
            className="w-full h-auto"
          />
        </div>

        <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
          레이어별 그래디언트 크기를 시각화합니다. 너무 작으면 Vanishing Gradient, 너무 크면 Exploding Gradient 문제가 발생할 수 있습니다.
        </div>
      </div>

      {/* Training Log */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Recent Training Log
        </h3>

        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-xs max-h-48 overflow-y-auto">
          {metrics.length === 0 ? (
            <div className="text-gray-500 dark:text-gray-400">
              Training not started. Click "Start" to begin training.
            </div>
          ) : (
            metrics.slice(-10).reverse().map((metric, idx) => (
              <div key={idx} className="mb-1 text-gray-700 dark:text-gray-300">
                <span className="text-blue-600 dark:text-blue-400">[Epoch {metric.epoch}]</span>{' '}
                Loss: {metric.trainLoss.toFixed(4)} / {metric.valLoss.toFixed(4)}{' '}
                | Acc: {metric.trainAcc.toFixed(2)}% / {metric.valAcc.toFixed(2)}%{' '}
                | LR: {metric.learningRate.toExponential(2)}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Info */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-300 mb-3">
          💡 Training Monitoring 이해하기
        </h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm text-blue-800 dark:text-blue-200">
          <div>
            <strong>Loss:</strong> 모델의 예측과 실제 값 사이의 오차
          </div>
          <div>
            <strong>Accuracy:</strong> 정확하게 예측한 데이터의 비율
          </div>
          <div>
            <strong>Learning Rate:</strong> 파라미터 업데이트 속도 조절
          </div>
          <div>
            <strong>Gradient:</strong> 손실 함수의 기울기, 파라미터 업데이트 방향
          </div>
        </div>
      </div>
    </div>
  )
}
