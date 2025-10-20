'use client'

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Play, Pause, RotateCcw, Info, Zap, Layers, Target, Cpu } from 'lucide-react'

interface ModelMetrics {
  size: number // MB
  accuracy: number // %
  inferenceTime: number // ms
  power: number // W
}

interface OptimizationConfig {
  quantization: 'none' | 'int8' | 'int4' | 'mixed'
  pruningRate: number // 0-90%
  distillation: boolean
}

interface Hardware {
  name: string
  price: number
  memory: number // GB
  tflops: number
  power: number // W
}

const HARDWARE_OPTIONS: Hardware[] = [
  { name: 'Jetson Nano', price: 59, memory: 4, tflops: 0.5, power: 10 },
  { name: 'Jetson Orin Nano', price: 499, memory: 8, tflops: 20, power: 15 },
  { name: 'Jetson AGX Orin', price: 1999, memory: 64, tflops: 275, power: 60 }
]

const BASE_MODEL: ModelMetrics = {
  size: 500, // MB
  accuracy: 95.2,
  inferenceTime: 120, // ms
  power: 25 // W
}

export default function EdgeAIOptimizer() {
  const [isRunning, setIsRunning] = useState(false)
  const [config, setConfig] = useState<OptimizationConfig>({
    quantization: 'none',
    pruningRate: 0,
    distillation: false
  })
  const [selectedHardware, setSelectedHardware] = useState<Hardware>(HARDWARE_OPTIONS[0])
  const [currentMetrics, setCurrentMetrics] = useState<ModelMetrics>(BASE_MODEL)
  const [optimizationHistory, setOptimizationHistory] = useState<{ time: number; accuracy: number; size: number }[]>([])
  const [progress, setProgress] = useState(0)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartCanvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()

  // Calculate optimized metrics based on configuration
  const calculateMetrics = useCallback((config: OptimizationConfig, hardware: Hardware): ModelMetrics => {
    let size = BASE_MODEL.size
    let accuracy = BASE_MODEL.accuracy
    let inferenceTime = BASE_MODEL.inferenceTime
    let power = BASE_MODEL.power

    // Quantization effects
    switch (config.quantization) {
      case 'int8':
        size *= 0.25 // 4x compression
        accuracy -= 0.3
        inferenceTime *= 0.4
        power *= 0.6
        break
      case 'int4':
        size *= 0.125 // 8x compression
        accuracy -= 1.2
        inferenceTime *= 0.3
        power *= 0.4
        break
      case 'mixed':
        size *= 0.35 // ~3x compression
        accuracy -= 0.15
        inferenceTime *= 0.5
        power *= 0.7
        break
    }

    // Pruning effects
    const pruningFactor = config.pruningRate / 100
    size *= (1 - pruningFactor * 0.8) // Can remove up to 80% of parameters
    accuracy -= pruningFactor * 2 // Lose some accuracy
    inferenceTime *= (1 - pruningFactor * 0.5)

    // Knowledge distillation effects
    if (config.distillation) {
      size *= 0.3 // Student model much smaller
      accuracy -= 1.5 // Student slightly less accurate
      inferenceTime *= 0.4
      power *= 0.5
    }

    // Hardware effects
    const hardwareSpeedup = hardware.tflops / HARDWARE_OPTIONS[0].tflops
    inferenceTime /= Math.sqrt(hardwareSpeedup) // Diminishing returns
    power *= (hardware.power / HARDWARE_OPTIONS[0].power) * 0.8

    return {
      size: Math.max(1, size),
      accuracy: Math.max(70, Math.min(100, accuracy)),
      inferenceTime: Math.max(5, inferenceTime),
      power: Math.max(2, power)
    }
  }, [])

  // Update metrics when config changes
  useEffect(() => {
    const metrics = calculateMetrics(config, selectedHardware)
    setCurrentMetrics(metrics)
  }, [config, selectedHardware, calculateMetrics])

  // Visualization: Neural network with pruning
  const drawNeuralNetwork = useCallback((ctx: CanvasRenderingContext2D, width: number, height: number) => {
    ctx.clearRect(0, 0, width, height)

    const layers = [8, 16, 16, 8, 4] // Network architecture
    const layerSpacing = width / (layers.length + 1)
    const nodeRadius = 8

    const pruningRate = config.pruningRate / 100
    const quantBits = config.quantization === 'int8' ? 8 : config.quantization === 'int4' ? 4 : 32

    // Draw connections first
    for (let l = 0; l < layers.length - 1; l++) {
      const currentLayerSize = layers[l]
      const nextLayerSize = layers[l + 1]
      const currentX = layerSpacing * (l + 1)
      const nextX = layerSpacing * (l + 2)

      for (let i = 0; i < currentLayerSize; i++) {
        const currentY = (height / (currentLayerSize + 1)) * (i + 1)

        // Randomly prune connections
        const isPruned = Math.random() < pruningRate

        for (let j = 0; j < nextLayerSize; j++) {
          const nextY = (height / (nextLayerSize + 1)) * (j + 1)

          // Skip pruned connections
          if (isPruned && Math.random() > 0.5) continue

          // Weight visualization
          const weight = Math.random()
          const alpha = isPruned ? 0.1 : 0.3 + weight * 0.4

          ctx.strokeStyle = `rgba(59, 130, 246, ${alpha})`
          ctx.lineWidth = isPruned ? 0.5 : 1 + weight * 1.5
          ctx.beginPath()
          ctx.moveTo(currentX, currentY)
          ctx.lineTo(nextX, nextY)
          ctx.stroke()
        }
      }
    }

    // Draw nodes
    for (let l = 0; l < layers.length; l++) {
      const layerSize = layers[l]
      const x = layerSpacing * (l + 1)

      for (let i = 0; i < layerSize; i++) {
        const y = (height / (layerSize + 1)) * (i + 1)
        const isPruned = Math.random() < pruningRate

        // Node color based on quantization
        let nodeColor = 'rgb(59, 130, 246)' // FP32 - blue
        if (config.quantization === 'int8') nodeColor = 'rgb(34, 197, 94)' // INT8 - green
        if (config.quantization === 'int4') nodeColor = 'rgb(251, 146, 60)' // INT4 - orange
        if (config.quantization === 'mixed') nodeColor = 'rgb(168, 85, 247)' // Mixed - purple

        ctx.fillStyle = isPruned ? 'rgba(107, 114, 128, 0.3)' : nodeColor
        ctx.beginPath()
        ctx.arc(x, y, nodeRadius, 0, Math.PI * 2)
        ctx.fill()

        // Outline
        ctx.strokeStyle = isPruned ? 'rgba(107, 114, 128, 0.5)' : 'rgba(255, 255, 255, 0.8)'
        ctx.lineWidth = 2
        ctx.stroke()
      }
    }

    // Legend
    ctx.font = '14px Inter, sans-serif'
    ctx.fillStyle = '#e5e7eb'
    ctx.fillText(`Pruning: ${config.pruningRate}%`, 20, 30)
    ctx.fillText(`Quantization: ${quantBits}-bit`, 20, 50)
    if (config.distillation) {
      ctx.fillText('Knowledge Distillation: ON', 20, 70)
    }
  }, [config])

  // Draw performance charts
  const drawCharts = useCallback((ctx: CanvasRenderingContext2D, width: number, height: number) => {
    ctx.clearRect(0, 0, width, height)

    const chartHeight = height - 80
    const chartWidth = width - 80
    const baseX = 50
    const baseY = height - 40

    // Grid
    ctx.strokeStyle = 'rgba(107, 114, 128, 0.3)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 5; i++) {
      const y = baseY - (chartHeight / 5) * i
      ctx.beginPath()
      ctx.moveTo(baseX, y)
      ctx.lineTo(baseX + chartWidth, y)
      ctx.stroke()
    }

    // Axes
    ctx.strokeStyle = 'rgba(229, 231, 235, 0.8)'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(baseX, baseY)
    ctx.lineTo(baseX, baseY - chartHeight)
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(baseX, baseY)
    ctx.lineTo(baseX + chartWidth, baseY)
    ctx.stroke()

    // Base model vs Optimized model comparison
    const metrics = [
      { label: 'Size (MB)', base: BASE_MODEL.size, current: currentMetrics.size, max: 500 },
      { label: 'Latency (ms)', base: BASE_MODEL.inferenceTime, current: currentMetrics.inferenceTime, max: 150 },
      { label: 'Accuracy (%)', base: BASE_MODEL.accuracy, current: currentMetrics.accuracy, max: 100 },
      { label: 'Power (W)', base: BASE_MODEL.power, current: currentMetrics.power, max: 30 }
    ]

    const barWidth = chartWidth / (metrics.length * 2.5)
    const barSpacing = chartWidth / (metrics.length + 1)

    metrics.forEach((metric, i) => {
      const x = baseX + barSpacing * (i + 0.5)

      // Base model bar (gray)
      const baseHeight = (metric.base / metric.max) * chartHeight
      ctx.fillStyle = 'rgba(107, 114, 128, 0.6)'
      ctx.fillRect(x - barWidth * 0.6, baseY - baseHeight, barWidth * 0.5, baseHeight)

      // Optimized model bar (blue)
      const currentHeight = (metric.current / metric.max) * chartHeight
      ctx.fillStyle = 'rgba(59, 130, 246, 0.8)'
      ctx.fillRect(x + barWidth * 0.1, baseY - currentHeight, barWidth * 0.5, currentHeight)

      // Labels
      ctx.font = '12px Inter, sans-serif'
      ctx.fillStyle = '#e5e7eb'
      ctx.textAlign = 'center'
      ctx.fillText(metric.label.split(' ')[0], x, baseY + 20)

      // Values
      ctx.font = '10px Inter, sans-serif'
      ctx.fillStyle = 'rgba(229, 231, 235, 0.7)'
      ctx.fillText(metric.base.toFixed(1), x - barWidth * 0.35, baseY - baseHeight - 5)
      ctx.fillStyle = 'rgba(59, 130, 246, 1)'
      ctx.fillText(metric.current.toFixed(1), x + barWidth * 0.35, baseY - currentHeight - 5)
    })

    // Title
    ctx.font = 'bold 16px Inter, sans-serif'
    ctx.fillStyle = '#f3f4f6'
    ctx.textAlign = 'left'
    ctx.fillText('Base vs Optimized Model', baseX, 25)

    // Legend
    ctx.font = '12px Inter, sans-serif'
    ctx.fillStyle = 'rgba(107, 114, 128, 1)'
    ctx.fillRect(baseX + chartWidth - 180, 10, 15, 15)
    ctx.fillStyle = '#e5e7eb'
    ctx.fillText('Base Model', baseX + chartWidth - 160, 22)

    ctx.fillStyle = 'rgba(59, 130, 246, 1)'
    ctx.fillRect(baseX + chartWidth - 70, 10, 15, 15)
    ctx.fillStyle = '#e5e7eb'
    ctx.fillText('Optimized', baseX + chartWidth - 50, 22)
  }, [currentMetrics])

  // Animation loop
  useEffect(() => {
    if (!isRunning) return

    const animate = () => {
      const canvas = canvasRef.current
      const chartCanvas = chartCanvasRef.current

      if (canvas && chartCanvas) {
        const ctx = canvas.getContext('2d')
        const chartCtx = chartCanvas.getContext('2d')

        if (ctx && chartCtx) {
          drawNeuralNetwork(ctx, canvas.width, canvas.height)
          drawCharts(chartCtx, chartCanvas.width, chartCanvas.height)
        }
      }

      setProgress(prev => {
        if (prev >= 100) {
          setIsRunning(false)
          return 100
        }
        return prev + 0.5
      })

      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning, drawNeuralNetwork, drawCharts])

  // Initial draw
  useEffect(() => {
    const canvas = canvasRef.current
    const chartCanvas = chartCanvasRef.current

    if (canvas && chartCanvas) {
      const ctx = canvas.getContext('2d')
      const chartCtx = chartCanvas.getContext('2d')

      if (ctx && chartCtx) {
        drawNeuralNetwork(ctx, canvas.width, canvas.height)
        drawCharts(chartCtx, chartCanvas.width, chartCanvas.height)
      }
    }
  }, [config, drawNeuralNetwork, drawCharts])

  const handleStart = () => {
    setIsRunning(true)
    setProgress(0)
  }

  const handlePause = () => {
    setIsRunning(false)
  }

  const handleReset = () => {
    setIsRunning(false)
    setProgress(0)
    setConfig({
      quantization: 'none',
      pruningRate: 0,
      distillation: false
    })
    setCurrentMetrics(BASE_MODEL)
    setOptimizationHistory([])
  }

  const calculateSpeedup = () => {
    return (BASE_MODEL.inferenceTime / currentMetrics.inferenceTime).toFixed(2)
  }

  const calculateCompression = () => {
    return (BASE_MODEL.size / currentMetrics.size).toFixed(2)
  }

  const calculateAccuracyLoss = () => {
    return (BASE_MODEL.accuracy - currentMetrics.accuracy).toFixed(2)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Zap className="w-10 h-10 text-yellow-400" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              Edge AI Optimizer
            </h1>
          </div>
          <p className="text-slate-300 text-lg">
            모델 양자화, 프루닝, 지식 증류로 엣지 디바이스 최적화 시뮬레이션
          </p>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Left Panel - Neural Network Visualization */}
          <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Layers className="w-5 h-5 text-blue-400" />
              Neural Network Structure
            </h2>
            <canvas
              ref={canvasRef}
              width={800}
              height={400}
              className="w-full h-[400px] bg-slate-900/50 rounded-lg"
            />
          </div>

          {/* Right Panel - Controls */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-green-400" />
              Optimization Settings
            </h2>

            {/* Quantization */}
            <div className="mb-6">
              <label className="block text-sm font-medium mb-2 text-slate-300">
                Quantization
              </label>
              <select
                value={config.quantization}
                onChange={(e) => setConfig({ ...config, quantization: e.target.value as any })}
                className="w-full bg-slate-900/50 border border-slate-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="none">FP32 (None)</option>
                <option value="int8">INT8 (4x compression)</option>
                <option value="int4">INT4 (8x compression)</option>
                <option value="mixed">Mixed Precision</option>
              </select>
            </div>

            {/* Pruning */}
            <div className="mb-6">
              <label className="block text-sm font-medium mb-2 text-slate-300">
                Pruning Rate: {config.pruningRate}%
              </label>
              <input
                type="range"
                min="0"
                max="90"
                step="10"
                value={config.pruningRate}
                onChange={(e) => setConfig({ ...config, pruningRate: Number(e.target.value) })}
                className="w-full accent-blue-500"
              />
              <div className="flex justify-between text-xs text-slate-400 mt-1">
                <span>0%</span>
                <span>50%</span>
                <span>90%</span>
              </div>
            </div>

            {/* Knowledge Distillation */}
            <div className="mb-6">
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.distillation}
                  onChange={(e) => setConfig({ ...config, distillation: e.target.checked })}
                  className="w-5 h-5 accent-blue-500"
                />
                <span className="text-sm font-medium text-slate-300">
                  Knowledge Distillation
                </span>
              </label>
            </div>

            {/* Hardware Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium mb-2 text-slate-300 flex items-center gap-2">
                <Cpu className="w-4 h-4" />
                Target Hardware
              </label>
              <select
                value={selectedHardware.name}
                onChange={(e) => {
                  const hw = HARDWARE_OPTIONS.find(h => h.name === e.target.value)
                  if (hw) setSelectedHardware(hw)
                }}
                className="w-full bg-slate-900/50 border border-slate-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {HARDWARE_OPTIONS.map(hw => (
                  <option key={hw.name} value={hw.name}>
                    {hw.name} (${hw.price})
                  </option>
                ))}
              </select>
              <div className="mt-2 text-xs text-slate-400 space-y-1">
                <div>Memory: {selectedHardware.memory}GB</div>
                <div>Performance: {selectedHardware.tflops} TFLOPS</div>
                <div>Power: {selectedHardware.power}W</div>
              </div>
            </div>

            {/* Control Buttons */}
            <div className="flex gap-2 mb-4">
              {!isRunning ? (
                <button
                  onClick={handleStart}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  Optimize
                </button>
              ) : (
                <button
                  onClick={handlePause}
                  className="flex-1 bg-yellow-600 hover:bg-yellow-700 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  <Pause className="w-4 h-4" />
                  Pause
                </button>
              )}
              <button
                onClick={handleReset}
                className="bg-slate-700 hover:bg-slate-600 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>

            {/* Progress */}
            {progress > 0 && (
              <div className="mb-4">
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-400">Optimization Progress</span>
                  <span className="text-blue-400">{progress.toFixed(0)}%</span>
                </div>
                <div className="w-full bg-slate-700 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Key Metrics */}
            <div className="bg-slate-900/50 rounded-lg p-4 space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Speedup:</span>
                <span className="text-green-400 font-semibold">{calculateSpeedup()}x</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Compression:</span>
                <span className="text-blue-400 font-semibold">{calculateCompression()}x</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Accuracy Loss:</span>
                <span className="text-red-400 font-semibold">-{calculateAccuracyLoss()}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Panel - Performance Charts */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Info className="w-5 h-5 text-purple-400" />
            Performance Comparison
          </h2>
          <canvas
            ref={chartCanvasRef}
            width={1200}
            height={300}
            className="w-full h-[300px] bg-slate-900/50 rounded-lg"
          />
        </div>

        {/* Info Panel */}
        <div className="mt-6 bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
          <h3 className="text-lg font-semibold mb-3 text-blue-400">Optimization Techniques</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-slate-300">
            <div>
              <h4 className="font-semibold text-white mb-2">Quantization</h4>
              <p>FP32 → INT8/INT4로 변환하여 모델 크기와 추론 속도를 대폭 개선. 정확도 손실은 최소화.</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">Pruning</h4>
              <p>중요도가 낮은 가중치를 제거하여 50-90% 파라미터 감소. Sparse 연산으로 효율 향상.</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">Knowledge Distillation</h4>
              <p>큰 Teacher 모델의 지식을 작은 Student 모델로 전이. 70% 크기 감소, 정확도 유지.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
