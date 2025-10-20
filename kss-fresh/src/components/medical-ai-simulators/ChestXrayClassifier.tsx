'use client'

import React, { useState, useEffect, useRef } from 'react'

type Disease = 'Normal' | 'Pneumonia' | 'Tuberculosis' | 'COVID-19'
type ViewMode = 'xray' | 'gradcam' | 'activation' | 'metrics'

interface ClassificationResult {
  disease: Disease
  confidence: number
  timestamp: number
}

interface Metrics {
  accuracy: number
  sensitivity: number
  specificity: number
  aucRoc: number
}

interface ConfusionMatrix {
  truePositive: number
  trueNegative: number
  falsePositive: number
  falseNegative: number
}

export default function ChestXrayClassifier() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [selectedDisease, setSelectedDisease] = useState<Disease>('Normal')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [viewMode, setViewMode] = useState<ViewMode>('xray')
  const [results, setResults] = useState<ClassificationResult[]>([])
  const [confusionMatrix, setConfusionMatrix] = useState<ConfusionMatrix>({
    truePositive: 342,
    trueNegative: 289,
    falsePositive: 23,
    falseNegative: 18
  })
  const [metrics, setMetrics] = useState<Metrics>({
    accuracy: 0.939,
    sensitivity: 0.950,
    specificity: 0.926,
    aucRoc: 0.839
  })
  const [layerIndex, setLayerIndex] = useState(0)

  const diseases: Disease[] = ['Normal', 'Pneumonia', 'Tuberculosis', 'COVID-19']

  const diseaseColors: Record<Disease, string> = {
    'Normal': '#10b981',
    'Pneumonia': '#f59e0b',
    'Tuberculosis': '#ef4444',
    'COVID-19': '#8b5cf6'
  }

  useEffect(() => {
    drawXray()
  }, [selectedDisease, viewMode, layerIndex])

  const drawXray = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    // Clear canvas
    ctx.fillStyle = '#1a1a1a'
    ctx.fillRect(0, 0, width, height)

    if (viewMode === 'xray') {
      drawChestXray(ctx, width, height)
    } else if (viewMode === 'gradcam') {
      drawChestXray(ctx, width, height)
      drawGradCAM(ctx, width, height)
    } else if (viewMode === 'activation') {
      drawActivationMap(ctx, width, height)
    } else if (viewMode === 'metrics') {
      drawMetricsVisualization(ctx, width, height)
    }
  }

  const drawChestXray = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    // Create grayscale gradient for X-ray background
    const gradient = ctx.createRadialGradient(width / 2, height / 2, 50, width / 2, height / 2, width / 2)
    gradient.addColorStop(0, '#4a4a4a')
    gradient.addColorStop(0.6, '#2a2a2a')
    gradient.addColorStop(1, '#1a1a1a')
    ctx.fillStyle = gradient
    ctx.fillRect(0, 0, width, height)

    // Draw ribcage structure
    ctx.strokeStyle = '#6a6a6a'
    ctx.lineWidth = 2

    // Ribs
    for (let i = 0; i < 8; i++) {
      const y = height * 0.2 + i * (height * 0.6 / 8)
      const curve = Math.sin(i * 0.3) * 40

      ctx.beginPath()
      ctx.moveTo(width * 0.3, y)
      ctx.quadraticCurveTo(width * 0.2 + curve, y + 20, width * 0.3, y + 40)
      ctx.stroke()

      ctx.beginPath()
      ctx.moveTo(width * 0.7, y)
      ctx.quadraticCurveTo(width * 0.8 - curve, y + 20, width * 0.7, y + 40)
      ctx.stroke()
    }

    // Spine
    ctx.beginPath()
    ctx.moveTo(width / 2, height * 0.15)
    ctx.lineTo(width / 2, height * 0.85)
    ctx.stroke()

    // Lungs
    ctx.fillStyle = 'rgba(80, 80, 80, 0.3)'

    // Left lung
    ctx.beginPath()
    ctx.ellipse(width * 0.35, height * 0.45, width * 0.12, height * 0.25, 0, 0, Math.PI * 2)
    ctx.fill()

    // Right lung
    ctx.beginPath()
    ctx.ellipse(width * 0.65, height * 0.45, width * 0.12, height * 0.25, 0, 0, Math.PI * 2)
    ctx.fill()

    // Heart
    ctx.fillStyle = 'rgba(100, 100, 100, 0.4)'
    ctx.beginPath()
    ctx.ellipse(width * 0.45, height * 0.5, width * 0.08, height * 0.12, -0.3, 0, Math.PI * 2)
    ctx.fill()

    // Add pathology indicators based on disease
    if (selectedDisease === 'Pneumonia') {
      // Pneumonia - cloudy opacities in lower right lung
      ctx.fillStyle = 'rgba(200, 200, 200, 0.6)'
      ctx.beginPath()
      ctx.ellipse(width * 0.65, height * 0.6, width * 0.08, height * 0.12, 0, 0, Math.PI * 2)
      ctx.fill()

      // Additional patches
      ctx.beginPath()
      ctx.arc(width * 0.68, height * 0.55, width * 0.04, 0, Math.PI * 2)
      ctx.fill()
    } else if (selectedDisease === 'Tuberculosis') {
      // TB - cavitary lesions in upper lobes
      ctx.strokeStyle = 'rgba(255, 100, 100, 0.8)'
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.arc(width * 0.35, height * 0.3, width * 0.03, 0, Math.PI * 2)
      ctx.stroke()

      ctx.beginPath()
      ctx.arc(width * 0.65, height * 0.32, width * 0.025, 0, Math.PI * 2)
      ctx.stroke()

      // Fibrotic changes
      ctx.fillStyle = 'rgba(180, 180, 180, 0.5)'
      ctx.beginPath()
      ctx.ellipse(width * 0.35, height * 0.35, width * 0.06, height * 0.08, 0, 0, Math.PI * 2)
      ctx.fill()
    } else if (selectedDisease === 'COVID-19') {
      // COVID-19 - bilateral ground-glass opacities
      ctx.fillStyle = 'rgba(220, 220, 220, 0.5)'

      // Multiple patches in both lungs
      const patches = [
        { x: 0.32, y: 0.4, rx: 0.05, ry: 0.06 },
        { x: 0.38, y: 0.5, rx: 0.04, ry: 0.05 },
        { x: 0.62, y: 0.42, rx: 0.055, ry: 0.065 },
        { x: 0.68, y: 0.52, rx: 0.045, ry: 0.055 }
      ]

      patches.forEach(patch => {
        ctx.beginPath()
        ctx.ellipse(width * patch.x, height * patch.y, width * patch.rx, height * patch.ry, 0, 0, Math.PI * 2)
        ctx.fill()
      })
    }
  }

  const drawGradCAM = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    if (selectedDisease === 'Normal') return

    // Grad-CAM heatmap overlay
    const heatmapData = ctx.createImageData(width, height)
    const data = heatmapData.data

    // Define region of interest based on disease
    let centerX = width * 0.65
    let centerY = height * 0.6

    if (selectedDisease === 'Tuberculosis') {
      centerY = height * 0.3
    } else if (selectedDisease === 'COVID-19') {
      centerX = width * 0.5
      centerY = height * 0.45
    }

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4
        const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2)
        const maxDist = width * 0.15

        if (distance < maxDist) {
          const intensity = 1 - (distance / maxDist)
          const alpha = intensity * 0.6

          // Heat color map (red-yellow)
          data[idx] = 255
          data[idx + 1] = Math.floor(255 * (1 - intensity * 0.5))
          data[idx + 2] = 0
          data[idx + 3] = alpha * 255
        }
      }
    }

    ctx.putImageData(heatmapData, 0, 0)

    // Add label
    ctx.fillStyle = '#fff'
    ctx.font = 'bold 14px Inter'
    ctx.fillText('Grad-CAM: Region of Interest', 10, 25)
  }

  const drawActivationMap = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const layers = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'MaxPool', 'Dense']
    const gridSize = 8
    const cellSize = Math.min(width, height) / gridSize - 2

    ctx.fillStyle = '#fff'
    ctx.font = 'bold 16px Inter'
    ctx.fillText(`Layer ${layerIndex + 1}: ${layers[layerIndex]}`, 10, 25)

    // Draw activation grid
    for (let row = 0; row < gridSize; row++) {
      for (let col = 0; col < gridSize; col++) {
        const x = col * (cellSize + 2) + 10
        const y = row * (cellSize + 2) + 40

        // Simulate activation values
        const seed = (row * gridSize + col + layerIndex * 100) * 0.1
        const activation = Math.abs(Math.sin(seed) * Math.cos(seed * 1.5))

        // Color based on activation strength
        const intensity = Math.floor(activation * 255)
        ctx.fillStyle = `rgb(${intensity}, ${Math.floor(intensity * 0.5)}, ${255 - intensity})`
        ctx.fillRect(x, y, cellSize, cellSize)

        // Add border
        ctx.strokeStyle = '#333'
        ctx.lineWidth = 1
        ctx.strokeRect(x, y, cellSize, cellSize)
      }
    }

    // Add info
    ctx.fillStyle = '#aaa'
    ctx.font = '12px Inter'
    ctx.fillText(`Activation map size: ${gridSize}x${gridSize}`, 10, height - 10)
  }

  const drawMetricsVisualization = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    // Draw confusion matrix
    const matrixSize = Math.min(width, height) * 0.4
    const startX = (width - matrixSize) / 2
    const startY = 50
    const cellSize = matrixSize / 2

    // Labels
    ctx.fillStyle = '#fff'
    ctx.font = 'bold 14px Inter'
    ctx.fillText('Confusion Matrix', startX, startY - 10)

    ctx.font = '12px Inter'
    ctx.fillText('Predicted', startX + matrixSize / 2 - 30, startY - 25)

    ctx.save()
    ctx.translate(startX - 40, startY + matrixSize / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText('Actual', -20, 0)
    ctx.restore()

    // Positive/Negative labels
    ctx.fillText('Pos', startX + cellSize / 2 - 15, startY - 5)
    ctx.fillText('Neg', startX + cellSize * 1.5 - 15, startY - 5)
    ctx.fillText('Pos', startX - 30, startY + cellSize / 2 + 5)
    ctx.fillText('Neg', startX - 30, startY + cellSize * 1.5 + 5)

    // Matrix cells
    const matrix = [
      [confusionMatrix.truePositive, confusionMatrix.falseNegative],
      [confusionMatrix.falsePositive, confusionMatrix.trueNegative]
    ]

    const colors = [
      ['rgba(34, 197, 94, 0.6)', 'rgba(239, 68, 68, 0.6)'],
      ['rgba(239, 68, 68, 0.6)', 'rgba(34, 197, 94, 0.6)']
    ]

    for (let row = 0; row < 2; row++) {
      for (let col = 0; col < 2; col++) {
        const x = startX + col * cellSize
        const y = startY + row * cellSize

        ctx.fillStyle = colors[row][col]
        ctx.fillRect(x, y, cellSize, cellSize)

        ctx.strokeStyle = '#555'
        ctx.lineWidth = 2
        ctx.strokeRect(x, y, cellSize, cellSize)

        ctx.fillStyle = '#fff'
        ctx.font = 'bold 24px Inter'
        ctx.textAlign = 'center'
        ctx.fillText(matrix[row][col].toString(), x + cellSize / 2, y + cellSize / 2 + 8)
      }
    }

    // Draw metrics bars
    const metricsY = startY + matrixSize + 60
    const barWidth = width * 0.8
    const barHeight = 30
    const barSpacing = 50
    const barStartX = (width - barWidth) / 2

    const metricsArray = [
      { name: 'Accuracy', value: metrics.accuracy, color: '#3b82f6' },
      { name: 'Sensitivity', value: metrics.sensitivity, color: '#10b981' },
      { name: 'Specificity', value: metrics.specificity, color: '#8b5cf6' },
      { name: 'AUC-ROC', value: metrics.aucRoc, color: '#f59e0b' }
    ]

    metricsArray.forEach((metric, idx) => {
      const y = metricsY + idx * barSpacing

      // Label
      ctx.fillStyle = '#fff'
      ctx.font = '14px Inter'
      ctx.textAlign = 'left'
      ctx.fillText(metric.name, barStartX, y - 5)

      // Background bar
      ctx.fillStyle = '#2a2a2a'
      ctx.fillRect(barStartX, y, barWidth, barHeight)

      // Value bar
      ctx.fillStyle = metric.color
      ctx.fillRect(barStartX, y, barWidth * metric.value, barHeight)

      // Border
      ctx.strokeStyle = '#555'
      ctx.lineWidth = 1
      ctx.strokeRect(barStartX, y, barWidth, barHeight)

      // Value text
      ctx.fillStyle = '#fff'
      ctx.font = 'bold 14px Inter'
      ctx.textAlign = 'center'
      ctx.fillText(`${(metric.value * 100).toFixed(1)}%`, barStartX + barWidth * metric.value / 2, y + barHeight / 2 + 5)
    })

    ctx.textAlign = 'left'
  }

  const analyzeXray = () => {
    setIsAnalyzing(true)

    setTimeout(() => {
      // Simulate CNN inference
      const baseConfidence = 0.75 + Math.random() * 0.2
      const result: ClassificationResult = {
        disease: selectedDisease,
        confidence: baseConfidence,
        timestamp: Date.now()
      }

      setResults(prev => [result, ...prev.slice(0, 4)])
      setIsAnalyzing(false)

      // Update confusion matrix
      const isCorrect = Math.random() > 0.1 // 90% accuracy simulation
      setConfusionMatrix(prev => ({
        truePositive: prev.truePositive + (isCorrect && selectedDisease !== 'Normal' ? 1 : 0),
        trueNegative: prev.trueNegative + (isCorrect && selectedDisease === 'Normal' ? 1 : 0),
        falsePositive: prev.falsePositive + (!isCorrect && selectedDisease === 'Normal' ? 1 : 0),
        falseNegative: prev.falseNegative + (!isCorrect && selectedDisease !== 'Normal' ? 1 : 0)
      }))

      // Recalculate metrics
      calculateMetrics()
    }, 2000)
  }

  const calculateMetrics = () => {
    const { truePositive, trueNegative, falsePositive, falseNegative } = confusionMatrix
    const total = truePositive + trueNegative + falsePositive + falseNegative

    setMetrics({
      accuracy: (truePositive + trueNegative) / total,
      sensitivity: truePositive / (truePositive + falseNegative),
      specificity: trueNegative / (trueNegative + falsePositive),
      aucRoc: 0.839 + (Math.random() - 0.5) * 0.05
    })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-500 via-pink-600 to-red-500 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 mb-6">
          <h1 className="text-4xl font-bold text-white mb-2">Chest X-ray Classifier</h1>
          <p className="text-white/80">CNN-based pneumonia, tuberculosis, and COVID-19 detection using deep learning</p>
          <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">Architecture</div>
              <div className="text-white font-bold">CheXNet (121-layer DenseNet)</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">Dataset</div>
              <div className="text-white font-bold">ChestX-ray14 (112K images)</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">Performance</div>
              <div className="text-white font-bold">AUC 0.839 (Expert-level)</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">Reference</div>
              <div className="text-white font-bold">Rajpurkar et al. (2017)</div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-white/10 backdrop-blur-md rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-white">X-ray Visualization</h2>
              <div className="flex gap-2">
                {(['xray', 'gradcam', 'activation', 'metrics'] as ViewMode[]).map(mode => (
                  <button
                    key={mode}
                    onClick={() => setViewMode(mode)}
                    className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                      viewMode === mode
                        ? 'bg-white text-pink-600'
                        : 'bg-white/20 text-white hover:bg-white/30'
                    }`}
                  >
                    {mode === 'xray' ? 'X-ray' : mode === 'gradcam' ? 'Grad-CAM' : mode === 'activation' ? 'Layers' : 'Metrics'}
                  </button>
                ))}
              </div>
            </div>

            <canvas
              ref={canvasRef}
              width={600}
              height={600}
              className="w-full bg-gray-900 rounded-lg"
            />

            {viewMode === 'activation' && (
              <div className="mt-4">
                <label className="text-white font-medium block mb-2">
                  CNN Layer: {layerIndex + 1} / 7
                </label>
                <input
                  type="range"
                  min="0"
                  max="6"
                  value={layerIndex}
                  onChange={(e) => setLayerIndex(Number(e.target.value))}
                  className="w-full"
                />
              </div>
            )}
          </div>

          <div className="space-y-6">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-4">Select Disease</h3>
              <div className="space-y-2">
                {diseases.map(disease => (
                  <button
                    key={disease}
                    onClick={() => setSelectedDisease(disease)}
                    className={`w-full p-3 rounded-lg font-medium transition-all ${
                      selectedDisease === disease
                        ? 'bg-white text-gray-900'
                        : 'bg-white/20 text-white hover:bg-white/30'
                    }`}
                  >
                    {disease}
                  </button>
                ))}
              </div>

              <button
                onClick={analyzeXray}
                disabled={isAnalyzing}
                className={`w-full mt-4 py-3 rounded-lg font-bold transition-all ${
                  isAnalyzing
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-white text-pink-600 hover:bg-gray-100'
                }`}
              >
                {isAnalyzing ? 'Analyzing...' : 'Run Classification'}
              </button>
            </div>

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-4">Recent Results</h3>
              <div className="space-y-2">
                {results.length === 0 ? (
                  <p className="text-white/60 text-sm">No results yet</p>
                ) : (
                  results.map((result, idx) => (
                    <div
                      key={result.timestamp}
                      className="bg-white/10 rounded-lg p-3"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-bold text-white">{result.disease}</span>
                        <span
                          className="px-2 py-0.5 rounded text-xs font-bold"
                          style={{ backgroundColor: diseaseColors[result.disease] }}
                        >
                          {(result.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="text-white/60 text-xs">
                        {new Date(result.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
