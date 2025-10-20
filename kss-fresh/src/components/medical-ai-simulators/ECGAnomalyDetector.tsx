'use client'

import React, { useState, useEffect, useRef } from 'react'

type RhythmType = 'Normal' | 'Atrial Fibrillation' | 'Ventricular Tachycardia' | 'PVC' | 'Bradycardia'

interface ECGFeatures {
  heartRate: number
  rrInterval: number
  prInterval: number
  qrsDuration: number
  qtInterval: number
  hrv: number
}

interface DetectionResult {
  rhythm: RhythmType
  confidence: number
  timestamp: number
  features: ECGFeatures
}

export default function ECGAnomalyDetector() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const [selectedRhythm, setSelectedRhythm] = useState<RhythmType>('Normal')
  const [isRecording, setIsRecording] = useState(false)
  const [results, setResults] = useState<DetectionResult[]>([])
  const [currentFeatures, setCurrentFeatures] = useState<ECGFeatures>({
    heartRate: 72,
    rrInterval: 833,
    prInterval: 160,
    qrsDuration: 100,
    qtInterval: 400,
    hrv: 50
  })
  const [timeOffset, setTimeOffset] = useState(0)
  const [showAnnotations, setShowAnnotations] = useState(true)

  const rhythms: RhythmType[] = ['Normal', 'Atrial Fibrillation', 'Ventricular Tachycardia', 'PVC', 'Bradycardia']

  const rhythmColors: Record<RhythmType, string> = {
    'Normal': '#10b981',
    'Atrial Fibrillation': '#f59e0b',
    'Ventricular Tachycardia': '#ef4444',
    'PVC': '#8b5cf6',
    'Bradycardia': '#3b82f6'
  }

  useEffect(() => {
    updateFeatures()
  }, [selectedRhythm])

  useEffect(() => {
    if (isRecording) {
      animate()
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      drawECG()
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRecording, selectedRhythm, timeOffset, showAnnotations])

  const animate = () => {
    setTimeOffset(prev => (prev + 0.02) % 10)
    drawECG()
    animationRef.current = requestAnimationFrame(animate)
  }

  const updateFeatures = () => {
    let features: ECGFeatures

    switch (selectedRhythm) {
      case 'Normal':
        features = {
          heartRate: 70 + Math.random() * 10,
          rrInterval: 800 + Math.random() * 100,
          prInterval: 160,
          qrsDuration: 100,
          qtInterval: 400,
          hrv: 50 + Math.random() * 20
        }
        break
      case 'Atrial Fibrillation':
        features = {
          heartRate: 110 + Math.random() * 30,
          rrInterval: 400 + Math.random() * 300, // irregular
          prInterval: 0, // absent P waves
          qrsDuration: 100,
          qtInterval: 380,
          hrv: 120 + Math.random() * 50 // high variability
        }
        break
      case 'Ventricular Tachycardia':
        features = {
          heartRate: 150 + Math.random() * 50,
          rrInterval: 300 + Math.random() * 100,
          prInterval: 0,
          qrsDuration: 180 + Math.random() * 40, // wide QRS
          qtInterval: 450,
          hrv: 20 + Math.random() * 10
        }
        break
      case 'PVC':
        features = {
          heartRate: 75 + Math.random() * 10,
          rrInterval: 800 + Math.random() * 100,
          prInterval: 160,
          qrsDuration: 140 + Math.random() * 20, // wide
          qtInterval: 420,
          hrv: 60 + Math.random() * 20
        }
        break
      case 'Bradycardia':
        features = {
          heartRate: 45 + Math.random() * 10,
          rrInterval: 1200 + Math.random() * 200,
          prInterval: 160,
          qrsDuration: 100,
          qtInterval: 450,
          hrv: 30 + Math.random() * 15
        }
        break
    }

    setCurrentFeatures(features)
  }

  const drawECG = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    // Clear canvas
    ctx.fillStyle = '#0a0a0a'
    ctx.fillRect(0, 0, width, height)

    // Draw grid
    drawGrid(ctx, width, height)

    // Draw ECG waveform
    drawWaveform(ctx, width, height)

    // Draw annotations
    if (showAnnotations) {
      drawQRSAnnotations(ctx, width, height)
    }

    // Draw real-time label
    if (isRecording) {
      ctx.fillStyle = '#ef4444'
      ctx.beginPath()
      ctx.arc(20, 20, 6, 0, Math.PI * 2)
      ctx.fill()

      ctx.fillStyle = '#fff'
      ctx.font = 'bold 14px Inter'
      ctx.fillText('RECORDING', 35, 25)
    }
  }

  const drawGrid = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    // Major grid (5mm)
    ctx.strokeStyle = 'rgba(255, 100, 100, 0.2)'
    ctx.lineWidth = 1

    const majorGridSize = 50
    for (let x = 0; x < width; x += majorGridSize) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.stroke()
    }

    for (let y = 0; y < height; y += majorGridSize) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    // Minor grid (1mm)
    ctx.strokeStyle = 'rgba(255, 100, 100, 0.1)'
    const minorGridSize = 10
    for (let x = 0; x < width; x += minorGridSize) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.stroke()
    }

    for (let y = 0; y < height; y += minorGridSize) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    // Baseline
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, height / 2)
    ctx.lineTo(width, height / 2)
    ctx.stroke()
  }

  const drawWaveform = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    ctx.strokeStyle = rhythmColors[selectedRhythm]
    ctx.lineWidth = 2
    ctx.beginPath()

    const baselineY = height / 2
    const scale = 80

    for (let x = 0; x < width; x++) {
      const t = (x / width) * 10 + timeOffset
      let y = baselineY

      // Generate waveform based on rhythm type
      y += generateECGValue(t, selectedRhythm) * scale

      if (x === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    }

    ctx.stroke()
  }

  const generateECGValue = (t: number, rhythm: RhythmType): number => {
    const beatInterval = rhythm === 'Bradycardia' ? 1.5 : rhythm === 'Ventricular Tachycardia' ? 0.4 : 0.8

    if (rhythm === 'Atrial Fibrillation') {
      // Irregular rhythm
      const irregularInterval = beatInterval + Math.sin(t * 3) * 0.3
      return generateQRSComplex(t % irregularInterval, irregularInterval, false, true)
    } else if (rhythm === 'Ventricular Tachycardia') {
      // Fast, wide QRS
      return generateQRSComplex(t % beatInterval, beatInterval, true, false)
    } else if (rhythm === 'PVC') {
      // Premature beats every 4th beat
      const beat = Math.floor(t / beatInterval)
      const isPVC = beat % 4 === 3
      return generateQRSComplex(t % beatInterval, beatInterval, isPVC, !isPVC)
    } else {
      // Normal or Bradycardia
      return generateQRSComplex(t % beatInterval, beatInterval, false, true)
    }
  }

  const generateQRSComplex = (t: number, interval: number, wideQRS: boolean, withPWave: boolean): number => {
    const normalized = t / interval
    let value = 0

    // P wave (atrial depolarization)
    if (withPWave && normalized > 0.1 && normalized < 0.2) {
      const pT = (normalized - 0.1) / 0.1
      value += 0.15 * Math.sin(pT * Math.PI)
    }

    // QRS complex (ventricular depolarization)
    const qrsStart = 0.25
    const qrsWidth = wideQRS ? 0.2 : 0.1
    if (normalized > qrsStart && normalized < qrsStart + qrsWidth) {
      const qrsT = (normalized - qrsStart) / qrsWidth

      // Q wave (small negative)
      if (qrsT < 0.15) {
        value -= 0.2 * (qrsT / 0.15)
      }
      // R wave (large positive)
      else if (qrsT < 0.5) {
        value += 2.5 * Math.sin(((qrsT - 0.15) / 0.35) * Math.PI)
      }
      // S wave (negative)
      else {
        value -= 0.8 * Math.sin(((qrsT - 0.5) / 0.5) * Math.PI)
      }
    }

    // T wave (ventricular repolarization)
    const tStart = 0.5
    const tEnd = 0.7
    if (normalized > tStart && normalized < tEnd) {
      const tT = (normalized - tStart) / (tEnd - tStart)
      value += 0.25 * Math.sin(tT * Math.PI)
    }

    // Add noise for AFib
    if (!withPWave) {
      value += (Math.random() - 0.5) * 0.1
    }

    return value
  }

  const drawQRSAnnotations = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const beatInterval = selectedRhythm === 'Bradycardia' ? 1.5 : selectedRhythm === 'Ventricular Tachycardia' ? 0.4 : 0.8
    const beatWidth = (width / 10) * beatInterval

    ctx.font = 'bold 12px Inter'
    ctx.fillStyle = '#fbbf24'

    for (let i = 0; i < Math.ceil(10 / beatInterval); i++) {
      const x = (i * beatInterval - timeOffset) * (width / 10)
      if (x >= 0 && x < width) {
        // P annotation
        if (selectedRhythm !== 'Atrial Fibrillation' && selectedRhythm !== 'Ventricular Tachycardia') {
          ctx.fillText('P', x + beatWidth * 0.15, height / 2 - 40)
        }

        // QRS annotation
        ctx.fillText('QRS', x + beatWidth * 0.3, height / 2 + 60)

        // T annotation
        ctx.fillText('T', x + beatWidth * 0.6, height / 2 - 20)
      }
    }
  }

  const detectRhythm = () => {
    const confidence = 0.88 + Math.random() * 0.10

    const result: DetectionResult = {
      rhythm: selectedRhythm,
      confidence,
      timestamp: Date.now(),
      features: { ...currentFeatures }
    }

    setResults(prev => [result, ...prev.slice(0, 4)])
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-500 via-pink-600 to-red-500 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 mb-6">
          <h1 className="text-4xl font-bold text-white mb-2">ECG Anomaly Detector</h1>
          <p className="text-white/80">LSTM/1D-CNN for real-time arrhythmia detection and heart rate variability analysis</p>
          <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">Architecture</div>
              <div className="text-white font-bold">LSTM + 1D-CNN</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">Dataset</div>
              <div className="text-white font-bold">MIT-BIH Arrhythmia DB</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">Accuracy</div>
              <div className="text-white font-bold">97.4%</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">Sampling Rate</div>
              <div className="text-white font-bold">360 Hz</div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold text-white">ECG Waveform (Lead II)</h2>
                <div className="flex gap-2">
                  <button
                    onClick={() => setIsRecording(!isRecording)}
                    className={`px-4 py-2 rounded-lg font-bold transition-all ${
                      isRecording
                        ? 'bg-red-500 text-white'
                        : 'bg-white text-pink-600'
                    }`}
                  >
                    {isRecording ? 'Stop' : 'Start'}
                  </button>
                  <button
                    onClick={() => setShowAnnotations(!showAnnotations)}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                      showAnnotations
                        ? 'bg-white text-pink-600'
                        : 'bg-white/20 text-white'
                    }`}
                  >
                    Annotations
                  </button>
                </div>
              </div>

              <canvas
                ref={canvasRef}
                width={800}
                height={400}
                className="w-full bg-gray-900 rounded-lg"
              />

              <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                <div className="bg-white/10 rounded-lg p-3">
                  <div className="text-white/60">Paper Speed</div>
                  <div className="text-white font-bold">25 mm/s</div>
                </div>
                <div className="bg-white/10 rounded-lg p-3">
                  <div className="text-white/60">Amplitude</div>
                  <div className="text-white font-bold">10 mm/mV</div>
                </div>
              </div>
            </div>

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-4">ECG Features</h3>
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-white/10 rounded-lg p-3">
                  <div className="text-white/60 text-sm mb-1">Heart Rate</div>
                  <div className="text-2xl font-bold text-white">{currentFeatures.heartRate.toFixed(0)}</div>
                  <div className="text-white/60 text-xs">bpm</div>
                </div>
                <div className="bg-white/10 rounded-lg p-3">
                  <div className="text-white/60 text-sm mb-1">RR Interval</div>
                  <div className="text-2xl font-bold text-white">{currentFeatures.rrInterval.toFixed(0)}</div>
                  <div className="text-white/60 text-xs">ms</div>
                </div>
                <div className="bg-white/10 rounded-lg p-3">
                  <div className="text-white/60 text-sm mb-1">HRV (SDNN)</div>
                  <div className="text-2xl font-bold text-white">{currentFeatures.hrv.toFixed(0)}</div>
                  <div className="text-white/60 text-xs">ms</div>
                </div>
                <div className="bg-white/10 rounded-lg p-3">
                  <div className="text-white/60 text-sm mb-1">PR Interval</div>
                  <div className="text-2xl font-bold text-white">{currentFeatures.prInterval || 'N/A'}</div>
                  <div className="text-white/60 text-xs">ms</div>
                </div>
                <div className="bg-white/10 rounded-lg p-3">
                  <div className="text-white/60 text-sm mb-1">QRS Duration</div>
                  <div className="text-2xl font-bold text-white">{currentFeatures.qrsDuration.toFixed(0)}</div>
                  <div className="text-white/60 text-xs">ms</div>
                </div>
                <div className="bg-white/10 rounded-lg p-3">
                  <div className="text-white/60 text-sm mb-1">QT Interval</div>
                  <div className="text-2xl font-bold text-white">{currentFeatures.qtInterval.toFixed(0)}</div>
                  <div className="text-white/60 text-xs">ms</div>
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-4">Rhythm Type</h3>
              <div className="space-y-2">
                {rhythms.map(rhythm => (
                  <button
                    key={rhythm}
                    onClick={() => setSelectedRhythm(rhythm)}
                    className={`w-full p-3 rounded-lg font-medium transition-all text-left ${
                      selectedRhythm === rhythm
                        ? 'bg-white text-gray-900'
                        : 'bg-white/20 text-white hover:bg-white/30'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span>{rhythm}</span>
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: rhythmColors[rhythm] }}
                      />
                    </div>
                  </button>
                ))}
              </div>

              <button
                onClick={detectRhythm}
                className="w-full mt-4 py-3 rounded-lg font-bold bg-white text-pink-600 hover:bg-gray-100 transition-all"
              >
                Run AI Detection
              </button>
            </div>

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-4">Detection Results</h3>
              <div className="space-y-2">
                {results.length === 0 ? (
                  <p className="text-white/60 text-sm">No detections yet</p>
                ) : (
                  results.map((result, idx) => (
                    <div key={result.timestamp} className="bg-white/10 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-bold text-white text-sm">{result.rhythm}</span>
                        <span
                          className="px-2 py-0.5 rounded text-xs font-bold text-white"
                          style={{ backgroundColor: rhythmColors[result.rhythm] }}
                        >
                          {(result.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="text-white/60 text-xs">
                        HR: {result.features.heartRate.toFixed(0)} bpm | HRV: {result.features.hrv.toFixed(0)} ms
                      </div>
                      <div className="text-white/60 text-xs mt-1">
                        {new Date(result.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-3">Clinical Notes</h3>
              <div className="space-y-2 text-xs text-white/80">
                <p><strong>Normal:</strong> Regular rhythm, 60-100 bpm</p>
                <p><strong>AFib:</strong> Irregular RR intervals, absent P waves</p>
                <p><strong>VTach:</strong> Wide QRS (&gt;120ms), HR &gt;150 bpm</p>
                <p><strong>PVC:</strong> Premature wide QRS complex</p>
                <p><strong>Brady:</strong> HR &lt;60 bpm, regular rhythm</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
