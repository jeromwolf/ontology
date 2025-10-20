'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, Pause } from 'lucide-react'

export default function WaveInterference() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [source1, setSource1] = useState({ x: 300, y: 300 })
  const [source2, setSource2] = useState({ x: 500, y: 300 })
  const [frequency, setFrequency] = useState(0.05)
  const [wavelength, setWavelength] = useState(40)
  const [isRunning, setIsRunning] = useState(true)
  const [time, setTime] = useState(0)
  const animationRef = useRef<number>()

  useEffect(() => {
    if (isRunning) {
      const animate = () => {
        setTime((t) => t + 0.1)
        animationRef.current = requestAnimationFrame(animate)
      }
      animationRef.current = requestAnimationFrame(animate)
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning])

  useEffect(() => {
    drawCanvas()
  }, [time, source1, source2, frequency, wavelength])

  const drawCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, width, height)

    const imageData = ctx.createImageData(width, height)
    const data = imageData.data

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const r1 = Math.sqrt((x - source1.x) ** 2 + (y - source1.y) ** 2)
        const r2 = Math.sqrt((x - source2.x) ** 2 + (y - source2.y) ** 2)

        const wave1 = Math.sin((2 * Math.PI * r1) / wavelength - frequency * time)
        const wave2 = Math.sin((2 * Math.PI * r2) / wavelength - frequency * time)

        const amplitude = wave1 + wave2
        const intensity = ((amplitude + 2) / 4) * 255

        const idx = (y * width + x) * 4
        data[idx] = intensity // R
        data[idx + 1] = intensity * 0.7 // G
        data[idx + 2] = intensity * 1.2 // B
        data[idx + 3] = 255 // A
      }
    }

    ctx.putImageData(imageData, 0, 0)

    // Draw sources
    ctx.fillStyle = '#10b981'
    ctx.beginPath()
    ctx.arc(source1.x, source1.y, 10, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.arc(source2.x, source2.y, 10, 0, Math.PI * 2)
    ctx.fill()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">파동 간섭 시뮬레이터</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">간섭 패턴</h2>
            <canvas
              ref={canvasRef}
              width={800}
              height={600}
              className="w-full border border-purple-600 rounded-lg"
            />
            <div className="mt-4 space-y-2 text-sm text-slate-300">
              <p>💡 밝은 부분: 보강 간섭</p>
              <p>💡 어두운 부분: 상쇄 간섭</p>
              <p>💡 초록색 점: 파동 source</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">파라미터</h3>
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-slate-300 mb-2 block">진동수: {frequency.toFixed(2)}</label>
                  <input
                    type="range"
                    min="0.01"
                    max="0.2"
                    step="0.01"
                    value={frequency}
                    onChange={(e) => setFrequency(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="text-sm text-slate-300 mb-2 block">파장: {wavelength} px</label>
                  <input
                    type="range"
                    min="20"
                    max="80"
                    value={wavelength}
                    onChange={(e) => setWavelength(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">제어</h3>
              <button
                onClick={() => setIsRunning(!isRunning)}
                className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-500 rounded-lg"
              >
                {isRunning ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                <span>{isRunning ? '일시정지' : '시작'}</span>
              </button>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">간섭 조건</h3>
              <div className="text-xs text-slate-300 space-y-2">
                <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3">
                  <p className="font-semibold text-green-400 mb-1">보강 간섭</p>
                  <p className="font-mono">Δr = nλ (n = 0, 1, 2, ...)</p>
                </div>
                <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                  <p className="font-semibold text-red-400 mb-1">상쇄 간섭</p>
                  <p className="font-mono">Δr = (n + ½)λ</p>
                </div>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">공식</h3>
              <div className="text-xs text-slate-300 space-y-1 font-mono">
                <p>y = A sin(kx - ωt)</p>
                <p>k = 2π/λ (파수)</p>
                <p>ω = 2πf (각진동수)</p>
                <p>v = λf (파동 속력)</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
