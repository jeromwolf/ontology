'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, Search } from 'lucide-react'

export default function OptimizationLab() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [problem, setProblem] = useState<'box' | 'fence' | 'cylinder'>('box')
  const [parameter, setParameter] = useState<number>(10)

  useEffect(() => {
    drawCanvas()
  }, [problem, parameter])

  const calculateOptimum = () => {
    switch (problem) {
      case 'box': return { x: parameter / 6, max: (parameter / 6) ** 2 * (parameter / 3) }
      case 'fence': return { x: parameter / 4, max: parameter ** 2 / 16 }
      case 'cylinder': return { r: Math.sqrt(parameter / (4 * Math.PI)), h: parameter / (2 * Math.PI * Math.sqrt(parameter / (4 * Math.PI))) }
      default: return { x: 0, max: 0 }
    }
  }

  const objectiveFunction = (x: number): number => {
    switch (problem) {
      case 'box': return x * x * (parameter - 2 * x)
      case 'fence': return x * (parameter - 2 * x)
      case 'cylinder': {
        const r = x
        const h = (parameter - 2 * Math.PI * r * r) / (2 * Math.PI * r)
        return h > 0 ? Math.PI * r * r * h : 0
      }
      default: return 0
    }
  }

  const drawCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, width, height)

    // Draw objective function
    const padding = 50
    const graphWidth = width - 2 * padding
    const graphHeight = height - 2 * padding

    // Find max value for scaling
    let maxY = 0
    const maxX = problem === 'box' ? parameter / 2 : problem === 'fence' ? parameter / 2 : parameter / (2 * Math.PI)

    for (let x = 0; x <= maxX; x += maxX / 100) {
      const y = objectiveFunction(x)
      if (y > maxY) maxY = y
    }

    // Draw axes
    ctx.strokeStyle = '#475569'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(padding, padding)
    ctx.lineTo(padding, height - padding)
    ctx.lineTo(width - padding, height - padding)
    ctx.stroke()

    // Draw grid
    ctx.strokeStyle = '#1e293b'
    ctx.lineWidth = 1
    for (let i = 0; i <= 10; i++) {
      const y = padding + (graphHeight * i) / 10
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(width - padding, y)
      ctx.stroke()
    }

    // Draw function
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 3
    ctx.beginPath()
    for (let i = 0; i <= 200; i++) {
      const x = (i / 200) * maxX
      const y = objectiveFunction(x)
      const px = padding + (x / maxX) * graphWidth
      const py = height - padding - (y / maxY) * graphHeight
      if (i === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    ctx.stroke()

    // Draw optimum point
    const opt = calculateOptimum()
    const optX = problem === 'cylinder' ? (opt as any).r : (opt as any).x
    const optY = problem === 'cylinder' ? Math.PI * (opt as any).r ** 2 * (opt as any).h : (opt as any).max
    const optPx = padding + (optX / maxX) * graphWidth
    const optPy = height - padding - (optY / maxY) * graphHeight

    ctx.fillStyle = '#10b981'
    ctx.beginPath()
    ctx.arc(optPx, optPy, 8, 0, Math.PI * 2)
    ctx.fill()

    // Vertical line to optimum
    ctx.strokeStyle = '#10b981'
    ctx.setLineDash([5, 5])
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(optPx, height - padding)
    ctx.lineTo(optPx, optPy)
    ctx.stroke()
    ctx.setLineDash([])

    // Labels
    ctx.fillStyle = '#94a3b8'
    ctx.font = '12px Inter'
    ctx.fillText('0', padding - 10, height - padding + 20)
    ctx.fillText(maxX.toFixed(1), width - padding, height - padding + 20)
    ctx.fillText(maxY.toFixed(1), 10, padding)

    ctx.fillStyle = '#10b981'
    ctx.font = 'bold 14px Inter'
    ctx.fillText(`최댓값: ${optY.toFixed(2)}`, optPx + 15, optPy - 10)
    ctx.fillText(`x = ${optX.toFixed(2)}`, optPx + 15, optPy + 20)
  }

  const problems = [
    { id: 'box', label: '상자 부피 최대화', desc: '정사각형을 잘라 상자 만들기' },
    { id: 'fence', label: '울타리 넓이 최대화', desc: '고정 길이로 최대 넓이' },
    { id: 'cylinder', label: '원기둥 부피 최대화', desc: '고정 표면적으로 최대 부피' }
  ]

  const opt = calculateOptimum()

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-green-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">최적화 실험실</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h2 className="text-xl font-semibold mb-4">목적 함수 그래프</h2>
              <canvas ref={canvasRef} width={800} height={500} className="w-full border border-slate-600 rounded-lg" />
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Search className="w-5 h-5 text-green-400" />
                최적해
              </h2>
              <div className="grid grid-cols-2 gap-4">
                {problem === 'cylinder' ? (
                  <>
                    <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                      <div className="text-sm text-slate-400">반지름 r</div>
                      <div className="text-2xl font-bold text-green-400">{((opt as any).r).toFixed(3)}</div>
                    </div>
                    <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                      <div className="text-sm text-slate-400">높이 h</div>
                      <div className="text-2xl font-bold text-green-400">{((opt as any).h).toFixed(3)}</div>
                    </div>
                    <div className="col-span-2 bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                      <div className="text-sm text-slate-400">최대 부피</div>
                      <div className="text-3xl font-bold text-blue-400">{(Math.PI * (opt as any).r ** 2 * (opt as any).h).toFixed(3)}</div>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                      <div className="text-sm text-slate-400">최적 x</div>
                      <div className="text-2xl font-bold text-green-400">{((opt as any).x).toFixed(3)}</div>
                    </div>
                    <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                      <div className="text-sm text-slate-400">최댓값</div>
                      <div className="text-2xl font-bold text-blue-400">{((opt as any).max).toFixed(3)}</div>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">문제 선택</h3>
              <div className="space-y-2">
                {problems.map((p) => (
                  <button key={p.id} onClick={() => setProblem(p.id as any)} className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${problem === p.id ? 'bg-green-600 text-white' : 'bg-slate-700/50 hover:bg-slate-700'}`}>
                    <div className="font-semibold text-sm">{p.label}</div>
                    <div className="text-xs text-slate-400">{p.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">파라미터</h3>
              <div>
                <label className="text-sm text-slate-300 mb-2 block">
                  {problem === 'box' ? '종이 한 변 길이' : problem === 'fence' ? '울타리 총 길이' : '표면적'}: {parameter}
                </label>
                <input type="range" min="5" max="50" value={parameter} onChange={(e) => setParameter(parseInt(e.target.value))} className="w-full" />
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">문제 설명</h3>
              <div className="text-sm text-slate-300 space-y-2">
                {problem === 'box' && (
                  <>
                    <p>정사각형 종이의 네 모서리에서 작은 정사각형을 잘라내어 상자를 만듭니다.</p>
                    <p className="font-mono text-xs bg-slate-900/50 p-2 rounded">V(x) = x²(L - 2x)</p>
                  </>
                )}
                {problem === 'fence' && (
                  <>
                    <p>고정된 길이의 울타리로 직사각형 영역을 만들 때 최대 넓이를 구합니다.</p>
                    <p className="font-mono text-xs bg-slate-900/50 p-2 rounded">A(x) = x(P - 2x)</p>
                  </>
                )}
                {problem === 'cylinder' && (
                  <>
                    <p>고정된 표면적으로 원기둥을 만들 때 최대 부피를 구합니다.</p>
                    <p className="font-mono text-xs bg-slate-900/50 p-2 rounded">V = πr²h, S = 2πr² + 2πrh</p>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
