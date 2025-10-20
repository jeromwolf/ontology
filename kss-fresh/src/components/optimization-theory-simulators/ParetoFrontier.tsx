'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, RefreshCw, Download, TrendingUp, Target } from 'lucide-react'

interface Solution {
  id: string
  obj1: number
  obj2: number
  isDominated: boolean
  dominationCount: number
}

interface Population {
  solutions: Solution[]
  generation: number
}

export default function ParetoFrontier() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [population, setPopulation] = useState<Population>({ solutions: [], generation: 0 })
  const [selectedSolution, setSelectedSolution] = useState<Solution | null>(null)
  const [populationSize, setPopulationSize] = useState(50)
  const [isRunning, setIsRunning] = useState(false)
  const [maxGenerations, setMaxGenerations] = useState(100)
  const animationRef = useRef<number>()

  const generateRandomSolution = (): Solution => {
    const obj1 = Math.random() * 100
    const obj2 = Math.random() * 100
    return {
      id: Math.random().toString(36).substr(2, 9),
      obj1,
      obj2,
      isDominated: false,
      dominationCount: 0,
    }
  }

  const dominates = (sol1: Solution, sol2: Solution): boolean => {
    // sol1 dominates sol2 if it's better in at least one objective and not worse in any
    return (
      (sol1.obj1 >= sol2.obj1 && sol1.obj2 >= sol2.obj2) &&
      (sol1.obj1 > sol2.obj1 || sol1.obj2 > sol2.obj2)
    )
  }

  const calculateParetoFrontier = (solutions: Solution[]): Solution[] => {
    // Mark dominated solutions
    const updatedSolutions = solutions.map((sol) => {
      let dominated = false
      let count = 0
      for (const other of solutions) {
        if (other.id !== sol.id && dominates(other, sol)) {
          dominated = true
          count++
        }
      }
      return { ...sol, isDominated: dominated, dominationCount: count }
    })

    return updatedSolutions
  }

  const initializePopulation = () => {
    const solutions: Solution[] = []
    for (let i = 0; i < populationSize; i++) {
      solutions.push(generateRandomSolution())
    }
    const updated = calculateParetoFrontier(solutions)
    setPopulation({ solutions: updated, generation: 0 })
  }

  const evolvePopulation = () => {
    if (population.generation >= maxGenerations) {
      setIsRunning(false)
      return
    }

    const currentSolutions = population.solutions

    // NSGA-II style evolution
    const paretoFront = currentSolutions.filter((s) => !s.isDominated)
    const newSolutions: Solution[] = []

    // Keep elites (Pareto front)
    newSolutions.push(...paretoFront.slice(0, Math.floor(populationSize * 0.3)))

    // Generate offspring through crossover and mutation
    while (newSolutions.length < populationSize) {
      // Select parents (tournament selection)
      const parent1 = paretoFront[Math.floor(Math.random() * paretoFront.length)] || currentSolutions[0]
      const parent2 = paretoFront[Math.floor(Math.random() * paretoFront.length)] || currentSolutions[1]

      // Crossover
      const alpha = Math.random()
      let child1 = alpha * parent1.obj1 + (1 - alpha) * parent2.obj1
      let child2 = alpha * parent1.obj2 + (1 - alpha) * parent2.obj2

      // Mutation
      if (Math.random() < 0.2) {
        child1 += (Math.random() - 0.5) * 10
        child2 += (Math.random() - 0.5) * 10
      }

      // Bounds
      child1 = Math.max(0, Math.min(100, child1))
      child2 = Math.max(0, Math.min(100, child2))

      newSolutions.push({
        id: Math.random().toString(36).substr(2, 9),
        obj1: child1,
        obj2: child2,
        isDominated: false,
        dominationCount: 0,
      })
    }

    const updated = calculateParetoFrontier(newSolutions)
    setPopulation({ solutions: updated, generation: population.generation + 1 })
  }

  const drawVisualization = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.clearRect(0, 0, width, height)
    ctx.fillStyle = '#1f2937'
    ctx.fillRect(0, 0, width, height)

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 10; i++) {
      const x = (i / 10) * width
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.stroke()

      const y = (i / 10) * height
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    // Draw axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(50, height - 50)
    ctx.lineTo(width - 20, height - 50)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(50, height - 50)
    ctx.lineTo(50, 20)
    ctx.stroke()

    // Draw arrow heads
    ctx.fillStyle = 'rgba(255, 255, 255, 0.3)'
    ctx.beginPath()
    ctx.moveTo(width - 20, height - 50)
    ctx.lineTo(width - 30, height - 45)
    ctx.lineTo(width - 30, height - 55)
    ctx.fill()
    ctx.beginPath()
    ctx.moveTo(50, 20)
    ctx.lineTo(45, 30)
    ctx.lineTo(55, 30)
    ctx.fill()

    // Draw labels
    ctx.fillStyle = '#9ca3af'
    ctx.font = 'bold 14px Inter'
    ctx.fillText('목적함수 1 (최대화) →', width - 180, height - 20)
    ctx.save()
    ctx.translate(20, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText('목적함수 2 (최대화) →', -100, 0)
    ctx.restore()

    const padding = 60
    const chartWidth = width - padding - 20
    const chartHeight = height - padding - 20

    // Draw dominated solutions
    population.solutions
      .filter((s) => s.isDominated)
      .forEach((sol) => {
        const x = (sol.obj1 / 100) * chartWidth + padding
        const y = height - padding - (sol.obj2 / 100) * chartHeight
        ctx.fillStyle = 'rgba(156, 163, 175, 0.4)'
        ctx.beginPath()
        ctx.arc(x, y, 4, 0, Math.PI * 2)
        ctx.fill()
      })

    // Draw Pareto front solutions
    const paretoSolutions = population.solutions
      .filter((s) => !s.isDominated)
      .sort((a, b) => a.obj1 - b.obj1)

    paretoSolutions.forEach((sol, idx) => {
      const x = (sol.obj1 / 100) * chartWidth + padding
      const y = height - padding - (sol.obj2 / 100) * chartHeight

      // Draw point
      ctx.fillStyle = '#10b981'
      ctx.beginPath()
      ctx.arc(x, y, 6, 0, Math.PI * 2)
      ctx.fill()

      ctx.strokeStyle = '#1f2937'
      ctx.lineWidth = 2
      ctx.stroke()

      // Highlight selected
      if (selectedSolution && selectedSolution.id === sol.id) {
        ctx.strokeStyle = '#fbbf24'
        ctx.lineWidth = 3
        ctx.beginPath()
        ctx.arc(x, y, 10, 0, Math.PI * 2)
        ctx.stroke()
      }
    })

    // Draw Pareto frontier line
    if (paretoSolutions.length > 1) {
      ctx.strokeStyle = '#10b981'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      paretoSolutions.forEach((sol, idx) => {
        const x = (sol.obj1 / 100) * chartWidth + padding
        const y = height - padding - (sol.obj2 / 100) * chartHeight
        if (idx === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()
      ctx.setLineDash([])
    }

    // Draw scale markers
    ctx.fillStyle = '#6b7280'
    ctx.font = '11px Inter'
    for (let i = 0; i <= 5; i++) {
      const value = (i / 5) * 100
      const x = (i / 5) * chartWidth + padding
      ctx.fillText(value.toFixed(0), x - 10, height - 30)

      const y = height - padding - (i / 5) * chartHeight
      ctx.fillText(value.toFixed(0), 20, y + 5)
    }
  }

  useEffect(() => {
    initializePopulation()
  }, [])

  useEffect(() => {
    drawVisualization()
  }, [population, selectedSolution])

  useEffect(() => {
    if (isRunning && population.generation < maxGenerations) {
      animationRef.current = window.setTimeout(() => {
        evolvePopulation()
      }, 100)
    } else if (population.generation >= maxGenerations) {
      setIsRunning(false)
    }
    return () => {
      if (animationRef.current) clearTimeout(animationRef.current)
    }
  }, [isRunning, population])

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    const padding = 60
    const chartWidth = canvas.width - padding - 20
    const chartHeight = canvas.height - padding - 20

    // Find closest solution
    let closest: Solution | null = null
    let minDist = Infinity

    population.solutions.forEach((sol) => {
      const sx = (sol.obj1 / 100) * chartWidth + padding
      const sy = canvas.height - padding - (sol.obj2 / 100) * chartHeight
      const dist = Math.sqrt((x - sx) ** 2 + (y - sy) ** 2)
      if (dist < minDist && dist < 20) {
        minDist = dist
        closest = sol
      }
    })

    setSelectedSolution(closest)
  }

  const downloadResults = () => {
    const paretoFront = population.solutions.filter((s) => !s.isDominated)
    const data = {
      generation: population.generation,
      paretoFrontSize: paretoFront.length,
      solutions: paretoFront.map((s) => ({
        objective1: s.obj1,
        objective2: s.obj2,
      })),
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'pareto-frontier.json'
    a.click()
  }

  const paretoFront = population.solutions.filter((s) => !s.isDominated)

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-emerald-600 to-teal-700 rounded-xl">
              <Target className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-400 to-teal-400 bg-clip-text text-transparent">
                파레토 프론티어 (Pareto Frontier)
              </h1>
              <p className="text-gray-400 mt-1">다목적 최적화에서 비지배 해집합을 시각화합니다</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Controls */}
          <div className="lg:col-span-1 space-y-4">
            {/* Population Size */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                개체수: {populationSize}
              </label>
              <input
                type="range"
                min="20"
                max="100"
                value={populationSize}
                onChange={(e) => {
                  setPopulationSize(parseInt(e.target.value))
                  if (!isRunning) initializePopulation()
                }}
                className="w-full accent-emerald-500"
              />
            </div>

            {/* Max Generations */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                최대 세대: {maxGenerations}
              </label>
              <input
                type="range"
                min="50"
                max="200"
                value={maxGenerations}
                onChange={(e) => setMaxGenerations(parseInt(e.target.value))}
                className="w-full accent-emerald-500"
              />
            </div>

            {/* Controls */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 space-y-3">
              <button
                onClick={() => setIsRunning(!isRunning)}
                className="w-full bg-gradient-to-r from-emerald-600 to-teal-700 hover:from-emerald-700 hover:to-teal-800 text-white px-4 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2"
              >
                {isRunning ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5" />}
                {isRunning ? '진화중...' : '진화 시작'}
              </button>
              <button
                onClick={initializePopulation}
                className="w-full bg-gray-700 hover:bg-gray-600 text-white px-4 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2"
              >
                <RefreshCw className="w-5 h-5" />
                새로운 개체군
              </button>
              <button
                onClick={downloadResults}
                className="w-full bg-gray-700 hover:bg-gray-600 text-white px-4 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2"
              >
                <Download className="w-5 h-5" />
                결과 다운로드
              </button>
            </div>

            {/* Stats */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-sm font-semibold text-emerald-400 mb-3">통계</h3>
              <div className="space-y-3">
                <div>
                  <div className="text-xs text-gray-400">현재 세대</div>
                  <div className="text-2xl font-bold text-emerald-400">{population.generation}</div>
                </div>
                <div>
                  <div className="text-xs text-gray-400">파레토 프론티어 크기</div>
                  <div className="text-2xl font-bold text-emerald-400">{paretoFront.length}</div>
                </div>
                <div>
                  <div className="text-xs text-gray-400">지배된 해</div>
                  <div className="text-2xl font-bold text-gray-400">
                    {population.solutions.length - paretoFront.length}
                  </div>
                </div>
              </div>
            </div>

            {/* Selected Solution */}
            {selectedSolution && (
              <div className="bg-gradient-to-br from-yellow-900/30 to-yellow-800/30 border border-yellow-600 rounded-xl p-6">
                <h3 className="text-sm font-semibold text-yellow-400 mb-3">선택된 해</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-300">목적함수 1:</span>
                    <span className="font-mono text-yellow-400">{selectedSolution.obj1.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">목적함수 2:</span>
                    <span className="font-mono text-yellow-400">{selectedSolution.obj2.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">상태:</span>
                    <span
                      className={`font-medium ${
                        selectedSolution.isDominated ? 'text-gray-400' : 'text-emerald-400'
                      }`}
                    >
                      {selectedSolution.isDominated ? '지배됨' : '비지배'}
                    </span>
                  </div>
                  {selectedSolution.isDominated && (
                    <div className="flex justify-between">
                      <span className="text-gray-300">지배 횟수:</span>
                      <span className="text-gray-400">{selectedSolution.dominationCount}</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Legend */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-sm font-semibold text-emerald-400 mb-3">범례</h3>
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-emerald-500 rounded-full"></div>
                  <span className="text-gray-300">파레토 최적해</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-gray-500 rounded-full"></div>
                  <span className="text-gray-300">지배된 해</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-6 h-0.5 border-t-2 border-dashed border-emerald-500"></div>
                  <span className="text-gray-300">파레토 프론티어</span>
                </div>
              </div>
            </div>
          </div>

          {/* Visualization */}
          <div className="lg:col-span-3">
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 text-emerald-400">목적함수 공간</h3>
              <canvas
                ref={canvasRef}
                width={900}
                height={700}
                className="w-full rounded-lg cursor-crosshair"
                onClick={handleCanvasClick}
              />
              <div className="mt-6 bg-gray-700 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-emerald-400 mb-2">파레토 지배 (Pareto Dominance)</h4>
                <p className="text-sm text-gray-300">
                  해 A가 해 B를 지배한다는 것은 A가 모든 목적함수에서 B보다 나쁘지 않고, 적어도 하나의 목적함수에서는
                  더 좋다는 의미입니다. 파레토 프론티어는 어떤 해에게도 지배되지 않는 비지배 해들의 집합입니다.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
