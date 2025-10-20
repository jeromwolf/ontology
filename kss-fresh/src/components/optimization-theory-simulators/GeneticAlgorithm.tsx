'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, Pause, RotateCcw, Download, Dna, TrendingUp } from 'lucide-react'

interface Chromosome {
  id: string
  genes: number[]
  fitness: number
}

interface Population {
  chromosomes: Chromosome[]
  generation: number
  bestFitness: number
  avgFitness: number
}

export default function GeneticAlgorithm() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartRef = useRef<HTMLCanvasElement>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [populationSize, setPopulationSize] = useState(50)
  const [chromosomeLength, setChromosomeLength] = useState(20)
  const [mutationRate, setMutationRate] = useState(0.01)
  const [crossoverRate, setCrossoverRate] = useState(0.7)
  const [eliteCount, setEliteCount] = useState(2)
  const [population, setPopulation] = useState<Population>({
    chromosomes: [],
    generation: 0,
    bestFitness: 0,
    avgFitness: 0,
  })
  const [history, setHistory] = useState<{ generation: number; best: number; avg: number }[]>([])
  const animationRef = useRef<number>()

  // Fitness function: count of 1s in binary string (One-Max problem)
  const calculateFitness = (genes: number[]): number => {
    return genes.reduce((sum, gene) => sum + gene, 0)
  }

  const createChromosome = (): Chromosome => {
    const genes = Array.from({ length: chromosomeLength }, () => Math.random() > 0.5 ? 1 : 0)
    return {
      id: Math.random().toString(36).substr(2, 9),
      genes,
      fitness: calculateFitness(genes),
    }
  }

  const initializePopulation = () => {
    const chromosomes = Array.from({ length: populationSize }, createChromosome)
    const fitnesses = chromosomes.map((c) => c.fitness)
    const best = Math.max(...fitnesses)
    const avg = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length

    setPopulation({
      chromosomes,
      generation: 0,
      bestFitness: best,
      avgFitness: avg,
    })
    setHistory([{ generation: 0, best, avg }])
  }

  const selectParent = (chromosomes: Chromosome[]): Chromosome => {
    // Tournament selection
    const tournamentSize = 3
    const tournament = Array.from(
      { length: tournamentSize },
      () => chromosomes[Math.floor(Math.random() * chromosomes.length)]
    )
    return tournament.reduce((best, current) => (current.fitness > best.fitness ? current : best))
  }

  const crossover = (parent1: Chromosome, parent2: Chromosome): [number[], number[]] => {
    if (Math.random() > crossoverRate) {
      return [parent1.genes, parent2.genes]
    }

    // Single-point crossover
    const point = Math.floor(Math.random() * chromosomeLength)
    const child1 = [...parent1.genes.slice(0, point), ...parent2.genes.slice(point)]
    const child2 = [...parent2.genes.slice(0, point), ...parent1.genes.slice(point)]
    return [child1, child2]
  }

  const mutate = (genes: number[]): number[] => {
    return genes.map((gene) => (Math.random() < mutationRate ? 1 - gene : gene))
  }

  const evolve = () => {
    const currentChromosomes = population.chromosomes

    // Sort by fitness
    const sorted = [...currentChromosomes].sort((a, b) => b.fitness - a.fitness)

    // Elitism: keep best chromosomes
    const newChromosomes: Chromosome[] = sorted.slice(0, eliteCount).map((c) => ({ ...c, id: Math.random().toString(36).substr(2, 9) }))

    // Generate offspring
    while (newChromosomes.length < populationSize) {
      const parent1 = selectParent(currentChromosomes)
      const parent2 = selectParent(currentChromosomes)

      const [childGenes1, childGenes2] = crossover(parent1, parent2)
      const mutatedGenes1 = mutate(childGenes1)
      const mutatedGenes2 = mutate(childGenes2)

      newChromosomes.push({
        id: Math.random().toString(36).substr(2, 9),
        genes: mutatedGenes1,
        fitness: calculateFitness(mutatedGenes1),
      })

      if (newChromosomes.length < populationSize) {
        newChromosomes.push({
          id: Math.random().toString(36).substr(2, 9),
          genes: mutatedGenes2,
          fitness: calculateFitness(mutatedGenes2),
        })
      }
    }

    const fitnesses = newChromosomes.map((c) => c.fitness)
    const best = Math.max(...fitnesses)
    const avg = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length

    const newPopulation = {
      chromosomes: newChromosomes,
      generation: population.generation + 1,
      bestFitness: best,
      avgFitness: avg,
    }

    setPopulation(newPopulation)
    setHistory((prev) => [...prev, { generation: newPopulation.generation, best, avg }])
  }

  const drawChromosomes = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.clearRect(0, 0, width, height)
    ctx.fillStyle = '#1f2937'
    ctx.fillRect(0, 0, width, height)

    const sorted = [...population.chromosomes].sort((a, b) => b.fitness - a.fitness)
    const maxFitness = chromosomeLength
    const cellWidth = Math.min(20, width / chromosomeLength - 2)
    const cellHeight = Math.min(20, (height - 40) / populationSize - 1)

    sorted.forEach((chromosome, rowIdx) => {
      chromosome.genes.forEach((gene, colIdx) => {
        const x = colIdx * (cellWidth + 2) + 10
        const y = rowIdx * (cellHeight + 1) + 30

        // Color based on fitness
        const fitnessRatio = chromosome.fitness / maxFitness
        const hue = fitnessRatio * 120 // 0 (red) to 120 (green)
        ctx.fillStyle = `hsl(${hue}, 70%, ${gene === 1 ? 50 : 20}%)`
        ctx.fillRect(x, y, cellWidth, cellHeight)

        // Border
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)'
        ctx.lineWidth = 1
        ctx.strokeRect(x, y, cellWidth, cellHeight)
      })

      // Draw fitness label
      ctx.fillStyle = '#9ca3af'
      ctx.font = '10px Inter'
      ctx.fillText(
        chromosome.fitness.toString(),
        width - 40,
        rowIdx * (cellHeight + 1) + 30 + cellHeight / 2 + 3
      )
    })

    // Draw header
    ctx.fillStyle = '#10b981'
    ctx.font = 'bold 12px Inter'
    ctx.fillText(`세대: ${population.generation}`, 10, 20)
    ctx.fillText(`최고 적합도: ${population.bestFitness}/${maxFitness}`, 150, 20)
    ctx.fillText(`평균 적합도: ${population.avgFitness.toFixed(2)}`, 350, 20)
  }

  const drawFitnessChart = () => {
    const canvas = chartRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.clearRect(0, 0, width, height)
    ctx.fillStyle = '#1f2937'
    ctx.fillRect(0, 0, width, height)

    if (history.length === 0) return

    const maxGen = history[history.length - 1].generation
    const maxFitness = chromosomeLength

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 5; i++) {
      const y = (i / 5) * height
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    // Draw best fitness line
    ctx.strokeStyle = '#10b981'
    ctx.lineWidth = 3
    ctx.beginPath()
    history.forEach((point, idx) => {
      const x = (point.generation / Math.max(maxGen, 1)) * width
      const y = height - (point.best / maxFitness) * height
      if (idx === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // Draw average fitness line
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 2
    ctx.beginPath()
    history.forEach((point, idx) => {
      const x = (point.generation / Math.max(maxGen, 1)) * width
      const y = height - (point.avg / maxFitness) * height
      if (idx === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // Draw labels
    ctx.fillStyle = '#9ca3af'
    ctx.font = '12px Inter'
    ctx.fillText('적합도', 10, 20)
    ctx.fillText('세대', width - 50, height - 10)

    // Draw legend
    ctx.fillStyle = '#10b981'
    ctx.fillRect(width - 150, 10, 20, 3)
    ctx.fillStyle = '#9ca3af'
    ctx.fillText('최고 적합도', width - 120, 15)

    ctx.fillStyle = '#3b82f6'
    ctx.fillRect(width - 150, 25, 20, 2)
    ctx.fillStyle = '#9ca3af'
    ctx.fillText('평균 적합도', width - 120, 30)
  }

  useEffect(() => {
    initializePopulation()
  }, [])

  useEffect(() => {
    drawChromosomes()
    drawFitnessChart()
  }, [population, history])

  useEffect(() => {
    if (isRunning) {
      animationRef.current = window.setTimeout(() => {
        evolve()
      }, 100)
    }
    return () => {
      if (animationRef.current) clearTimeout(animationRef.current)
    }
  }, [isRunning, population])

  const handleReset = () => {
    setIsRunning(false)
    initializePopulation()
  }

  const downloadResults = () => {
    const data = {
      finalGeneration: population.generation,
      bestFitness: population.bestFitness,
      avgFitness: population.avgFitness,
      parameters: {
        populationSize,
        chromosomeLength,
        mutationRate,
        crossoverRate,
        eliteCount,
      },
      history: history,
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'genetic-algorithm-results.json'
    a.click()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-emerald-600 to-teal-700 rounded-xl">
              <Dna className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-400 to-teal-400 bg-clip-text text-transparent">
                유전 알고리즘 실험실
              </h1>
              <p className="text-gray-400 mt-1">염색체의 진화 과정을 실시간으로 관찰합니다 (One-Max 문제)</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Controls */}
          <div className="lg:col-span-1 space-y-4">
            {/* Population Size */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                개체군 크기: {populationSize}
              </label>
              <input
                type="range"
                min="20"
                max="100"
                value={populationSize}
                onChange={(e) => setPopulationSize(parseInt(e.target.value))}
                disabled={isRunning}
                className="w-full accent-emerald-500"
              />
            </div>

            {/* Chromosome Length */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                염색체 길이: {chromosomeLength}
              </label>
              <input
                type="range"
                min="10"
                max="30"
                value={chromosomeLength}
                onChange={(e) => setChromosomeLength(parseInt(e.target.value))}
                disabled={isRunning}
                className="w-full accent-emerald-500"
              />
            </div>

            {/* Mutation Rate */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                돌연변이율: {(mutationRate * 100).toFixed(1)}%
              </label>
              <input
                type="range"
                min="0.001"
                max="0.1"
                step="0.001"
                value={mutationRate}
                onChange={(e) => setMutationRate(parseFloat(e.target.value))}
                className="w-full accent-emerald-500"
              />
            </div>

            {/* Crossover Rate */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                교차율: {(crossoverRate * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.5"
                max="1"
                step="0.05"
                value={crossoverRate}
                onChange={(e) => setCrossoverRate(parseFloat(e.target.value))}
                className="w-full accent-emerald-500"
              />
            </div>

            {/* Elite Count */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                엘리트 개체 수: {eliteCount}
              </label>
              <input
                type="range"
                min="0"
                max="10"
                value={eliteCount}
                onChange={(e) => setEliteCount(parseInt(e.target.value))}
                className="w-full accent-emerald-500"
              />
            </div>

            {/* Controls */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 space-y-3">
              <button
                onClick={() => setIsRunning(!isRunning)}
                className="w-full bg-gradient-to-r from-emerald-600 to-teal-700 hover:from-emerald-700 hover:to-teal-800 text-white px-4 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2"
              >
                {isRunning ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                {isRunning ? '일시정지' : '진화 시작'}
              </button>
              <button
                onClick={handleReset}
                className="w-full bg-gray-700 hover:bg-gray-600 text-white px-4 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2"
              >
                <RotateCcw className="w-5 h-5" />
                초기화
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
              <h3 className="text-sm font-semibold text-emerald-400 mb-3">진화 통계</h3>
              <div className="space-y-3">
                <div>
                  <div className="text-xs text-gray-400">현재 세대</div>
                  <div className="text-2xl font-bold text-emerald-400">{population.generation}</div>
                </div>
                <div>
                  <div className="text-xs text-gray-400">최고 적합도</div>
                  <div className="text-2xl font-bold text-emerald-400">
                    {population.bestFitness}/{chromosomeLength}
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2 mt-1">
                    <div
                      className="bg-gradient-to-r from-emerald-600 to-teal-700 h-2 rounded-full"
                      style={{ width: `${(population.bestFitness / chromosomeLength) * 100}%` }}
                    ></div>
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-400">평균 적합도</div>
                  <div className="text-xl font-bold text-blue-400">{population.avgFitness.toFixed(2)}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Visualizations */}
          <div className="lg:col-span-3 space-y-6">
            {/* Chromosome Visualization */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 text-emerald-400">
                개체군 시각화 (위에서 아래로 적합도 순)
              </h3>
              <canvas ref={canvasRef} width={900} height={600} className="w-full rounded-lg" />
              <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="font-medium text-gray-300 mb-1">밝은 초록색</div>
                  <p className="text-xs text-gray-500">유전자 값 1 (높은 적합도)</p>
                </div>
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="font-medium text-gray-300 mb-1">어두운 색</div>
                  <p className="text-xs text-gray-500">유전자 값 0 (낮은 적합도)</p>
                </div>
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="font-medium text-gray-300 mb-1">색상 그라데이션</div>
                  <p className="text-xs text-gray-500">빨강(낮음) → 초록(높음)</p>
                </div>
              </div>
            </div>

            {/* Fitness Chart */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 text-emerald-400">적합도 진화 곡선</h3>
              <canvas ref={chartRef} width={900} height={300} className="w-full rounded-lg" />
            </div>

            {/* Problem Description */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-3 text-emerald-400">One-Max 문제</h3>
              <p className="text-sm text-gray-300 mb-3">
                이진 문자열에서 1의 개수를 최대화하는 고전적인 유전 알고리즘 벤치마크 문제입니다.
                각 염색체는 이진 유전자(0 또는 1)의 배열로 표현되며, 적합도는 1의 총 개수입니다.
              </p>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="font-medium text-emerald-400 mb-1">선택 (Selection)</div>
                  <p className="text-xs text-gray-400">토너먼트 선택으로 적합도가 높은 개체 선호</p>
                </div>
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="font-medium text-emerald-400 mb-1">교차 (Crossover)</div>
                  <p className="text-xs text-gray-400">단일점 교차로 부모의 유전자 조합</p>
                </div>
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="font-medium text-emerald-400 mb-1">돌연변이 (Mutation)</div>
                  <p className="text-xs text-gray-400">낮은 확률로 비트 반전</p>
                </div>
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="font-medium text-emerald-400 mb-1">엘리트주의 (Elitism)</div>
                  <p className="text-xs text-gray-400">최고 개체는 다음 세대로 직접 복사</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
