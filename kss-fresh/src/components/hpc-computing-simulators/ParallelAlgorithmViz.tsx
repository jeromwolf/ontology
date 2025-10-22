'use client'

import { useState, useEffect, useRef } from 'react'
import { Play, Pause, RotateCcw, SkipForward, Layers } from 'lucide-react'

type Algorithm = 'parallel-sum' | 'parallel-sort' | 'map-reduce' | 'scan-prefix'

interface ProcessorState {
  id: number
  value: number
  active: boolean
  computed: boolean
}

export default function ParallelAlgorithmViz() {
  const [algorithm, setAlgorithm] = useState<Algorithm>('parallel-sum')
  const [isRunning, setIsRunning] = useState(false)
  const [step, setStep] = useState(0)
  const [processors, setProcessors] = useState<ProcessorState[]>([])
  const [speed, setSpeed] = useState(500)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<NodeJS.Timeout | null>(null)

  const PROCESSOR_COUNT = 16

  // 초기화
  useEffect(() => {
    initializeAlgorithm()
  }, [algorithm])

  const initializeAlgorithm = () => {
    setStep(0)
    setIsRunning(false)

    const initialData: ProcessorState[] = Array.from({ length: PROCESSOR_COUNT }, (_, i) => ({
      id: i,
      value: algorithm === 'parallel-sort' ? Math.floor(Math.random() * 100) : i + 1,
      active: false,
      computed: false,
    }))

    setProcessors(initialData)
  }

  // 애니메이션 스텝
  useEffect(() => {
    if (!isRunning) return

    animationRef.current = setTimeout(() => {
      executeStep()
    }, speed)

    return () => {
      if (animationRef.current) clearTimeout(animationRef.current)
    }
  }, [isRunning, step, processors])

  const executeStep = () => {
    switch (algorithm) {
      case 'parallel-sum':
        executeParallelSum()
        break
      case 'parallel-sort':
        executeParallelSort()
        break
      case 'map-reduce':
        executeMapReduce()
        break
      case 'scan-prefix':
        executeScanPrefix()
        break
    }
  }

  const executeParallelSum = () => {
    const maxSteps = Math.log2(PROCESSOR_COUNT)
    if (step >= maxSteps) {
      setIsRunning(false)
      return
    }

    const newProcessors = [...processors]
    const stride = Math.pow(2, step + 1)

    for (let i = 0; i < PROCESSOR_COUNT; i += stride) {
      const partner = i + Math.pow(2, step)
      if (partner < PROCESSOR_COUNT) {
        newProcessors[i].value += newProcessors[partner].value
        newProcessors[i].active = true
        newProcessors[partner].active = true
      }
    }

    setProcessors(newProcessors)
    setStep(step + 1)
  }

  const executeParallelSort = () => {
    // Bitonic Sort 구현
    const n = PROCESSOR_COUNT
    const totalSteps = (Math.log2(n) * (Math.log2(n) + 1)) / 2

    if (step >= totalSteps) {
      setIsRunning(false)
      return
    }

    const newProcessors = [...processors]
    const k = Math.floor(Math.log2(step + 1)) + 1
    const j = step - (k * (k - 1)) / 2

    for (let i = 0; i < n; i++) {
      const ixj = i ^ (1 << j)
      if (ixj > i) {
        if ((i & (1 << k)) === 0) {
          // Ascending
          if (newProcessors[i].value > newProcessors[ixj].value) {
            [newProcessors[i].value, newProcessors[ixj].value] =
            [newProcessors[ixj].value, newProcessors[i].value]
          }
        } else {
          // Descending
          if (newProcessors[i].value < newProcessors[ixj].value) {
            [newProcessors[i].value, newProcessors[ixj].value] =
            [newProcessors[ixj].value, newProcessors[i].value]
          }
        }
        newProcessors[i].active = true
        newProcessors[ixj].active = true
      }
    }

    setProcessors(newProcessors)
    setStep(step + 1)
  }

  const executeMapReduce = () => {
    const totalSteps = Math.log2(PROCESSOR_COUNT) + 2

    if (step >= totalSteps) {
      setIsRunning(false)
      return
    }

    const newProcessors = [...processors]

    if (step === 0) {
      // Map phase
      newProcessors.forEach(p => {
        p.value = p.value * 2 // Example: multiply by 2
        p.active = true
      })
    } else if (step === 1) {
      // Shuffle phase
      newProcessors.forEach(p => {
        p.active = true
      })
    } else {
      // Reduce phase (parallel sum)
      const reduceStep = step - 2
      const stride = Math.pow(2, reduceStep + 1)

      for (let i = 0; i < PROCESSOR_COUNT; i += stride) {
        const partner = i + Math.pow(2, reduceStep)
        if (partner < PROCESSOR_COUNT) {
          newProcessors[i].value += newProcessors[partner].value
          newProcessors[i].active = true
          newProcessors[partner].active = true
        }
      }
    }

    setProcessors(newProcessors)
    setStep(step + 1)
  }

  const executeScanPrefix = () => {
    // Parallel Prefix Sum (Blelloch Scan)
    const totalSteps = 2 * Math.log2(PROCESSOR_COUNT)

    if (step >= totalSteps) {
      setIsRunning(false)
      return
    }

    const newProcessors = [...processors]
    const halfSteps = Math.log2(PROCESSOR_COUNT)

    if (step < halfSteps) {
      // Up-sweep (reduction)
      const d = step
      const stride = Math.pow(2, d + 1)

      for (let i = 0; i < PROCESSOR_COUNT; i += stride) {
        const left = i + Math.pow(2, d) - 1
        const right = i + stride - 1
        if (right < PROCESSOR_COUNT) {
          newProcessors[right].value += newProcessors[left].value
          newProcessors[left].active = true
          newProcessors[right].active = true
        }
      }
    } else {
      // Down-sweep
      if (step === halfSteps) {
        newProcessors[PROCESSOR_COUNT - 1].value = 0
      }

      const d = totalSteps - step - 1
      const stride = Math.pow(2, d + 1)

      for (let i = 0; i < PROCESSOR_COUNT; i += stride) {
        const left = i + Math.pow(2, d) - 1
        const right = i + stride - 1
        if (right < PROCESSOR_COUNT) {
          const temp = newProcessors[left].value
          newProcessors[left].value = newProcessors[right].value
          newProcessors[right].value = temp + newProcessors[right].value
          newProcessors[left].active = true
          newProcessors[right].active = true
        }
      }
    }

    setProcessors(newProcessors)
    setStep(step + 1)
  }

  // Canvas 시각화
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.fillStyle = '#1a1a1a'
    ctx.fillRect(0, 0, width, height)

    const boxWidth = (width - 100) / PROCESSOR_COUNT
    const boxHeight = 60

    processors.forEach((proc, i) => {
      const x = 50 + i * boxWidth
      const y = height / 2 - boxHeight / 2

      // Box
      if (proc.active) {
        ctx.fillStyle = 'rgba(251, 191, 36, 0.8)' // Yellow
      } else if (proc.computed) {
        ctx.fillStyle = 'rgba(34, 197, 94, 0.6)' // Green
      } else {
        ctx.fillStyle = 'rgba(100, 100, 100, 0.6)'
      }

      ctx.fillRect(x, y, boxWidth - 4, boxHeight)

      // Border
      ctx.strokeStyle = proc.active ? '#f59e0b' : '#555'
      ctx.lineWidth = 2
      ctx.strokeRect(x, y, boxWidth - 4, boxHeight)

      // Processor ID
      ctx.fillStyle = '#ffffff'
      ctx.font = '10px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(`P${i}`, x + boxWidth / 2 - 2, y - 5)

      // Value
      ctx.font = 'bold 14px monospace'
      ctx.fillText(proc.value.toString(), x + boxWidth / 2 - 2, y + boxHeight / 2 + 5)
    })

    // Step info
    ctx.fillStyle = '#ffffff'
    ctx.font = '16px monospace'
    ctx.textAlign = 'left'
    ctx.fillText(`Step: ${step}`, 50, 30)

    // Reset active状態
    setTimeout(() => {
      setProcessors(prev => prev.map(p => ({ ...p, active: false })))
    }, 200)

  }, [processors, step])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg p-6 border-l-4 border-yellow-500">
        <h3 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
          병렬 알고리즘 시각화
        </h3>
        <p className="text-gray-700 dark:text-gray-300">
          분산 알고리즘의 실행 흐름을 단계별로 시각화합니다 (16개 프로세서)
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left: Controls */}
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              알고리즘 선택
            </h4>

            <div className="space-y-2">
              {[
                { id: 'parallel-sum', name: 'Parallel Sum', desc: '병렬 합산 (Tree)' },
                { id: 'parallel-sort', name: 'Bitonic Sort', desc: '병렬 정렬' },
                { id: 'map-reduce', name: 'MapReduce', desc: 'Map-Shuffle-Reduce' },
                { id: 'scan-prefix', name: 'Prefix Sum', desc: 'Parallel Scan' },
              ].map((algo) => (
                <button
                  key={algo.id}
                  onClick={() => {
                    setAlgorithm(algo.id as Algorithm)
                    setIsRunning(false)
                  }}
                  className={`w-full text-left px-4 py-3 rounded-lg border-2 transition ${
                    algorithm === algo.id
                      ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-yellow-300'
                  }`}
                >
                  <div className="font-semibold text-gray-900 dark:text-white">
                    {algo.name}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    {algo.desc}
                  </div>
                </button>
              ))}
            </div>

            <div className="mt-6 space-y-3">
              <div>
                <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">
                  Animation Speed: {speed}ms
                </label>
                <input
                  type="range"
                  min="100"
                  max="2000"
                  step="100"
                  value={speed}
                  onChange={(e) => setSpeed(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>

              <div className="flex gap-2">
                <button
                  onClick={() => setIsRunning(!isRunning)}
                  className={`flex-1 px-4 py-2 rounded-lg transition flex items-center justify-center gap-2 ${
                    isRunning
                      ? 'bg-red-500 hover:bg-red-600 text-white'
                      : 'bg-yellow-500 hover:bg-yellow-600 text-white'
                  }`}
                >
                  {isRunning ? (
                    <>
                      <Pause className="w-4 h-4" />
                      Pause
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      Play
                    </>
                  )}
                </button>
                <button
                  onClick={() => {
                    if (!isRunning) executeStep()
                  }}
                  className="px-4 py-2 bg-orange-500 hover:bg-orange-600 text-white rounded-lg transition"
                  disabled={isRunning}
                >
                  <SkipForward className="w-4 h-4" />
                </button>
                <button
                  onClick={initializeAlgorithm}
                  className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg transition"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>

          {/* Algorithm Info */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
              <Layers className="w-5 h-5 text-yellow-500" />
              Algorithm Info
            </h4>

            {algorithm === 'parallel-sum' && (
              <div className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <p><strong>시간 복잡도:</strong> O(log n)</p>
                <p><strong>프로세서:</strong> n/2</p>
                <p><strong>단계:</strong> log₂(n) = {Math.log2(PROCESSOR_COUNT)}</p>
                <p className="text-xs">Tree 구조로 pair-wise 합산</p>
              </div>
            )}

            {algorithm === 'parallel-sort' && (
              <div className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <p><strong>시간 복잡도:</strong> O(log² n)</p>
                <p><strong>프로세서:</strong> n</p>
                <p><strong>단계:</strong> {(Math.log2(PROCESSOR_COUNT) * (Math.log2(PROCESSOR_COUNT) + 1)) / 2}</p>
                <p className="text-xs">Bitonic Merge Network</p>
              </div>
            )}

            {algorithm === 'map-reduce' && (
              <div className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <p><strong>Map:</strong> 병렬 변환</p>
                <p><strong>Shuffle:</strong> 데이터 재분배</p>
                <p><strong>Reduce:</strong> 병렬 집계</p>
                <p className="text-xs">Hadoop/Spark 모델</p>
              </div>
            )}

            {algorithm === 'scan-prefix' && (
              <div className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <p><strong>시간 복잡도:</strong> O(log n)</p>
                <p><strong>Work:</strong> O(n)</p>
                <p><strong>단계:</strong> 2 × log₂(n) = {2 * Math.log2(PROCESSOR_COUNT)}</p>
                <p className="text-xs">Blelloch Scan (Work-Efficient)</p>
              </div>
            )}
          </div>
        </div>

        {/* Right: Visualization */}
        <div className="lg:col-span-3">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              Processor Activity
            </h4>
            <canvas
              ref={canvasRef}
              width={900}
              height={300}
              className="w-full border border-gray-300 dark:border-gray-600 rounded"
            />
          </div>

          {/* Explanation */}
          <div className="mt-6 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4 border-l-4 border-yellow-500">
            <h5 className="font-semibold mb-2 text-gray-900 dark:text-white">
              📊 시각화 설명
            </h5>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong className="text-yellow-600">노란색 박스</strong>: 현재 단계에서 활성화된 프로세서</li>
              <li>• <strong>회색 박스</strong>: 대기 중인 프로세서</li>
              <li>• <strong>박스 내 숫자</strong>: 프로세서가 현재 보유한 값</li>
              <li>• 각 단계마다 병렬로 작동하는 프로세서들의 패턴을 관찰하세요</li>
              <li>• Tree 구조에서는 단계마다 활성 프로세서가 절반씩 줄어듭니다</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
