'use client'

import { useState, useEffect, useRef } from 'react'
import { Play, Pause, RotateCcw, Zap, Cpu, MemoryStick, Clock } from 'lucide-react'

interface KernelConfig {
  gridSize: number
  blockSize: number
  sharedMemory: number
  registers: number
}

interface PerformanceMetrics {
  occupancy: number
  throughput: number
  latency: number
  bandwidth: number
  efficiency: number
}

export default function CUDAKernelAnalyzer() {
  const [config, setConfig] = useState<KernelConfig>({
    gridSize: 128,
    blockSize: 256,
    sharedMemory: 48,
    registers: 32,
  })

  const [isRunning, setIsRunning] = useState(false)
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    occupancy: 0,
    throughput: 0,
    latency: 0,
    bandwidth: 0,
    efficiency: 0,
  })

  const canvasRef = useRef<HTMLCanvasElement>(null)

  // GPU 사양 (NVIDIA A100 기준)
  const GPU_SPECS = {
    smCount: 108,
    maxThreadsPerSM: 2048,
    maxBlocksPerSM: 32,
    warpSize: 32,
    sharedMemoryPerSM: 164, // KB
    registersPerSM: 65536,
  }

  // 성능 메트릭 계산
  useEffect(() => {
    const calculateMetrics = () => {
      const threadsPerBlock = config.blockSize
      const warpsPerBlock = Math.ceil(threadsPerBlock / GPU_SPECS.warpSize)

      // 1. Occupancy 계산
      const blockLimitByThreads = Math.floor(GPU_SPECS.maxThreadsPerSM / threadsPerBlock)
      const blockLimitBySharedMem = Math.floor(GPU_SPECS.sharedMemoryPerSM / config.sharedMemory)
      const blockLimitByRegisters = Math.floor(GPU_SPECS.registersPerSM / (config.registers * threadsPerBlock))
      const blockLimitByMax = GPU_SPECS.maxBlocksPerSM

      const activeBlocksPerSM = Math.min(
        blockLimitByThreads,
        blockLimitBySharedMem,
        blockLimitByRegisters,
        blockLimitByMax
      )

      const activeWarpsPerSM = activeBlocksPerSM * warpsPerBlock
      const maxWarpsPerSM = GPU_SPECS.maxThreadsPerSM / GPU_SPECS.warpSize
      const occupancy = (activeWarpsPerSM / maxWarpsPerSM) * 100

      // 2. Throughput (GFLOPS 추정)
      const clockSpeed = 1.41 // GHz (A100)
      const coresPerSM = 64
      const throughput = (occupancy / 100) * GPU_SPECS.smCount * coresPerSM * clockSpeed * 2 // FP32 ops

      // 3. Memory Bandwidth (GB/s)
      const peakBandwidth = 1555 // GB/s (A100 HBM2)
      const bandwidth = peakBandwidth * (occupancy / 100) * 0.7 // 실제 활용률 70% 가정

      // 4. Latency (μs)
      const latency = (1000000 / (throughput * 1000)) * config.gridSize

      // 5. Overall Efficiency
      const efficiency = (occupancy * 0.4 + (bandwidth / peakBandwidth * 100) * 0.3 + (throughput / 20000) * 100 * 0.3)

      setMetrics({
        occupancy: Math.min(occupancy, 100),
        throughput: throughput,
        latency: latency,
        bandwidth: bandwidth,
        efficiency: Math.min(efficiency, 100),
      })
    }

    calculateMetrics()
  }, [config])

  // Canvas 시각화
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    // Clear
    ctx.fillStyle = '#1a1a1a'
    ctx.fillRect(0, 0, width, height)

    // Draw SM Grid (9x12 = 108 SMs)
    const smRows = 9
    const smCols = 12
    const smWidth = (width - 40) / smCols
    const smHeight = (height - 100) / smRows

    for (let row = 0; row < smRows; row++) {
      for (let col = 0; col < smCols; col++) {
        const x = 20 + col * smWidth
        const y = 20 + row * smHeight

        // SM 활성화 정도 (Occupancy 기반)
        const intensity = metrics.occupancy / 100
        const r = Math.floor(255 * intensity)
        const g = Math.floor(200 * intensity)
        const b = 50

        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.8)`
        ctx.fillRect(x, y, smWidth - 2, smHeight - 2)

        // SM 번호
        ctx.fillStyle = '#ffffff'
        ctx.font = '8px monospace'
        ctx.textAlign = 'center'
        ctx.fillText(`SM${row * smCols + col}`, x + smWidth / 2, y + smHeight / 2)
      }
    }

    // Legend
    ctx.fillStyle = '#ffffff'
    ctx.font = '12px monospace'
    ctx.textAlign = 'left'
    ctx.fillText(`Active Warps per SM: ${Math.floor((metrics.occupancy / 100) * 64)}`, 20, height - 50)
    ctx.fillText(`Total Active Threads: ${Math.floor((metrics.occupancy / 100) * GPU_SPECS.smCount * GPU_SPECS.maxThreadsPerSM)}`, 20, height - 30)
    ctx.fillText(`Grid: ${config.gridSize} blocks × ${config.blockSize} threads`, 20, height - 10)

  }, [metrics, config])

  const handleReset = () => {
    setConfig({
      gridSize: 128,
      blockSize: 256,
      sharedMemory: 48,
      registers: 32,
    })
    setIsRunning(false)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg p-6 border-l-4 border-yellow-500">
        <h3 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
          CUDA 커널 분석기
        </h3>
        <p className="text-gray-700 dark:text-gray-300">
          GPU 커널 설정에 따른 성능 메트릭을 실시간으로 분석합니다 (NVIDIA A100 기준)
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Configuration */}
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              커널 설정
            </h4>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">
                  Grid Size (블록 수): {config.gridSize}
                </label>
                <input
                  type="range"
                  min="1"
                  max="512"
                  value={config.gridSize}
                  onChange={(e) => setConfig({ ...config, gridSize: parseInt(e.target.value) })}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">
                  Block Size (블록당 스레드): {config.blockSize}
                </label>
                <input
                  type="range"
                  min="32"
                  max="1024"
                  step="32"
                  value={config.blockSize}
                  onChange={(e) => setConfig({ ...config, blockSize: parseInt(e.target.value) })}
                  className="w-full"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Warps: {Math.ceil(config.blockSize / 32)}
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">
                  Shared Memory (KB/block): {config.sharedMemory}
                </label>
                <input
                  type="range"
                  min="0"
                  max="164"
                  value={config.sharedMemory}
                  onChange={(e) => setConfig({ ...config, sharedMemory: parseInt(e.target.value) })}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">
                  Registers (per thread): {config.registers}
                </label>
                <input
                  type="range"
                  min="8"
                  max="255"
                  value={config.registers}
                  onChange={(e) => setConfig({ ...config, registers: parseInt(e.target.value) })}
                  className="w-full"
                />
              </div>
            </div>

            <div className="flex gap-2 mt-6">
              <button
                onClick={handleReset}
                className="flex-1 px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg transition flex items-center justify-center gap-2"
              >
                <RotateCcw className="w-4 h-4" />
                Reset
              </button>
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              성능 메트릭
            </h4>

            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-700 dark:text-gray-300 flex items-center gap-2">
                    <Zap className="w-4 h-4 text-yellow-500" />
                    Occupancy
                  </span>
                  <span className="font-mono font-semibold text-yellow-600 dark:text-yellow-400">
                    {metrics.occupancy.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-yellow-500 h-2 rounded-full transition-all"
                    style={{ width: `${metrics.occupancy}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-700 dark:text-gray-300 flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-orange-500" />
                    Throughput
                  </span>
                  <span className="font-mono font-semibold text-orange-600 dark:text-orange-400">
                    {metrics.throughput.toFixed(0)} GFLOPS
                  </span>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-700 dark:text-gray-300 flex items-center gap-2">
                    <MemoryStick className="w-4 h-4 text-blue-500" />
                    Bandwidth
                  </span>
                  <span className="font-mono font-semibold text-blue-600 dark:text-blue-400">
                    {metrics.bandwidth.toFixed(0)} GB/s
                  </span>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-700 dark:text-gray-300 flex items-center gap-2">
                    <Clock className="w-4 h-4 text-green-500" />
                    Latency
                  </span>
                  <span className="font-mono font-semibold text-green-600 dark:text-green-400">
                    {metrics.latency.toFixed(2)} μs
                  </span>
                </div>
              </div>

              <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-700 dark:text-gray-300 font-semibold">
                    Overall Efficiency
                  </span>
                  <span className="font-mono font-bold text-yellow-600 dark:text-yellow-400">
                    {metrics.efficiency.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                  <div
                    className="bg-gradient-to-r from-yellow-500 to-orange-500 h-3 rounded-full transition-all"
                    style={{ width: `${metrics.efficiency}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Visualization */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              SM Activity Map (108 Streaming Multiprocessors)
            </h4>
            <canvas
              ref={canvasRef}
              width={800}
              height={600}
              className="w-full border border-gray-300 dark:border-gray-600 rounded"
            />
          </div>

          {/* Optimization Tips */}
          <div className="mt-6 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4 border-l-4 border-yellow-500">
            <h5 className="font-semibold mb-2 text-gray-900 dark:text-white">
              ⚡ 최적화 팁
            </h5>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>Occupancy 75% 이상</strong>이 이상적입니다 (100%가 항상 최적은 아님)</li>
              <li>• Block Size는 <strong>Warp(32)의 배수</strong>로 설정하세요</li>
              <li>• Shared Memory 사용을 줄이면 더 많은 블록이 동시 실행됩니다</li>
              <li>• Register 사용량이 많으면 Occupancy가 급격히 떨어집니다</li>
              <li>• 작업량이 많으면 Grid Size를 늘려 병렬성을 높이세요</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
