'use client'

import { useState } from 'react'
import { Play, Award, TrendingUp, Zap } from 'lucide-react'

interface BenchmarkResult {
  name: string
  score: number
  unit: string
  rank: 'excellent' | 'good' | 'average' | 'poor'
}

interface System {
  name: string
  cpu: string
  gpu: string
  memory: string
  network: string
}

export default function HPCBenchmark() {
  const [selectedSystem, setSelectedSystem] = useState<'a100' | 'h100' | 'mi250x' | 'custom'>('a100')
  const [isRunning, setIsRunning] = useState(false)
  const [results, setResults] = useState<BenchmarkResult[]>([])

  const systems: Record<string, System> = {
    a100: {
      name: 'NVIDIA A100 Cluster',
      cpu: 'AMD EPYC 7763 (64 cores)',
      gpu: '8× NVIDIA A100 80GB',
      memory: '1 TB DDR4-3200',
      network: '400 Gbps InfiniBand',
    },
    h100: {
      name: 'NVIDIA H100 Cluster',
      cpu: 'Intel Xeon Platinum 8480 (56 cores)',
      gpu: '8× NVIDIA H100 80GB',
      memory: '2 TB DDR5-4800',
      network: '400 Gbps InfiniBand',
    },
    mi250x: {
      name: 'AMD MI250X Cluster',
      cpu: 'AMD EPYC 7763 (64 cores)',
      gpu: '4× AMD MI250X (220 CUs)',
      memory: '512 GB DDR4-3200',
      network: '200 Gbps InfiniBand',
    },
    custom: {
      name: 'Custom HPC System',
      cpu: 'Custom CPU Configuration',
      gpu: 'Custom GPU Configuration',
      memory: 'Custom Memory',
      network: 'Custom Network',
    },
  }

  const runBenchmark = () => {
    setIsRunning(true)
    setResults([])

    setTimeout(() => {
      const benchmarks: BenchmarkResult[] = []

      // LINPACK (HPL)
      let linpackScore = 0
      if (selectedSystem === 'a100') linpackScore = 450
      else if (selectedSystem === 'h100') linpackScore = 650
      else if (selectedSystem === 'mi250x') linpackScore = 380
      else linpackScore = 300

      benchmarks.push({
        name: 'LINPACK (HPL)',
        score: linpackScore + Math.random() * 20 - 10,
        unit: 'TFLOPS',
        rank: linpackScore > 500 ? 'excellent' : linpackScore > 400 ? 'good' : linpackScore > 300 ? 'average' : 'poor',
      })

      // STREAM Triad
      let streamScore = 0
      if (selectedSystem === 'a100') streamScore = 1550
      else if (selectedSystem === 'h100') streamScore = 2400
      else if (selectedSystem === 'mi250x') streamScore = 1350
      else streamScore = 1000

      benchmarks.push({
        name: 'STREAM Triad',
        score: streamScore + Math.random() * 100 - 50,
        unit: 'GB/s',
        rank: streamScore > 2000 ? 'excellent' : streamScore > 1500 ? 'good' : streamScore > 1000 ? 'average' : 'poor',
      })

      // MPI Latency
      let mpiLatency = 0
      if (selectedSystem === 'a100') mpiLatency = 1.2
      else if (selectedSystem === 'h100') mpiLatency = 1.0
      else if (selectedSystem === 'mi250x') mpiLatency = 1.5
      else mpiLatency = 2.0

      benchmarks.push({
        name: 'MPI Latency',
        score: mpiLatency + Math.random() * 0.2 - 0.1,
        unit: 'μs',
        rank: mpiLatency < 1.2 ? 'excellent' : mpiLatency < 1.5 ? 'good' : mpiLatency < 2.0 ? 'average' : 'poor',
      })

      // MPI Bandwidth
      let mpiBandwidth = 0
      if (selectedSystem === 'a100') mpiBandwidth = 380
      else if (selectedSystem === 'h100') mpiBandwidth = 390
      else if (selectedSystem === 'mi250x') mpiBandwidth = 180
      else mpiBandwidth = 150

      benchmarks.push({
        name: 'MPI Bandwidth',
        score: mpiBandwidth + Math.random() * 10 - 5,
        unit: 'GB/s',
        rank: mpiBandwidth > 350 ? 'excellent' : mpiBandwidth > 250 ? 'good' : mpiBandwidth > 150 ? 'average' : 'poor',
      })

      // AI Performance (FP16)
      let aiPerf = 0
      if (selectedSystem === 'a100') aiPerf = 2400
      else if (selectedSystem === 'h100') aiPerf = 4000
      else if (selectedSystem === 'mi250x') aiPerf = 1800
      else aiPerf = 1000

      benchmarks.push({
        name: 'AI Performance (FP16)',
        score: aiPerf + Math.random() * 100 - 50,
        unit: 'TFLOPS',
        rank: aiPerf > 3000 ? 'excellent' : aiPerf > 2000 ? 'good' : aiPerf > 1000 ? 'average' : 'poor',
      })

      // Parallel I/O
      let ioPerf = 0
      if (selectedSystem === 'a100') ioPerf = 45
      else if (selectedSystem === 'h100') ioPerf = 52
      else if (selectedSystem === 'mi250x') ioPerf = 38
      else ioPerf = 30

      benchmarks.push({
        name: 'Parallel I/O',
        score: ioPerf + Math.random() * 5 - 2.5,
        unit: 'GB/s',
        rank: ioPerf > 50 ? 'excellent' : ioPerf > 40 ? 'good' : ioPerf > 30 ? 'average' : 'poor',
      })

      setResults(benchmarks)
      setIsRunning(false)
    }, 2000)
  }

  const getRankColor = (rank: string) => {
    switch (rank) {
      case 'excellent': return 'text-green-600 dark:text-green-400'
      case 'good': return 'text-blue-600 dark:text-blue-400'
      case 'average': return 'text-yellow-600 dark:text-yellow-400'
      case 'poor': return 'text-red-600 dark:text-red-400'
      default: return 'text-gray-600'
    }
  }

  const getOverallScore = () => {
    if (results.length === 0) return 0
    const excellentCount = results.filter(r => r.rank === 'excellent').length
    const goodCount = results.filter(r => r.rank === 'good').length
    const avgCount = results.filter(r => r.rank === 'average').length

    return ((excellentCount * 100 + goodCount * 75 + avgCount * 50) / results.length).toFixed(0)
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg p-6 border-l-4 border-yellow-500">
        <h3 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
          HPC 벤치마크 도구
        </h3>
        <p className="text-gray-700 dark:text-gray-300">
          시스템 성능을 측정하고 비교합니다 (LINPACK, STREAM, MPI, AI)
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              System Configuration
            </h4>

            <div className="space-y-2">
              {Object.entries(systems).map(([id, system]) => (
                <button
                  key={id}
                  onClick={() => setSelectedSystem(id as any)}
                  className={`w-full text-left px-4 py-3 rounded-lg border-2 transition ${
                    selectedSystem === id
                      ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <div className="font-semibold text-sm text-gray-900 dark:text-white">
                    {system.name}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    {system.gpu}
                  </div>
                </button>
              ))}
            </div>

            <button
              onClick={runBenchmark}
              disabled={isRunning}
              className={`w-full mt-4 px-4 py-2 rounded-lg transition flex items-center justify-center gap-2 ${
                isRunning
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-yellow-500 hover:bg-yellow-600 text-white'
              }`}
            >
              <Play className="w-4 h-4" />
              {isRunning ? 'Running...' : 'Run Benchmark'}
            </button>
          </div>

          {results.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
                <Award className="w-5 h-5 text-yellow-500" />
                Overall Score
              </h4>

              <div className="text-center">
                <div className="text-4xl font-bold text-yellow-600 dark:text-yellow-400">
                  {getOverallScore()}
                </div>
                <div className="text-sm text-gray-500 mt-2">
                  / 100
                </div>

                <div className="mt-4 w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                  <div
                    className="bg-gradient-to-r from-yellow-500 to-orange-500 h-3 rounded-full transition-all"
                    style={{ width: `${getOverallScore()}%` }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="lg:col-span-2 space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              System Specifications
            </h4>

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-gray-500 dark:text-gray-400">CPU</div>
                <div className="font-semibold text-gray-900 dark:text-white">
                  {systems[selectedSystem].cpu}
                </div>
              </div>

              <div>
                <div className="text-gray-500 dark:text-gray-400">GPU</div>
                <div className="font-semibold text-gray-900 dark:text-white">
                  {systems[selectedSystem].gpu}
                </div>
              </div>

              <div>
                <div className="text-gray-500 dark:text-gray-400">Memory</div>
                <div className="font-semibold text-gray-900 dark:text-white">
                  {systems[selectedSystem].memory}
                </div>
              </div>

              <div>
                <div className="text-gray-500 dark:text-gray-400">Network</div>
                <div className="font-semibold text-gray-900 dark:text-white">
                  {systems[selectedSystem].network}
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-yellow-500" />
              Benchmark Results
            </h4>

            {results.length === 0 ? (
              <div className="text-center py-12 text-gray-500">
                Run benchmark to see results
              </div>
            ) : (
              <div className="space-y-3">
                {results.map((result, idx) => (
                  <div
                    key={idx}
                    className="p-4 rounded-lg border border-gray-200 dark:border-gray-700"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="font-semibold text-gray-900 dark:text-white">
                        {result.name}
                      </div>
                      <div className={`text-xs px-2 py-1 rounded font-semibold ${
                        result.rank === 'excellent' ? 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-400' :
                        result.rank === 'good' ? 'bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400' :
                        result.rank === 'average' ? 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-400' :
                        'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-400'
                      }`}>
                        {result.rank.toUpperCase()}
                      </div>
                    </div>

                    <div className="flex items-baseline gap-2">
                      <span className={`text-2xl font-bold ${getRankColor(result.rank)}`}>
                        {result.score.toFixed(1)}
                      </span>
                      <span className="text-sm text-gray-500">
                        {result.unit}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4 border-l-4 border-yellow-500">
            <h5 className="font-semibold mb-2 text-gray-900 dark:text-white flex items-center gap-2">
              <Zap className="w-5 h-5 text-yellow-600" />
              벤치마크 설명
            </h5>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>LINPACK (HPL)</strong>: 선형대수 연산 성능 (Top500 순위 기준)</li>
              <li>• <strong>STREAM Triad</strong>: 메모리 대역폭 측정</li>
              <li>• <strong>MPI Latency</strong>: 프로세스 간 통신 지연시간</li>
              <li>• <strong>MPI Bandwidth</strong>: 대용량 데이터 전송 속도</li>
              <li>• <strong>AI Performance</strong>: FP16 텐서 연산 성능</li>
              <li>• <strong>Parallel I/O</strong>: 병렬 파일시스템 처리량</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
