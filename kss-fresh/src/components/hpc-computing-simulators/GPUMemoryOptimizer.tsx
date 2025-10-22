'use client'

import { useState, useEffect } from 'react'
import { MemoryStick, Zap, TrendingUp, AlertCircle } from 'lucide-react'

type MemoryPattern = 'coalesced' | 'strided' | 'random' | 'shared-memory'

interface MemoryAccess {
  threadId: number
  address: number
  latency: number
  cached: boolean
}

export default function GPUMemoryOptimizer() {
  const [pattern, setPattern] = useState<MemoryPattern>('coalesced')
  const [warpSize] = useState(32)
  const [accesses, setAccesses] = useState<MemoryAccess[]>([])
  const [metrics, setMetrics] = useState({
    efficiency: 0,
    transactions: 0,
    bandwidth: 0,
    avgLatency: 0,
  })

  useEffect(() => {
    simulateMemoryPattern()
  }, [pattern])

  const simulateMemoryPattern = () => {
    const newAccesses: MemoryAccess[] = []
    let transactions = 0
    let totalLatency = 0

    for (let i = 0; i < warpSize; i++) {
      let address = 0
      let latency = 0
      let cached = false

      switch (pattern) {
        case 'coalesced':
          // 연속된 메모리 접근 (최적)
          address = i * 4 // 4 bytes per element
          latency = 200 // L2 cache
          cached = true
          break

        case 'strided':
          // Stride 접근 (비효율적)
          address = i * 128 // Large stride
          latency = 400 // Cache miss
          cached = false
          break

        case 'random':
          // 랜덤 접근 (최악)
          address = Math.floor(Math.random() * 10000) * 4
          latency = 600 // DRAM
          cached = false
          break

        case 'shared-memory':
          // Shared Memory (매우 빠름)
          address = i * 4
          latency = 20 // Shared memory
          cached = true
          break
      }

      newAccesses.push({ threadId: i, address, latency, cached })
      totalLatency += latency
    }

    // Calculate transactions (cache line = 128 bytes, 32 threads × 4 bytes = 128 bytes)
    if (pattern === 'coalesced') {
      transactions = 1 // Single transaction
    } else if (pattern === 'strided') {
      transactions = warpSize // Each thread different cache line
    } else if (pattern === 'random') {
      transactions = warpSize // Worst case
    } else {
      transactions = 0 // Shared memory (no global mem transactions)
    }

    const efficiency = pattern === 'coalesced' ? 100 :
                      pattern === 'shared-memory' ? 100 :
                      pattern === 'strided' ? 3.125 : 3.125

    const bandwidth = (warpSize * 4) / (totalLatency / warpSize) * 1000 // GB/s (theoretical)

    setAccesses(newAccesses)
    setMetrics({
      efficiency,
      transactions,
      bandwidth,
      avgLatency: totalLatency / warpSize,
    })
  }

  const getPatternColor = (p: MemoryPattern) => {
    switch (p) {
      case 'coalesced': return 'from-green-500 to-emerald-600'
      case 'strided': return 'from-yellow-500 to-orange-600'
      case 'random': return 'from-red-500 to-pink-600'
      case 'shared-memory': return 'from-blue-500 to-cyan-600'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg p-6 border-l-4 border-yellow-500">
        <h3 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
          GPU 메모리 최적화기
        </h3>
        <p className="text-gray-700 dark:text-gray-300">
          메모리 접근 패턴에 따른 성능 차이를 분석합니다
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Pattern Selection */}
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              메모리 접근 패턴
            </h4>

            <div className="space-y-3">
              {[
                {
                  id: 'coalesced',
                  name: 'Coalesced Access',
                  desc: '연속된 메모리 접근 (최적)',
                  example: 'arr[tid]',
                },
                {
                  id: 'strided',
                  name: 'Strided Access',
                  desc: 'Stride 패턴 (비효율)',
                  example: 'arr[tid * stride]',
                },
                {
                  id: 'random',
                  name: 'Random Access',
                  desc: '랜덤 메모리 접근 (최악)',
                  example: 'arr[random()]',
                },
                {
                  id: 'shared-memory',
                  name: 'Shared Memory',
                  desc: 'On-chip 고속 메모리',
                  example: '__shared__ arr[]',
                },
              ].map((p) => (
                <button
                  key={p.id}
                  onClick={() => setPattern(p.id as MemoryPattern)}
                  className={`w-full text-left px-4 py-3 rounded-lg border-2 transition ${
                    pattern === p.id
                      ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-yellow-300'
                  }`}
                >
                  <div className="font-semibold text-gray-900 dark:text-white">
                    {p.name}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    {p.desc}
                  </div>
                  <code className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded mt-1 inline-block">
                    {p.example}
                  </code>
                </button>
              ))}
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              성능 지표
            </h4>

            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-700 dark:text-gray-300 flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-green-500" />
                    Memory Efficiency
                  </span>
                  <span className="font-mono font-semibold text-green-600 dark:text-green-400">
                    {metrics.efficiency.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className={`bg-gradient-to-r ${getPatternColor(pattern)} h-2 rounded-full transition-all`}
                    style={{ width: `${metrics.efficiency}%` }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 pt-3 border-t border-gray-200 dark:border-gray-700">
                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Transactions</div>
                  <div className="text-xl font-bold text-gray-900 dark:text-white">
                    {metrics.transactions}
                  </div>
                  <div className="text-xs text-gray-500">{pattern === 'coalesced' ? 'Optimal' : 'Inefficient'}</div>
                </div>

                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Avg Latency</div>
                  <div className="text-xl font-bold text-gray-900 dark:text-white">
                    {metrics.avgLatency.toFixed(0)} ns
                  </div>
                  <div className="text-xs text-gray-500">
                    {pattern === 'shared-memory' ? 'On-chip' : 'Global Mem'}
                  </div>
                </div>

                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Bandwidth</div>
                  <div className="text-xl font-bold text-gray-900 dark:text-white">
                    {metrics.bandwidth.toFixed(1)} GB/s
                  </div>
                </div>

                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Warp Size</div>
                  <div className="text-xl font-bold text-gray-900 dark:text-white">
                    {warpSize}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Memory Access Visualization */}
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              Memory Access Pattern (Warp 0)
            </h4>

            <div className="space-y-1 max-h-96 overflow-y-auto">
              {accesses.map((access, idx) => (
                <div
                  key={idx}
                  className="flex items-center gap-3 p-2 rounded border border-gray-200 dark:border-gray-700"
                >
                  <div className="w-16 text-xs font-mono text-gray-500">
                    T{access.threadId}
                  </div>
                  <div className="flex-1">
                    <div className="text-xs font-mono text-gray-700 dark:text-gray-300">
                      Addr: 0x{access.address.toString(16).padStart(8, '0')}
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <div
                        className={`h-1 rounded-full transition-all ${
                          access.latency < 100 ? 'bg-green-500' :
                          access.latency < 300 ? 'bg-yellow-500' :
                          'bg-red-500'
                        }`}
                        style={{ width: `${(access.latency / 600) * 100}%` }}
                      />
                      <span className="text-xs text-gray-500">
                        {access.latency}ns
                      </span>
                    </div>
                  </div>
                  {access.cached && (
                    <Zap className="w-4 h-4 text-green-500" />
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Optimization Tips */}
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4 border-l-4 border-yellow-500">
            <h5 className="font-semibold mb-2 text-gray-900 dark:text-white flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-yellow-600" />
              최적화 가이드
            </h5>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>Coalesced Access</strong>: 인접한 스레드가 연속된 메모리 접근</li>
              <li>• <strong>Cache Line</strong>: 128 bytes 단위로 로드됨</li>
              <li>• <strong>Shared Memory</strong>: 레지스터 다음으로 빠른 메모리 (20-40ns)</li>
              <li>• <strong>Bank Conflict</strong>: 같은 뱅크 접근 시 성능 저하</li>
              <li>• <strong>Global Memory</strong>: 200-600ns 레이턴시</li>
              <li>• <strong>실전 팁</strong>: 배열 인덱스를 threadIdx.x로 직접 사용</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Code Examples */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
          코드 예시: {pattern === 'coalesced' ? '✅ 최적화된 접근' :
                      pattern === 'shared-memory' ? '✅ Shared Memory 활용' :
                      '❌ 비효율적 접근'}
        </h4>

        <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
          <pre className="text-sm text-gray-100">
            <code>{
              pattern === 'coalesced' ? `// ✅ Coalesced Memory Access
__global__ void optimized_kernel(float *input, float *output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        output[tid] = input[tid] * 2.0f;  // Sequential access
    }
}

// Warp의 모든 스레드가 연속된 메모리에 접근
// → Single memory transaction (128 bytes)` :
              pattern === 'strided' ? `// ❌ Strided Access (비효율적)
__global__ void strided_kernel(float *input, float *output, int N, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        output[tid] = input[tid * stride];  // Large stride
    }
}

// Stride가 크면 각 스레드가 다른 cache line 접근
// → 32 memory transactions (비효율)` :
              pattern === 'random' ? `// ❌ Random Access (최악)
__global__ void random_kernel(float *input, float *output, int *indices, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int idx = indices[tid];  // Random index
        output[tid] = input[idx];
    }
}

// 예측 불가능한 메모리 패턴
// → Cache miss 빈발, 최악의 성능` :
              `// ✅ Shared Memory Optimization
__global__ void shared_mem_kernel(float *input, float *output, int N) {
    __shared__ float shared[256];  // On-chip memory

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    // Global → Shared (coalesced)
    if (tid < N) shared[local_id] = input[tid];
    __syncthreads();

    // Shared memory에서 연산 (매우 빠름!)
    if (tid < N) output[tid] = shared[local_id] * 2.0f;
}

// Shared memory 레이턴시: 20-40ns (Global 대비 10-30배 빠름)`
            }</code>
          </pre>
        </div>
      </div>
    </div>
  )
}
