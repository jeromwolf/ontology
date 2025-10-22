'use client'

import { useState, useEffect, useRef } from 'react'
import { Activity, Cpu, MemoryStick, Network } from 'lucide-react'

interface ProfileData {
  time: number
  cpu: number
  memory: number
  network: number
  gpu: number
}

export default function PerformanceProfiler() {
  const [data, setData] = useState<ProfileData[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [workload, setWorkload] = useState<'compute' | 'memory' | 'communication'>('compute')
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(() => {
      setData(prev => {
        const newData = [...prev]
        const time = prev.length

        let cpu = 0, memory = 0, network = 0, gpu = 0

        if (workload === 'compute') {
          cpu = 80 + Math.random() * 20
          gpu = 90 + Math.random() * 10
          memory = 40 + Math.random() * 10
          network = 10 + Math.random() * 5
        } else if (workload === 'memory') {
          cpu = 30 + Math.random() * 10
          gpu = 40 + Math.random() * 10
          memory = 85 + Math.random() * 15
          network = 15 + Math.random() * 10
        } else {
          cpu = 25 + Math.random() * 10
          gpu = 30 + Math.random() * 10
          memory = 30 + Math.random() * 10
          network = 80 + Math.random() * 20
        }

        newData.push({ time, cpu, memory, network, gpu })
        if (newData.length > 100) newData.shift()

        return newData
      })
    }, 100)

    return () => clearInterval(interval)
  }, [isRunning, workload])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || data.length === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.fillStyle = '#1a1a1a'
    ctx.fillRect(0, 0, width, height)

    const drawLine = (values: number[], color: string) => {
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.beginPath()

      values.forEach((val, i) => {
        const x = (i / values.length) * width
        const y = height - (val / 100) * height
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })

      ctx.stroke()
    }

    drawLine(data.map(d => d.cpu), '#f59e0b')
    drawLine(data.map(d => d.memory), '#3b82f6')
    drawLine(data.map(d => d.network), '#10b981')
    drawLine(data.map(d => d.gpu), '#8b5cf6')

    // Grid
    ctx.strokeStyle = '#333'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = (i / 4) * height
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()

      ctx.fillStyle = '#666'
      ctx.font = '10px monospace'
      ctx.fillText(`${100 - i * 25}%`, 5, y - 3)
    }
  }, [data])

  const getBottleneck = () => {
    if (data.length === 0) return 'N/A'
    const latest = data[data.length - 1]
    const max = Math.max(latest.cpu, latest.memory, latest.network, latest.gpu)

    if (max === latest.cpu) return 'CPU'
    if (max === latest.memory) return 'Memory'
    if (max === latest.network) return 'Network'
    return 'GPU'
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg p-6 border-l-4 border-yellow-500">
        <h3 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
          성능 프로파일러
        </h3>
        <p className="text-gray-700 dark:text-gray-300">
          HPC 애플리케이션의 리소스 사용량을 실시간 모니터링합니다
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              Workload Type
            </h4>

            <div className="space-y-2">
              {[
                { id: 'compute', name: 'Compute Intensive', icon: Cpu },
                { id: 'memory', name: 'Memory Bound', icon: MemoryStick },
                { id: 'communication', name: 'Network Bound', icon: Network },
              ].map(({ id, name, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => setWorkload(id as any)}
                  className={`w-full text-left px-4 py-3 rounded-lg border-2 transition flex items-center gap-3 ${
                    workload === id
                      ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <Icon className="w-5 h-5 text-yellow-600" />
                  <span className="font-semibold text-sm text-gray-900 dark:text-white">
                    {name}
                  </span>
                </button>
              ))}
            </div>

            <button
              onClick={() => setIsRunning(!isRunning)}
              className={`w-full mt-4 px-4 py-2 rounded-lg transition ${
                isRunning ? 'bg-red-500 text-white' : 'bg-yellow-500 text-white'
              }`}
            >
              {isRunning ? 'Stop' : 'Start'} Profiling
            </button>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              Bottleneck
            </h4>

            <div className="text-center">
              <div className="text-3xl font-bold text-yellow-600 dark:text-yellow-400">
                {getBottleneck()}
              </div>
              <div className="text-sm text-gray-500 mt-2">
                Primary resource constraint
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-3">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              Resource Utilization Timeline
            </h4>

            <canvas
              ref={canvasRef}
              width={900}
              height={300}
              className="w-full border border-gray-300 dark:border-gray-600 rounded"
            />

            <div className="grid grid-cols-4 gap-4 mt-4">
              {[
                { label: 'CPU', color: '#f59e0b', value: data[data.length - 1]?.cpu || 0 },
                { label: 'Memory', color: '#3b82f6', value: data[data.length - 1]?.memory || 0 },
                { label: 'Network', color: '#10b981', value: data[data.length - 1]?.network || 0 },
                { label: 'GPU', color: '#8b5cf6', value: data[data.length - 1]?.gpu || 0 },
              ].map(({ label, color, value }) => (
                <div key={label} className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
                  <div>
                    <div className="text-xs text-gray-500">{label}</div>
                    <div className="text-lg font-bold" style={{ color }}>
                      {value.toFixed(1)}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
