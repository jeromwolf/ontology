'use client'

import { useState, useEffect, useRef } from 'react'
import { Activity, TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react'

interface Metric {
  timestamp: number
  accuracy: number
  latency: number
  throughput: number
  errorRate: number
}

export default function ModelMonitor() {
  const [metrics, setMetrics] = useState<Metric[]>([])
  const [isMonitoring, setIsMonitoring] = useState(false)
  const [alerts, setAlerts] = useState<string[]>([])
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!isMonitoring) return

    const interval = setInterval(() => {
      setMetrics((prev) => {
        const newMetric: Metric = {
          timestamp: Date.now(),
          accuracy: 0.92 + (Math.random() - 0.5) * 0.08,
          latency: 50 + Math.random() * 30,
          throughput: 1000 + Math.random() * 200,
          errorRate: Math.random() * 2,
        }

        const updated = [...prev, newMetric]
        if (updated.length > 50) updated.shift()

        // Check for alerts
        if (newMetric.accuracy < 0.88) {
          setAlerts((a) => [...a, `정확도 하락: ${(newMetric.accuracy * 100).toFixed(1)}%`])
        }
        if (newMetric.latency > 75) {
          setAlerts((a) => [...a, `지연시간 증가: ${newMetric.latency.toFixed(0)}ms`])
        }

        return updated
      })
    }, 500)

    return () => clearInterval(interval)
  }, [isMonitoring])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || metrics.length === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.fillStyle = '#1a1a1a'
    ctx.fillRect(0, 0, width, height)

    // Draw accuracy line
    ctx.strokeStyle = '#10b981'
    ctx.lineWidth = 2
    ctx.beginPath()
    metrics.forEach((m, i) => {
      const x = (i / metrics.length) * width
      const y = height - (m.accuracy * height)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // Draw grid
    ctx.strokeStyle = '#333'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = (i / 4) * height
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }
  }, [metrics])

  const getLatestMetric = () => {
    return metrics[metrics.length - 1]
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-slate-50 to-gray-50 dark:from-slate-900/20 dark:to-gray-900/20 rounded-lg p-6 border-l-4 border-slate-600">
        <h3 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
          모델 모니터링 대시보드
        </h3>
        <p className="text-gray-700 dark:text-gray-300">
          프로덕션 모델의 성능 지표를 실시간으로 모니터링합니다
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <button
              onClick={() => setIsMonitoring(!isMonitoring)}
              className={`w-full px-4 py-2 rounded-lg transition ${
                isMonitoring ? 'bg-red-500 text-white' : 'bg-slate-600 text-white'
              }`}
            >
              {isMonitoring ? 'Stop Monitoring' : 'Start Monitoring'}
            </button>
          </div>

          {getLatestMetric() && (
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
                <Activity className="w-5 h-5 text-slate-600" />
                Current Metrics
              </h4>

              <div className="space-y-3">
                {[
                  { label: 'Accuracy', value: (getLatestMetric().accuracy * 100).toFixed(2) + '%', icon: TrendingUp, color: 'text-green-600' },
                  { label: 'Latency', value: getLatestMetric().latency.toFixed(0) + 'ms', icon: TrendingDown, color: 'text-blue-600' },
                  { label: 'Throughput', value: getLatestMetric().throughput.toFixed(0) + '/s', icon: TrendingUp, color: 'text-purple-600' },
                  { label: 'Error Rate', value: getLatestMetric().errorRate.toFixed(2) + '%', icon: AlertTriangle, color: 'text-red-600' },
                ].map((metric) => (
                  <div key={metric.label} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <metric.icon className={`w-4 h-4 ${metric.color}`} />
                      <span className="text-sm text-gray-600 dark:text-gray-400">{metric.label}</span>
                    </div>
                    <span className="font-bold text-gray-900 dark:text-white">{metric.value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {alerts.length > 0 && (
            <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4 border-l-4 border-red-500">
              <h5 className="font-semibold mb-2 text-gray-900 dark:text-white flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-red-600" />
                Alerts
              </h5>
              <div className="space-y-1 max-h-40 overflow-y-auto">
                {alerts.slice(-5).map((alert, idx) => (
                  <div key={idx} className="text-xs text-red-700 dark:text-red-400">
                    • {alert}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="lg:col-span-3">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              Accuracy Timeline
            </h4>

            <canvas
              ref={canvasRef}
              width={900}
              height={300}
              className="w-full border border-gray-300 dark:border-gray-600 rounded"
            />

            <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
              <strong>Monitoring Tip</strong>: 정확도가 88% 이하로 떨어지거나 지연시간이 75ms를 초과하면 알림이 발생합니다.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
