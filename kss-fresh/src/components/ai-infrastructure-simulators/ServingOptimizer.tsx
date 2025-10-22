'use client'

import { useState } from 'react'
import { Zap, Cpu, HardDrive, CheckCircle } from 'lucide-react'

interface OptimizationConfig {
  batchSize: number
  modelQuantization: 'fp32' | 'fp16' | 'int8'
  instanceType: 'cpu' | 'gpu-t4' | 'gpu-a10' | 'gpu-a100'
  autoscaling: boolean
  caching: boolean
}

export default function ServingOptimizer() {
  const [config, setConfig] = useState<OptimizationConfig>({
    batchSize: 8,
    modelQuantization: 'fp32',
    instanceType: 'gpu-t4',
    autoscaling: true,
    caching: false,
  })

  const calculateMetrics = () => {
    let latency = 100 // base latency in ms
    let throughput = 10 // base throughput in requests/sec
    let cost = 5 // base cost in $/hour

    // Batch size impact
    latency += config.batchSize * 2
    throughput += config.batchSize * 3

    // Quantization impact
    if (config.modelQuantization === 'fp16') {
      latency *= 0.7
      throughput *= 1.4
    } else if (config.modelQuantization === 'int8') {
      latency *= 0.5
      throughput *= 1.8
    }

    // Instance type impact
    if (config.instanceType === 'cpu') {
      latency *= 3
      throughput *= 0.3
      cost = 2
    } else if (config.instanceType === 'gpu-t4') {
      cost = 5
    } else if (config.instanceType === 'gpu-a10') {
      latency *= 0.7
      throughput *= 1.5
      cost = 8
    } else if (config.instanceType === 'gpu-a100') {
      latency *= 0.5
      throughput *= 2
      cost = 15
    }

    // Caching impact
    if (config.caching) {
      latency *= 0.8
      throughput *= 1.3
      cost += 1
    }

    // Autoscaling doesn't affect single instance metrics but adds cost
    if (config.autoscaling) {
      cost += 2
    }

    return {
      latency: Math.round(latency),
      throughput: Math.round(throughput),
      cost: Math.round(cost * 10) / 10,
    }
  }

  const metrics = calculateMetrics()

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-slate-50 to-gray-50 dark:from-slate-900/20 dark:to-gray-900/20 rounded-lg p-6 border-l-4 border-slate-600">
        <h3 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
          모델 서빙 최적화
        </h3>
        <p className="text-gray-700 dark:text-gray-300">
          서빙 설정을 조정하여 성능과 비용의 균형을 찾습니다
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              Optimization Settings
            </h4>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Batch Size: {config.batchSize}
                </label>
                <input
                  type="range"
                  min="1"
                  max="32"
                  value={config.batchSize}
                  onChange={(e) => setConfig({ ...config, batchSize: parseInt(e.target.value) })}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>1</span>
                  <span>32</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Model Quantization
                </label>
                <div className="space-y-2">
                  {[
                    { value: 'fp32', label: 'FP32 (Full Precision)', desc: '최고 정확도' },
                    { value: 'fp16', label: 'FP16 (Half Precision)', desc: '균형잡힌 선택' },
                    { value: 'int8', label: 'INT8 (Quantized)', desc: '최고 속도' },
                  ].map((option) => (
                    <button
                      key={option.value}
                      onClick={() => setConfig({ ...config, modelQuantization: option.value as any })}
                      className={`w-full text-left px-4 py-3 rounded-lg border-2 transition ${
                        config.modelQuantization === option.value
                          ? 'border-slate-600 bg-slate-50 dark:bg-slate-900/20'
                          : 'border-gray-200 dark:border-gray-700'
                      }`}
                    >
                      <div className="font-semibold text-sm text-gray-900 dark:text-white">
                        {option.label}
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">{option.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Instance Type
                </label>
                <div className="space-y-2">
                  {[
                    { value: 'cpu', label: 'CPU', desc: '저렴하지만 느림' },
                    { value: 'gpu-t4', label: 'GPU T4', desc: '비용 효율적' },
                    { value: 'gpu-a10', label: 'GPU A10', desc: '고성능' },
                    { value: 'gpu-a100', label: 'GPU A100', desc: '최고 성능' },
                  ].map((option) => (
                    <button
                      key={option.value}
                      onClick={() => setConfig({ ...config, instanceType: option.value as any })}
                      className={`w-full text-left px-4 py-3 rounded-lg border-2 transition ${
                        config.instanceType === option.value
                          ? 'border-slate-600 bg-slate-50 dark:bg-slate-900/20'
                          : 'border-gray-200 dark:border-gray-700'
                      }`}
                    >
                      <div className="font-semibold text-sm text-gray-900 dark:text-white">
                        {option.label}
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">{option.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={config.autoscaling}
                    onChange={(e) => setConfig({ ...config, autoscaling: e.target.checked })}
                    className="w-4 h-4"
                  />
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Auto-scaling 활성화
                  </span>
                </label>

                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={config.caching}
                    onChange={(e) => setConfig({ ...config, caching: e.target.checked })}
                    className="w-4 h-4"
                  />
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Response Caching 활성화
                  </span>
                </label>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
              <Zap className="w-5 h-5 text-slate-600" />
              Performance Metrics
            </h4>

            <div className="space-y-4">
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <div className="flex items-center gap-2 mb-1">
                  <Cpu className="w-5 h-5 text-blue-600" />
                  <div className="text-sm text-gray-600 dark:text-gray-400">Latency (P50)</div>
                </div>
                <div className="text-3xl font-bold text-blue-600">{metrics.latency}ms</div>
              </div>

              <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <div className="flex items-center gap-2 mb-1">
                  <CheckCircle className="w-5 h-5 text-green-600" />
                  <div className="text-sm text-gray-600 dark:text-gray-400">Throughput</div>
                </div>
                <div className="text-3xl font-bold text-green-600">{metrics.throughput} req/s</div>
              </div>

              <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                <div className="flex items-center gap-2 mb-1">
                  <HardDrive className="w-5 h-5 text-yellow-600" />
                  <div className="text-sm text-gray-600 dark:text-gray-400">Estimated Cost</div>
                </div>
                <div className="text-3xl font-bold text-yellow-600">${metrics.cost}/hour</div>
              </div>
            </div>
          </div>

          <div className="bg-slate-50 dark:bg-slate-900/20 rounded-lg p-4 border-l-4 border-slate-600">
            <h5 className="font-semibold mb-2 text-gray-900 dark:text-white">
              최적화 가이드
            </h5>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>Batch Size</strong>: 높이면 처리량 증가하지만 지연시간도 증가</li>
              <li>• <strong>Quantization</strong>: 정확도 약간 감소하지만 속도 대폭 개선</li>
              <li>• <strong>Instance Type</strong>: 워크로드 특성에 맞는 인스턴스 선택</li>
              <li>• <strong>Caching</strong>: 반복 요청이 많을 때 효과적</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
