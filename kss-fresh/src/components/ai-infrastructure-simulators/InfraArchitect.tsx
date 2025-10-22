'use client'

import { useState } from 'react'
import { Server, Database, Network, Cpu, HardDrive, Shield, Zap, CheckCircle } from 'lucide-react'

interface Component {
  id: string
  name: string
  type: 'compute' | 'storage' | 'network' | 'ml-service'
  specs: string
  cost: number
}

interface Architecture {
  name: string
  components: Component[]
  totalCost: number
  estimatedPerformance: number
}

export default function InfraArchitect() {
  const [architecture, setArchitecture] = useState<Architecture>({
    name: 'My AI Infrastructure',
    components: [],
    totalCost: 0,
    estimatedPerformance: 0,
  })

  const [selectedWorkload, setSelectedWorkload] = useState<'training' | 'inference' | 'both'>('training')

  const availableComponents: Component[] = [
    // Compute
    { id: 'gpu-a100', name: 'NVIDIA A100 (8×)', type: 'compute', specs: '80GB HBM2e, 19.5 TFLOPS', cost: 25000 },
    { id: 'gpu-h100', name: 'NVIDIA H100 (8×)', type: 'compute', specs: '80GB HBM3, 30 TFLOPS', cost: 40000 },
    { id: 'cpu-epyc', name: 'AMD EPYC 7763 (2×)', type: 'compute', specs: '128 cores, 512GB RAM', cost: 15000 },
    { id: 'cpu-xeon', name: 'Intel Xeon 8480 (2×)', type: 'compute', specs: '112 cores, 1TB RAM', cost: 18000 },

    // Storage
    { id: 'nvme-ssd', name: 'NVMe SSD Array', type: 'storage', specs: '100TB, 7GB/s read', cost: 8000 },
    { id: 'object-storage', name: 'Object Storage', type: 'storage', specs: '1PB, S3-compatible', cost: 12000 },
    { id: 'parallel-fs', name: 'Parallel File System', type: 'storage', specs: 'Lustre, 50GB/s', cost: 20000 },

    // Network
    { id: 'infiniband', name: 'InfiniBand Network', type: 'network', specs: '400 Gbps, RDMA', cost: 10000 },
    { id: 'roce', name: 'RoCE Network', type: 'network', specs: '200 Gbps, Ethernet', cost: 5000 },

    // ML Services
    { id: 'mlops-platform', name: 'MLOps Platform', type: 'ml-service', specs: 'Kubeflow + MLflow', cost: 3000 },
    { id: 'monitoring', name: 'Monitoring Stack', type: 'ml-service', specs: 'Prometheus + Grafana', cost: 2000 },
    { id: 'serving', name: 'Model Serving', type: 'ml-service', specs: 'TensorFlow Serving + Triton', cost: 4000 },
  ]

  const addComponent = (component: Component) => {
    const newComponents = [...architecture.components, component]
    const totalCost = newComponents.reduce((sum, c) => sum + c.cost, 0)

    // Calculate performance score based on components
    let performance = 0
    const gpuCount = newComponents.filter(c => c.id.includes('gpu')).length
    const storageCount = newComponents.filter(c => c.type === 'storage').length
    const networkCount = newComponents.filter(c => c.type === 'network').length

    performance = (gpuCount * 30) + (storageCount * 15) + (networkCount * 20)

    setArchitecture({
      ...architecture,
      components: newComponents,
      totalCost,
      estimatedPerformance: Math.min(performance, 100),
    })
  }

  const removeComponent = (index: number) => {
    const newComponents = architecture.components.filter((_, i) => i !== index)
    const totalCost = newComponents.reduce((sum, c) => sum + c.cost, 0)

    let performance = 0
    const gpuCount = newComponents.filter(c => c.id.includes('gpu')).length
    const storageCount = newComponents.filter(c => c.type === 'storage').length
    const networkCount = newComponents.filter(c => c.type === 'network').length

    performance = (gpuCount * 30) + (storageCount * 15) + (networkCount * 20)

    setArchitecture({
      ...architecture,
      components: newComponents,
      totalCost,
      estimatedPerformance: Math.min(performance, 100),
    })
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'compute': return <Cpu className="w-5 h-5 text-slate-600" />
      case 'storage': return <HardDrive className="w-5 h-5 text-blue-600" />
      case 'network': return <Network className="w-5 h-5 text-green-600" />
      case 'ml-service': return <Zap className="w-5 h-5 text-purple-600" />
      default: return <Server className="w-5 h-5" />
    }
  }

  const getRecommendations = () => {
    const hasGPU = architecture.components.some(c => c.id.includes('gpu'))
    const hasStorage = architecture.components.some(c => c.type === 'storage')
    const hasNetwork = architecture.components.some(c => c.type === 'network')
    const hasMLOps = architecture.components.some(c => c.type === 'ml-service')

    const recommendations = []

    if (!hasGPU && selectedWorkload !== 'inference') {
      recommendations.push('훈련을 위해 GPU 추가를 권장합니다')
    }
    if (!hasStorage) {
      recommendations.push('데이터 저장을 위한 스토리지가 필요합니다')
    }
    if (!hasNetwork && architecture.components.length > 1) {
      recommendations.push('고속 네트워크로 컴포넌트 연결이 필요합니다')
    }
    if (!hasMLOps) {
      recommendations.push('MLOps 플랫폼으로 운영 효율성을 높이세요')
    }

    return recommendations
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-slate-50 to-gray-50 dark:from-slate-900/20 dark:to-gray-900/20 rounded-lg p-6 border-l-4 border-slate-600">
        <h3 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
          AI 인프라 설계 도구
        </h3>
        <p className="text-gray-700 dark:text-gray-300">
          최적의 AI 인프라 아키텍처를 설계하고 비용을 추정합니다
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Workload Selection */}
        <div className="lg:col-span-3">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              워크로드 타입
            </h4>
            <div className="flex gap-3">
              {[
                { id: 'training', label: '모델 훈련', desc: '대규모 GPU 필요' },
                { id: 'inference', label: '추론 서빙', desc: '낮은 지연시간 중요' },
                { id: 'both', label: '훈련 + 추론', desc: '하이브리드 환경' },
              ].map((workload) => (
                <button
                  key={workload.id}
                  onClick={() => setSelectedWorkload(workload.id as any)}
                  className={`flex-1 p-4 rounded-lg border-2 transition ${
                    selectedWorkload === workload.id
                      ? 'border-slate-600 bg-slate-50 dark:bg-slate-900/20'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <div className="font-semibold text-gray-900 dark:text-white">{workload.label}</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">{workload.desc}</div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Available Components */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              사용 가능한 컴포넌트
            </h4>
            <div className="space-y-2 max-h-[500px] overflow-y-auto">
              {availableComponents.map((component) => (
                <div
                  key={component.id}
                  className="flex items-center justify-between p-3 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-900/20 transition"
                >
                  <div className="flex items-center gap-3 flex-1">
                    {getTypeIcon(component.type)}
                    <div className="flex-1">
                      <div className="font-semibold text-sm text-gray-900 dark:text-white">
                        {component.name}
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">
                        {component.specs}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="text-sm font-mono text-slate-600 dark:text-slate-400">
                      ${component.cost.toLocaleString()}
                    </div>
                    <button
                      onClick={() => addComponent(component)}
                      className="px-3 py-1 bg-slate-600 text-white rounded text-sm hover:bg-slate-700 transition"
                    >
                      추가
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Current Architecture */}
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              현재 아키텍처
            </h4>

            {architecture.components.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                컴포넌트를 추가하세요
              </div>
            ) : (
              <div className="space-y-2">
                {architecture.components.map((component, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-900/20 rounded-lg"
                  >
                    <div className="flex items-center gap-2 flex-1">
                      {getTypeIcon(component.type)}
                      <div className="text-sm text-gray-900 dark:text-white">
                        {component.name}
                      </div>
                    </div>
                    <button
                      onClick={() => removeComponent(index)}
                      className="text-red-600 hover:text-red-700 text-sm"
                    >
                      제거
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700 space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600 dark:text-gray-400">총 비용</span>
                <span className="font-bold text-slate-600 dark:text-slate-400 font-mono">
                  ${architecture.totalCost.toLocaleString()}
                </span>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600 dark:text-gray-400">예상 성능</span>
                  <span className="font-bold text-slate-600 dark:text-slate-400">
                    {architecture.estimatedPerformance}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-slate-600 h-2 rounded-full transition-all"
                    style={{ width: `${architecture.estimatedPerformance}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          {getRecommendations().length > 0 && (
            <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4 border-l-4 border-yellow-500">
              <h5 className="font-semibold mb-2 text-gray-900 dark:text-white flex items-center gap-2">
                <Shield className="w-5 h-5 text-yellow-600" />
                권장 사항
              </h5>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                {getRecommendations().map((rec, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-yellow-600 mt-0.5" />
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
