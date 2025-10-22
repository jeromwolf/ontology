'use client'

import { useState } from 'react'
import { Play, CheckCircle, XCircle, Clock, GitBranch, Database, Package, Rocket } from 'lucide-react'

interface Stage {
  id: string
  name: string
  status: 'pending' | 'running' | 'success' | 'failed'
  duration: number
  icon: React.ReactNode
}

export default function MLOpsPipeline() {
  const [pipeline, setPipeline] = useState<Stage[]>([
    { id: 'data-validation', name: 'Data Validation', status: 'pending', duration: 0, icon: <Database className="w-5 h-5" /> },
    { id: 'feature-engineering', name: 'Feature Engineering', status: 'pending', duration: 0, icon: <GitBranch className="w-5 h-5" /> },
    { id: 'model-training', name: 'Model Training', status: 'pending', duration: 0, icon: <Package className="w-5 h-5" /> },
    { id: 'model-evaluation', name: 'Model Evaluation', status: 'pending', duration: 0, icon: <CheckCircle className="w-5 h-5" /> },
    { id: 'model-registry', name: 'Model Registry', status: 'pending', duration: 0, icon: <Database className="w-5 h-5" /> },
    { id: 'deployment', name: 'Deployment', status: 'pending', duration: 0, icon: <Rocket className="w-5 h-5" /> },
  ])

  const [isRunning, setIsRunning] = useState(false)
  const [currentStage, setCurrentStage] = useState(0)

  const runPipeline = () => {
    setIsRunning(true)
    setCurrentStage(0)
    setPipeline((prev) => prev.map((stage) => ({ ...stage, status: 'pending', duration: 0 })))

    let stageIndex = 0

    const interval = setInterval(() => {
      if (stageIndex >= pipeline.length) {
        clearInterval(interval)
        setIsRunning(false)
        return
      }

      const duration = Math.floor(Math.random() * 20) + 10
      const success = Math.random() > 0.15 // 85% success rate

      setPipeline((prev) => {
        const updated = [...prev]
        updated[stageIndex].status = 'running'
        return updated
      })

      setTimeout(() => {
        setPipeline((prev) => {
          const updated = [...prev]
          updated[stageIndex].status = success ? 'success' : 'failed'
          updated[stageIndex].duration = duration
          return updated
        })

        if (!success) {
          clearInterval(interval)
          setIsRunning(false)
        } else {
          stageIndex++
          setCurrentStage(stageIndex)
        }
      }, duration * 100)
    }, 100)
  }

  const getStatusColor = (status: Stage['status']) => {
    switch (status) {
      case 'pending': return 'border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-800'
      case 'running': return 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
      case 'success': return 'border-green-500 bg-green-50 dark:bg-green-900/20'
      case 'failed': return 'border-red-500 bg-red-50 dark:bg-red-900/20'
    }
  }

  const getStatusIcon = (status: Stage['status']) => {
    switch (status) {
      case 'pending': return <Clock className="w-5 h-5 text-gray-400" />
      case 'running': return <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      case 'success': return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'failed': return <XCircle className="w-5 h-5 text-red-500" />
    }
  }

  const getTotalDuration = () => {
    return pipeline.reduce((sum, stage) => sum + stage.duration, 0)
  }

  const getSuccessRate = () => {
    const completed = pipeline.filter(s => s.status === 'success' || s.status === 'failed').length
    const succeeded = pipeline.filter(s => s.status === 'success').length
    return completed > 0 ? (succeeded / completed) * 100 : 0
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-slate-50 to-gray-50 dark:from-slate-900/20 dark:to-gray-900/20 rounded-lg p-6 border-l-4 border-slate-600">
        <h3 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
          MLOps 파이프라인 구축
        </h3>
        <p className="text-gray-700 dark:text-gray-300">
          자동화된 ML 파이프라인의 각 단계를 시뮬레이션합니다
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between mb-6">
              <h4 className="text-lg font-semibold text-gray-900 dark:text-white">
                Pipeline Stages
              </h4>
              <button
                onClick={runPipeline}
                disabled={isRunning}
                className={`px-4 py-2 rounded-lg transition flex items-center gap-2 ${
                  isRunning
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-slate-600 hover:bg-slate-700 text-white'
                }`}
              >
                <Play className="w-4 h-4" />
                Run Pipeline
              </button>
            </div>

            <div className="space-y-3">
              {pipeline.map((stage, index) => (
                <div key={stage.id}>
                  <div
                    className={`p-4 rounded-lg border-2 transition ${getStatusColor(stage.status)}`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3 flex-1">
                        <div className="text-slate-600 dark:text-slate-400">
                          {stage.icon}
                        </div>
                        <div className="flex-1">
                          <div className="font-semibold text-gray-900 dark:text-white">
                            {stage.name}
                          </div>
                          {stage.duration > 0 && (
                            <div className="text-xs text-gray-600 dark:text-gray-400">
                              Duration: {stage.duration}s
                            </div>
                          )}
                        </div>
                      </div>
                      {getStatusIcon(stage.status)}
                    </div>
                  </div>

                  {index < pipeline.length - 1 && (
                    <div className="flex justify-center py-1">
                      <div className={`w-0.5 h-4 ${
                        stage.status === 'success' ? 'bg-green-500' : 'bg-gray-300 dark:bg-gray-700'
                      }`} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              Pipeline Metrics
            </h4>

            <div className="space-y-4">
              <div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Total Duration</div>
                <div className="text-2xl font-bold text-slate-600 dark:text-slate-400">
                  {getTotalDuration()}s
                </div>
              </div>

              <div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Success Rate</div>
                <div className="text-2xl font-bold text-slate-600 dark:text-slate-400">
                  {getSuccessRate().toFixed(0)}%
                </div>
              </div>

              <div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Current Stage</div>
                <div className="text-lg font-semibold text-slate-600 dark:text-slate-400">
                  {isRunning ? pipeline[currentStage]?.name || 'Complete' : 'Not Running'}
                </div>
              </div>

              <div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">Progress</div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                  <div
                    className="bg-slate-600 h-3 rounded-full transition-all"
                    style={{ width: `${(currentStage / pipeline.length) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="bg-slate-50 dark:bg-slate-900/20 rounded-lg p-4 border-l-4 border-slate-600">
            <h5 className="font-semibold mb-2 text-gray-900 dark:text-white">
              MLOps Best Practices
            </h5>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>자동화</strong>: CI/CD 파이프라인 완전 자동화</li>
              <li>• <strong>버전 관리</strong>: 모델, 데이터, 코드 버전 추적</li>
              <li>• <strong>모니터링</strong>: 실시간 성능 모니터링</li>
              <li>• <strong>재현성</strong>: 실험 결과 재현 가능</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
