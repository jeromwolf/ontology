'use client'

import { useState, useEffect } from 'react'
import { Play, Pause, RotateCcw, Cpu, TrendingUp } from 'lucide-react'

interface Worker {
  id: number
  gpu: string
  progress: number
  loss: number
  throughput: number
  status: 'idle' | 'training' | 'syncing' | 'error'
}

type Strategy = 'data-parallel' | 'model-parallel' | 'pipeline-parallel' | 'hybrid'

export default function DistributedTrainer() {
  const [workers, setWorkers] = useState<Worker[]>([])
  const [strategy, setStrategy] = useState<Strategy>('data-parallel')
  const [isTraining, setIsTraining] = useState(false)
  const [epoch, setEpoch] = useState(0)
  const [globalLoss, setGlobalLoss] = useState(2.5)

  const WORKER_COUNT = 8

  useEffect(() => {
    initializeWorkers()
  }, [strategy])

  useEffect(() => {
    if (!isTraining) return

    const interval = setInterval(() => {
      updateTraining()
    }, 500)

    return () => clearInterval(interval)
  }, [isTraining, workers, strategy])

  const initializeWorkers = () => {
    const gpuTypes = ['A100', 'H100', 'V100', 'A100']
    const initialWorkers: Worker[] = Array.from({ length: WORKER_COUNT }, (_, i) => ({
      id: i,
      gpu: gpuTypes[i % 4],
      progress: 0,
      loss: 2.5 + Math.random() * 0.5,
      throughput: 0,
      status: 'idle',
    }))
    setWorkers(initialWorkers)
    setEpoch(0)
    setGlobalLoss(2.5)
  }

  const updateTraining = () => {
    setWorkers((prev) => {
      const updated = prev.map((worker) => {
        let newProgress = worker.progress + (Math.random() * 5 + 2)
        let newStatus: Worker['status'] = 'training'

        // Simulate synchronization every 20% progress
        if (Math.floor(newProgress / 20) > Math.floor(worker.progress / 20)) {
          newStatus = 'syncing'
        }

        if (newProgress >= 100) {
          newProgress = 0
          setEpoch((e) => e + 1)
        }

        const newLoss = Math.max(0.1, worker.loss - Math.random() * 0.02)
        const throughput = strategy === 'data-parallel' ? 120 + Math.random() * 20 :
                          strategy === 'model-parallel' ? 80 + Math.random() * 15 :
                          strategy === 'pipeline-parallel' ? 100 + Math.random() * 20 : 95 + Math.random() * 15

        return {
          ...worker,
          progress: newProgress,
          loss: newLoss,
          throughput: Math.round(throughput),
          status: newStatus,
        }
      })

      // Calculate global loss
      const avgLoss = updated.reduce((sum, w) => sum + w.loss, 0) / updated.length
      setGlobalLoss(avgLoss)

      return updated
    })
  }

  const getEfficiency = () => {
    if (workers.length === 0) return 0

    const baseEfficiency = {
      'data-parallel': 95,
      'model-parallel': 75,
      'pipeline-parallel': 85,
      'hybrid': 88,
    }[strategy]

    return baseEfficiency + Math.random() * 5
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-slate-50 to-gray-50 dark:from-slate-900/20 dark:to-gray-900/20 rounded-lg p-6 border-l-4 border-slate-600">
        <h3 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
          분산 학습 시뮬레이터
        </h3>
        <p className="text-gray-700 dark:text-gray-300">
          다양한 분산 학습 전략의 성능과 효율성을 비교합니다
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              분산 전략
            </h4>

            <div className="space-y-2">
              {[
                { id: 'data-parallel', name: 'Data Parallel', desc: '데이터 분할' },
                { id: 'model-parallel', name: 'Model Parallel', desc: '모델 분할' },
                { id: 'pipeline-parallel', name: 'Pipeline Parallel', desc: '파이프라인' },
                { id: 'hybrid', name: 'Hybrid', desc: '혼합 전략' },
              ].map((s) => (
                <button
                  key={s.id}
                  onClick={() => { setStrategy(s.id as Strategy); initializeWorkers(); }}
                  disabled={isTraining}
                  className={`w-full text-left px-4 py-3 rounded-lg border-2 transition ${
                    strategy === s.id
                      ? 'border-slate-600 bg-slate-50 dark:bg-slate-900/20'
                      : 'border-gray-200 dark:border-gray-700'
                  } ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <div className="font-semibold text-sm text-gray-900 dark:text-white">
                    {s.name}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    {s.desc}
                  </div>
                </button>
              ))}
            </div>

            <div className="flex gap-2 mt-4">
              <button
                onClick={() => setIsTraining(!isTraining)}
                className={`flex-1 px-4 py-2 rounded-lg transition flex items-center justify-center gap-2 ${
                  isTraining ? 'bg-red-500 text-white' : 'bg-slate-600 text-white'
                }`}
              >
                {isTraining ? <><Pause className="w-4 h-4" /> Pause</> : <><Play className="w-4 h-4" /> Start</>}
              </button>
              <button
                onClick={initializeWorkers}
                className="px-4 py-2 bg-gray-500 text-white rounded-lg"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-slate-600" />
              Training Metrics
            </h4>

            <div className="space-y-3">
              <div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Epoch</div>
                <div className="text-2xl font-bold text-slate-600 dark:text-slate-400">
                  {epoch}
                </div>
              </div>

              <div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Global Loss</div>
                <div className="text-2xl font-bold text-slate-600 dark:text-slate-400">
                  {globalLoss.toFixed(4)}
                </div>
              </div>

              <div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Efficiency</div>
                <div className="text-2xl font-bold text-slate-600 dark:text-slate-400">
                  {getEfficiency().toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-3">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              Worker Status (8 GPUs)
            </h4>

            <div className="grid grid-cols-2 gap-3">
              {workers.map((worker) => (
                <div
                  key={worker.id}
                  className={`p-4 rounded-lg border-2 transition ${
                    worker.status === 'training' ? 'border-green-500 bg-green-50 dark:bg-green-900/20' :
                    worker.status === 'syncing' ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20' :
                    worker.status === 'error' ? 'border-red-500 bg-red-50 dark:bg-red-900/20' :
                    'border-gray-300 dark:border-gray-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Cpu className="w-4 h-4 text-slate-600" />
                      <span className="font-mono font-bold text-sm text-gray-900 dark:text-white">
                        Worker {worker.id}
                      </span>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded ${
                      worker.status === 'training' ? 'bg-green-200 dark:bg-green-800 text-green-900 dark:text-green-100' :
                      worker.status === 'syncing' ? 'bg-yellow-200 dark:bg-yellow-800 text-yellow-900 dark:text-yellow-100' :
                      worker.status === 'error' ? 'bg-red-200 dark:bg-red-800' :
                      'bg-gray-200 dark:bg-gray-700'
                    }`}>
                      {worker.status}
                    </span>
                  </div>

                  <div className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                    GPU: {worker.gpu} | Loss: {worker.loss.toFixed(3)}
                  </div>

                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mb-2">
                    <div
                      className="bg-slate-600 h-2 rounded-full transition-all"
                      style={{ width: `${worker.progress}%` }}
                    />
                  </div>

                  <div className="text-xs text-gray-500">
                    {worker.throughput} samples/sec
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 bg-slate-50 dark:bg-slate-900/20 rounded-lg p-4 border-l-4 border-slate-600">
            <h5 className="font-semibold mb-2 text-gray-900 dark:text-white">
              전략 비교
            </h5>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>Data Parallel</strong>: 각 GPU가 다른 데이터 배치 처리 (가장 효율적)</li>
              <li>• <strong>Model Parallel</strong>: 모델을 여러 GPU에 분할 (대형 모델 필수)</li>
              <li>• <strong>Pipeline Parallel</strong>: 레이어를 순차적으로 분배 (메모리 효율)</li>
              <li>• <strong>Hybrid</strong>: 위 전략들을 결합 (최대 확장성)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
