'use client'

import { useState, useEffect } from 'react'
import { Play, Pause, RotateCcw, Scale } from 'lucide-react'

interface Worker {
  id: number
  load: number
  tasks: number[]
  status: 'idle' | 'busy' | 'overloaded'
}

interface Task {
  id: number
  cost: number
  assignedTo?: number
}

type Strategy = 'round-robin' | 'least-loaded' | 'work-stealing' | 'random'

export default function LoadBalancer() {
  const [workers, setWorkers] = useState<Worker[]>([])
  const [tasks, setTasks] = useState<Task[]>([])
  const [strategy, setStrategy] = useState<Strategy>('round-robin')
  const [currentTaskId, setCurrentTaskId] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const [stats, setStats] = useState({ balanced: 0, imbalance: 0 })

  const WORKER_COUNT = 8

  useEffect(() => {
    reset()
  }, [])

  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(() => {
      addTask()
    }, 500)

    return () => clearInterval(interval)
  }, [isRunning, tasks, workers, strategy])

  const reset = () => {
    const initialWorkers: Worker[] = Array.from({ length: WORKER_COUNT }, (_, i) => ({
      id: i,
      load: 0,
      tasks: [],
      status: 'idle',
    }))

    setWorkers(initialWorkers)
    setTasks([])
    setCurrentTaskId(0)
    setIsRunning(false)
  }

  const addTask = () => {
    const newTask: Task = {
      id: currentTaskId,
      cost: Math.floor(Math.random() * 20) + 5,
    }

    setCurrentTaskId(prev => prev + 1)

    let targetWorkerId = 0

    switch (strategy) {
      case 'round-robin':
        targetWorkerId = currentTaskId % WORKER_COUNT
        break

      case 'least-loaded':
        targetWorkerId = workers.reduce((min, worker) =>
          worker.load < workers[min].load ? worker.id : min, 0
        )
        break

      case 'work-stealing':
        targetWorkerId = workers.reduce((min, worker) =>
          worker.load < workers[min].load ? worker.id : min, 0
        )
        break

      case 'random':
        targetWorkerId = Math.floor(Math.random() * WORKER_COUNT)
        break
    }

    newTask.assignedTo = targetWorkerId

    setWorkers(prev => {
      const updated = [...prev]
      updated[targetWorkerId].load += newTask.cost
      updated[targetWorkerId].tasks.push(newTask.id)
      updated[targetWorkerId].status =
        updated[targetWorkerId].load > 100 ? 'overloaded' :
        updated[targetWorkerId].load > 50 ? 'busy' : 'idle'
      return updated
    })

    setTasks(prev => [...prev, newTask])

    // Calculate imbalance
    setTimeout(() => {
      calculateStats()
    }, 100)
  }

  const calculateStats = () => {
    const loads = workers.map(w => w.load)
    const avgLoad = loads.reduce((sum, l) => sum + l, 0) / WORKER_COUNT
    const variance = loads.reduce((sum, l) => sum + Math.pow(l - avgLoad, 2), 0) / WORKER_COUNT
    const stdDev = Math.sqrt(variance)

    const balanced = avgLoad > 0 ? Math.max(0, 100 - (stdDev / avgLoad) * 100) : 100
    const imbalance = avgLoad > 0 ? (stdDev / avgLoad) * 100 : 0

    setStats({ balanced, imbalance })
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg p-6 border-l-4 border-yellow-500">
        <h3 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
          부하 균형 시뮬레이터
        </h3>
        <p className="text-gray-700 dark:text-gray-300">
          다양한 로드 밸런싱 전략의 효율성을 비교합니다
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              Strategy
            </h4>

            <div className="space-y-2">
              {[
                { id: 'round-robin', name: 'Round Robin', desc: '순환 할당' },
                { id: 'least-loaded', name: 'Least Loaded', desc: '최소 부하' },
                { id: 'work-stealing', name: 'Work Stealing', desc: '작업 훔치기' },
                { id: 'random', name: 'Random', desc: '무작위 할당' },
              ].map(s => (
                <button
                  key={s.id}
                  onClick={() => setStrategy(s.id as Strategy)}
                  className={`w-full text-left px-4 py-3 rounded-lg border-2 transition ${
                    strategy === s.id
                      ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
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
                onClick={() => setIsRunning(!isRunning)}
                className={`flex-1 px-4 py-2 rounded-lg transition ${
                  isRunning ? 'bg-red-500 text-white' : 'bg-yellow-500 text-white'
                }`}
              >
                {isRunning ? <><Pause className="w-4 h-4 inline" /> Pause</> : <><Play className="w-4 h-4 inline" /> Start</>}
              </button>
              <button
                onClick={reset}
                className="px-4 py-2 bg-gray-500 text-white rounded-lg"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
              <Scale className="w-5 h-5 text-yellow-500" />
              Balance Score
            </h4>

            <div className="text-center">
              <div className="text-3xl font-bold text-yellow-600 dark:text-yellow-400">
                {stats.balanced.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-500 mt-2">
                Load balance efficiency
              </div>

              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Imbalance: {stats.imbalance.toFixed(1)}%
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Total tasks: {tasks.length}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-3">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              Worker Loads
            </h4>

            <div className="space-y-3">
              {workers.map(worker => (
                <div key={worker.id} className="space-y-1">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-mono text-gray-700 dark:text-gray-300">
                      Worker {worker.id}
                    </span>
                    <div className="flex items-center gap-3">
                      <span
                        className={`text-xs px-2 py-0.5 rounded ${
                          worker.status === 'idle' ? 'bg-gray-200 dark:bg-gray-700' :
                          worker.status === 'busy' ? 'bg-yellow-200 dark:bg-yellow-800' :
                          'bg-red-200 dark:bg-red-800'
                        }`}
                      >
                        {worker.status}
                      </span>
                      <span className="font-mono font-semibold text-gray-900 dark:text-white">
                        {worker.load.toFixed(0)}
                      </span>
                    </div>
                  </div>

                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4 overflow-hidden">
                    <div
                      className={`h-4 rounded-full transition-all ${
                        worker.load > 100 ? 'bg-red-500' :
                        worker.load > 50 ? 'bg-yellow-500' :
                        'bg-green-500'
                      }`}
                      style={{ width: `${Math.min(worker.load, 100)}%` }}
                    />
                  </div>

                  <div className="text-xs text-gray-500">
                    Tasks: {worker.tasks.length}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4 border-l-4 border-yellow-500">
            <h5 className="font-semibold mb-2 text-gray-900 dark:text-white">
              전략 비교
            </h5>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>Round Robin</strong>: 간단하지만 작업 크기가 다르면 불균형</li>
              <li>• <strong>Least Loaded</strong>: 항상 최소 부하 워커에 할당 (최적)</li>
              <li>• <strong>Work Stealing</strong>: 유휴 워커가 바쁜 워커에서 작업 가져옴</li>
              <li>• <strong>Random</strong>: 무작위 할당 (가장 비효율적)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
