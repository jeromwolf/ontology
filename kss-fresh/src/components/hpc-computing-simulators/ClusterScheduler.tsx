'use client'

import { useState, useEffect } from 'react'
import { Play, Pause, RotateCcw, Server, Clock, Cpu } from 'lucide-react'

interface Job {
  id: number
  name: string
  nodes: number
  runtime: number
  priority: 'high' | 'normal' | 'low'
  status: 'waiting' | 'running' | 'completed'
  startTime?: number
  endTime?: number
  color: string
}

interface Node {
  id: number
  status: 'idle' | 'busy'
  currentJob?: number
}

export default function ClusterScheduler() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [nodes, setNodes] = useState<Node[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [schedulerType, setSchedulerType] = useState<'fifo' | 'sjf' | 'priority'>('fifo')

  const TOTAL_NODES = 64

  useEffect(() => {
    initializeCluster()
  }, [])

  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(() => {
      setCurrentTime(prev => prev + 1)
      updateCluster()
    }, 100)

    return () => clearInterval(interval)
  }, [isRunning, currentTime, jobs, nodes])

  const initializeCluster = () => {
    const initialNodes: Node[] = Array.from({ length: TOTAL_NODES }, (_, i) => ({
      id: i,
      status: 'idle',
    }))

    const colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b', '#ef4444']

    const initialJobs: Job[] = [
      { id: 1, name: 'Molecular Dynamics', nodes: 16, runtime: 20, priority: 'high', status: 'waiting', color: colors[0] },
      { id: 2, name: 'Climate Simulation', nodes: 32, runtime: 30, priority: 'normal', status: 'waiting', color: colors[1] },
      { id: 3, name: 'CFD Analysis', nodes: 8, runtime: 15, priority: 'low', status: 'waiting', color: colors[2] },
      { id: 4, name: 'Genomics Pipeline', nodes: 4, runtime: 10, priority: 'high', status: 'waiting', color: colors[3] },
      { id: 5, name: 'AI Training', nodes: 16, runtime: 25, priority: 'normal', status: 'waiting', color: colors[4] },
      { id: 6, name: 'Quantum Sim', nodes: 8, runtime: 12, priority: 'low', status: 'waiting', color: colors[5] },
    ]

    setNodes(initialNodes)
    setJobs(initialJobs)
    setCurrentTime(0)
  }

  const updateCluster = () => {
    setJobs(prevJobs => {
      const newJobs = [...prevJobs]

      // Complete running jobs
      newJobs.forEach(job => {
        if (job.status === 'running' && job.endTime && currentTime >= job.endTime) {
          job.status = 'completed'

          setNodes(prevNodes => {
            return prevNodes.map(node => {
              if (node.currentJob === job.id) {
                return { ...node, status: 'idle', currentJob: undefined }
              }
              return node
            })
          })
        }
      })

      // Schedule new jobs
      const waitingJobs = newJobs.filter(j => j.status === 'waiting')
      if (waitingJobs.length > 0) {
        const sortedJobs = sortJobsByScheduler(waitingJobs)

        for (const job of sortedJobs) {
          const availableNodes = nodes.filter(n => n.status === 'idle').length
          if (availableNodes >= job.nodes) {
            // Allocate nodes
            let allocated = 0
            setNodes(prevNodes => {
              return prevNodes.map(node => {
                if (node.status === 'idle' && allocated < job.nodes) {
                  allocated++
                  return { ...node, status: 'busy', currentJob: job.id }
                }
                return node
              })
            })

            // Start job
            job.status = 'running'
            job.startTime = currentTime
            job.endTime = currentTime + job.runtime
            break
          }
        }
      }

      return newJobs
    })
  }

  const sortJobsByScheduler = (waitingJobs: Job[]) => {
    switch (schedulerType) {
      case 'fifo':
        return waitingJobs
      case 'sjf':
        return [...waitingJobs].sort((a, b) => a.runtime - b.runtime)
      case 'priority':
        const priorityMap = { high: 3, normal: 2, low: 1 }
        return [...waitingJobs].sort((a, b) => priorityMap[b.priority] - priorityMap[a.priority])
      default:
        return waitingJobs
    }
  }

  const addJob = () => {
    const colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b', '#ef4444']
    const newJob: Job = {
      id: jobs.length + 1,
      name: `Job ${jobs.length + 1}`,
      nodes: Math.floor(Math.random() * 24) + 4,
      runtime: Math.floor(Math.random() * 20) + 10,
      priority: ['high', 'normal', 'low'][Math.floor(Math.random() * 3)] as 'high' | 'normal' | 'low',
      status: 'waiting',
      color: colors[Math.floor(Math.random() * colors.length)],
    }
    setJobs([...jobs, newJob])
  }

  const getUtilization = () => {
    const busyNodes = nodes.filter(n => n.status === 'busy').length
    return ((busyNodes / TOTAL_NODES) * 100).toFixed(1)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg p-6 border-l-4 border-yellow-500">
        <h3 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
          HPC 클러스터 스케줄러
        </h3>
        <p className="text-gray-700 dark:text-gray-300">
          Slurm 스타일 작업 스케줄링 시뮬레이터 ({TOTAL_NODES} 노드 클러스터)
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Controls */}
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              스케줄러 정책
            </h4>

            <div className="space-y-2">
              {[
                { id: 'fifo', name: 'FIFO', desc: 'First In First Out' },
                { id: 'sjf', name: 'SJF', desc: 'Shortest Job First' },
                { id: 'priority', name: 'Priority', desc: '우선순위 기반' },
              ].map((policy) => (
                <button
                  key={policy.id}
                  onClick={() => setSchedulerType(policy.id as any)}
                  className={`w-full text-left px-4 py-3 rounded-lg border-2 transition ${
                    schedulerType === policy.id
                      ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <div className="font-semibold text-gray-900 dark:text-white">
                    {policy.name}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    {policy.desc}
                  </div>
                </button>
              ))}
            </div>

            <div className="flex gap-2 mt-6">
              <button
                onClick={() => setIsRunning(!isRunning)}
                className={`flex-1 px-4 py-2 rounded-lg transition flex items-center justify-center gap-2 ${
                  isRunning ? 'bg-red-500 text-white' : 'bg-yellow-500 text-white'
                }`}
              >
                {isRunning ? <><Pause className="w-4 h-4" />Pause</> : <><Play className="w-4 h-4" />Start</>}
              </button>
              <button
                onClick={initializeCluster}
                className="px-4 py-2 bg-gray-500 text-white rounded-lg"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>

            <button
              onClick={addJob}
              className="w-full mt-3 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition"
            >
              + Add Random Job
            </button>
          </div>

          {/* Stats */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              클러스터 상태
            </h4>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400 flex items-center gap-2">
                  <Clock className="w-4 h-4" />
                  Current Time
                </span>
                <span className="font-mono font-semibold text-lg">{currentTime}s</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400 flex items-center gap-2">
                  <Server className="w-4 h-4" />
                  Utilization
                </span>
                <span className="font-mono font-semibold text-lg text-green-600">
                  {getUtilization()}%
                </span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400 flex items-center gap-2">
                  <Cpu className="w-4 h-4" />
                  Busy Nodes
                </span>
                <span className="font-mono font-semibold text-lg">
                  {nodes.filter(n => n.status === 'busy').length} / {TOTAL_NODES}
                </span>
              </div>

              <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                <div className="text-xs text-gray-500 space-y-1">
                  <div>Waiting: {jobs.filter(j => j.status === 'waiting').length}</div>
                  <div>Running: {jobs.filter(j => j.status === 'running').length}</div>
                  <div>Completed: {jobs.filter(j => j.status === 'completed').length}</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Visualization */}
        <div className="lg:col-span-2 space-y-4">
          {/* Node Grid */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              Cluster Nodes (8×8 Grid)
            </h4>

            <div className="grid grid-cols-8 gap-1">
              {nodes.map(node => {
                const job = jobs.find(j => j.id === node.currentJob)
                return (
                  <div
                    key={node.id}
                    className={`aspect-square rounded flex items-center justify-center text-xs font-mono transition-all ${
                      node.status === 'idle'
                        ? 'bg-gray-200 dark:bg-gray-700 text-gray-500'
                        : 'text-white font-semibold'
                    }`}
                    style={{
                      backgroundColor: node.status === 'busy' && job ? job.color : undefined
                    }}
                    title={node.status === 'busy' ? `Node ${node.id} - ${job?.name}` : `Node ${node.id} - Idle`}
                  >
                    {node.id}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Job Queue */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              Job Queue
            </h4>

            <div className="space-y-2 max-h-64 overflow-y-auto">
              {jobs.map(job => (
                <div
                  key={job.id}
                  className={`p-3 rounded-lg border-l-4 ${
                    job.status === 'waiting' ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-500' :
                    job.status === 'running' ? 'bg-green-50 dark:bg-green-900/20 border-green-500' :
                    'bg-gray-50 dark:bg-gray-900/20 border-gray-500'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: job.color }}
                      />
                      <div>
                        <div className="font-semibold text-sm text-gray-900 dark:text-white">
                          {job.name}
                        </div>
                        <div className="text-xs text-gray-600 dark:text-gray-400">
                          {job.nodes} nodes • {job.runtime}s • {job.priority}
                        </div>
                      </div>
                    </div>

                    <div className="text-right">
                      <div className={`text-xs font-semibold ${
                        job.status === 'waiting' ? 'text-yellow-600' :
                        job.status === 'running' ? 'text-green-600' :
                        'text-gray-600'
                      }`}>
                        {job.status.toUpperCase()}
                      </div>
                      {job.startTime !== undefined && (
                        <div className="text-xs text-gray-500">
                          {job.status === 'running' ? `${currentTime - job.startTime}s elapsed` :
                           job.status === 'completed' ? `Finished at ${job.endTime}s` : ''}
                        </div>
                      )}
                    </div>
                  </div>

                  {job.status === 'running' && job.startTime && job.endTime && (
                    <div className="mt-2">
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                        <div
                          className="bg-green-500 h-1.5 rounded-full transition-all"
                          style={{
                            width: `${((currentTime - job.startTime) / (job.endTime - job.startTime)) * 100}%`
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
