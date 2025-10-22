'use client'

import { useState } from 'react'
import { Play, Bug, CheckCircle, XCircle, AlertTriangle } from 'lucide-react'

type IssueType = 'deadlock' | 'race-condition' | 'load-imbalance' | 'memory-leak' | 'none'

interface Process {
  rank: number
  status: 'idle' | 'sending' | 'receiving' | 'computing' | 'blocked'
  message?: string
  issues: IssueType[]
}

export default function MPIDebugger() {
  const [processes, setProcesses] = useState<Process[]>(
    Array.from({ length: 8 }, (_, i) => ({
      rank: i,
      status: 'idle',
      issues: [],
    }))
  )

  const [selectedPattern, setSelectedPattern] = useState<'correct' | 'deadlock' | 'race'>('correct')
  const [detectedIssues, setDetectedIssues] = useState<IssueType[]>([])

  const runPattern = () => {
    switch (selectedPattern) {
      case 'correct':
        runCorrectPattern()
        break
      case 'deadlock':
        runDeadlockPattern()
        break
      case 'race':
        runRaceConditionPattern()
        break
    }
  }

  const runCorrectPattern = () => {
    const newProcesses = processes.map(p => ({ ...p, status: 'computing' as const, issues: [] }))
    setProcesses(newProcesses)
    setDetectedIssues([])

    setTimeout(() => {
      setProcesses(prev => prev.map(p => ({ ...p, status: 'idle' as const })))
    }, 1000)
  }

  const runDeadlockPattern = () => {
    const newProcesses = processes.map(p => ({
      ...p,
      status: p.rank % 2 === 0 ? 'sending' as const : 'receiving' as const,
      issues: ['deadlock'] as IssueType[],
      message: p.rank % 2 === 0 ? `Waiting to send to P${p.rank + 1}` : `Waiting to receive from P${p.rank - 1}`
    }))

    setProcesses(newProcesses)
    setDetectedIssues(['deadlock'])
  }

  const runRaceConditionPattern = () => {
    const newProcesses = processes.map(p => ({
      ...p,
      status: 'computing' as const,
      issues: ['race-condition'] as IssueType[],
      message: 'Accessing shared variable without synchronization'
    }))

    setProcesses(newProcesses)
    setDetectedIssues(['race-condition'])
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg p-6 border-l-4 border-yellow-500">
        <h3 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
          MPI 디버거
        </h3>
        <p className="text-gray-700 dark:text-gray-300">
          분산 프로그램의 일반적인 버그를 시뮬레이션하고 감지합니다
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              Test Pattern
            </h4>

            <div className="space-y-2">
              {[
                { id: 'correct', name: '✅ Correct Pattern', desc: 'Non-blocking communication' },
                { id: 'deadlock', name: '🔴 Deadlock', desc: 'Circular wait condition' },
                { id: 'race', name: '⚠️ Race Condition', desc: 'Unsynchronized access' },
              ].map(pattern => (
                <button
                  key={pattern.id}
                  onClick={() => setSelectedPattern(pattern.id as any)}
                  className={`w-full text-left px-4 py-3 rounded-lg border-2 transition ${
                    selectedPattern === pattern.id
                      ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <div className="font-semibold text-sm text-gray-900 dark:text-white">
                    {pattern.name}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    {pattern.desc}
                  </div>
                </button>
              ))}
            </div>

            <button
              onClick={runPattern}
              className="w-full mt-4 px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-white rounded-lg transition flex items-center justify-center gap-2"
            >
              <Play className="w-4 h-4" />
              Run Test
            </button>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
              <Bug className="w-5 h-5 text-red-500" />
              Detected Issues
            </h4>

            {detectedIssues.length === 0 ? (
              <div className="flex items-center gap-2 text-green-600">
                <CheckCircle className="w-5 h-5" />
                <span className="text-sm">No issues detected</span>
              </div>
            ) : (
              <div className="space-y-2">
                {detectedIssues.map((issue, idx) => (
                  <div key={idx} className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 rounded border-l-4 border-red-500">
                    <XCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <div className="font-semibold text-sm text-red-700 dark:text-red-400">
                        {issue.replace('-', ' ').toUpperCase()}
                      </div>
                      <div className="text-xs text-red-600 dark:text-red-500 mt-1">
                        {issue === 'deadlock' && 'Circular dependency detected in MPI_Send/Recv'}
                        {issue === 'race-condition' && 'Multiple processes accessing shared data'}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              Process States (8 MPI Ranks)
            </h4>

            <div className="grid grid-cols-2 gap-3">
              {processes.map(proc => (
                <div
                  key={proc.rank}
                  className={`p-4 rounded-lg border-2 transition ${
                    proc.issues.length > 0
                      ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                      : proc.status === 'computing'
                      ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                      : 'border-gray-300 dark:border-gray-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono font-bold text-gray-900 dark:text-white">
                      Rank {proc.rank}
                    </span>
                    <span
                      className={`text-xs px-2 py-1 rounded ${
                        proc.status === 'idle' ? 'bg-gray-200 dark:bg-gray-700' :
                        proc.status === 'computing' ? 'bg-green-200 dark:bg-green-800 text-green-900 dark:text-green-100' :
                        proc.status === 'blocked' ? 'bg-red-200 dark:bg-red-800 text-red-900 dark:text-red-100' :
                        'bg-yellow-200 dark:bg-yellow-800 text-yellow-900 dark:text-yellow-100'
                      }`}
                    >
                      {proc.status}
                    </span>
                  </div>

                  {proc.message && (
                    <div className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                      {proc.message}
                    </div>
                  )}

                  {proc.issues.length > 0 && (
                    <div className="mt-2 flex items-center gap-1 text-red-600 dark:text-red-400">
                      <AlertTriangle className="w-3 h-3" />
                      <span className="text-xs font-semibold">
                        {proc.issues[0].toUpperCase()}
                      </span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="mt-6 bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              Code Examples
            </h4>

            <div className="space-y-4">
              <div>
                <div className="text-sm font-semibold text-green-600 mb-2">✅ Correct: Non-blocking</div>
                <div className="bg-gray-900 rounded p-3 overflow-x-auto">
                  <pre className="text-xs text-gray-100">
                    <code>{`MPI_Isend(&data, 1, MPI_INT, dest, tag, comm, &request);
MPI_Irecv(&recv_data, 1, MPI_INT, source, tag, comm, &request);
MPI_Wait(&request, &status);  // No deadlock`}</code>
                  </pre>
                </div>
              </div>

              <div>
                <div className="text-sm font-semibold text-red-600 mb-2">❌ Deadlock: Circular Wait</div>
                <div className="bg-gray-900 rounded p-3 overflow-x-auto">
                  <pre className="text-xs text-gray-100">
                    <code>{`// Even ranks
MPI_Send(&data, 1, MPI_INT, rank+1, tag, comm);
MPI_Recv(&recv, 1, MPI_INT, rank+1, tag, comm, &status);

// Odd ranks
MPI_Send(&data, 1, MPI_INT, rank-1, tag, comm);  // Deadlock!
MPI_Recv(&recv, 1, MPI_INT, rank-1, tag, comm, &status);`}</code>
                  </pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
