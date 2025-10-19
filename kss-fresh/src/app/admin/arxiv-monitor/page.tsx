'use client'

/**
 * ArXiv Monitor - Admin Dashboard
 * /admin/arxiv-monitor
 */

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  PlayCircle,
  Database,
  FileText,
  CheckCircle,
  XCircle,
  Clock,
  RefreshCw,
  Loader2,
  AlertCircle,
  Download,
  Trash2
} from 'lucide-react'

interface StatusData {
  summary: {
    total: number
    crawled: number
    summarized: number
    mdxGenerated: number
    published: number
    failed: number
    estimatedCost: string
  }
  recentPapers: Array<{
    id: string
    arxivId: string
    title: string
    status: string
    createdAt: string
    publishedDate: string
  }>
  recentLogs: Array<{
    id: string
    stage: string
    status: string
    message: string | null
    createdAt: string
    paper: {
      arxivId: string
      title: string
    }
  }>
}

interface LogEntry {
  level: string
  message: string
  timestamp: string
}

export default function ArxivMonitorAdminPage() {
  const [status, setStatus] = useState<StatusData | null>(null)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [selectedAction, setSelectedAction] = useState<string>('')

  // 상태 조회
  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/arxiv-monitor/status')
      const data = await res.json()
      if (data.success) {
        setStatus(data.data)
      }
    } catch (error) {
      console.error('Failed to fetch status:', error)
    }
  }

  // 로그 조회
  const fetchLogs = async () => {
    try {
      const res = await fetch('/api/arxiv-monitor/logs?lines=50')
      const data = await res.json()
      if (data.success) {
        setLogs(data.data.logs)
      }
    } catch (error) {
      console.error('Failed to fetch logs:', error)
    }
  }

  // 파이프라인 실행
  const runPipeline = async (action: string) => {
    setIsRunning(true)
    setSelectedAction(action)

    try {
      const res = await fetch('/api/arxiv-monitor/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      })

      const data = await res.json()
      if (data.success) {
        alert(`${action} 파이프라인이 시작되었습니다!`)

        // 5초 후 상태 갱신
        setTimeout(() => {
          fetchStatus()
          fetchLogs()
        }, 5000)
      } else {
        alert(`실행 실패: ${data.error}`)
      }
    } catch (error) {
      console.error('Failed to run pipeline:', error)
      alert('실행 중 오류가 발생했습니다.')
    } finally {
      setIsRunning(false)
      setSelectedAction('')
    }
  }

  // 초기 로드 및 자동 갱신
  useEffect(() => {
    fetchStatus()
    fetchLogs()

    const interval = setInterval(() => {
      fetchStatus()
      fetchLogs()
    }, 10000) // 10초마다 갱신

    return () => clearInterval(interval)
  }, [])

  // 상태 색상
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'CRAWLED': return 'bg-blue-500'
      case 'SUMMARIZED': return 'bg-yellow-500'
      case 'MDX_GENERATED': return 'bg-green-500'
      case 'PUBLISHED': return 'bg-purple-500'
      case 'FAILED': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  // 로그 레벨 색상
  const getLogLevelColor = (level: string) => {
    switch (level) {
      case 'error': return 'text-red-500'
      case 'warn': return 'text-yellow-500'
      case 'info': return 'text-blue-500'
      case 'debug': return 'text-gray-500'
      default: return 'text-gray-700 dark:text-gray-300'
    }
  }

  if (!status) {
    return (
      <div className="container mx-auto p-8 flex items-center justify-center min-h-screen">
        <Loader2 className="w-8 h-8 animate-spin" />
      </div>
    )
  }

  return (
    <div className="container mx-auto p-8 space-y-6">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">ArXiv Monitor 관리자</h1>
          <p className="text-gray-600 dark:text-gray-400">
            자동 논문 수집 및 요약 시스템
          </p>
        </div>
        <Button
          variant="outline"
          onClick={() => {
            fetchStatus()
            fetchLogs()
          }}
        >
          <RefreshCw className="w-4 h-4 mr-2" />
          새로고침
        </Button>
      </div>

      {/* 통계 대시보드 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
              전체 논문
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{status.summary.total}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
              수집 완료
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-blue-600">
              {status.summary.crawled}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
              MDX 생성
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-green-600">
              {status.summary.mdxGenerated}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
              예상 비용
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-purple-600">
              ${status.summary.estimatedCost}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 파이프라인 실행 */}
      <Card>
        <CardHeader>
          <CardTitle>파이프라인 실행</CardTitle>
          <CardDescription>
            ArXiv Monitor 파이프라인을 수동으로 실행합니다
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 md:grid-cols-6 gap-2">
            <Button
              onClick={() => runPipeline('full')}
              disabled={isRunning}
              className="w-full"
            >
              {isRunning && selectedAction === 'full' ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <PlayCircle className="w-4 h-4 mr-2" />
              )}
              전체 실행
            </Button>

            <Button
              variant="outline"
              onClick={() => runPipeline('crawler')}
              disabled={isRunning}
            >
              {isRunning && selectedAction === 'crawler' ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Database className="w-4 h-4 mr-2" />
              )}
              크롤러
            </Button>

            <Button
              variant="outline"
              onClick={() => runPipeline('summarizer')}
              disabled={isRunning}
            >
              {isRunning && selectedAction === 'summarizer' ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <FileText className="w-4 h-4 mr-2" />
              )}
              요약
            </Button>

            <Button
              variant="outline"
              onClick={() => runPipeline('generator')}
              disabled={isRunning}
            >
              {isRunning && selectedAction === 'generator' ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Download className="w-4 h-4 mr-2" />
              )}
              MDX 생성
            </Button>

            <Button
              variant="outline"
              onClick={() => runPipeline('notifier')}
              disabled={isRunning}
            >
              {isRunning && selectedAction === 'notifier' ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <CheckCircle className="w-4 h-4 mr-2" />
              )}
              알림
            </Button>

            <Button
              variant="destructive"
              onClick={() => {
                if (confirm('정말 모든 데이터를 삭제하시겠습니까?')) {
                  runPipeline('reset')
                }
              }}
              disabled={isRunning}
            >
              {isRunning && selectedAction === 'reset' ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Trash2 className="w-4 h-4 mr-2" />
              )}
              초기화
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* 최근 논문 */}
      <Card>
        <CardHeader>
          <CardTitle>최근 논문 (5개)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {status.recentPapers.map((paper) => (
              <div
                key={paper.id}
                className="flex items-start justify-between p-3 border rounded-lg"
              >
                <div className="flex-1">
                  <div className="font-mono text-sm text-blue-600 dark:text-blue-400">
                    {paper.arxivId}
                  </div>
                  <div className="font-medium line-clamp-1">{paper.title}</div>
                  <div className="text-xs text-gray-500 mt-1">
                    {new Date(paper.publishedDate).toLocaleDateString('ko-KR')}
                  </div>
                </div>
                <Badge className={getStatusColor(paper.status)}>
                  {paper.status}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* 실시간 로그 */}
      <Card>
        <CardHeader>
          <CardTitle>실시간 로그 (최근 50개)</CardTitle>
          <CardDescription>자동으로 10초마다 갱신됩니다</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-1 max-h-96 overflow-y-auto font-mono text-sm">
            {logs.length === 0 ? (
              <div className="text-center text-gray-500 py-8">
                로그가 없습니다
              </div>
            ) : (
              logs.map((log, index) => (
                <div key={index} className="flex items-start gap-2 py-1">
                  <span className="text-xs text-gray-400 whitespace-nowrap">
                    {new Date(log.timestamp).toLocaleTimeString('ko-KR')}
                  </span>
                  <span className={`text-xs font-semibold uppercase ${getLogLevelColor(log.level)}`}>
                    [{log.level}]
                  </span>
                  <span className="flex-1 text-xs">{log.message}</span>
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>

      {/* 처리 로그 */}
      <Card>
        <CardHeader>
          <CardTitle>처리 로그 (최근 10개)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {status.recentLogs.map((log) => (
              <div
                key={log.id}
                className="flex items-center justify-between p-2 border rounded text-sm"
              >
                <div className="flex items-center gap-3">
                  {log.status === 'success' ? (
                    <CheckCircle className="w-4 h-4 text-green-500" />
                  ) : log.status === 'failed' ? (
                    <XCircle className="w-4 h-4 text-red-500" />
                  ) : (
                    <Clock className="w-4 h-4 text-yellow-500" />
                  )}
                  <div>
                    <div className="font-medium">{log.stage}</div>
                    <div className="text-xs text-gray-500">
                      {log.paper.arxivId} - {log.paper.title.slice(0, 50)}...
                    </div>
                  </div>
                </div>
                <div className="text-xs text-gray-400">
                  {new Date(log.createdAt).toLocaleString('ko-KR')}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
