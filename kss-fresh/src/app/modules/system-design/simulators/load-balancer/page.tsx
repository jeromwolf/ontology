'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import { ArrowLeft, Play, Pause, RotateCcw, Server, Users, Activity, Settings, Zap } from 'lucide-react'

interface Server {
  id: number
  load: number
  requestCount: number
  responseTime: number
  status: 'healthy' | 'degraded' | 'down'
}

interface Request {
  id: number
  startTime: number
  serverId?: number
  status: 'pending' | 'processing' | 'completed' | 'failed'
}

type Algorithm = 'round-robin' | 'least-connections' | 'weighted' | 'ip-hash' | 'random'

export default function LoadBalancerSimulator() {
  const [servers, setServers] = useState<Server[]>([
    { id: 1, load: 0, requestCount: 0, responseTime: 50, status: 'healthy' },
    { id: 2, load: 0, requestCount: 0, responseTime: 60, status: 'healthy' },
    { id: 3, load: 0, requestCount: 0, responseTime: 45, status: 'healthy' },
    { id: 4, load: 0, requestCount: 0, responseTime: 55, status: 'healthy' }
  ])
  
  const [requests, setRequests] = useState<Request[]>([])
  const [algorithm, setAlgorithm] = useState<Algorithm>('round-robin')
  const [isRunning, setIsRunning] = useState(false)
  const [requestRate, setRequestRate] = useState(2) // requests per second
  const [totalRequests, setTotalRequests] = useState(0)
  const [successRate, setSuccessRate] = useState(100)
  const [avgResponseTime, setAvgResponseTime] = useState(0)
  
  const roundRobinIndex = useRef(0)
  const requestIdCounter = useRef(0)
  const animationFrameId = useRef<number>()
  const lastRequestTime = useRef(0)

  // 서버 선택 알고리즘
  const selectServer = (algo: Algorithm): number => {
    const healthyServers = servers.filter(s => s.status !== 'down')
    if (healthyServers.length === 0) return -1

    switch (algo) {
      case 'round-robin':
        const selected = healthyServers[roundRobinIndex.current % healthyServers.length]
        roundRobinIndex.current++
        return selected.id

      case 'least-connections':
        const leastLoaded = healthyServers.reduce((min, server) => 
          server.load < min.load ? server : min
        )
        return leastLoaded.id

      case 'weighted':
        // 응답 시간이 빠른 서버에 더 많은 가중치
        const weights = healthyServers.map(s => 100 / s.responseTime)
        const totalWeight = weights.reduce((sum, w) => sum + w, 0)
        let random = Math.random() * totalWeight
        
        for (let i = 0; i < healthyServers.length; i++) {
          random -= weights[i]
          if (random <= 0) return healthyServers[i].id
        }
        return healthyServers[0].id

      case 'ip-hash':
        // 시뮬레이션을 위해 요청 ID를 IP처럼 사용
        const hash = requestIdCounter.current % healthyServers.length
        return healthyServers[hash].id

      case 'random':
        const randomIndex = Math.floor(Math.random() * healthyServers.length)
        return healthyServers[randomIndex].id

      default:
        return healthyServers[0].id
    }
  }

  // 요청 생성 및 처리
  const processRequests = (timestamp: number) => {
    if (!isRunning) return

    // 새 요청 생성
    const timeSinceLastRequest = timestamp - lastRequestTime.current
    if (timeSinceLastRequest >= 1000 / requestRate) {
      const serverId = selectServer(algorithm)
      
      if (serverId !== -1) {
        const newRequest: Request = {
          id: requestIdCounter.current++,
          startTime: Date.now(),
          serverId,
          status: 'processing'
        }
        
        setRequests(prev => [...prev.slice(-20), newRequest])
        setTotalRequests(prev => prev + 1)
        
        // 서버 부하 증가
        setServers(prev => prev.map(server => 
          server.id === serverId 
            ? { ...server, load: Math.min(100, server.load + 20), requestCount: server.requestCount + 1 }
            : server
        ))
        
        // 요청 완료 처리
        setTimeout(() => {
          setRequests(prev => prev.map(req => 
            req.id === newRequest.id ? { ...req, status: 'completed' } : req
          ))
          
          setServers(prev => prev.map(server => 
            server.id === serverId 
              ? { ...server, load: Math.max(0, server.load - 20) }
              : server
          ))
        }, servers.find(s => s.id === serverId)?.responseTime || 50)
      }
      
      lastRequestTime.current = timestamp
    }

    // 서버 상태 업데이트
    setServers(prev => prev.map(server => {
      let status: Server['status'] = 'healthy'
      if (server.load > 80) status = 'degraded'
      if (server.load >= 100) status = 'down'
      
      return { ...server, status }
    }))

    // 메트릭 업데이트
    const completed = requests.filter(r => r.status === 'completed').length
    const failed = requests.filter(r => r.status === 'failed').length
    const total = completed + failed
    
    if (total > 0) {
      setSuccessRate(Math.round((completed / total) * 100))
    }
    
    const avgTime = servers.reduce((sum, s) => sum + s.responseTime, 0) / servers.length
    setAvgResponseTime(Math.round(avgTime))

    animationFrameId.current = requestAnimationFrame(processRequests)
  }

  useEffect(() => {
    if (isRunning) {
      animationFrameId.current = requestAnimationFrame(processRequests)
    } else {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current)
      }
    }
    
    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current)
      }
    }
  }, [isRunning, algorithm, requestRate])

  const reset = () => {
    setIsRunning(false)
    setServers(servers.map(s => ({ ...s, load: 0, requestCount: 0, status: 'healthy' })))
    setRequests([])
    setTotalRequests(0)
    setSuccessRate(100)
    roundRobinIndex.current = 0
    requestIdCounter.current = 0
  }

  const toggleServerHealth = (serverId: number) => {
    setServers(prev => prev.map(server => 
      server.id === serverId 
        ? { ...server, status: server.status === 'healthy' ? 'down' : 'healthy', load: 0 }
        : server
    ))
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/system-design"
          className="inline-flex items-center text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 mb-4"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          System Design 모듈로 돌아가기
        </Link>
        
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            로드 밸런서 시뮬레이터
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            다양한 로드 밸런싱 알고리즘을 시각적으로 비교하고 학습합니다
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-8">
        <div className="flex flex-wrap items-center gap-4">
          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`px-6 py-3 rounded-lg font-semibold transition-colors flex items-center gap-2 ${
              isRunning 
                ? 'bg-red-500 hover:bg-red-600 text-white' 
                : 'bg-green-500 hover:bg-green-600 text-white'
            }`}
          >
            {isRunning ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
            {isRunning ? '정지' : '시작'}
          </button>
          
          <button
            onClick={reset}
            className="px-6 py-3 bg-gray-500 hover:bg-gray-600 text-white rounded-lg font-semibold transition-colors flex items-center gap-2"
          >
            <RotateCcw className="w-5 h-5" />
            리셋
          </button>
          
          <div className="flex items-center gap-3">
            <Settings className="w-5 h-5 text-gray-600 dark:text-gray-400" />
            <select
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value as Algorithm)}
              className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="round-robin">Round Robin</option>
              <option value="least-connections">Least Connections</option>
              <option value="weighted">Weighted Response Time</option>
              <option value="ip-hash">IP Hash</option>
              <option value="random">Random</option>
            </select>
          </div>
          
          <div className="flex items-center gap-3">
            <Zap className="w-5 h-5 text-gray-600 dark:text-gray-400" />
            <label className="text-sm text-gray-600 dark:text-gray-400">요청 속도:</label>
            <input
              type="range"
              min="1"
              max="10"
              value={requestRate}
              onChange={(e) => setRequestRate(Number(e.target.value))}
              className="w-32"
            />
            <span className="text-sm font-semibold text-gray-900 dark:text-white">
              {requestRate}/초
            </span>
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">총 요청</span>
            <Users className="w-4 h-4 text-purple-500" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {totalRequests}
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">성공률</span>
            <Activity className="w-4 h-4 text-green-500" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {successRate}%
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">평균 응답 시간</span>
            <Zap className="w-4 h-4 text-yellow-500" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {avgResponseTime}ms
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">활성 서버</span>
            <Server className="w-4 h-4 text-blue-500" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {servers.filter(s => s.status !== 'down').length}/{servers.length}
          </div>
        </div>
      </div>

      {/* Main Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Request Queue */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            요청 큐
          </h3>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {requests.slice(-10).reverse().map((request) => (
              <div
                key={request.id}
                className={`p-3 rounded-lg text-sm ${
                  request.status === 'completed' 
                    ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
                    : request.status === 'processing'
                    ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                <div className="flex justify-between items-center">
                  <span>Request #{request.id}</span>
                  {request.serverId && (
                    <span className="text-xs">→ Server {request.serverId}</span>
                  )}
                </div>
              </div>
            ))}
            {requests.length === 0 && (
              <p className="text-gray-500 dark:text-gray-400 text-center py-8">
                요청이 없습니다
              </p>
            )}
          </div>
        </div>

        {/* Servers */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            서버 상태
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {servers.map((server) => (
              <div
                key={server.id}
                onClick={() => toggleServerHealth(server.id)}
                className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                  server.status === 'healthy' 
                    ? 'border-green-500 bg-green-50 dark:bg-green-950/20'
                    : server.status === 'degraded'
                    ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-950/20'
                    : 'border-red-500 bg-red-50 dark:bg-red-950/20'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Server className={`w-6 h-6 ${
                      server.status === 'healthy' ? 'text-green-600 dark:text-green-400' :
                      server.status === 'degraded' ? 'text-yellow-600 dark:text-yellow-400' :
                      'text-red-600 dark:text-red-400'
                    }`} />
                    <span className="font-semibold text-gray-900 dark:text-white">
                      Server {server.id}
                    </span>
                  </div>
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    server.status === 'healthy' 
                      ? 'bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200'
                      : server.status === 'degraded'
                      ? 'bg-yellow-200 dark:bg-yellow-800 text-yellow-800 dark:text-yellow-200'
                      : 'bg-red-200 dark:bg-red-800 text-red-800 dark:text-red-200'
                  }`}>
                    {server.status}
                  </span>
                </div>
                
                <div className="space-y-2">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600 dark:text-gray-400">부하</span>
                      <span className="text-gray-900 dark:text-white font-medium">
                        {server.load}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all duration-300 ${
                          server.load > 80 ? 'bg-red-500' :
                          server.load > 50 ? 'bg-yellow-500' :
                          'bg-green-500'
                        }`}
                        style={{ width: `${server.load}%` }}
                      />
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">요청 수:</span>
                      <span className="ml-1 font-medium text-gray-900 dark:text-white">
                        {server.requestCount}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">응답:</span>
                      <span className="ml-1 font-medium text-gray-900 dark:text-white">
                        {server.responseTime}ms
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg">
            <p className="text-sm text-blue-700 dark:text-blue-300">
              💡 서버를 클릭하여 상태를 변경할 수 있습니다
            </p>
          </div>
        </div>
      </div>

      {/* Algorithm Description */}
      <div className="mt-8 bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
          알고리즘 설명
        </h3>
        <div className="prose dark:prose-invert max-w-none">
          {algorithm === 'round-robin' && (
            <p className="text-gray-700 dark:text-gray-300">
              <strong>Round Robin:</strong> 서버에 순차적으로 요청을 분배합니다. 
              가장 단순하고 공평한 방식이지만, 서버의 성능 차이를 고려하지 않습니다.
            </p>
          )}
          {algorithm === 'least-connections' && (
            <p className="text-gray-700 dark:text-gray-300">
              <strong>Least Connections:</strong> 현재 연결 수가 가장 적은 서버에 요청을 전달합니다. 
              서버 부하를 균등하게 분산시키는 데 효과적입니다.
            </p>
          )}
          {algorithm === 'weighted' && (
            <p className="text-gray-700 dark:text-gray-300">
              <strong>Weighted Response Time:</strong> 서버의 응답 시간을 기반으로 가중치를 부여합니다. 
              성능이 좋은 서버가 더 많은 요청을 처리합니다.
            </p>
          )}
          {algorithm === 'ip-hash' && (
            <p className="text-gray-700 dark:text-gray-300">
              <strong>IP Hash:</strong> 클라이언트 IP를 해싱하여 특정 서버에 매핑합니다. 
              같은 클라이언트는 항상 같은 서버로 연결되어 세션 지속성을 보장합니다.
            </p>
          )}
          {algorithm === 'random' && (
            <p className="text-gray-700 dark:text-gray-300">
              <strong>Random:</strong> 무작위로 서버를 선택합니다. 
              구현이 간단하지만 부하 분산이 불균등할 수 있습니다.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}