'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import { ArrowLeft, Play, Pause, RotateCcw, Droplet, Timer, TrendingUp, AlertTriangle, CheckCircle, XCircle } from 'lucide-react'

type Algorithm = 'token-bucket' | 'leaky-bucket' | 'fixed-window' | 'sliding-window'

interface Request {
  id: number
  timestamp: number
  status: 'allowed' | 'rejected'
  remainingTokens?: number
}

interface TokenBucket {
  tokens: number
  capacity: number
  refillRate: number
  lastRefill: number
}

interface LeakyBucket {
  queue: Request[]
  capacity: number
  leakRate: number
  lastLeak: number
}

interface FixedWindow {
  count: number
  windowStart: number
  windowSize: number
  limit: number
}

interface SlidingWindow {
  requests: number[]
  windowSize: number
  limit: number
}

export default function RateLimiterSimulator() {
  const [algorithm, setAlgorithm] = useState<Algorithm>('token-bucket')
  const [isRunning, setIsRunning] = useState(false)
  const [requestRate, setRequestRate] = useState(5) // requests per second
  const [burstSize, setBurstSize] = useState(10)
  const [limit, setLimit] = useState(10) // requests per window
  const [windowSize, setWindowSize] = useState(5000) // milliseconds
  
  // Algorithm states
  const [tokenBucket, setTokenBucket] = useState<TokenBucket>({
    tokens: burstSize,
    capacity: burstSize,
    refillRate: 2,
    lastRefill: Date.now()
  })
  
  const [leakyBucket, setLeakyBucket] = useState<LeakyBucket>({
    queue: [],
    capacity: burstSize,
    leakRate: 2,
    lastLeak: Date.now()
  })
  
  const [fixedWindow, setFixedWindow] = useState<FixedWindow>({
    count: 0,
    windowStart: Date.now(),
    windowSize: windowSize,
    limit: limit
  })
  
  const [slidingWindow, setSlidingWindow] = useState<SlidingWindow>({
    requests: [],
    windowSize: windowSize,
    limit: limit
  })
  
  // Statistics
  const [requests, setRequests] = useState<Request[]>([])
  const [totalRequests, setTotalRequests] = useState(0)
  const [allowedRequests, setAllowedRequests] = useState(0)
  const [rejectedRequests, setRejectedRequests] = useState(0)
  
  const requestIdCounter = useRef(0)
  const animationFrameId = useRef<number>()
  const lastRequestTime = useRef(0)

  // Token Bucket Algorithm
  const processTokenBucket = (): boolean => {
    const now = Date.now()
    const timePassed = (now - tokenBucket.lastRefill) / 1000
    const tokensToAdd = Math.floor(timePassed * tokenBucket.refillRate)
    
    if (tokensToAdd > 0) {
      setTokenBucket(prev => ({
        ...prev,
        tokens: Math.min(prev.capacity, prev.tokens + tokensToAdd),
        lastRefill: now
      }))
    }
    
    if (tokenBucket.tokens >= 1) {
      setTokenBucket(prev => ({
        ...prev,
        tokens: prev.tokens - 1
      }))
      return true
    }
    
    return false
  }

  // Leaky Bucket Algorithm
  const processLeakyBucket = (request: Request): boolean => {
    const now = Date.now()
    const timePassed = (now - leakyBucket.lastLeak) / 1000
    const requestsToLeak = Math.floor(timePassed * leakyBucket.leakRate)
    
    if (requestsToLeak > 0) {
      setLeakyBucket(prev => ({
        ...prev,
        queue: prev.queue.slice(requestsToLeak),
        lastLeak: now
      }))
    }
    
    if (leakyBucket.queue.length < leakyBucket.capacity) {
      setLeakyBucket(prev => ({
        ...prev,
        queue: [...prev.queue, request]
      }))
      return true
    }
    
    return false
  }

  // Fixed Window Algorithm
  const processFixedWindow = (): boolean => {
    const now = Date.now()
    
    // Check if we need to reset the window
    if (now - fixedWindow.windowStart >= fixedWindow.windowSize) {
      setFixedWindow({
        count: 0,
        windowStart: now,
        windowSize: windowSize,
        limit: limit
      })
    }
    
    if (fixedWindow.count < fixedWindow.limit) {
      setFixedWindow(prev => ({
        ...prev,
        count: prev.count + 1
      }))
      return true
    }
    
    return false
  }

  // Sliding Window Algorithm
  const processSlidingWindow = (): boolean => {
    const now = Date.now()
    const windowStart = now - slidingWindow.windowSize
    
    // Remove old requests outside the window
    const validRequests = slidingWindow.requests.filter(time => time > windowStart)
    
    setSlidingWindow(prev => ({
      ...prev,
      requests: validRequests
    }))
    
    if (validRequests.length < slidingWindow.limit) {
      setSlidingWindow(prev => ({
        ...prev,
        requests: [...prev.requests, now]
      }))
      return true
    }
    
    return false
  }

  // Process request based on algorithm
  const processRequest = () => {
    const request: Request = {
      id: requestIdCounter.current++,
      timestamp: Date.now(),
      status: 'rejected'
    }
    
    let allowed = false
    
    switch (algorithm) {
      case 'token-bucket':
        allowed = processTokenBucket()
        request.remainingTokens = tokenBucket.tokens
        break
      case 'leaky-bucket':
        allowed = processLeakyBucket(request)
        break
      case 'fixed-window':
        allowed = processFixedWindow()
        break
      case 'sliding-window':
        allowed = processSlidingWindow()
        break
    }
    
    request.status = allowed ? 'allowed' : 'rejected'
    
    setRequests(prev => [...prev.slice(-20), request])
    setTotalRequests(prev => prev + 1)
    
    if (allowed) {
      setAllowedRequests(prev => prev + 1)
    } else {
      setRejectedRequests(prev => prev + 1)
    }
  }

  // Animation loop
  const animate = (timestamp: number) => {
    if (!isRunning) return
    
    const timeSinceLastRequest = timestamp - lastRequestTime.current
    if (timeSinceLastRequest >= 1000 / requestRate) {
      processRequest()
      lastRequestTime.current = timestamp
    }
    
    animationFrameId.current = requestAnimationFrame(animate)
  }

  useEffect(() => {
    if (isRunning) {
      animationFrameId.current = requestAnimationFrame(animate)
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
  }, [isRunning, requestRate, algorithm])

  const reset = () => {
    setIsRunning(false)
    setRequests([])
    setTotalRequests(0)
    setAllowedRequests(0)
    setRejectedRequests(0)
    requestIdCounter.current = 0
    
    // Reset algorithm states
    setTokenBucket({
      tokens: burstSize,
      capacity: burstSize,
      refillRate: 2,
      lastRefill: Date.now()
    })
    setLeakyBucket({
      queue: [],
      capacity: burstSize,
      leakRate: 2,
      lastLeak: Date.now()
    })
    setFixedWindow({
      count: 0,
      windowStart: Date.now(),
      windowSize: windowSize,
      limit: limit
    })
    setSlidingWindow({
      requests: [],
      windowSize: windowSize,
      limit: limit
    })
  }

  const acceptanceRate = totalRequests > 0 ? ((allowedRequests / totalRequests) * 100).toFixed(1) : '0'

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
            Rate Limiter 구현
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Token Bucket, Leaky Bucket, Fixed/Sliding Window 알고리즘을 체험합니다
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              알고리즘
            </label>
            <select
              value={algorithm}
              onChange={(e) => {
                setAlgorithm(e.target.value as Algorithm)
                reset()
              }}
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="token-bucket">Token Bucket</option>
              <option value="leaky-bucket">Leaky Bucket</option>
              <option value="fixed-window">Fixed Window</option>
              <option value="sliding-window">Sliding Window</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              요청 속도: {requestRate}/초
            </label>
            <input
              type="range"
              min="1"
              max="20"
              value={requestRate}
              onChange={(e) => setRequestRate(Number(e.target.value))}
              className="w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              {algorithm.includes('bucket') ? `버스트 크기: ${burstSize}` : `윈도우 한계: ${limit}`}
            </label>
            <input
              type="range"
              min="5"
              max="20"
              value={algorithm.includes('bucket') ? burstSize : limit}
              onChange={(e) => {
                const value = Number(e.target.value)
                if (algorithm.includes('bucket')) {
                  setBurstSize(value)
                  setTokenBucket(prev => ({ ...prev, capacity: value, tokens: Math.min(prev.tokens, value) }))
                  setLeakyBucket(prev => ({ ...prev, capacity: value }))
                } else {
                  setLimit(value)
                  setFixedWindow(prev => ({ ...prev, limit: value }))
                  setSlidingWindow(prev => ({ ...prev, limit: value }))
                }
              }}
              className="w-full"
            />
          </div>
          
          <div className="flex items-end gap-2">
            <button
              onClick={() => setIsRunning(!isRunning)}
              className={`flex-1 px-4 py-2 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
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
              className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg font-semibold transition-colors"
            >
              <RotateCcw className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">총 요청</span>
            <TrendingUp className="w-4 h-4 text-purple-500" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {totalRequests}
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">허용됨</span>
            <CheckCircle className="w-4 h-4 text-green-500" />
          </div>
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">
            {allowedRequests}
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">거부됨</span>
            <XCircle className="w-4 h-4 text-red-500" />
          </div>
          <div className="text-2xl font-bold text-red-600 dark:text-red-400">
            {rejectedRequests}
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">수락률</span>
            <Timer className="w-4 h-4 text-blue-500" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {acceptanceRate}%
          </div>
        </div>
      </div>

      {/* Main Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Request Stream */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            요청 스트림
          </h3>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {requests.slice().reverse().map((request) => (
              <div
                key={request.id}
                className={`p-2 rounded-lg text-sm flex items-center justify-between ${
                  request.status === 'allowed' 
                    ? 'bg-green-100 dark:bg-green-900/30'
                    : 'bg-red-100 dark:bg-red-900/30'
                }`}
              >
                <span className="text-gray-700 dark:text-gray-300">
                  Request #{request.id}
                </span>
                <span className={`font-semibold ${
                  request.status === 'allowed'
                    ? 'text-green-700 dark:text-green-300'
                    : 'text-red-700 dark:text-red-300'
                }`}>
                  {request.status === 'allowed' ? '✓ 허용' : '✗ 거부'}
                </span>
              </div>
            ))}
            {requests.length === 0 && (
              <p className="text-gray-500 dark:text-gray-400 text-center py-8">
                요청이 없습니다
              </p>
            )}
          </div>
        </div>

        {/* Algorithm State */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            알고리즘 상태
          </h3>
          
          {algorithm === 'token-bucket' && (
            <div>
              <div className="mb-6">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-600 dark:text-gray-400">토큰 버킷</span>
                  <span className="text-gray-900 dark:text-white font-semibold">
                    {Math.floor(tokenBucket.tokens)} / {tokenBucket.capacity} 토큰
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-8 relative overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-blue-400 to-blue-600 rounded-full transition-all duration-300 flex items-center justify-center"
                    style={{ width: `${(tokenBucket.tokens / tokenBucket.capacity) * 100}%` }}
                  >
                    <Droplet className="w-5 h-5 text-white" />
                  </div>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  재충전 속도: {tokenBucket.refillRate} 토큰/초
                </p>
              </div>
              
              <div className="p-4 bg-blue-50 dark:bg-blue-950/20 rounded-lg">
                <h4 className="font-semibold text-blue-900 dark:text-blue-300 mb-2">
                  Token Bucket 알고리즘
                </h4>
                <p className="text-sm text-blue-700 dark:text-blue-400">
                  토큰이 일정 속도로 버킷에 추가됩니다. 요청이 오면 토큰을 소비하고, 
                  토큰이 없으면 요청을 거부합니다. 버스트 트래픽을 처리할 수 있습니다.
                </p>
              </div>
            </div>
          )}
          
          {algorithm === 'leaky-bucket' && (
            <div>
              <div className="mb-6">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-600 dark:text-gray-400">누수 버킷 큐</span>
                  <span className="text-gray-900 dark:text-white font-semibold">
                    {leakyBucket.queue.length} / {leakyBucket.capacity} 요청
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-8 relative">
                  <div
                    className="h-full bg-gradient-to-r from-green-400 to-green-600 rounded-full transition-all duration-300"
                    style={{ width: `${(leakyBucket.queue.length / leakyBucket.capacity) * 100}%` }}
                  />
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  누수 속도: {leakyBucket.leakRate} 요청/초
                </p>
              </div>
              
              <div className="p-4 bg-green-50 dark:bg-green-950/20 rounded-lg">
                <h4 className="font-semibold text-green-900 dark:text-green-300 mb-2">
                  Leaky Bucket 알고리즘
                </h4>
                <p className="text-sm text-green-700 dark:text-green-400">
                  요청이 버킷(큐)에 들어가고 일정 속도로 처리됩니다. 
                  버킷이 가득 차면 새 요청을 거부합니다. 일정한 출력 속도를 보장합니다.
                </p>
              </div>
            </div>
          )}
          
          {algorithm === 'fixed-window' && (
            <div>
              <div className="mb-6">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-600 dark:text-gray-400">현재 윈도우</span>
                  <span className="text-gray-900 dark:text-white font-semibold">
                    {fixedWindow.count} / {fixedWindow.limit} 요청
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-8 relative">
                  <div
                    className="h-full bg-gradient-to-r from-yellow-400 to-yellow-600 rounded-full transition-all duration-300"
                    style={{ width: `${(fixedWindow.count / fixedWindow.limit) * 100}%` }}
                  />
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  윈도우 크기: {fixedWindow.windowSize / 1000}초 | 
                  남은 시간: {Math.max(0, (fixedWindow.windowSize - (Date.now() - fixedWindow.windowStart)) / 1000).toFixed(1)}초
                </p>
              </div>
              
              <div className="p-4 bg-yellow-50 dark:bg-yellow-950/20 rounded-lg">
                <h4 className="font-semibold text-yellow-900 dark:text-yellow-300 mb-2">
                  Fixed Window 알고리즘
                </h4>
                <p className="text-sm text-yellow-700 dark:text-yellow-400">
                  고정된 시간 윈도우 내에서 요청 수를 제한합니다. 
                  윈도우가 끝나면 카운터가 리셋됩니다. 구현이 간단하지만 경계 문제가 있을 수 있습니다.
                </p>
              </div>
            </div>
          )}
          
          {algorithm === 'sliding-window' && (
            <div>
              <div className="mb-6">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-600 dark:text-gray-400">슬라이딩 윈도우</span>
                  <span className="text-gray-900 dark:text-white font-semibold">
                    {slidingWindow.requests.length} / {slidingWindow.limit} 요청
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-8 relative">
                  <div
                    className="h-full bg-gradient-to-r from-purple-400 to-purple-600 rounded-full transition-all duration-300"
                    style={{ width: `${(slidingWindow.requests.length / slidingWindow.limit) * 100}%` }}
                  />
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  윈도우 크기: {slidingWindow.windowSize / 1000}초 (연속적으로 이동)
                </p>
              </div>
              
              <div className="p-4 bg-purple-50 dark:bg-purple-950/20 rounded-lg">
                <h4 className="font-semibold text-purple-900 dark:text-purple-300 mb-2">
                  Sliding Window 알고리즘
                </h4>
                <p className="text-sm text-purple-700 dark:text-purple-400">
                  이동하는 시간 윈도우를 사용하여 요청을 추적합니다. 
                  Fixed Window의 경계 문제를 해결하지만 더 많은 메모리를 사용합니다.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Algorithm Comparison */}
      <div className="mt-8 bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          알고리즘 비교
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Droplet className="w-5 h-5 text-blue-500" />
              <h4 className="font-semibold text-gray-900 dark:text-white">Token Bucket</h4>
            </div>
            <div className="text-xs space-y-1">
              <div className="text-green-600 dark:text-green-400">✓ 버스트 허용</div>
              <div className="text-green-600 dark:text-green-400">✓ 부드러운 처리</div>
              <div className="text-red-600 dark:text-red-400">✗ 메모리 사용</div>
              <div className="text-gray-600 dark:text-gray-400">용도: API Gateway</div>
            </div>
          </div>
          
          <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Timer className="w-5 h-5 text-green-500" />
              <h4 className="font-semibold text-gray-900 dark:text-white">Leaky Bucket</h4>
            </div>
            <div className="text-xs space-y-1">
              <div className="text-green-600 dark:text-green-400">✓ 일정한 출력</div>
              <div className="text-green-600 dark:text-green-400">✓ 트래픽 평활화</div>
              <div className="text-red-600 dark:text-red-400">✗ 버스트 불가</div>
              <div className="text-gray-600 dark:text-gray-400">용도: 네트워크 대역폭</div>
            </div>
          </div>
          
          <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="w-5 h-5 text-yellow-500" />
              <h4 className="font-semibold text-gray-900 dark:text-white">Fixed Window</h4>
            </div>
            <div className="text-xs space-y-1">
              <div className="text-green-600 dark:text-green-400">✓ 구현 간단</div>
              <div className="text-green-600 dark:text-green-400">✓ 메모리 효율</div>
              <div className="text-red-600 dark:text-red-400">✗ 경계 문제</div>
              <div className="text-gray-600 dark:text-gray-400">용도: 간단한 제한</div>
            </div>
          </div>
          
          <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-5 h-5 text-purple-500" />
              <h4 className="font-semibold text-gray-900 dark:text-white">Sliding Window</h4>
            </div>
            <div className="text-xs space-y-1">
              <div className="text-green-600 dark:text-green-400">✓ 정확한 제한</div>
              <div className="text-green-600 dark:text-green-400">✓ 부드러운 처리</div>
              <div className="text-red-600 dark:text-red-400">✗ 복잡한 구현</div>
              <div className="text-gray-600 dark:text-gray-400">용도: 정밀한 제어</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}