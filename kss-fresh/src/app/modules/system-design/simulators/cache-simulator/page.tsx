'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import { ArrowLeft, Play, Pause, RotateCcw, Database, Clock, Zap, TrendingUp, HardDrive } from 'lucide-react'

interface CacheEntry {
  key: string
  value: any
  accessCount: number
  lastAccess: number
  insertTime: number
  size: number
}

interface Request {
  id: number
  key: string
  timestamp: number
  hitOrMiss: 'hit' | 'miss'
}

type CachePolicy = 'LRU' | 'LFU' | 'FIFO' | 'Random'

export default function CacheSimulator() {
  const [cacheSize] = useState(8) // 최대 캐시 항목 수
  const [cache, setCache] = useState<Map<string, CacheEntry>>(new Map())
  const [requests, setRequests] = useState<Request[]>([])
  const [policy, setPolicy] = useState<CachePolicy>('LRU')
  const [isRunning, setIsRunning] = useState(false)
  const [requestRate, setRequestRate] = useState(1) // requests per second
  
  // 통계
  const [totalRequests, setTotalRequests] = useState(0)
  const [cacheHits, setCacheHits] = useState(0)
  const [cacheMisses, setCacheMisses] = useState(0)
  const [evictions, setEvictions] = useState(0)
  
  const requestIdCounter = useRef(0)
  const animationFrameId = useRef<number>()
  const lastRequestTime = useRef(0)
  
  // 데이터 아이템 풀 (시뮬레이션용)
  const dataItems = [
    'user_profile_123', 'product_456', 'session_789', 'cart_012',
    'order_345', 'review_678', 'inventory_901', 'payment_234',
    'shipping_567', 'notification_890', 'analytics_123', 'recommendation_456',
    'search_789', 'category_012', 'promotion_345', 'wishlist_678'
  ]

  // 캐시 항목 제거 정책
  const evictItem = (currentPolicy: CachePolicy, cacheMap: Map<string, CacheEntry>): string | null => {
    if (cacheMap.size < cacheSize) return null
    
    let keyToEvict: string | null = null
    
    switch (currentPolicy) {
      case 'LRU': // Least Recently Used
        let oldestAccess = Date.now()
        cacheMap.forEach((entry, key) => {
          if (entry.lastAccess < oldestAccess) {
            oldestAccess = entry.lastAccess
            keyToEvict = key
          }
        })
        break
        
      case 'LFU': // Least Frequently Used
        let minAccess = Infinity
        cacheMap.forEach((entry, key) => {
          if (entry.accessCount < minAccess) {
            minAccess = entry.accessCount
            keyToEvict = key
          }
        })
        break
        
      case 'FIFO': // First In First Out
        let oldestInsert = Date.now()
        cacheMap.forEach((entry, key) => {
          if (entry.insertTime < oldestInsert) {
            oldestInsert = entry.insertTime
            keyToEvict = key
          }
        })
        break
        
      case 'Random':
        const keys = Array.from(cacheMap.keys())
        keyToEvict = keys[Math.floor(Math.random() * keys.length)]
        break
    }
    
    return keyToEvict
  }

  // 요청 처리
  const processRequest = () => {
    const randomKey = dataItems[Math.floor(Math.random() * dataItems.length)]
    const timestamp = Date.now()
    
    setCache(prevCache => {
      const newCache = new Map(prevCache)
      let hitOrMiss: 'hit' | 'miss' = 'miss'
      
      if (newCache.has(randomKey)) {
        // Cache Hit
        hitOrMiss = 'hit'
        const entry = newCache.get(randomKey)!
        entry.accessCount++
        entry.lastAccess = timestamp
        setCacheHits(prev => prev + 1)
      } else {
        // Cache Miss
        if (newCache.size >= cacheSize) {
          // 캐시가 꽉 찼으면 제거
          const keyToEvict = evictItem(policy, newCache)
          if (keyToEvict) {
            newCache.delete(keyToEvict)
            setEvictions(prev => prev + 1)
          }
        }
        
        // 새 항목 추가
        newCache.set(randomKey, {
          key: randomKey,
          value: `Data for ${randomKey}`,
          accessCount: 1,
          lastAccess: timestamp,
          insertTime: timestamp,
          size: Math.floor(Math.random() * 100) + 50 // 50-150 KB
        })
        setCacheMisses(prev => prev + 1)
      }
      
      // 요청 기록
      const newRequest: Request = {
        id: requestIdCounter.current++,
        key: randomKey,
        timestamp,
        hitOrMiss
      }
      
      setRequests(prev => [...prev.slice(-15), newRequest])
      setTotalRequests(prev => prev + 1)
      
      return newCache
    })
  }

  // 애니메이션 루프
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
  }, [isRunning, requestRate, policy])

  const reset = () => {
    setIsRunning(false)
    setCache(new Map())
    setRequests([])
    setTotalRequests(0)
    setCacheHits(0)
    setCacheMisses(0)
    setEvictions(0)
    requestIdCounter.current = 0
  }

  const hitRate = totalRequests > 0 ? ((cacheHits / totalRequests) * 100) : 0
  const hitRateDisplay = hitRate.toFixed(1)
  const avgCacheSize = cache.size
  const totalCacheMemory = Array.from(cache.values()).reduce((sum, entry) => sum + entry.size, 0)

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
            캐시 전략 시뮬레이터
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            LRU, LFU, FIFO, Random 캐시 정책을 비교하고 학습합니다
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
            <HardDrive className="w-5 h-5 text-gray-600 dark:text-gray-400" />
            <select
              value={policy}
              onChange={(e) => setPolicy(e.target.value as CachePolicy)}
              className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="LRU">LRU (Least Recently Used)</option>
              <option value="LFU">LFU (Least Frequently Used)</option>
              <option value="FIFO">FIFO (First In First Out)</option>
              <option value="Random">Random</option>
            </select>
          </div>
          
          <div className="flex items-center gap-3">
            <Zap className="w-5 h-5 text-gray-600 dark:text-gray-400" />
            <label className="text-sm text-gray-600 dark:text-gray-400">요청 속도:</label>
            <input
              type="range"
              min="0.5"
              max="5"
              step="0.5"
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
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-8">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">총 요청</span>
            <Database className="w-4 h-4 text-purple-500" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {totalRequests}
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">캐시 적중률</span>
            <TrendingUp className="w-4 h-4 text-green-500" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {hitRateDisplay}%
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">적중/실패</span>
            <Zap className="w-4 h-4 text-yellow-500" />
          </div>
          <div className="text-lg font-bold text-gray-900 dark:text-white">
            <span className="text-green-600 dark:text-green-400">{cacheHits}</span>
            {' / '}
            <span className="text-red-600 dark:text-red-400">{cacheMisses}</span>
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">제거된 항목</span>
            <Clock className="w-4 h-4 text-red-500" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {evictions}
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">캐시 사용량</span>
            <HardDrive className="w-4 h-4 text-blue-500" />
          </div>
          <div className="text-lg font-bold text-gray-900 dark:text-white">
            {avgCacheSize}/{cacheSize}
            <span className="text-xs text-gray-500 ml-1">({totalCacheMemory} KB)</span>
          </div>
        </div>
      </div>

      {/* Main Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Request History */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            요청 기록
          </h3>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {requests.slice().reverse().map((request) => (
              <div
                key={request.id}
                className={`p-2 rounded-lg text-sm flex items-center justify-between ${
                  request.hitOrMiss === 'hit' 
                    ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
                    : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
                }`}
              >
                <span className="font-mono text-xs">{request.key}</span>
                <span className="font-semibold uppercase">
                  {request.hitOrMiss}
                </span>
              </div>
            ))}
            {requests.length === 0 && (
              <p className="text-gray-500 dark:text-gray-400 text-center py-8">
                요청 기록이 없습니다
              </p>
            )}
          </div>
        </div>

        {/* Cache State */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            캐시 상태
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Array.from({ length: cacheSize }).map((_, index) => {
              const entries = Array.from(cache.entries())
              const [key, entry] = entries[index] || [null, null]
              const isOccupied = entry !== null
              
              return (
                <div
                  key={index}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    isOccupied
                      ? 'border-purple-500 bg-purple-50 dark:bg-purple-950/20'
                      : 'border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-900/50'
                  }`}
                >
                  {isOccupied ? (
                    <div>
                      <div className="font-mono text-xs text-gray-700 dark:text-gray-300 truncate mb-1">
                        {key}
                      </div>
                      <div className="grid grid-cols-2 gap-1 text-xs">
                        <div>
                          <span className="text-gray-500 dark:text-gray-400">액세스:</span>
                          <span className="ml-1 font-semibold text-gray-900 dark:text-white">
                            {entry.accessCount}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-500 dark:text-gray-400">크기:</span>
                          <span className="ml-1 font-semibold text-gray-900 dark:text-white">
                            {entry.size}KB
                          </span>
                        </div>
                      </div>
                      {policy === 'LRU' && (
                        <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                          최근: {new Date(entry.lastAccess).toLocaleTimeString()}
                        </div>
                      )}
                      {policy === 'FIFO' && (
                        <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                          추가: {new Date(entry.insertTime).toLocaleTimeString()}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-2">
                      <div className="text-gray-400 dark:text-gray-500 text-sm">
                        비어있음
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
          
          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-950/20 rounded-lg">
              <h4 className="font-semibold text-blue-900 dark:text-blue-300 mb-2">
                캐시 정책: {policy}
              </h4>
              <p className="text-sm text-blue-700 dark:text-blue-400">
                {policy === 'LRU' && '가장 오래 사용되지 않은 항목을 제거합니다.'}
                {policy === 'LFU' && '가장 적게 사용된 항목을 제거합니다.'}
                {policy === 'FIFO' && '가장 먼저 들어온 항목을 제거합니다.'}
                {policy === 'Random' && '무작위로 항목을 제거합니다.'}
              </p>
            </div>
            
            <div className="p-4 bg-green-50 dark:bg-green-950/20 rounded-lg">
              <h4 className="font-semibold text-green-900 dark:text-green-300 mb-2">
                성능 팁
              </h4>
              <p className="text-sm text-green-700 dark:text-green-400">
                {hitRate >= 80 && '훌륭한 캐시 적중률입니다!'}
                {hitRate >= 50 && hitRate < 80 && '적절한 캐시 성능입니다.'}
                {hitRate < 50 && '캐시 정책을 재검토해보세요.'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Policy Comparison */}
      <div className="mt-8 bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          캐시 정책 비교
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">LRU</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              Least Recently Used - 시간적 지역성 활용
            </p>
            <div className="text-xs">
              <span className="text-green-600 dark:text-green-400">✓ 장점:</span> 최근 데이터 우선
              <br />
              <span className="text-red-600 dark:text-red-400">✗ 단점:</span> 타임스탬프 관리 필요
            </div>
          </div>
          
          <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">LFU</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              Least Frequently Used - 빈도 기반 관리
            </p>
            <div className="text-xs">
              <span className="text-green-600 dark:text-green-400">✓ 장점:</span> 인기 데이터 보호
              <br />
              <span className="text-red-600 dark:text-red-400">✗ 단점:</span> 오래된 인기 데이터 문제
            </div>
          </div>
          
          <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">FIFO</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              First In First Out - 단순 큐 방식
            </p>
            <div className="text-xs">
              <span className="text-green-600 dark:text-green-400">✓ 장점:</span> 구현이 간단
              <br />
              <span className="text-red-600 dark:text-red-400">✗ 단점:</span> 사용 패턴 무시
            </div>
          </div>
          
          <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Random</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              무작위 선택 - 비결정적 방식
            </p>
            <div className="text-xs">
              <span className="text-green-600 dark:text-green-400">✓ 장점:</span> 오버헤드 없음
              <br />
              <span className="text-red-600 dark:text-red-400">✗ 단점:</span> 예측 불가능
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}