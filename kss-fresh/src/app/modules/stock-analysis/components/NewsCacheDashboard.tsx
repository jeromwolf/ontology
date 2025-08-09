'use client'

import { useState, useEffect } from 'react'
import { Clock, Database, TrendingUp, DollarSign, Activity, RefreshCw, CheckCircle, AlertCircle } from 'lucide-react'

interface CacheStats {
  timestamp: string
  market: {
    status: string
    nextOpen: string
    currentUpdateFreq: string
    schedule: string
  }
  cache: {
    totalCached: number
    memoryCacheSize: number
    hitRate: string
  }
  api: {
    last24hCalls: number
    estimatedMonthlyCost: string
    remaining: number
  }
  performance: {
    avgResponseTime: string
    cacheHitRatio: string
    apiCallsSaved: number
  }
}

export default function NewsCacheDashboard() {
  const [stats, setStats] = useState<CacheStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [autoRefresh, setAutoRefresh] = useState(true)

  const fetchStats = async () => {
    try {
      const response = await fetch('/api/news/stats')
      const data = await response.json()
      if (data.success) {
        setStats(data)
      }
    } catch (error) {
      console.error('통계 조회 실패:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStats()
    
    if (autoRefresh) {
      const interval = setInterval(fetchStats, 30000) // 30초마다 갱신
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  const getMarketStatusColor = (status: string) => {
    switch (status) {
      case '장중': return 'text-green-600 bg-green-100 dark:bg-green-900/30'
      case '장시작준비':
      case '장마감정리': return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30'
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30'
    }
  }

  const getApiUsageColor = (remaining: number) => {
    if (remaining > 700) return 'text-green-600'
    if (remaining > 300) return 'text-yellow-600'
    return 'text-red-600'
  }

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="w-8 h-8 animate-spin text-indigo-600" />
        </div>
      </div>
    )
  }

  if (!stats) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <div className="text-center text-gray-600 dark:text-gray-400">
          통계를 불러올 수 없습니다
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
      {/* 헤더 */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <Database className="w-6 h-6 text-indigo-600" />
          뉴스 캐시 대시보드
        </h3>
        <div className="flex items-center gap-4">
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors
              ${autoRefresh 
                ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' 
                : 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400'}`}
          >
            {autoRefresh ? '자동 갱신 ON' : '자동 갱신 OFF'}
          </button>
          <button
            onClick={fetchStats}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* 시장 상태 */}
      <div className="mb-6 p-4 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <Clock className="w-5 h-5 text-indigo-600" />
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
                현재 시장 상태
              </span>
            </div>
            <div className="flex items-center gap-3">
              <span className={`px-3 py-1 rounded-full text-sm font-bold ${getMarketStatusColor(stats.market.status)}`}>
                {stats.market.status}
              </span>
              <span className="text-gray-700 dark:text-gray-300">
                업데이트 주기: <strong>{stats.market.schedule}</strong>
              </span>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-indigo-600 dark:text-indigo-400">
              {stats.market.currentUpdateFreq}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400">
              현재 업데이트 주기
            </div>
          </div>
        </div>
      </div>

      {/* 통계 그리드 */}
      <div className="grid md:grid-cols-4 gap-4 mb-6">
        {/* 캐시 상태 */}
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <Database className="w-5 h-5 text-blue-600" />
            <span className="text-xs text-gray-500">캐시</span>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {stats.cache.totalCached}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400">
            저장된 쿼리
          </div>
          <div className="mt-2 text-xs">
            <span className="text-green-600 dark:text-green-400 font-medium">
              {stats.cache.hitRate} 적중률
            </span>
          </div>
        </div>

        {/* API 사용량 */}
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <Activity className="w-5 h-5 text-purple-600" />
            <span className="text-xs text-gray-500">API</span>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {stats.api.last24hCalls}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400">
            24시간 호출
          </div>
          <div className="mt-2 text-xs">
            <span className={`font-medium ${getApiUsageColor(stats.api.remaining)}`}>
              {stats.api.remaining} 남음
            </span>
          </div>
        </div>

        {/* 비용 */}
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <DollarSign className="w-5 h-5 text-green-600" />
            <span className="text-xs text-gray-500">비용</span>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {stats.api.estimatedMonthlyCost}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400">
            월 예상 비용
          </div>
          <div className="mt-2 text-xs">
            <span className="text-green-600 dark:text-green-400 font-medium">
              80% 절감
            </span>
          </div>
        </div>

        {/* 성능 */}
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <TrendingUp className="w-5 h-5 text-orange-600" />
            <span className="text-xs text-gray-500">성능</span>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {stats.performance.avgResponseTime}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400">
            평균 응답시간
          </div>
          <div className="mt-2 text-xs">
            <span className="text-blue-600 dark:text-blue-400 font-medium">
              {stats.performance.apiCallsSaved} 절약
            </span>
          </div>
        </div>
      </div>

      {/* 업데이트 스케줄 */}
      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
          <Clock className="w-4 h-4" />
          시간대별 업데이트 스케줄
        </h4>
        <div className="grid md:grid-cols-3 gap-3 text-sm">
          <div className="flex items-center justify-between p-2 bg-white dark:bg-gray-800 rounded">
            <span className="text-gray-600 dark:text-gray-400">장 시작 (08:30-09:30)</span>
            <span className="font-medium text-yellow-600">5분마다</span>
          </div>
          <div className="flex items-center justify-between p-2 bg-white dark:bg-gray-800 rounded">
            <span className="text-gray-600 dark:text-gray-400">장중 (09:30-15:30)</span>
            <span className="font-medium text-green-600">15분마다</span>
          </div>
          <div className="flex items-center justify-between p-2 bg-white dark:bg-gray-800 rounded">
            <span className="text-gray-600 dark:text-gray-400">장 마감 (15:30-16:00)</span>
            <span className="font-medium text-yellow-600">5분마다</span>
          </div>
          <div className="flex items-center justify-between p-2 bg-white dark:bg-gray-800 rounded">
            <span className="text-gray-600 dark:text-gray-400">장외시간</span>
            <span className="font-medium text-gray-600">60분마다</span>
          </div>
          <div className="flex items-center justify-between p-2 bg-white dark:bg-gray-800 rounded">
            <span className="text-gray-600 dark:text-gray-400">주말</span>
            <span className="font-medium text-gray-600">3시간마다</span>
          </div>
          <div className="flex items-center justify-between p-2 bg-green-50 dark:bg-green-900/20 rounded">
            <span className="text-green-700 dark:text-green-400">
              <CheckCircle className="w-4 h-4 inline mr-1" />
              캐시 효율
            </span>
            <span className="font-bold text-green-600">85%+</span>
          </div>
        </div>
      </div>

      {/* 하단 정보 */}
      <div className="mt-4 text-xs text-gray-500 dark:text-gray-400 text-center">
        마지막 업데이트: {new Date(stats.timestamp).toLocaleString('ko-KR')}
      </div>
    </div>
  )
}