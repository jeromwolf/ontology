'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ArrowLeft, Database, Hash, Globe, Calendar, BarChart, Server, Users } from 'lucide-react'

type ShardingStrategy = 'range' | 'hash' | 'geographic' | 'composite'

interface DataItem {
  id: string
  userId: number
  country: string
  timestamp: Date
  value: number
}

interface Shard {
  id: number
  name: string
  items: DataItem[]
  color: string
  criteria: string
}

export default function ShardingVisualizer() {
  const [strategy, setStrategy] = useState<ShardingStrategy>('hash')
  const [dataItems, setDataItems] = useState<DataItem[]>([])
  const [shards, setShards] = useState<Shard[]>([])
  const [selectedItem, setSelectedItem] = useState<DataItem | null>(null)
  const [dataSize, setDataSize] = useState(100)
  const [shardCount, setShardCount] = useState(4)

  const countries = ['USA', 'UK', 'Korea', 'Japan', 'Germany', 'France', 'Brazil', 'India']
  const colors = ['purple', 'blue', 'green', 'yellow', 'red', 'indigo', 'pink', 'cyan']

  // 데이터 생성
  const generateData = () => {
    const items: DataItem[] = []
    const now = Date.now()
    
    for (let i = 0; i < dataSize; i++) {
      items.push({
        id: `item_${i}`,
        userId: Math.floor(Math.random() * 10000),
        country: countries[Math.floor(Math.random() * countries.length)],
        timestamp: new Date(now - Math.random() * 30 * 24 * 60 * 60 * 1000), // 30일 범위
        value: Math.floor(Math.random() * 1000)
      })
    }
    
    setDataItems(items)
  }

  // 해시 함수
  const hashFunction = (key: string | number, modulo: number): number => {
    let hash = 0
    const str = String(key)
    for (let i = 0; i < str.length; i++) {
      hash = (hash * 31 + str.charCodeAt(i)) % Number.MAX_SAFE_INTEGER
    }
    return Math.abs(hash) % modulo
  }

  // 샤딩 전략에 따른 데이터 분배
  const distributeData = () => {
    const newShards: Shard[] = []
    
    // 샤드 초기화
    for (let i = 0; i < shardCount; i++) {
      newShards.push({
        id: i,
        name: `Shard ${i + 1}`,
        items: [],
        color: colors[i % colors.length],
        criteria: ''
      })
    }
    
    // 전략에 따른 분배
    dataItems.forEach(item => {
      let shardIndex = 0
      
      switch (strategy) {
        case 'hash':
          // 사용자 ID 기반 해시 샤딩
          shardIndex = hashFunction(item.userId, shardCount)
          newShards[shardIndex].criteria = 'Hash(UserID)'
          break
          
        case 'range':
          // 사용자 ID 범위 기반 샤딩
          const rangeSize = 10000 / shardCount
          shardIndex = Math.min(Math.floor(item.userId / rangeSize), shardCount - 1)
          newShards[shardIndex].criteria = `UserID: ${Math.floor(shardIndex * rangeSize)}-${Math.floor((shardIndex + 1) * rangeSize - 1)}`
          break
          
        case 'geographic':
          // 지리적 위치 기반 샤딩
          const geoMap: { [key: string]: number } = {
            'USA': 0, 'Brazil': 0,
            'UK': 1, 'Germany': 1, 'France': 1,
            'Korea': 2, 'Japan': 2,
            'India': 3
          }
          shardIndex = geoMap[item.country] % shardCount
          newShards[shardIndex].criteria = 'Geographic Region'
          break
          
        case 'composite':
          // 복합 키 샤딩 (국가 + 시간)
          const monthIndex = item.timestamp.getMonth()
          const countryIndex = countries.indexOf(item.country)
          shardIndex = (countryIndex + monthIndex) % shardCount
          newShards[shardIndex].criteria = 'Country + Time'
          break
      }
      
      newShards[shardIndex].items.push(item)
    })
    
    setShards(newShards)
  }

  useEffect(() => {
    generateData()
  }, [dataSize])

  useEffect(() => {
    if (dataItems.length > 0) {
      distributeData()
    }
  }, [dataItems, strategy, shardCount])

  // 통계 계산
  const calculateStats = () => {
    if (shards.length === 0) return { avg: 0, min: 0, max: 0, stdDev: 0 }
    
    const counts = shards.map(s => s.items.length)
    const avg = counts.reduce((a, b) => a + b, 0) / counts.length
    const min = Math.min(...counts)
    const max = Math.max(...counts)
    
    const variance = counts.reduce((sum, count) => sum + Math.pow(count - avg, 2), 0) / counts.length
    const stdDev = Math.sqrt(variance)
    
    return { avg, min, max, stdDev }
  }

  const stats = calculateStats()
  const balance = stats.stdDev < 5 ? 'excellent' : stats.stdDev < 10 ? 'good' : 'poor'

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
            샤딩 전략 시각화
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Range, Hash, Geographic, Composite 샤딩 전략을 비교합니다
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Strategy Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              샤딩 전략
            </label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as ShardingStrategy)}
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="hash">Hash Sharding</option>
              <option value="range">Range Sharding</option>
              <option value="geographic">Geographic Sharding</option>
              <option value="composite">Composite Key</option>
            </select>
          </div>
          
          {/* Data Size */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              데이터 크기: {dataSize}
            </label>
            <input
              type="range"
              min="50"
              max="500"
              step="50"
              value={dataSize}
              onChange={(e) => setDataSize(Number(e.target.value))}
              className="w-full"
            />
          </div>
          
          {/* Shard Count */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              샤드 개수: {shardCount}
            </label>
            <input
              type="range"
              min="2"
              max="8"
              value={shardCount}
              onChange={(e) => setShardCount(Number(e.target.value))}
              className="w-full"
            />
          </div>
          
          {/* Regenerate Button */}
          <div className="flex items-end">
            <button
              onClick={generateData}
              className="w-full px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg font-semibold transition-colors"
            >
              데이터 재생성
            </button>
          </div>
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">평균 항목</span>
            <BarChart className="w-4 h-4 text-purple-500" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {stats.avg.toFixed(1)}
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">최소/최대</span>
            <Database className="w-4 h-4 text-blue-500" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {stats.min} / {stats.max}
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">표준편차</span>
            <Hash className="w-4 h-4 text-green-500" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {stats.stdDev.toFixed(2)}
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">균형도</span>
            <Server className="w-4 h-4 text-yellow-500" />
          </div>
          <div className={`text-2xl font-bold ${
            balance === 'excellent' ? 'text-green-600 dark:text-green-400' :
            balance === 'good' ? 'text-yellow-600 dark:text-yellow-400' :
            'text-red-600 dark:text-red-400'
          }`}>
            {balance === 'excellent' ? '우수' : balance === 'good' ? '양호' : '불균형'}
          </div>
        </div>
      </div>

      {/* Shards Visualization */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-8">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          샤드 분포
        </h3>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {shards.map((shard) => (
            <div
              key={shard.id}
              className={`relative p-4 rounded-lg border-2 border-${shard.color}-500 bg-${shard.color}-50 dark:bg-${shard.color}-950/20`}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Database className={`w-5 h-5 text-${shard.color}-600 dark:text-${shard.color}-400`} />
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {shard.name}
                  </span>
                </div>
                <span className={`text-lg font-bold text-${shard.color}-600 dark:text-${shard.color}-400`}>
                  {shard.items.length}
                </span>
              </div>
              
              <div className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                {shard.criteria}
              </div>
              
              {/* Progress Bar */}
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mb-2">
                <div
                  className={`h-2 rounded-full bg-${shard.color}-500 transition-all duration-300`}
                  style={{ width: `${(shard.items.length / dataSize) * 100}%` }}
                />
              </div>
              
              <div className="text-xs text-gray-500 dark:text-gray-500">
                {((shard.items.length / dataSize) * 100).toFixed(1)}% of data
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Data Distribution Chart */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-8">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          데이터 분포 차트
        </h3>
        
        <div className="relative h-64">
          <div className="absolute bottom-0 left-0 right-0 flex items-end justify-around h-full">
            {shards.map((shard) => {
              const height = (shard.items.length / Math.max(...shards.map(s => s.items.length))) * 100
              return (
                <div
                  key={shard.id}
                  className="flex-1 mx-1 flex flex-col items-center justify-end"
                >
                  <div className="text-sm font-semibold text-gray-900 dark:text-white mb-2">
                    {shard.items.length}
                  </div>
                  <div
                    className={`w-full bg-${shard.color}-500 rounded-t-lg transition-all duration-500`}
                    style={{ height: `${height}%` }}
                  />
                  <div className="mt-2 text-xs text-gray-600 dark:text-gray-400">
                    Shard {shard.id + 1}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Strategy Comparison */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          샤딩 전략 비교
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Hash className="w-5 h-5 text-purple-500" />
              <h4 className="font-semibold text-gray-900 dark:text-white">Hash Sharding</h4>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              키의 해시값으로 균등 분배
            </p>
            <div className="text-xs">
              <span className="text-green-600 dark:text-green-400">✓ 균등한 분포</span><br />
              <span className="text-green-600 dark:text-green-400">✓ 확장성 우수</span><br />
              <span className="text-red-600 dark:text-red-400">✗ 범위 쿼리 어려움</span>
            </div>
          </div>
          
          <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Calendar className="w-5 h-5 text-blue-500" />
              <h4 className="font-semibold text-gray-900 dark:text-white">Range Sharding</h4>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              연속된 키 범위로 분할
            </p>
            <div className="text-xs">
              <span className="text-green-600 dark:text-green-400">✓ 범위 쿼리 효율적</span><br />
              <span className="text-green-600 dark:text-green-400">✓ 구현 간단</span><br />
              <span className="text-red-600 dark:text-red-400">✗ 핫스팟 발생 가능</span>
            </div>
          </div>
          
          <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Globe className="w-5 h-5 text-green-500" />
              <h4 className="font-semibold text-gray-900 dark:text-white">Geographic</h4>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              지리적 위치 기반 분할
            </p>
            <div className="text-xs">
              <span className="text-green-600 dark:text-green-400">✓ 지연시간 최소화</span><br />
              <span className="text-green-600 dark:text-green-400">✓ 규정 준수 용이</span><br />
              <span className="text-red-600 dark:text-red-400">✗ 불균등 분포</span>
            </div>
          </div>
          
          <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Users className="w-5 h-5 text-yellow-500" />
              <h4 className="font-semibold text-gray-900 dark:text-white">Composite Key</h4>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              복합 키로 다차원 분할
            </p>
            <div className="text-xs">
              <span className="text-green-600 dark:text-green-400">✓ 유연한 분배</span><br />
              <span className="text-green-600 dark:text-green-400">✓ 다양한 쿼리 지원</span><br />
              <span className="text-red-600 dark:text-red-400">✗ 복잡한 라우팅</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}