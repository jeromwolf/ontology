'use client'

import { useState, useEffect, useRef } from 'react'
import { Play, Pause, RotateCcw, Shuffle, Zap, Target, Info, Settings } from 'lucide-react'

interface DataPoint {
  x: number
  y: number
  cluster?: number
  id: string
}

interface Centroid {
  x: number
  y: number
  id: number
}

export default function ClusteringVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([])
  const [centroids, setCentroids] = useState<Centroid[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [iteration, setIteration] = useState(0)
  const [converged, setConverged] = useState(false)
  
  const [algorithm, setAlgorithm] = useState<'kmeans' | 'dbscan' | 'hierarchical'>('kmeans')
  const [numClusters, setNumClusters] = useState(3)
  const [datasetType, setDatasetType] = useState<'blobs' | 'moons' | 'circles' | 'random'>('blobs')
  const [numPoints, setNumPoints] = useState(150)
  
  // DBSCAN 파라미터
  const [epsilon, setEpsilon] = useState(50)
  const [minPoints, setMinPoints] = useState(5)
  
  // 색상 팔레트
  const clusterColors = [
    '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
    '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16'
  ]
  
  // 데이터 생성
  const generateData = () => {
    const points: DataPoint[] = []
    
    switch (datasetType) {
      case 'blobs':
        // 뭉친 형태의 클러스터
        for (let c = 0; c < 3; c++) {
          const centerX = 150 + (c - 1) * 200
          const centerY = 200 + (c - 1) * 50
          
          for (let i = 0; i < numPoints / 3; i++) {
            const angle = Math.random() * Math.PI * 2
            const radius = Math.random() * 60 + Math.random() * 40
            
            points.push({
              x: centerX + radius * Math.cos(angle),
              y: centerY + radius * Math.sin(angle),
              id: `${c}-${i}`
            })
          }
        }
        break
        
      case 'moons':
        // 초승달 모양
        for (let i = 0; i < numPoints / 2; i++) {
          const angle = Math.random() * Math.PI
          const radius = 100 + (Math.random() - 0.5) * 30
          
          points.push({
            x: 200 + radius * Math.cos(angle),
            y: 200 + radius * Math.sin(angle),
            id: `moon1-${i}`
          })
          
          points.push({
            x: 300 + radius * Math.cos(angle + Math.PI),
            y: 300 + radius * Math.sin(angle + Math.PI),
            id: `moon2-${i}`
          })
        }
        break
        
      case 'circles':
        // 동심원
        for (let r = 0; r < 2; r++) {
          const radius = 50 + r * 80
          
          for (let i = 0; i < numPoints / 2; i++) {
            const angle = (Math.PI * 2 * i) / (numPoints / 2) + Math.random() * 0.3
            
            points.push({
              x: 300 + radius * Math.cos(angle) + (Math.random() - 0.5) * 20,
              y: 200 + radius * Math.sin(angle) + (Math.random() - 0.5) * 20,
              id: `circle${r}-${i}`
            })
          }
        }
        break
        
      case 'random':
        // 무작위
        for (let i = 0; i < numPoints; i++) {
          points.push({
            x: Math.random() * 500 + 50,
            y: Math.random() * 300 + 50,
            id: `random-${i}`
          })
        }
        break
    }
    
    setDataPoints(points)
    setIteration(0)
    setConverged(false)
    
    // K-means 초기 중심점
    if (algorithm === 'kmeans') {
      initializeCentroids(points)
    }
  }
  
  // K-means 중심점 초기화
  const initializeCentroids = (points: DataPoint[]) => {
    const newCentroids: Centroid[] = []
    const selectedIndices = new Set<number>()
    
    // K-means++ 초기화
    if (points.length > 0) {
      // 첫 번째 중심점은 무작위 선택
      let firstIndex = Math.floor(Math.random() * points.length)
      selectedIndices.add(firstIndex)
      newCentroids.push({
        x: points[firstIndex].x,
        y: points[firstIndex].y,
        id: 0
      })
      
      // 나머지 중심점은 거리 기반 확률로 선택
      for (let k = 1; k < numClusters; k++) {
        const distances = points.map((point, i) => {
          if (selectedIndices.has(i)) return 0
          
          let minDist = Infinity
          newCentroids.forEach(centroid => {
            const dist = Math.sqrt(
              Math.pow(point.x - centroid.x, 2) + 
              Math.pow(point.y - centroid.y, 2)
            )
            minDist = Math.min(minDist, dist)
          })
          
          return minDist * minDist
        })
        
        const totalDist = distances.reduce((sum, d) => sum + d, 0)
        let random = Math.random() * totalDist
        
        for (let i = 0; i < points.length; i++) {
          random -= distances[i]
          if (random <= 0 && !selectedIndices.has(i)) {
            selectedIndices.add(i)
            newCentroids.push({
              x: points[i].x,
              y: points[i].y,
              id: k
            })
            break
          }
        }
      }
    }
    
    setCentroids(newCentroids)
  }
  
  // K-means 스텝
  const kmeansStep = () => {
    // 1. 할당 단계: 각 점을 가장 가까운 중심점에 할당
    const newPoints = dataPoints.map(point => {
      let minDist = Infinity
      let closestCentroid = 0
      
      centroids.forEach(centroid => {
        const dist = Math.sqrt(
          Math.pow(point.x - centroid.x, 2) + 
          Math.pow(point.y - centroid.y, 2)
        )
        
        if (dist < minDist) {
          minDist = dist
          closestCentroid = centroid.id
        }
      })
      
      return { ...point, cluster: closestCentroid }
    })
    
    // 2. 업데이트 단계: 중심점을 클러스터의 평균으로 이동
    const newCentroids = centroids.map(centroid => {
      const clusterPoints = newPoints.filter(p => p.cluster === centroid.id)
      
      if (clusterPoints.length === 0) {
        return centroid
      }
      
      const avgX = clusterPoints.reduce((sum, p) => sum + p.x, 0) / clusterPoints.length
      const avgY = clusterPoints.reduce((sum, p) => sum + p.y, 0) / clusterPoints.length
      
      return { ...centroid, x: avgX, y: avgY }
    })
    
    // 수렴 확인
    const hasConverged = centroids.every((centroid, i) => 
      Math.abs(centroid.x - newCentroids[i].x) < 1 &&
      Math.abs(centroid.y - newCentroids[i].y) < 1
    )
    
    setDataPoints(newPoints)
    setCentroids(newCentroids)
    setIteration(prev => prev + 1)
    
    if (hasConverged) {
      setConverged(true)
      setIsRunning(false)
    }
  }
  
  // DBSCAN 알고리즘
  const runDBSCAN = () => {
    const newPoints = [...dataPoints]
    const visited = new Set<string>()
    const noise = new Set<string>()
    let clusterNum = 0
    
    const getNeighbors = (point: DataPoint) => {
      return dataPoints.filter(p => {
        const dist = Math.sqrt(
          Math.pow(point.x - p.x, 2) + 
          Math.pow(point.y - p.y, 2)
        )
        return dist <= epsilon
      })
    }
    
    const expandCluster = (point: DataPoint, neighbors: DataPoint[], cluster: number) => {
      point.cluster = cluster
      
      const queue = [...neighbors]
      
      while (queue.length > 0) {
        const current = queue.shift()!
        
        if (!visited.has(current.id)) {
          visited.add(current.id)
          
          const currentNeighbors = getNeighbors(current)
          
          if (currentNeighbors.length >= minPoints) {
            queue.push(...currentNeighbors.filter(n => !visited.has(n.id)))
          }
        }
        
        if (current.cluster === undefined) {
          current.cluster = cluster
          noise.delete(current.id)
        }
      }
    }
    
    // DBSCAN 메인 루프
    dataPoints.forEach(point => {
      if (visited.has(point.id)) return
      
      visited.add(point.id)
      const neighbors = getNeighbors(point)
      
      if (neighbors.length < minPoints) {
        noise.add(point.id)
        point.cluster = -1 // 노이즈
      } else {
        expandCluster(point, neighbors, clusterNum)
        clusterNum++
      }
    })
    
    setDataPoints([...dataPoints])
    setConverged(true)
    setIsRunning(false)
  }
  
  // 계층적 클러스터링 (간단한 구현)
  const runHierarchical = () => {
    // 초기에 각 점이 하나의 클러스터
    let clusters = dataPoints.map((point, i) => ({
      points: [point],
      id: i
    }))
    
    // 목표 클러스터 수에 도달할 때까지 병합
    while (clusters.length > numClusters) {
      let minDist = Infinity
      let mergeI = 0, mergeJ = 1
      
      // 가장 가까운 두 클러스터 찾기
      for (let i = 0; i < clusters.length; i++) {
        for (let j = i + 1; j < clusters.length; j++) {
          // 평균 연결법 사용
          let totalDist = 0
          let count = 0
          
          clusters[i].points.forEach(p1 => {
            clusters[j].points.forEach(p2 => {
              totalDist += Math.sqrt(
                Math.pow(p1.x - p2.x, 2) + 
                Math.pow(p1.y - p2.y, 2)
              )
              count++
            })
          })
          
          const avgDist = totalDist / count
          
          if (avgDist < minDist) {
            minDist = avgDist
            mergeI = i
            mergeJ = j
          }
        }
      }
      
      // 클러스터 병합
      clusters[mergeI].points.push(...clusters[mergeJ].points)
      clusters.splice(mergeJ, 1)
    }
    
    // 클러스터 번호 할당
    const newPoints = [...dataPoints]
    clusters.forEach((cluster, i) => {
      cluster.points.forEach(point => {
        const foundPoint = newPoints.find(p => p.id === point.id)
        if (foundPoint) {
          foundPoint.cluster = i
        }
      })
    })
    
    setDataPoints(newPoints)
    setConverged(true)
    setIsRunning(false)
  }
  
  // 캔버스 그리기
  const drawCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    ctx.clearRect(0, 0, 600, 400)
    
    // 그리드
    ctx.strokeStyle = '#e5e7eb'
    ctx.lineWidth = 1
    for (let i = 0; i <= 600; i += 50) {
      ctx.beginPath()
      ctx.moveTo(i, 0)
      ctx.lineTo(i, 400)
      ctx.stroke()
    }
    for (let i = 0; i <= 400; i += 50) {
      ctx.beginPath()
      ctx.moveTo(0, i)
      ctx.lineTo(600, i)
      ctx.stroke()
    }
    
    // DBSCAN의 경우 epsilon 범위 표시
    if (algorithm === 'dbscan' && dataPoints.length > 0) {
      ctx.strokeStyle = 'rgba(156, 163, 175, 0.3)'
      ctx.lineWidth = 1
      ctx.setLineDash([5, 5])
      
      dataPoints.forEach(point => {
        ctx.beginPath()
        ctx.arc(point.x, point.y, epsilon, 0, Math.PI * 2)
        ctx.stroke()
      })
      
      ctx.setLineDash([])
    }
    
    // 데이터 포인트 그리기
    dataPoints.forEach(point => {
      ctx.beginPath()
      ctx.arc(point.x, point.y, 5, 0, Math.PI * 2)
      
      if (point.cluster !== undefined) {
        if (point.cluster === -1) {
          // 노이즈 포인트 (DBSCAN)
          ctx.fillStyle = '#6b7280'
        } else {
          ctx.fillStyle = clusterColors[point.cluster % clusterColors.length]
        }
      } else {
        ctx.fillStyle = '#9ca3af'
      }
      
      ctx.fill()
      ctx.strokeStyle = '#1f2937'
      ctx.lineWidth = 1
      ctx.stroke()
    })
    
    // K-means 중심점 그리기
    if (algorithm === 'kmeans') {
      centroids.forEach((centroid, i) => {
        // 중심점 십자가
        ctx.strokeStyle = clusterColors[i % clusterColors.length]
        ctx.lineWidth = 3
        
        ctx.beginPath()
        ctx.moveTo(centroid.x - 10, centroid.y)
        ctx.lineTo(centroid.x + 10, centroid.y)
        ctx.stroke()
        
        ctx.beginPath()
        ctx.moveTo(centroid.x, centroid.y - 10)
        ctx.lineTo(centroid.x, centroid.y + 10)
        ctx.stroke()
        
        // 중심점 원
        ctx.beginPath()
        ctx.arc(centroid.x, centroid.y, 8, 0, Math.PI * 2)
        ctx.fillStyle = 'white'
        ctx.fill()
        ctx.stroke()
      })
    }
  }
  
  // 애니메이션
  useEffect(() => {
    if (isRunning) {
      const timer = setTimeout(() => {
        switch (algorithm) {
          case 'kmeans':
            kmeansStep()
            break
          case 'dbscan':
            runDBSCAN()
            break
          case 'hierarchical':
            runHierarchical()
            break
        }
      }, 500)
      
      return () => clearTimeout(timer)
    }
  }, [isRunning, iteration])
  
  useEffect(() => {
    generateData()
  }, [datasetType, numPoints, algorithm])
  
  useEffect(() => {
    drawCanvas()
  }, [dataPoints, centroids, algorithm, epsilon])
  
  return (
    <div className="w-full max-w-6xl mx-auto">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-6">클러스터링 시각화</h2>
        
        <div className="grid lg:grid-cols-3 gap-6">
          {/* 시각화 영역 */}
          <div className="lg:col-span-2">
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <canvas
                ref={canvasRef}
                width={600}
                height={400}
                className="border border-gray-300 dark:border-gray-600 rounded"
              />
            </div>
            
            {/* 컨트롤 버튼 */}
            <div className="flex gap-2 mt-4">
              <button
                onClick={() => setIsRunning(!isRunning)}
                disabled={converged}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                  isRunning
                    ? 'bg-red-500 text-white hover:bg-red-600'
                    : 'bg-green-500 text-white hover:bg-green-600'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {isRunning ? (
                  <>
                    <Pause className="w-4 h-4" />
                    일시정지
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    {algorithm === 'kmeans' ? '클러스터링 시작' : '알고리즘 실행'}
                  </>
                )}
              </button>
              
              <button
                onClick={() => {
                  generateData()
                  setIsRunning(false)
                }}
                className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded-lg font-medium hover:bg-gray-600 transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                초기화
              </button>
              
              <button
                onClick={generateData}
                className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 transition-colors"
              >
                <Shuffle className="w-4 h-4" />
                데이터 재생성
              </button>
            </div>
            
            {/* 상태 정보 */}
            <div className="mt-4 bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-gray-600 dark:text-gray-400">알고리즘:</span>
                  <span className="ml-2 font-semibold">
                    {algorithm === 'kmeans' ? 'K-Means' : 
                     algorithm === 'dbscan' ? 'DBSCAN' : '계층적'}
                  </span>
                </div>
                {algorithm === 'kmeans' && (
                  <>
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">반복:</span>
                      <span className="ml-2 font-semibold">{iteration}</span>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">상태:</span>
                      <span className={`ml-2 font-semibold ${converged ? 'text-green-600' : 'text-yellow-600'}`}>
                        {converged ? '수렴 완료' : '학습 중'}
                      </span>
                    </div>
                  </>
                )}
                {algorithm === 'dbscan' && (
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">노이즈:</span>
                    <span className="ml-2 font-semibold">
                      {dataPoints.filter(p => p.cluster === -1).length}개
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
          
          {/* 설정 패널 */}
          <div className="space-y-6">
            {/* 알고리즘 선택 */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Zap className="w-5 h-5" />
                클러스터링 알고리즘
              </h3>
              <div className="space-y-2">
                {[
                  { id: 'kmeans', name: 'K-Means', desc: '중심점 기반 클러스터링' },
                  { id: 'dbscan', name: 'DBSCAN', desc: '밀도 기반 클러스터링' },
                  { id: 'hierarchical', name: '계층적', desc: '병합 클러스터링' }
                ].map(alg => (
                  <button
                    key={alg.id}
                    onClick={() => {
                      setAlgorithm(alg.id as any)
                      setConverged(false)
                      setIteration(0)
                    }}
                    className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                      algorithm === alg.id
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600'
                    }`}
                  >
                    <div className="font-medium">{alg.name}</div>
                    <div className="text-sm opacity-80">{alg.desc}</div>
                  </button>
                ))}
              </div>
            </div>
            
            {/* 데이터셋 설정 */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Target className="w-5 h-5" />
                데이터셋
              </h3>
              <select
                value={datasetType}
                onChange={(e) => setDatasetType(e.target.value as any)}
                className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700"
              >
                <option value="blobs">뭉친 형태 (Blobs)</option>
                <option value="moons">초승달 (Moons)</option>
                <option value="circles">동심원 (Circles)</option>
                <option value="random">무작위 (Random)</option>
              </select>
              
              <div className="mt-3">
                <label className="block text-sm font-medium mb-1">
                  데이터 포인트 수: {numPoints}
                </label>
                <input
                  type="range"
                  min="50"
                  max="300"
                  step="10"
                  value={numPoints}
                  onChange={(e) => setNumPoints(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>
            
            {/* 알고리즘별 파라미터 */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Settings className="w-5 h-5" />
                파라미터
              </h3>
              
              {algorithm === 'kmeans' && (
                <div>
                  <label className="block text-sm font-medium mb-1">
                    클러스터 수 (K): {numClusters}
                  </label>
                  <input
                    type="range"
                    min="2"
                    max="10"
                    value={numClusters}
                    onChange={(e) => {
                      setNumClusters(parseInt(e.target.value))
                      setConverged(false)
                      setIteration(0)
                    }}
                    className="w-full"
                  />
                </div>
              )}
              
              {algorithm === 'dbscan' && (
                <>
                  <div className="mb-3">
                    <label className="block text-sm font-medium mb-1">
                      Epsilon (ε): {epsilon}
                    </label>
                    <input
                      type="range"
                      min="10"
                      max="100"
                      value={epsilon}
                      onChange={(e) => setEpsilon(parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">
                      최소 포인트 수: {minPoints}
                    </label>
                    <input
                      type="range"
                      min="2"
                      max="20"
                      value={minPoints}
                      onChange={(e) => setMinPoints(parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </>
              )}
              
              {algorithm === 'hierarchical' && (
                <div>
                  <label className="block text-sm font-medium mb-1">
                    목표 클러스터 수: {numClusters}
                  </label>
                  <input
                    type="range"
                    min="2"
                    max="10"
                    value={numClusters}
                    onChange={(e) => {
                      setNumClusters(parseInt(e.target.value))
                      setConverged(false)
                    }}
                    className="w-full"
                  />
                </div>
              )}
            </div>
            
            {/* 정보 */}
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <Info className="w-4 h-4" />
                사용 방법
              </h4>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 다양한 데이터셋과 알고리즘을 선택해보세요</li>
                <li>• K-Means는 반복적으로 중심점을 업데이트합니다</li>
                <li>• DBSCAN은 밀도 기반으로 클러스터를 찾습니다</li>
                <li>• 파라미터를 조정하여 결과를 비교해보세요</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}