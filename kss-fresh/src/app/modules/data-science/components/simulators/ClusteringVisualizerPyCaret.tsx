'use client'

import { useState, useEffect, useRef } from 'react'
import { 
  Play, Shuffle, Zap, Target, Info, Settings, Download,
  Brain, BarChart3, Layers, GitBranch, CircleDot, Grid3x3,
  ChevronRight, Check, AlertCircle
} from 'lucide-react'

interface DataPoint {
  x: number
  y: number
  cluster?: number
  id: string
  features?: number[]
}

interface ClusteringResult {
  algorithm: string
  clusters: number
  silhouette: number
  calinski: number
  davies: number
  inertia?: number
  dataPoints: DataPoint[]
  centroids?: { x: number, y: number, id: number }[]
  elbow?: number[]
}

interface DatasetOption {
  name: string
  description: string
  type: 'blobs' | 'moons' | 'circles' | 'anisotropic' | 'varied' | 'no_structure'
  optimalClusters?: number
}

const DATASETS: DatasetOption[] = [
  {
    name: '고객 세분화',
    description: '구매 패턴에 따른 고객 그룹',
    type: 'blobs',
    optimalClusters: 3
  },
  {
    name: '이상치 탐지',
    description: '정상/비정상 패턴 분리',
    type: 'moons',
    optimalClusters: 2
  },
  {
    name: '동심원 데이터',
    description: '중첩된 원형 클러스터',
    type: 'circles',
    optimalClusters: 2
  },
  {
    name: '비등방성 데이터',
    description: '타원형 클러스터',
    type: 'anisotropic',
    optimalClusters: 3
  },
  {
    name: '다양한 밀도',
    description: '밀도가 다른 클러스터',
    type: 'varied',
    optimalClusters: 3
  },
  {
    name: '무작위 데이터',
    description: '클러스터 구조 없음',
    type: 'no_structure'
  }
]

export default function ClusteringVisualizerPyCaret() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const elbowCanvasRef = useRef<HTMLCanvasElement>(null)
  const [selectedDataset, setSelectedDataset] = useState<DatasetOption>(DATASETS[0])
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([])
  const [isTraining, setIsTraining] = useState(false)
  const [clusteringResults, setClusteringResults] = useState<ClusteringResult[]>([])
  const [selectedResult, setSelectedResult] = useState<ClusteringResult | null>(null)
  const [currentStep, setCurrentStep] = useState<'data' | 'setup' | 'compare' | 'visualize'>('data')
  const [numClusters, setNumClusters] = useState<'auto' | number>('auto')
  const [preprocessing, setPreprocessing] = useState({
    normalize: true,
    removeOutliers: false,
    pca: false,
    pcaComponents: 2
  })

  // 색상 팔레트
  const clusterColors = [
    '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
    '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16',
    '#6b7280' // 노이즈/이상치용
  ]

  // 데이터 생성
  const generateData = (dataset: DatasetOption): DataPoint[] => {
    const points: DataPoint[] = []
    const numPoints = 300

    switch (dataset.type) {
      case 'blobs':
        // 고객 세분화 시뮬레이션
        for (let c = 0; c < 3; c++) {
          const centerX = 150 + (c - 1) * 200
          const centerY = 250 + (c - 1) * 50
          const spread = 60
          
          for (let i = 0; i < numPoints / 3; i++) {
            const angle = Math.random() * Math.PI * 2
            const radius = Math.random() * spread
            points.push({
              x: centerX + Math.cos(angle) * radius,
              y: centerY + Math.sin(angle) * radius,
              id: `point-${points.length}`,
              features: [
                centerX + Math.cos(angle) * radius,
                centerY + Math.sin(angle) * radius,
                Math.random() * 100, // 구매 빈도
                Math.random() * 1000 // 평균 구매액
              ]
            })
          }
        }
        break

      case 'moons':
        // 이상치 탐지 패턴
        for (let i = 0; i < numPoints * 0.9; i++) {
          const angle = Math.random() * Math.PI
          const moon = i % 2
          const radius = 100 + (Math.random() - 0.5) * 20
          
          points.push({
            x: 250 + Math.cos(angle) * radius + (moon ? 100 : 0),
            y: 250 + Math.sin(angle) * radius + (moon ? -50 : 50),
            id: `point-${points.length}`
          })
        }
        // 이상치 추가
        for (let i = 0; i < numPoints * 0.1; i++) {
          points.push({
            x: 100 + Math.random() * 400,
            y: 100 + Math.random() * 300,
            id: `point-${points.length}`
          })
        }
        break

      case 'circles':
        // 동심원
        for (let ring = 0; ring < 2; ring++) {
          const radius = 50 + ring * 100
          const pointsPerRing = numPoints / 2
          
          for (let i = 0; i < pointsPerRing; i++) {
            const angle = (i / pointsPerRing) * Math.PI * 2
            const noise = (Math.random() - 0.5) * 20
            
            points.push({
              x: 250 + Math.cos(angle) * (radius + noise),
              y: 250 + Math.sin(angle) * (radius + noise),
              id: `point-${points.length}`
            })
          }
        }
        break

      case 'anisotropic':
        // 타원형 클러스터
        for (let c = 0; c < 3; c++) {
          const angle = (c * Math.PI * 2) / 3
          const centerX = 250 + Math.cos(angle) * 100
          const centerY = 250 + Math.sin(angle) * 100
          
          for (let i = 0; i < numPoints / 3; i++) {
            const theta = Math.random() * Math.PI * 2
            const radiusX = Math.random() * 80
            const radiusY = Math.random() * 30
            
            points.push({
              x: centerX + Math.cos(theta) * radiusX * Math.cos(angle) - Math.sin(theta) * radiusY * Math.sin(angle),
              y: centerY + Math.cos(theta) * radiusX * Math.sin(angle) + Math.sin(theta) * radiusY * Math.cos(angle),
              id: `point-${points.length}`
            })
          }
        }
        break

      case 'varied':
        // 다양한 밀도
        const densities = [100, 50, 30]
        const sizes = [40, 60, 80]
        
        for (let c = 0; c < 3; c++) {
          const centerX = 150 + (c - 1) * 150
          const centerY = 250
          
          for (let i = 0; i < densities[c]; i++) {
            const angle = Math.random() * Math.PI * 2
            const radius = Math.random() * sizes[c]
            
            points.push({
              x: centerX + Math.cos(angle) * radius,
              y: centerY + Math.sin(angle) * radius,
              id: `point-${points.length}`
            })
          }
        }
        break

      case 'no_structure':
        // 무작위
        for (let i = 0; i < numPoints; i++) {
          points.push({
            x: 50 + Math.random() * 400,
            y: 50 + Math.random() * 400,
            id: `point-${points.length}`
          })
        }
        break
    }

    return points
  }

  // PyCaret 클러스터링 시뮬레이션
  const trainClustering = async () => {
    setIsTraining(true)
    setClusteringResults([])
    setCurrentStep('compare')

    const algorithms = [
      { name: 'K-Means', supportsAuto: true, needsNumClusters: true },
      { name: 'Agglomerative', supportsAuto: false, needsNumClusters: true },
      { name: 'DBSCAN', supportsAuto: true, needsNumClusters: false },
      { name: 'Mean Shift', supportsAuto: true, needsNumClusters: false },
      { name: 'Spectral', supportsAuto: false, needsNumClusters: true },
      { name: 'HDBSCAN', supportsAuto: true, needsNumClusters: false }
    ]

    // K-means elbow method
    let elbowScores: number[] = []
    if (numClusters === 'auto') {
      for (let k = 2; k <= 10; k++) {
        elbowScores.push(1000 / k + Math.random() * 200)
      }
    }

    for (const algo of algorithms) {
      await new Promise(resolve => setTimeout(resolve, 800))

      // 클러스터 수 결정
      let actualClusters = 3
      if (numClusters === 'auto') {
        if (algo.supportsAuto) {
          actualClusters = selectedDataset.optimalClusters || Math.floor(Math.random() * 3) + 2
        } else {
          actualClusters = 3
        }
      } else {
        actualClusters = numClusters as number
      }

      // 클러스터링 수행
      const clusteredPoints = dataPoints.map(point => {
        let cluster = 0
        
        if (algo.name === 'K-Means') {
          // K-means는 거리 기반
          cluster = Math.floor(Math.random() * actualClusters)
        } else if (algo.name === 'DBSCAN' || algo.name === 'HDBSCAN') {
          // 밀도 기반 - 일부 노이즈 포인트
          cluster = Math.random() > 0.1 ? Math.floor(Math.random() * actualClusters) : -1
        } else {
          cluster = Math.floor(Math.random() * actualClusters)
        }

        return { ...point, cluster }
      })

      // 중심점 계산 (K-means만)
      let centroids = undefined
      if (algo.name === 'K-Means') {
        centroids = []
        for (let c = 0; c < actualClusters; c++) {
          const clusterPoints = clusteredPoints.filter(p => p.cluster === c)
          if (clusterPoints.length > 0) {
            const centerX = clusterPoints.reduce((sum, p) => sum + p.x, 0) / clusterPoints.length
            const centerY = clusterPoints.reduce((sum, p) => sum + p.y, 0) / clusterPoints.length
            centroids.push({ x: centerX, y: centerY, id: c })
          }
        }
      }

      // 성능 지표 계산
      const silhouette = 0.3 + Math.random() * 0.5
      const calinski = 100 + Math.random() * 400
      const davies = 0.5 + Math.random() * 1

      setClusteringResults(prev => [...prev, {
        algorithm: algo.name,
        clusters: actualClusters,
        silhouette,
        calinski,
        davies,
        dataPoints: clusteredPoints,
        centroids,
        elbow: algo.name === 'K-Means' ? elbowScores : undefined
      }])
    }

    setIsTraining(false)
  }

  // 시각화
  useEffect(() => {
    if (!canvasRef.current || !selectedResult) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')!
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width
    canvas.height = rect.height

    // Clear
    ctx.fillStyle = '#f3f4f6'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // 클러스터별로 점 그리기
    selectedResult.dataPoints.forEach(point => {
      ctx.fillStyle = point.cluster === -1 ? '#6b7280' : clusterColors[point.cluster! % clusterColors.length]
      ctx.beginPath()
      ctx.arc(point.x, point.y, 5, 0, Math.PI * 2)
      ctx.fill()
    })

    // 중심점 그리기 (K-means)
    if (selectedResult.centroids) {
      selectedResult.centroids.forEach(centroid => {
        ctx.fillStyle = clusterColors[centroid.id % clusterColors.length]
        ctx.strokeStyle = '#000'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(centroid.x, centroid.y, 10, 0, Math.PI * 2)
        ctx.fill()
        ctx.stroke()

        // X 표시
        ctx.strokeStyle = '#fff'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(centroid.x - 5, centroid.y - 5)
        ctx.lineTo(centroid.x + 5, centroid.y + 5)
        ctx.moveTo(centroid.x + 5, centroid.y - 5)
        ctx.lineTo(centroid.x - 5, centroid.y + 5)
        ctx.stroke()
      })
    }
  }, [selectedResult])

  // Elbow plot
  useEffect(() => {
    if (!elbowCanvasRef.current || !selectedResult?.elbow) return

    const canvas = elbowCanvasRef.current
    const ctx = canvas.getContext('2d')!
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width
    canvas.height = rect.height

    // Clear
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    const padding = 40
    const chartWidth = canvas.width - padding * 2
    const chartHeight = canvas.height - padding * 2

    // 축 그리기
    ctx.strokeStyle = '#6b7280'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(padding, padding)
    ctx.lineTo(padding, padding + chartHeight)
    ctx.lineTo(padding + chartWidth, padding + chartHeight)
    ctx.stroke()

    // Elbow curve
    const maxScore = Math.max(...selectedResult.elbow)
    const minScore = Math.min(...selectedResult.elbow)
    const scoreRange = maxScore - minScore

    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 2
    ctx.beginPath()

    selectedResult.elbow.forEach((score, i) => {
      const x = padding + (chartWidth / (selectedResult.elbow!.length - 1)) * i
      const y = padding + chartHeight - ((score - minScore) / scoreRange) * chartHeight

      if (i === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }

      // 점 그리기
      ctx.fillStyle = '#3b82f6'
      ctx.beginPath()
      ctx.arc(x, y, 4, 0, Math.PI * 2)
      ctx.fill()

      // 레이블
      ctx.fillStyle = '#6b7280'
      ctx.font = '12px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(`${i + 2}`, x, padding + chartHeight + 20)
    })
    ctx.stroke()

    // 축 레이블
    ctx.fillStyle = '#374151'
    ctx.font = '14px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('Number of Clusters', padding + chartWidth / 2, canvas.height - 10)
    
    ctx.save()
    ctx.translate(15, padding + chartHeight / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText('Inertia', 0, 0)
    ctx.restore()
  }, [selectedResult])

  // CSV 다운로드
  const downloadResults = () => {
    if (!selectedResult) return

    let csv = 'Point_ID,X,Y,Cluster\n'
    selectedResult.dataPoints.forEach(point => {
      csv += `${point.id},${point.x.toFixed(2)},${point.y.toFixed(2)},${point.cluster}\n`
    })

    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `clustering_results_${selectedResult.algorithm}_${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* 헤더 */}
      <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl p-6">
        <h2 className="text-2xl font-bold mb-2">클러스터링 분석 with PyCaret</h2>
        <p className="text-purple-100">
          PyCaret의 자동 클러스터링으로 최적의 그룹을 발견하고 패턴을 분석합니다
        </p>
      </div>

      {/* 진행 단계 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-4">
        <div className="flex items-center justify-between">
          {['data', 'setup', 'compare', 'visualize'].map((step, idx) => (
            <div key={step} className="flex items-center">
              <div className={`flex items-center justify-center w-10 h-10 rounded-full ${
                currentStep === step 
                  ? 'bg-purple-500 text-white' 
                  : idx < ['data', 'setup', 'compare', 'visualize'].indexOf(currentStep)
                    ? 'bg-green-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-500'
              }`}>
                {idx < ['data', 'setup', 'compare', 'visualize'].indexOf(currentStep) 
                  ? <Check className="w-5 h-5" />
                  : idx + 1
                }
              </div>
              <div className="ml-2">
                <div className="text-sm font-medium">
                  {step === 'data' && '데이터 선택'}
                  {step === 'setup' && '전처리 설정'}
                  {step === 'compare' && '알고리즘 비교'}
                  {step === 'visualize' && '시각화'}
                </div>
              </div>
              {idx < 3 && <ChevronRight className="w-5 h-5 mx-4 text-gray-400" />}
            </div>
          ))}
        </div>
      </div>

      {/* Step 1: 데이터 선택 */}
      {currentStep === 'data' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">데이터셋 선택</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {DATASETS.map((dataset) => (
              <button
                key={dataset.name}
                onClick={() => {
                  setSelectedDataset(dataset)
                  const points = generateData(dataset)
                  setDataPoints(points)
                  setCurrentStep('setup')
                }}
                className={`p-4 rounded-lg border-2 transition-all text-left ${
                  selectedDataset.name === dataset.name
                    ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                    : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                }`}
              >
                <h4 className="font-medium mb-1">{dataset.name}</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  {dataset.description}
                </p>
                {dataset.optimalClusters && (
                  <span className="text-xs bg-purple-100 dark:bg-purple-900/30 px-2 py-1 rounded">
                    최적 클러스터: {dataset.optimalClusters}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Step 2: 전처리 설정 */}
      {currentStep === 'setup' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">PyCaret 전처리 설정</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-3">클러스터 수</h4>
              <div className="space-y-2">
                <label className="flex items-center gap-2">
                  <input
                    type="radio"
                    checked={numClusters === 'auto'}
                    onChange={() => setNumClusters('auto')}
                    className="rounded"
                  />
                  <span>자동 결정 (Elbow method, Silhouette)</span>
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="radio"
                    checked={typeof numClusters === 'number'}
                    onChange={() => setNumClusters(3)}
                    className="rounded"
                  />
                  <span>수동 지정</span>
                </label>
                {typeof numClusters === 'number' && (
                  <input
                    type="number"
                    value={numClusters}
                    onChange={(e) => setNumClusters(parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg"
                    min="2"
                    max="10"
                  />
                )}
              </div>
            </div>
            
            <div>
              <h4 className="font-medium mb-3">데이터 전처리</h4>
              <div className="space-y-2">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={preprocessing.normalize}
                    onChange={(e) => setPreprocessing({...preprocessing, normalize: e.target.checked})}
                    className="rounded"
                  />
                  <span>정규화 (Min-Max Scaling)</span>
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={preprocessing.removeOutliers}
                    onChange={(e) => setPreprocessing({...preprocessing, removeOutliers: e.target.checked})}
                    className="rounded"
                  />
                  <span>이상치 제거 (Isolation Forest)</span>
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={preprocessing.pca}
                    onChange={(e) => setPreprocessing({...preprocessing, pca: e.target.checked})}
                    className="rounded"
                  />
                  <span>차원 축소 (PCA)</span>
                </label>
              </div>
            </div>
          </div>

          <div className="mt-6 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
            <div className="flex items-start gap-2">
              <Info className="w-5 h-5 text-purple-600 mt-0.5" />
              <div className="text-sm text-purple-700 dark:text-purple-300">
                <p className="font-medium">PyCaret 자동 최적화</p>
                <p>PyCaret은 자동으로 최적의 클러스터 수를 찾고, 여러 알고리즘을 비교합니다.</p>
              </div>
            </div>
          </div>
          
          <button
            onClick={trainClustering}
            className="mt-6 w-full px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors flex items-center justify-center gap-2"
          >
            <Brain className="w-5 h-5" />
            클러스터링 시작
          </button>
        </div>
      )}

      {/* Step 3: 알고리즘 비교 */}
      {currentStep === 'compare' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Layers className="w-5 h-5" />
            클러스터링 알고리즘 비교
          </h3>

          {isTraining && (
            <div className="mb-4 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <div className="flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-purple-500 border-t-transparent"></div>
                <span className="text-purple-700 dark:text-purple-300">
                  클러스터링 중... ({clusteringResults.length}/6)
                </span>
              </div>
            </div>
          )}

          {clusteringResults.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <th className="text-left py-3 px-4">알고리즘</th>
                    <th className="text-center py-3 px-4">클러스터 수</th>
                    <th className="text-center py-3 px-4">Silhouette</th>
                    <th className="text-center py-3 px-4">Calinski-Harabasz</th>
                    <th className="text-center py-3 px-4">Davies-Bouldin</th>
                    <th className="text-center py-3 px-4">선택</th>
                  </tr>
                </thead>
                <tbody>
                  {clusteringResults.map((result, idx) => {
                    const isBest = clusteringResults.reduce((best, r) => 
                      r.silhouette > best.silhouette ? r : best
                    ).algorithm === result.algorithm
                    
                    return (
                      <tr 
                        key={idx} 
                        className={`border-b border-gray-100 dark:border-gray-700 ${
                          isBest ? 'bg-green-50 dark:bg-green-900/20' : ''
                        }`}
                      >
                        <td className="py-3 px-4 font-medium">
                          {result.algorithm}
                          {isBest && (
                            <span className="ml-2 text-xs bg-green-500 text-white px-2 py-0.5 rounded">
                              Best
                            </span>
                          )}
                        </td>
                        <td className="text-center py-3 px-4">{result.clusters}</td>
                        <td className="text-center py-3 px-4">
                          <div className="flex items-center justify-center gap-1">
                            <span>{result.silhouette.toFixed(3)}</span>
                            <div className="w-12 bg-gray-200 rounded-full h-2">
                              <div 
                                className="bg-purple-500 h-2 rounded-full"
                                style={{ width: `${result.silhouette * 100}%` }}
                              />
                            </div>
                          </div>
                        </td>
                        <td className="text-center py-3 px-4">{result.calinski.toFixed(1)}</td>
                        <td className="text-center py-3 px-4">{result.davies.toFixed(2)}</td>
                        <td className="text-center py-3 px-4">
                          <button
                            onClick={() => {
                              setSelectedResult(result)
                              setCurrentStep('visualize')
                            }}
                            className="px-3 py-1 bg-purple-600 text-white text-sm rounded hover:bg-purple-700 transition-colors"
                          >
                            시각화
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}

          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-medium mb-2 flex items-center gap-2">
                <Target className="w-4 h-4" />
                Silhouette Score
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                클러스터 내 응집도와 분리도 측정 (높을수록 좋음, -1~1)
              </p>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-medium mb-2 flex items-center gap-2">
                <BarChart3 className="w-4 h-4" />
                Calinski-Harabasz
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                클러스터 간 분산 대비 클러스터 내 분산 (높을수록 좋음)
              </p>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-medium mb-2 flex items-center gap-2">
                <CircleDot className="w-4 h-4" />
                Davies-Bouldin
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                클러스터 간 유사도 (낮을수록 좋음)
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Step 4: 시각화 */}
      {currentStep === 'visualize' && selectedResult && (
        <>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">
                {selectedResult.algorithm} 클러스터링 결과
              </h3>
              <button
                onClick={downloadResults}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                결과 다운로드
              </button>
            </div>
            
            <canvas
              ref={canvasRef}
              className="w-full h-96 rounded-lg bg-gray-50"
            />
            
            {/* 클러스터 통계 */}
            <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
              {[...Array(selectedResult.clusters)].map((_, i) => {
                const clusterPoints = selectedResult.dataPoints.filter(p => p.cluster === i)
                return (
                  <div key={i} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <div 
                        className="w-4 h-4 rounded-full"
                        style={{ backgroundColor: clusterColors[i % clusterColors.length] }}
                      />
                      <span className="font-medium">클러스터 {i + 1}</span>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {clusterPoints.length}개 포인트
                    </p>
                  </div>
                )
              })}
              {selectedResult.dataPoints.some(p => p.cluster === -1) && (
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-4 h-4 rounded-full bg-gray-600" />
                    <span className="font-medium">노이즈</span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {selectedResult.dataPoints.filter(p => p.cluster === -1).length}개 포인트
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Elbow Plot (K-means only) */}
          {selectedResult.elbow && (
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">Elbow Method</h3>
              <canvas
                ref={elbowCanvasRef}
                className="w-full h-64 rounded-lg"
              />
              <p className="mt-4 text-sm text-gray-600 dark:text-gray-400">
                Elbow method는 최적의 클러스터 수를 찾기 위한 방법입니다. 
                그래프가 꺾이는 지점(elbow)이 최적의 클러스터 수를 나타냅니다.
              </p>
            </div>
          )}
        </>
      )}

      {/* 정보 패널 */}
      <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-purple-600 mt-0.5" />
          <div className="text-sm text-purple-700 dark:text-purple-300">
            <p className="font-medium mb-1">PyCaret 클러스터링</p>
            <p>
              이 시뮬레이터는 PyCaret의 클러스터링 기능을 시뮬레이션합니다.
              실제 PyCaret은 20개 이상의 클러스터링 알고리즘과 자동 하이퍼파라미터 튜닝을 제공합니다.
            </p>
            <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="font-medium mb-1">지원 알고리즘:</p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li>K-Means - 가장 일반적인 중심 기반</li>
                  <li>DBSCAN - 밀도 기반 (이상치 탐지)</li>
                  <li>Hierarchical - 계층적 클러스터링</li>
                  <li>Mean Shift - 밀도 추정 기반</li>
                </ul>
              </div>
              <div>
                <p className="font-medium mb-1">활용 사례:</p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li>고객 세분화</li>
                  <li>이상치 탐지</li>
                  <li>이미지 분할</li>
                  <li>문서 그룹화</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}