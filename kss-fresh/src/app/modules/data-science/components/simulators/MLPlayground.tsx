'use client'

import { useState, useEffect, useRef } from 'react'
import { Play, Pause, RotateCcw, Settings, Info, Download, ChevronDown, ChevronUp, TrendingDown } from 'lucide-react'

interface DataPoint {
  x: number
  y: number
  label?: number
}

interface ModelParams {
  algorithm: 'linear' | 'logistic' | 'svm' | 'decision-tree'
  learningRate: number
  iterations: number
  c?: number // for SVM
  maxDepth?: number // for Decision Tree
}

export default function MLPlayground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([])
  const [modelParams, setModelParams] = useState<ModelParams>({
    algorithm: 'logistic',
    learningRate: 0.01,
    iterations: 100
  })
  const [isTraining, setIsTraining] = useState(false)
  const [currentIteration, setCurrentIteration] = useState(0)
  const [accuracy, setAccuracy] = useState(0)
  const [selectedDataset, setSelectedDataset] = useState<'linear' | 'circular' | 'xor' | 'custom'>('circular')
  const [lossHistory, setLossHistory] = useState<number[]>([])
  const [showAdvanced, setShowAdvanced] = useState(false)

  // 데이터셋 생성
  const generateDataset = (type: string) => {
    const points: DataPoint[] = []
    
    switch(type) {
      case 'linear':
        // 선형 분리 가능한 데이터
        for (let i = 0; i < 100; i++) {
          const x = Math.random() * 400 + 50
          const y = Math.random() * 400 + 50
          const label = x + y > 500 ? 1 : 0
          points.push({ x, y, label })
        }
        break
        
      case 'circular':
        // 원형 데이터
        for (let i = 0; i < 100; i++) {
          const angle = Math.random() * Math.PI * 2
          const r1 = Math.random() * 50 + 50
          const r2 = Math.random() * 50 + 150
          
          if (i < 50) {
            points.push({
              x: 250 + r1 * Math.cos(angle),
              y: 250 + r1 * Math.sin(angle),
              label: 0
            })
          } else {
            points.push({
              x: 250 + r2 * Math.cos(angle),
              y: 250 + r2 * Math.sin(angle),
              label: 1
            })
          }
        }
        break
        
      case 'xor':
        // XOR 패턴
        for (let i = 0; i < 100; i++) {
          const x = Math.random() * 400 + 50
          const y = Math.random() * 400 + 50
          const label = ((x < 250 && y < 250) || (x > 250 && y > 250)) ? 1 : 0
          points.push({ x, y, label })
        }
        break
    }
    
    setDataPoints(points)
    setCurrentIteration(0)
    setAccuracy(0)
    setLossHistory([])
    // 가중치 재초기화
    setModelWeights({
      w1: Math.random() - 0.5,
      w2: Math.random() - 0.5,
      b: Math.random() - 0.5
    })
  }

  // 캔버스에 데이터 그리기
  const drawCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Clear canvas
    ctx.clearRect(0, 0, 500, 500)
    
    // Draw grid
    ctx.strokeStyle = '#e5e7eb'
    ctx.lineWidth = 1
    for (let i = 0; i <= 500; i += 50) {
      ctx.beginPath()
      ctx.moveTo(i, 0)
      ctx.lineTo(i, 500)
      ctx.stroke()
      
      ctx.beginPath()
      ctx.moveTo(0, i)
      ctx.lineTo(500, i)
      ctx.stroke()
    }
    
    // Draw decision boundary
    if (currentIteration > 0) {
      drawDecisionBoundary(ctx)
    }
    
    // Draw data points
    dataPoints.forEach(point => {
      ctx.beginPath()
      ctx.arc(point.x, point.y, 5, 0, Math.PI * 2)
      ctx.fillStyle = point.label === 1 ? '#3b82f6' : '#ef4444'
      ctx.fill()
      ctx.strokeStyle = '#1f2937'
      ctx.lineWidth = 1
      ctx.stroke()
    })
  }

  // 결정 경계 그리기 (정밀한 시각화)
  const drawDecisionBoundary = (ctx: CanvasRenderingContext2D) => {
    const imageData = ctx.createImageData(500, 500)
    const data = imageData.data
    
    // 각 픽셀에 대해 예측
    for (let x = 0; x < 500; x += 2) {
      for (let y = 0; y < 500; y += 2) {
        const prediction = predictPoint(x, y)
        
        // 연속적인 확률 값을 색상으로 매핑
        let color: number[]
        if (modelParams.algorithm === 'svm') {
          // SVM의 경우 마진 영역을 표시
          if (Math.abs(prediction - 0.5) < 0.01) {
            color = [128, 128, 128, 100] // 회색 마진
          } else {
            color = prediction > 0.5 ? [59, 130, 246, 40] : [239, 68, 68, 40]
          }
        } else {
          // 다른 알고리즘의 경우 확률에 따른 그라데이션
          const intensity = Math.abs(prediction - 0.5) * 2
          if (prediction > 0.5) {
            color = [59, 130, 246, Math.floor(20 + intensity * 60)]
          } else {
            color = [239, 68, 68, Math.floor(20 + intensity * 60)]
          }
        }
        
        // 2x2 픽셀 블록으로 그리기 (성능 향상)
        for (let dx = 0; dx < 2; dx++) {
          for (let dy = 0; dy < 2; dy++) {
            const px = x + dx
            const py = y + dy
            if (px < 500 && py < 500) {
              const index = (py * 500 + px) * 4
              data[index] = color[0]
              data[index + 1] = color[1]
              data[index + 2] = color[2]
              data[index + 3] = color[3]
            }
          }
        }
      }
    }
    
    ctx.putImageData(imageData, 0, 0)
  }

  // 모델 파라미터 (학습 중 업데이트됨)
  const [modelWeights, setModelWeights] = useState<{ w1: number, w2: number, b: number }>({
    w1: Math.random() - 0.5,
    w2: Math.random() - 0.5,
    b: Math.random() - 0.5
  })

  // 실제 예측 함수
  const predictPoint = (x: number, y: number): number => {
    // 정규화
    const normX = (x - 250) / 250
    const normY = (y - 250) / 250
    
    switch (modelParams.algorithm) {
      case 'linear':
        // 선형 회귀
        const linearScore = modelWeights.w1 * normX + modelWeights.w2 * normY + modelWeights.b
        return linearScore > 0 ? 1 : 0
        
      case 'logistic':
        // 로지스틱 회귀 (시그모이드 함수)
        const logisticScore = modelWeights.w1 * normX + modelWeights.w2 * normY + modelWeights.b
        return 1 / (1 + Math.exp(-logisticScore))
        
      case 'svm':
        // SVM (간단한 RBF 커널)
        const svmScore = modelWeights.w1 * normX + modelWeights.w2 * normY + modelWeights.b
        const margin = 0.1
        if (Math.abs(svmScore) < margin) return 0.5
        return svmScore > 0 ? 1 : 0
        
      case 'decision-tree':
        // 의사결정 트리 (간단한 2레벨 트리)
        if (normX > modelWeights.w1) {
          return normY > modelWeights.w2 ? 1 : 0
        } else {
          return normY > modelWeights.b ? 1 : 0
        }
        
      default:
        return 0.5
    }
  }

  // 학습 시뮬레이션
  const trainModel = () => {
    if (currentIteration >= modelParams.iterations) {
      setIsTraining(false)
      return
    }
    
    setCurrentIteration(prev => prev + 1)
    
    // 미니배치 그라디언트 디센트
    const batchSize = Math.min(32, dataPoints.length)
    const batch = dataPoints.slice(0, batchSize)
    
    // 그라디언트 계산
    let gradW1 = 0, gradW2 = 0, gradB = 0
    
    batch.forEach(point => {
      const normX = (point.x - 250) / 250
      const normY = (point.y - 250) / 250
      const prediction = predictPoint(point.x, point.y)
      const error = (point.label || 0) - prediction
      
      // 알고리즘별 그라디언트 계산
      if (modelParams.algorithm === 'logistic') {
        // 로지스틱 회귀 그라디언트
        gradW1 += error * normX
        gradW2 += error * normY
        gradB += error
      } else if (modelParams.algorithm === 'linear') {
        // 선형 회귀 그라디언트
        gradW1 += error * normX * 2
        gradW2 += error * normY * 2
        gradB += error * 2
      } else if (modelParams.algorithm === 'svm') {
        // SVM 힌지 손실 그라디언트
        const margin = 1 - (point.label || 0) * (modelWeights.w1 * normX + modelWeights.w2 * normY + modelWeights.b)
        if (margin > 0) {
          gradW1 += -(point.label || 0) * normX
          gradW2 += -(point.label || 0) * normY
          gradB += -(point.label || 0)
        }
      } else if (modelParams.algorithm === 'decision-tree') {
        // 의사결정 트리는 다른 방식으로 업데이트
        gradW1 += error * 0.01
        gradW2 += error * 0.01
        gradB += error * 0.01
      }
    })
    
    // 가중치 업데이트
    const lr = modelParams.learningRate
    setModelWeights(prev => ({
      w1: prev.w1 + lr * gradW1 / batchSize,
      w2: prev.w2 + lr * gradW2 / batchSize,
      b: prev.b + lr * gradB / batchSize
    }))
    
    // 정확도 및 손실 계산
    let totalLoss = 0
    const correctPredictions = dataPoints.filter(point => {
      const prediction = predictPoint(point.x, point.y)
      const label = point.label || 0
      
      // 손실 계산 (크로스 엔트로피)
      if (modelParams.algorithm === 'logistic') {
        totalLoss += -(label * Math.log(prediction + 1e-7) + (1 - label) * Math.log(1 - prediction + 1e-7))
      } else {
        totalLoss += Math.pow(label - prediction, 2)
      }
      
      return (prediction > 0.5 && label === 1) || (prediction <= 0.5 && label === 0)
    }).length
    
    setAccuracy(correctPredictions / dataPoints.length * 100)
    setLossHistory(prev => [...prev.slice(-99), totalLoss / dataPoints.length])
  }

  useEffect(() => {
    generateDataset(selectedDataset)
  }, [selectedDataset])

  useEffect(() => {
    drawCanvas()
  }, [dataPoints, currentIteration])

  useEffect(() => {
    if (isTraining) {
      const interval = setInterval(trainModel, 50)
      return () => clearInterval(interval)
    }
  }, [isTraining, currentIteration])

  // 캔버스 클릭으로 데이터 추가
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (selectedDataset !== 'custom') return
    
    const canvas = canvasRef.current
    if (!canvas) return
    
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    // Shift 키를 누르면 label 1, 아니면 0
    const label = e.shiftKey ? 1 : 0
    
    setDataPoints([...dataPoints, { x, y, label }])
  }

  return (
    <div className="w-full max-w-6xl mx-auto">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-6">머신러닝 알고리즘 플레이그라운드</h2>
        
        <div className="grid lg:grid-cols-2 gap-6">
          {/* 캔버스 영역 */}
          <div>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <canvas
                ref={canvasRef}
                width={500}
                height={500}
                className="border border-gray-300 dark:border-gray-600 rounded cursor-crosshair"
                onClick={handleCanvasClick}
              />
              
              {selectedDataset === 'custom' && (
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  클릭: 빨간 점 추가 | Shift+클릭: 파란 점 추가
                </p>
              )}
            </div>
            
            {/* 컨트롤 버튼 */}
            <div className="flex gap-2 mt-4">
              <button
                onClick={() => setIsTraining(!isTraining)}
                disabled={dataPoints.length === 0}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                  isTraining
                    ? 'bg-red-500 text-white hover:bg-red-600'
                    : 'bg-green-500 text-white hover:bg-green-600'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {isTraining ? (
                  <>
                    <Pause className="w-4 h-4" />
                    일시정지
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    학습 시작
                  </>
                )}
              </button>
              
              <button
                onClick={() => {
                  setCurrentIteration(0)
                  setAccuracy(0)
                  setIsTraining(false)
                  setLossHistory([])
                  setModelWeights({
                    w1: Math.random() - 0.5,
                    w2: Math.random() - 0.5,
                    b: Math.random() - 0.5
                  })
                }}
                className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded-lg font-medium hover:bg-gray-600 transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                초기화
              </button>
              
              <button
                onClick={() => {
                  const data = {
                    dataset: dataPoints,
                    model: {
                      algorithm: modelParams.algorithm,
                      weights: modelWeights,
                      accuracy: accuracy,
                      iterations: currentIteration
                    },
                    lossHistory: lossHistory
                  }
                  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `ml-playground-${Date.now()}.json`
                  a.click()
                  URL.revokeObjectURL(url)
                }}
                className="flex items-center gap-2 px-4 py-2 bg-purple-500 text-white rounded-lg font-medium hover:bg-purple-600 transition-colors"
              >
                <Download className="w-4 h-4" />
                내보내기
              </button>
            </div>
          </div>
          
          {/* 설정 패널 */}
          <div className="space-y-6">
            {/* 데이터셋 선택 */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Settings className="w-5 h-5" />
                데이터셋
              </h3>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { id: 'linear', name: '선형 분리' },
                  { id: 'circular', name: '원형' },
                  { id: 'xor', name: 'XOR' },
                  { id: 'custom', name: '사용자 정의' }
                ].map(dataset => (
                  <button
                    key={dataset.id}
                    onClick={() => setSelectedDataset(dataset.id as any)}
                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                      selectedDataset === dataset.id
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                    }`}
                  >
                    {dataset.name}
                  </button>
                ))}
              </div>
            </div>
            
            {/* 알고리즘 선택 */}
            <div>
              <h3 className="text-lg font-semibold mb-3">알고리즘</h3>
              <select
                value={modelParams.algorithm}
                onChange={(e) => setModelParams({...modelParams, algorithm: e.target.value as any})}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                <option value="linear">선형 회귀</option>
                <option value="logistic">로지스틱 회귀</option>
                <option value="svm">SVM</option>
                <option value="decision-tree">의사결정 트리</option>
              </select>
            </div>
            
            {/* 하이퍼파라미터 */}
            <div>
              <h3 className="text-lg font-semibold mb-3">하이퍼파라미터</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">
                    학습률: {modelParams.learningRate}
                  </label>
                  <input
                    type="range"
                    min="0.001"
                    max="0.1"
                    step="0.001"
                    value={modelParams.learningRate}
                    onChange={(e) => setModelParams({...modelParams, learningRate: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">
                    반복 횟수: {modelParams.iterations}
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="1000"
                    step="10"
                    value={modelParams.iterations}
                    onChange={(e) => setModelParams({...modelParams, iterations: parseInt(e.target.value)})}
                    className="w-full"
                  />
                </div>
              </div>
            </div>
            
            {/* 학습 진행 상황 */}
            <div>
              <h3 className="text-lg font-semibold mb-3">학습 진행</h3>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>진행률</span>
                    <span>{Math.round((currentIteration / modelParams.iterations) * 100)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${(currentIteration / modelParams.iterations) * 100}%` }}
                    />
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between text-sm">
                    <span>반복 횟수</span>
                    <span>{currentIteration} / {modelParams.iterations}</span>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between text-sm">
                    <span>정확도</span>
                    <span className="font-semibold text-green-600 dark:text-green-400">
                      {accuracy.toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* 손실 그래프 */}
            {lossHistory.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <TrendingDown className="w-5 h-5" />
                  손실 그래프
                </h3>
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <div className="h-32 relative">
                    <svg className="w-full h-full">
                      <polyline
                        fill="none"
                        stroke="url(#loss-gradient)"
                        strokeWidth="2"
                        points={lossHistory.map((loss, i) => {
                          const x = (i / Math.max(lossHistory.length - 1, 1)) * 100
                          const y = 100 - (Math.min(loss, 5) / 5) * 100
                          return `${x},${y}`
                        }).join(' ')}
                      />
                      <defs>
                        <linearGradient id="loss-gradient">
                          <stop offset="0%" stopColor="#ef4444" />
                          <stop offset="100%" stopColor="#f59e0b" />
                        </linearGradient>
                      </defs>
                    </svg>
                    <div className="absolute bottom-0 left-0 text-xs text-gray-500">0</div>
                    <div className="absolute bottom-0 right-0 text-xs text-gray-500">{lossHistory.length}</div>
                    <div className="absolute top-0 left-0 text-xs text-gray-500">
                      {lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].toFixed(3) : '0'}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 고급 설정 */}
            <div>
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="w-full flex items-center justify-between text-lg font-semibold mb-3 hover:text-blue-600 transition-colors"
              >
                <span>고급 설정</span>
                {showAdvanced ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </button>
              
              {showAdvanced && (
                <div className="space-y-4 bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">
                      정규화 파라미터 (SVM C값): {modelParams.c || 1.0}
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="10"
                      step="0.1"
                      value={modelParams.c || 1.0}
                      onChange={(e) => setModelParams({...modelParams, c: parseFloat(e.target.value)})}
                      className="w-full"
                      disabled={modelParams.algorithm !== 'svm'}
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium mb-1">
                      트리 최대 깊이: {modelParams.maxDepth || 3}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      step="1"
                      value={modelParams.maxDepth || 3}
                      onChange={(e) => setModelParams({...modelParams, maxDepth: parseInt(e.target.value)})}
                      className="w-full"
                      disabled={modelParams.algorithm !== 'decision-tree'}
                    />
                  </div>
                  
                  <div className="pt-2 border-t border-gray-300 dark:border-gray-600">
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">현재 가중치:</p>
                    <div className="font-mono text-xs space-y-1">
                      <div>w1: {modelWeights.w1.toFixed(4)}</div>
                      <div>w2: {modelWeights.w2.toFixed(4)}</div>
                      <div>b: {modelWeights.b.toFixed(4)}</div>
                    </div>
                  </div>
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
                <li>• 데이터셋을 선택하거나 직접 그려보세요</li>
                <li>• 알고리즘과 파라미터를 조정하세요</li>
                <li>• 학습 시작을 눌러 모델이 학습하는 과정을 관찰하세요</li>
                <li>• 결정 경계가 어떻게 변하는지 확인하세요</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}