'use client'

import { useState, useEffect, useRef } from 'react'
import { Play, Pause, RotateCcw, Settings, Info, TrendingUp, Brain, Sparkles } from 'lucide-react'

interface DataPoint {
  x: number
  y: number
  label: number
}

interface ModelWeights {
  w1: number
  w2: number
  b: number
}

export default function MLPlayground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([])
  const [algorithm, setAlgorithm] = useState<'linear' | 'logistic' | 'neural'>('logistic')
  const [isTraining, setIsTraining] = useState(false)
  const [learningRate, setLearningRate] = useState(0.1)
  const [iterations, setIterations] = useState(1000)
  const [currentIteration, setCurrentIteration] = useState(0)
  const [accuracy, setAccuracy] = useState(0)
  const [loss, setLoss] = useState(0)
  const [dataset, setDataset] = useState<'linear' | 'circular' | 'xor' | 'spiral'>('circular')
  const [weights, setWeights] = useState<ModelWeights>({ w1: 0, w2: 0, b: 0 })
  const [lossHistory, setLossHistory] = useState<number[]>([])
  // 신경망 가중치: [입력->은닉층, 은닉층->출력]
  const [neuralWeights, setNeuralWeights] = useState({
    hidden: [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], // 4개 은닉 뉴런
    output: [0, 0, 0, 0, 0] // 출력 가중치 + bias
  })
  const animationRef = useRef<number>()

  // 데이터셋 생성
  const generateDataset = (type: string) => {
    const points: DataPoint[] = []
    
    switch(type) {
      case 'linear':
        for (let i = 0; i < 200; i++) {
          const x = Math.random() * 400 + 50
          const y = Math.random() * 400 + 50
          const label = x + y > 500 ? 1 : 0
          points.push({ x, y, label })
        }
        break
        
      case 'circular':
        for (let i = 0; i < 200; i++) {
          const angle = Math.random() * Math.PI * 2
          const r1 = Math.random() * 80 + 20
          const r2 = Math.random() * 80 + 120
          
          if (i < 100) {
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
        for (let i = 0; i < 200; i++) {
          const x = Math.random() * 300 + 100
          const y = Math.random() * 300 + 100
          const label = ((x < 250 && y < 250) || (x > 250 && y > 250)) ? 1 : 0
          points.push({ x, y, label })
        }
        break
        
      case 'spiral':
        for (let i = 0; i < 200; i++) {
          const angle = (i / 100) * Math.PI * 4
          const r = (i / 200) * 150 + 50
          const noise = (Math.random() - 0.5) * 50
          
          if (i < 100) {
            points.push({
              x: 250 + (r + noise) * Math.cos(angle),
              y: 250 + (r + noise) * Math.sin(angle),
              label: 0
            })
          } else {
            points.push({
              x: 250 + (r + noise) * Math.cos(angle + Math.PI),
              y: 250 + (r + noise) * Math.sin(angle + Math.PI),
              label: 1
            })
          }
        }
        break
    }
    
    setDataPoints(points)
    resetModel()
  }

  // 모델 초기화
  const resetModel = () => {
    setWeights({
      w1: (Math.random() - 0.5) * 2,
      w2: (Math.random() - 0.5) * 2,
      b: (Math.random() - 0.5) * 2
    })
    // Xavier 초기화
    const scale = Math.sqrt(2 / 2) // 2 inputs
    setNeuralWeights({
      hidden: Array(4).fill(0).map(() => [
        (Math.random() - 0.5) * 2 * scale,
        (Math.random() - 0.5) * 2 * scale,
        (Math.random() - 0.5) * 2 * scale
      ]),
      output: Array(5).fill(0).map(() => (Math.random() - 0.5) * 2 * Math.sqrt(2 / 4))
    })
    setCurrentIteration(0)
    setAccuracy(0)
    setLoss(0)
    setLossHistory([])
  }

  // 예측 함수
  const predict = (x: number, y: number): number => {
    // 정규화
    const normX = (x - 250) / 250
    const normY = (y - 250) / 250
    
    switch (algorithm) {
      case 'linear':
        const linearScore = weights.w1 * normX + weights.w2 * normY + weights.b
        return linearScore > 0 ? 1 : 0
        
      case 'logistic':
        const logisticScore = weights.w1 * normX + weights.w2 * normY + weights.b
        return 1 / (1 + Math.exp(-logisticScore))
        
      case 'neural':
        // 4개 은닉 뉴런을 가진 신경망
        const hidden = []
        for (let i = 0; i < 4; i++) {
          const z = neuralWeights.hidden[i][0] * normX + 
                   neuralWeights.hidden[i][1] * normY + 
                   neuralWeights.hidden[i][2]
          hidden.push(Math.tanh(z)) // tanh 활성화
        }
        
        // 출력층
        let output = neuralWeights.output[4] // bias
        for (let i = 0; i < 4; i++) {
          output += neuralWeights.output[i] * hidden[i]
        }
        return 1 / (1 + Math.exp(-output)) // sigmoid
        
      default:
        return 0.5
    }
  }

  // 학습 스텝
  const trainStep = () => {
    if (dataPoints.length === 0) return
    
    // 미니배치
    const batchSize = 32
    const batch = dataPoints.slice(
      (currentIteration * batchSize) % dataPoints.length,
      ((currentIteration + 1) * batchSize) % dataPoints.length
    )
    
    let totalLoss = 0
    let gradW1 = 0, gradW2 = 0, gradB = 0
    const hiddenGrads = Array(4).fill(0).map(() => [0, 0, 0])
    const outputGrads = [0, 0, 0, 0, 0]
    
    batch.forEach(point => {
      const normX = (point.x - 250) / 250
      const normY = (point.y - 250) / 250
      const prediction = predict(point.x, point.y)
      const error = point.label - prediction
      
      if (algorithm === 'linear' || algorithm === 'logistic') {
        gradW1 += error * normX
        gradW2 += error * normY
        gradB += error
      } else if (algorithm === 'neural') {
        // 순전파로 은닉층 계산
        const hidden = []
        for (let i = 0; i < 4; i++) {
          const z = neuralWeights.hidden[i][0] * normX + 
                   neuralWeights.hidden[i][1] * normY + 
                   neuralWeights.hidden[i][2]
          hidden.push(Math.tanh(z))
        }
        
        // 출력층 그라디언트
        for (let i = 0; i < 4; i++) {
          outputGrads[i] += error * hidden[i]
        }
        outputGrads[4] += error // bias
        
        // 역전파: 은닉층 그라디언트
        for (let i = 0; i < 4; i++) {
          const dh = error * neuralWeights.output[i] * (1 - hidden[i] * hidden[i])
          hiddenGrads[i][0] += dh * normX
          hiddenGrads[i][1] += dh * normY
          hiddenGrads[i][2] += dh // bias
        }
      }
      
      // 손실 계산
      totalLoss += -point.label * Math.log(prediction + 1e-7) - (1 - point.label) * Math.log(1 - prediction + 1e-7)
    })
    
    // 가중치 업데이트
    const lr = learningRate / batch.length
    setWeights(prev => ({
      w1: prev.w1 + lr * gradW1,
      w2: prev.w2 + lr * gradW2,
      b: prev.b + lr * gradB
    }))
    
    if (algorithm === 'neural') {
      setNeuralWeights(prev => ({
        hidden: prev.hidden.map((weights, i) => [
          weights[0] + lr * hiddenGrads[i][0],
          weights[1] + lr * hiddenGrads[i][1],
          weights[2] + lr * hiddenGrads[i][2]
        ]),
        output: prev.output.map((weight, i) => weight + lr * outputGrads[i])
      }))
    }
    
    // 전체 데이터셋에 대한 정확도 계산
    let correct = 0
    let totalDataLoss = 0
    dataPoints.forEach(point => {
      const pred = predict(point.x, point.y)
      const predLabel = pred > 0.5 ? 1 : 0
      if (predLabel === point.label) correct++
      totalDataLoss += -point.label * Math.log(pred + 1e-7) - (1 - point.label) * Math.log(1 - pred + 1e-7)
    })
    
    setAccuracy(correct / dataPoints.length)
    setLoss(totalDataLoss / dataPoints.length)
    setLossHistory(prev => [...prev.slice(-99), totalDataLoss / dataPoints.length])
  }

  // 학습 루프
  useEffect(() => {
    if (!isTraining) return
    
    const animate = () => {
      if (currentIteration < iterations) {
        trainStep()
        setCurrentIteration(prev => prev + 1)
        animationRef.current = requestAnimationFrame(animate)
      } else {
        setIsTraining(false)
      }
    }
    
    animate()
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isTraining, currentIteration, iterations])

  // 캔버스 그리기
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // 캔버스 크기 설정
    canvas.width = 500
    canvas.height = 500
    
    // 배경
    ctx.fillStyle = '#f3f4f6'
    ctx.fillRect(0, 0, 500, 500)
    
    // Decision boundary 그리기
    const resolution = 5
    for (let x = 0; x < 500; x += resolution) {
      for (let y = 0; y < 500; y += resolution) {
        const prediction = predict(x, y)
        const intensity = Math.floor(prediction * 255)
        ctx.fillStyle = `rgb(${255 - intensity}, ${intensity}, ${intensity})`
        ctx.fillRect(x, y, resolution, resolution)
      }
    }
    
    // 데이터 포인트 그리기
    dataPoints.forEach(point => {
      ctx.beginPath()
      ctx.arc(point.x, point.y, 5, 0, Math.PI * 2)
      ctx.fillStyle = point.label === 1 ? '#3b82f6' : '#ef4444'
      ctx.fill()
      ctx.strokeStyle = '#000'
      ctx.lineWidth = 1
      ctx.stroke()
    })
    
    // 좌표축
    ctx.strokeStyle = '#6b7280'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(50, 250)
    ctx.lineTo(450, 250)
    ctx.moveTo(250, 50)
    ctx.lineTo(250, 450)
    ctx.stroke()
    
  }, [dataPoints, weights, neuralWeights, algorithm])

  // Loss 그래프 그리기
  const drawLossGraph = () => {
    if (lossHistory.length < 2) return null
    
    const maxLoss = Math.max(...lossHistory)
    const minLoss = Math.min(...lossHistory)
    const range = maxLoss - minLoss || 1
    
    return (
      <svg width="100%" height="100" className="mt-2">
        <polyline
          fill="none"
          stroke="#3b82f6"
          strokeWidth="2"
          points={lossHistory.map((loss, i) => 
            `${(i / (lossHistory.length - 1)) * 380},${80 - ((loss - minLoss) / range) * 60}`
          ).join(' ')}
        />
        <text x="0" y="95" className="text-xs fill-gray-500">0</text>
        <text x="350" y="95" className="text-xs fill-gray-500">{lossHistory.length}</text>
      </svg>
    )
  }

  // 초기 데이터셋 생성
  useEffect(() => {
    generateDataset(dataset)
  }, [])

  return (
    <div className="flex flex-col h-full bg-gray-50 dark:bg-gray-900">
      {/* 헤더 */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Brain className="w-6 h-6" />
          머신러닝 플레이그라운드
        </h2>
        <p className="text-blue-100 mt-1">실시간 학습 과정과 Decision Boundary 시각화</p>
      </div>

      {/* 메인 컨텐츠 */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-4 p-4">
        {/* 캔버스 영역 */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold">Decision Boundary</h3>
            <div className="flex gap-2">
              <button
                onClick={() => setIsTraining(!isTraining)}
                disabled={dataPoints.length === 0}
                className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
                  isTraining 
                    ? 'bg-red-500 hover:bg-red-600 text-white'
                    : 'bg-green-500 hover:bg-green-600 text-white'
                } disabled:opacity-50`}
              >
                {isTraining ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isTraining ? '정지' : '학습 시작'}
              </button>
              <button
                onClick={() => {
                  setIsTraining(false)
                  generateDataset(dataset)
                }}
                className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg flex items-center gap-2"
              >
                <RotateCcw className="w-4 h-4" />
                리셋
              </button>
            </div>
          </div>
          
          <canvas
            ref={canvasRef}
            className="border border-gray-300 dark:border-gray-600 rounded mx-auto"
            width={500}
            height={500}
          />
          
          <div className="mt-4 grid grid-cols-2 gap-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600">{(accuracy * 100).toFixed(1)}%</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">정확도</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600">{loss.toFixed(3)}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">손실</div>
            </div>
          </div>
        </div>

        {/* 컨트롤 패널 */}
        <div className="space-y-4">
          {/* 데이터셋 선택 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
            <h3 className="text-lg font-semibold mb-4">데이터셋</h3>
            <div className="grid grid-cols-2 gap-2">
              {['linear', 'circular', 'xor', 'spiral'].map((type) => (
                <button
                  key={type}
                  onClick={() => {
                    setDataset(type as any)
                    generateDataset(type)
                  }}
                  className={`px-3 py-2 rounded text-sm capitalize ${
                    dataset === type
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600'
                  }`}
                >
                  {type === 'xor' ? 'XOR' : type}
                </button>
              ))}
            </div>
          </div>

          {/* 알고리즘 선택 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
            <h3 className="text-lg font-semibold mb-4">알고리즘</h3>
            <div className="space-y-2">
              {[
                { value: 'linear', label: '선형 회귀', desc: '가장 간단한 모델' },
                { value: 'logistic', label: '로지스틱 회귀', desc: '확률적 분류' },
                { value: 'neural', label: '신경망', desc: '2층 신경망' }
              ].map((algo) => (
                <label
                  key={algo.value}
                  className={`block p-3 rounded cursor-pointer border ${
                    algorithm === algo.value
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800'
                  }`}
                >
                  <input
                    type="radio"
                    name="algorithm"
                    value={algo.value}
                    checked={algorithm === algo.value}
                    onChange={(e) => {
                      setAlgorithm(e.target.value as any)
                      resetModel()
                    }}
                    className="sr-only"
                  />
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium">{algo.label}</div>
                      <div className="text-xs text-gray-500">{algo.desc}</div>
                    </div>
                    {algorithm === algo.value && <Sparkles className="w-4 h-4 text-blue-500" />}
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* 하이퍼파라미터 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Settings className="w-5 h-5" />
              하이퍼파라미터
            </h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">
                  학습률: {learningRate.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0.01"
                  max="1"
                  step="0.01"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">
                  반복 횟수: {iterations}
                </label>
                <input
                  type="range"
                  min="100"
                  max="5000"
                  step="100"
                  value={iterations}
                  onChange={(e) => setIterations(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>
              
              <div className="text-sm text-gray-600 dark:text-gray-400">
                진행률: {currentIteration} / {iterations}
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${(currentIteration / iterations) * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* 손실 그래프 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
            <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              손실 함수
            </h3>
            {drawLossGraph()}
          </div>

          {/* 정보 */}
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <h4 className="flex items-center gap-2 font-semibold text-blue-700 dark:text-blue-300 mb-2">
              <Info className="w-4 h-4" />
              사용법
            </h4>
            <ul className="text-xs text-blue-600 dark:text-blue-400 space-y-1">
              <li>• 데이터셋을 선택하고 학습 시작</li>
              <li>• Decision Boundary가 실시간으로 변화</li>
              <li>• 빨간점: 클래스 0, 파란점: 클래스 1</li>
              <li>• XOR은 신경망으로만 해결 가능</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}