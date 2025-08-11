'use client'

import { useState, useEffect, useRef, Fragment } from 'react'
import React from 'react'
import { Play, Pause, RotateCcw, Plus, Minus, Layers, Zap, Brain, Activity, Settings, Download } from 'lucide-react'

interface Neuron {
  id: string
  value: number
  bias: number
  activation: number
}

interface Layer {
  id: string
  neurons: Neuron[]
  type: 'input' | 'hidden' | 'output'
}

interface Connection {
  from: string
  to: string
  weight: number
}

interface NetworkConfig {
  learningRate: number
  activation: 'sigmoid' | 'relu' | 'tanh'
  epochs: number
  batchSize: number
}

export default function NeuralNetworkBuilder() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [layers, setLayers] = useState<Layer[]>([
    {
      id: 'input',
      type: 'input',
      neurons: [
        { id: 'i1', value: 0, bias: 0, activation: 0 },
        { id: 'i2', value: 0, bias: 0, activation: 0 }
      ]
    },
    {
      id: 'hidden1',
      type: 'hidden',
      neurons: [
        { id: 'h1', value: 0, bias: Math.random() - 0.5, activation: 0 },
        { id: 'h2', value: 0, bias: Math.random() - 0.5, activation: 0 },
        { id: 'h3', value: 0, bias: Math.random() - 0.5, activation: 0 }
      ]
    },
    {
      id: 'output',
      type: 'output',
      neurons: [
        { id: 'o1', value: 0, bias: Math.random() - 0.5, activation: 0 }
      ]
    }
  ])
  
  const [connections, setConnections] = useState<Connection[]>([])
  const [isTraining, setIsTraining] = useState(false)
  const [currentEpoch, setCurrentEpoch] = useState(0)
  const [loss, setLoss] = useState(0)
  const [accuracy, setAccuracy] = useState(0)
  const [lossHistory, setLossHistory] = useState<number[]>([])
  
  const [config, setConfig] = useState<NetworkConfig>({
    learningRate: 0.1,
    activation: 'sigmoid',
    epochs: 100,
    batchSize: 10
  })
  
  const [selectedDataset, setSelectedDataset] = useState<'xor' | 'and' | 'or' | 'custom'>('xor')
  const [trainingData, setTrainingData] = useState<{inputs: number[], output: number}[]>([])
  
  // 초기 연결 생성
  useEffect(() => {
    const newConnections: Connection[] = []
    
    for (let i = 0; i < layers.length - 1; i++) {
      const fromLayer = layers[i]
      const toLayer = layers[i + 1]
      
      fromLayer.neurons.forEach(fromNeuron => {
        toLayer.neurons.forEach(toNeuron => {
          newConnections.push({
            from: fromNeuron.id,
            to: toNeuron.id,
            weight: (Math.random() - 0.5) * 2
          })
        })
      })
    }
    
    setConnections(newConnections)
  }, [])
  
  // 훈련 데이터 생성
  useEffect(() => {
    let data: {inputs: number[], output: number}[] = []
    
    switch (selectedDataset) {
      case 'xor':
        data = [
          { inputs: [0, 0], output: 0 },
          { inputs: [0, 1], output: 1 },
          { inputs: [1, 0], output: 1 },
          { inputs: [1, 1], output: 0 }
        ]
        break
      case 'and':
        data = [
          { inputs: [0, 0], output: 0 },
          { inputs: [0, 1], output: 0 },
          { inputs: [1, 0], output: 0 },
          { inputs: [1, 1], output: 1 }
        ]
        break
      case 'or':
        data = [
          { inputs: [0, 0], output: 0 },
          { inputs: [0, 1], output: 1 },
          { inputs: [1, 0], output: 1 },
          { inputs: [1, 1], output: 1 }
        ]
        break
    }
    
    setTrainingData(data)
  }, [selectedDataset])
  
  // 활성화 함수
  const activationFunction = (x: number): number => {
    switch (config.activation) {
      case 'sigmoid':
        return 1 / (1 + Math.exp(-x))
      case 'relu':
        return Math.max(0, x)
      case 'tanh':
        return Math.tanh(x)
      default:
        return x
    }
  }
  
  // 활성화 함수 미분
  const activationDerivative = (x: number): number => {
    switch (config.activation) {
      case 'sigmoid':
        const sig = activationFunction(x)
        return sig * (1 - sig)
      case 'relu':
        return x > 0 ? 1 : 0
      case 'tanh':
        const tanh = Math.tanh(x)
        return 1 - tanh * tanh
      default:
        return 1
    }
  }
  
  // 순전파
  const forwardPass = (inputs: number[]) => {
    const newLayers = [...layers]
    
    // 입력층 설정
    newLayers[0].neurons.forEach((neuron, i) => {
      neuron.value = inputs[i] || 0
      neuron.activation = neuron.value
    })
    
    // 은닉층과 출력층 계산
    for (let l = 1; l < newLayers.length; l++) {
      newLayers[l].neurons.forEach(neuron => {
        let sum = neuron.bias
        
        connections
          .filter(conn => conn.to === neuron.id)
          .forEach(conn => {
            const fromNeuron = newLayers.flat().flatMap(layer => layer.neurons).find(n => n.id === conn.from)
            if (fromNeuron) {
              sum += fromNeuron.activation * conn.weight
            }
          })
        
        neuron.value = sum
        neuron.activation = activationFunction(sum)
      })
    }
    
    setLayers(newLayers)
    return newLayers[newLayers.length - 1].neurons[0].activation
  }
  
  // 역전파
  const backwardPass = (target: number) => {
    const output = layers[layers.length - 1].neurons[0].activation
    const error = target - output
    
    // 각 뉴런의 그라디언트 계산
    const gradients: { [key: string]: number } = {}
    
    // 출력층 그라디언트
    const outputNeuron = layers[layers.length - 1].neurons[0]
    gradients[outputNeuron.id] = error * activationDerivative(outputNeuron.value)
    
    // 은닉층 그라디언트 (역순으로)
    for (let l = layers.length - 2; l >= 1; l--) {
      layers[l].neurons.forEach(neuron => {
        let errorSum = 0
        
        connections
          .filter(conn => conn.from === neuron.id)
          .forEach(conn => {
            errorSum += gradients[conn.to] * conn.weight
          })
        
        gradients[neuron.id] = errorSum * activationDerivative(neuron.value)
      })
    }
    
    // 가중치 업데이트
    const newConnections = connections.map(conn => {
      const fromNeuron = layers.flat().flatMap(layer => layer.neurons).find(n => n.id === conn.from)
      const gradient = gradients[conn.to]
      
      if (fromNeuron && gradient !== undefined) {
        return {
          ...conn,
          weight: conn.weight + config.learningRate * gradient * fromNeuron.activation
        }
      }
      return conn
    })
    
    // 편향 업데이트
    const newLayers = layers.map(layer => ({
      ...layer,
      neurons: layer.neurons.map(neuron => ({
        ...neuron,
        bias: neuron.bias + (gradients[neuron.id] || 0) * config.learningRate
      }))
    }))
    
    setConnections(newConnections)
    setLayers(newLayers)
    
    return Math.abs(error)
  }
  
  // 학습 단계
  const trainStep = () => {
    if (currentEpoch >= config.epochs) {
      setIsTraining(false)
      return
    }
    
    let totalLoss = 0
    let correct = 0
    
    // 모든 훈련 데이터에 대해
    trainingData.forEach(data => {
      const output = forwardPass(data.inputs)
      const loss = backwardPass(data.output)
      totalLoss += loss
      
      if ((output > 0.5 && data.output === 1) || (output <= 0.5 && data.output === 0)) {
        correct++
      }
    })
    
    const avgLoss = totalLoss / trainingData.length
    setLoss(avgLoss)
    setAccuracy((correct / trainingData.length) * 100)
    setLossHistory(prev => [...prev.slice(-99), avgLoss])
    setCurrentEpoch(prev => prev + 1)
  }
  
  // 신경망 그리기
  const drawNetwork = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    ctx.clearRect(0, 0, 800, 500)
    
    const layerSpacing = 800 / (layers.length + 1)
    const nodeRadius = 25
    
    // 뉴런 위치 계산
    const nodePositions: { [key: string]: { x: number, y: number } } = {}
    
    layers.forEach((layer, layerIndex) => {
      const x = layerSpacing * (layerIndex + 1)
      const spacing = 500 / (layer.neurons.length + 1)
      
      layer.neurons.forEach((neuron, neuronIndex) => {
        const y = spacing * (neuronIndex + 1)
        nodePositions[neuron.id] = { x, y }
      })
    })
    
    // 연결 그리기
    connections.forEach(conn => {
      const from = nodePositions[conn.from]
      const to = nodePositions[conn.to]
      
      if (from && to) {
        ctx.beginPath()
        ctx.moveTo(from.x, from.y)
        ctx.lineTo(to.x, to.y)
        
        // 가중치에 따른 색상
        const weight = Math.abs(conn.weight)
        const opacity = Math.min(weight, 1)
        ctx.strokeStyle = conn.weight > 0 
          ? `rgba(59, 130, 246, ${opacity})` 
          : `rgba(239, 68, 68, ${opacity})`
        ctx.lineWidth = Math.min(weight * 3, 5)
        ctx.stroke()
        
        // 가중치 표시
        const midX = (from.x + to.x) / 2
        const midY = (from.y + to.y) / 2
        ctx.fillStyle = '#666'
        ctx.font = '10px sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText(conn.weight.toFixed(2), midX, midY)
      }
    })
    
    // 뉴런 그리기
    layers.forEach((layer, layerIndex) => {
      layer.neurons.forEach(neuron => {
        const pos = nodePositions[neuron.id]
        if (!pos) return
        
        // 뉴런 원
        ctx.beginPath()
        ctx.arc(pos.x, pos.y, nodeRadius, 0, Math.PI * 2)
        
        // 활성화 값에 따른 색상
        const activation = neuron.activation
        const intensity = Math.floor(activation * 255)
        ctx.fillStyle = `rgb(${255 - intensity}, ${255 - intensity}, 255)`
        ctx.fill()
        
        ctx.strokeStyle = '#3b82f6'
        ctx.lineWidth = 2
        ctx.stroke()
        
        // 뉴런 ID
        ctx.fillStyle = '#333'
        ctx.font = 'bold 12px sans-serif'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(neuron.id.toUpperCase(), pos.x, pos.y)
        
        // 활성화 값
        ctx.fillStyle = '#666'
        ctx.font = '10px sans-serif'
        ctx.fillText(activation.toFixed(3), pos.x, pos.y + nodeRadius + 10)
        
        // 레이어 레이블
        if (neuron === layer.neurons[0]) {
          ctx.fillStyle = '#333'
          ctx.font = 'bold 14px sans-serif'
          ctx.fillText(
            layer.type === 'input' ? '입력층' : 
            layer.type === 'hidden' ? '은닉층' : '출력층',
            pos.x, 30
          )
        }
      })
    })
  }
  
  // 레이어 추가/제거
  const addLayer = () => {
    const newLayerId = `hidden${layers.length - 1}`
    const newLayer: Layer = {
      id: newLayerId,
      type: 'hidden',
      neurons: [
        { id: `${newLayerId}_1`, value: 0, bias: Math.random() - 0.5, activation: 0 },
        { id: `${newLayerId}_2`, value: 0, bias: Math.random() - 0.5, activation: 0 },
        { id: `${newLayerId}_3`, value: 0, bias: Math.random() - 0.5, activation: 0 }
      ]
    }
    
    const newLayers = [...layers.slice(0, -1), newLayer, layers[layers.length - 1]]
    setLayers(newLayers)
  }
  
  const removeLayer = () => {
    if (layers.length > 3) {
      const newLayers = [...layers.slice(0, -2), layers[layers.length - 1]]
      setLayers(newLayers)
    }
  }
  
  // 뉴런 추가/제거
  const addNeuron = (layerIndex: number) => {
    const layer = layers[layerIndex]
    if (layer.type !== 'input' && layer.type !== 'output') {
      const newNeuronId = `${layer.id}_${layer.neurons.length + 1}`
      const newNeuron: Neuron = {
        id: newNeuronId,
        value: 0,
        bias: Math.random() - 0.5,
        activation: 0
      }
      
      const newLayers = [...layers]
      newLayers[layerIndex].neurons.push(newNeuron)
      setLayers(newLayers)
    }
  }
  
  const removeNeuron = (layerIndex: number) => {
    const layer = layers[layerIndex]
    if (layer.type !== 'input' && layer.type !== 'output' && layer.neurons.length > 1) {
      const newLayers = [...layers]
      newLayers[layerIndex].neurons.pop()
      setLayers(newLayers)
    }
  }
  
  useEffect(() => {
    drawNetwork()
  }, [layers, connections])
  
  useEffect(() => {
    if (isTraining) {
      const interval = setInterval(trainStep, 50)
      return () => clearInterval(interval)
    }
  }, [isTraining, currentEpoch])
  
  return (
    <div className="w-full max-w-7xl mx-auto">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-6">신경망 빌더</h2>
        
        <div className="grid lg:grid-cols-3 gap-6">
          {/* 신경망 시각화 */}
          <div className="lg:col-span-2">
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <canvas
                ref={canvasRef}
                width={800}
                height={500}
                className="border border-gray-300 dark:border-gray-600 rounded"
              />
            </div>
            
            {/* 컨트롤 버튼 */}
            <div className="flex gap-2 mt-4">
              <button
                onClick={() => setIsTraining(!isTraining)}
                disabled={trainingData.length === 0}
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
                  setCurrentEpoch(0)
                  setLoss(0)
                  setAccuracy(0)
                  setLossHistory([])
                  setIsTraining(false)
                  // 가중치 재초기화
                  setConnections(connections.map(conn => ({
                    ...conn,
                    weight: (Math.random() - 0.5) * 2
                  })))
                  setLayers(layers.map(layer => ({
                    ...layer,
                    neurons: layer.neurons.map(neuron => ({
                      ...neuron,
                      bias: layer.type === 'input' ? 0 : Math.random() - 0.5
                    }))
                  })))
                }}
                className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded-lg font-medium hover:bg-gray-600 transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                초기화
              </button>
              
              <div className="flex-1" />
              
              <button
                onClick={addLayer}
                className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 transition-colors"
              >
                <Plus className="w-4 h-4" />
                레이어 추가
              </button>
              
              <button
                onClick={removeLayer}
                disabled={layers.length <= 3}
                className="flex items-center gap-2 px-4 py-2 bg-red-500 text-white rounded-lg font-medium hover:bg-red-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Minus className="w-4 h-4" />
                레이어 제거
              </button>
            </div>
            
            {/* 데이터 미리보기 */}
            <div className="mt-4 bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-2">훈련 데이터</h4>
              <div className="grid grid-cols-4 gap-2 text-sm">
                <div className="font-semibold">입력 1</div>
                <div className="font-semibold">입력 2</div>
                <div className="font-semibold">출력</div>
                <div className="font-semibold">예측</div>
                {trainingData.map((data, i) => {
                  const prediction = forwardPass(data.inputs)
                  return (
                    <React.Fragment key={i}>
                      <div>{data.inputs[0]}</div>
                      <div>{data.inputs[1]}</div>
                      <div>{data.output}</div>
                      <div className={Math.abs(prediction - data.output) < 0.5 ? 'text-green-600' : 'text-red-600'}>
                        {prediction.toFixed(3)}
                      </div>
                    </React.Fragment>
                  )
                })}
              </div>
            </div>
          </div>
          
          {/* 설정 패널 */}
          <div className="space-y-6">
            {/* 데이터셋 선택 */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Brain className="w-5 h-5" />
                데이터셋
              </h3>
              <div className="grid grid-cols-3 gap-2">
                {['xor', 'and', 'or'].map(dataset => (
                  <button
                    key={dataset}
                    onClick={() => setSelectedDataset(dataset as any)}
                    className={`px-3 py-2 rounded-lg font-medium transition-colors ${
                      selectedDataset === dataset
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                    }`}
                  >
                    {dataset.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
            
            {/* 네트워크 설정 */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Settings className="w-5 h-5" />
                네트워크 설정
              </h3>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium mb-1">
                    활성화 함수
                  </label>
                  <select
                    value={config.activation}
                    onChange={(e) => setConfig({...config, activation: e.target.value as any})}
                    className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700"
                  >
                    <option value="sigmoid">Sigmoid</option>
                    <option value="relu">ReLU</option>
                    <option value="tanh">Tanh</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">
                    학습률: {config.learningRate}
                  </label>
                  <input
                    type="range"
                    min="0.01"
                    max="1"
                    step="0.01"
                    value={config.learningRate}
                    onChange={(e) => setConfig({...config, learningRate: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">
                    에포크: {config.epochs}
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="1000"
                    step="10"
                    value={config.epochs}
                    onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
                    className="w-full"
                  />
                </div>
              </div>
            </div>
            
            {/* 학습 진행 상황 */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Activity className="w-5 h-5" />
                학습 진행
              </h3>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>진행률</span>
                    <span>{Math.round((currentEpoch / config.epochs) * 100)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${(currentEpoch / config.epochs) * 100}%` }}
                    />
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">에포크:</span>
                    <span className="ml-2 font-mono">{currentEpoch}/{config.epochs}</span>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">손실:</span>
                    <span className="ml-2 font-mono">{loss.toFixed(4)}</span>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">정확도:</span>
                    <span className="ml-2 font-mono text-green-600">{accuracy.toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* 손실 그래프 */}
            {lossHistory.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold mb-3">손실 그래프</h3>
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <div className="h-24 relative">
                    <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                      <polyline
                        fill="none"
                        stroke="#ef4444"
                        strokeWidth="2"
                        points={lossHistory.map((loss, i) => {
                          const x = (i / Math.max(lossHistory.length - 1, 1)) * 100
                          const y = 100 - (Math.min(loss, 1) * 100)
                          return `${x},${y}`
                        }).join(' ')}
                      />
                    </svg>
                  </div>
                </div>
              </div>
            )}
            
            {/* 레이어 관리 */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Layers className="w-5 h-5" />
                레이어 구조
              </h3>
              <div className="space-y-2">
                {layers.map((layer, i) => (
                  <div key={layer.id} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">
                        {layer.type === 'input' ? '입력층' : 
                         layer.type === 'hidden' ? `은닉층 ${i}` : '출력층'}
                      </span>
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        {layer.neurons.length} 뉴런
                      </span>
                    </div>
                    {layer.type === 'hidden' && (
                      <div className="flex gap-2 mt-2">
                        <button
                          onClick={() => addNeuron(i)}
                          className="text-xs px-2 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
                        >
                          <Plus className="w-3 h-3" />
                        </button>
                        <button
                          onClick={() => removeNeuron(i)}
                          disabled={layer.neurons.length <= 1}
                          className="text-xs px-2 py-1 bg-red-500 text-white rounded hover:bg-red-600 disabled:opacity-50"
                        >
                          <Minus className="w-3 h-3" />
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}