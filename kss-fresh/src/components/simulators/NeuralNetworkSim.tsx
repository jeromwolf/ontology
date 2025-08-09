'use client'

import { useEffect, useRef, useState } from 'react'

interface Node {
  x: number
  y: number
  id: string
  layer: number
  value: number
  activation: number
}

interface Connection {
  from: string
  to: string
  weight: number
  active: boolean
}

export default function NeuralNetworkSim() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [nodes, setNodes] = useState<Node[]>([])
  const [connections, setConnections] = useState<Connection[]>([])
  const [isTraining, setIsTraining] = useState(false)
  const [currentEpoch, setCurrentEpoch] = useState(0)
  const [animationFrame, setAnimationFrame] = useState(0)
  const animationRef = useRef<number>()

  // 네트워크 초기화
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const width = canvas.width = 400
    const height = canvas.height = 300

    // 노드 생성 (3-4-3 구조)
    const newNodes: Node[] = []
    const newConnections: Connection[] = []
    
    // Input layer (3 nodes)
    for (let i = 0; i < 3; i++) {
      newNodes.push({
        x: 60,
        y: 60 + i * 90,
        id: `input-${i}`,
        layer: 0,
        value: Math.random(),
        activation: 0
      })
    }
    
    // Hidden layer (4 nodes)
    for (let i = 0; i < 4; i++) {
      newNodes.push({
        x: 200,
        y: 35 + i * 70,
        id: `hidden-${i}`,
        layer: 1,
        value: 0,
        activation: 0
      })
    }
    
    // Output layer (3 nodes)
    for (let i = 0; i < 3; i++) {
      newNodes.push({
        x: 340,
        y: 60 + i * 90,
        id: `output-${i}`,
        layer: 2,
        value: 0,
        activation: 0
      })
    }

    // 연결 생성
    // Input to Hidden
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 4; j++) {
        newConnections.push({
          from: `input-${i}`,
          to: `hidden-${j}`,
          weight: (Math.random() - 0.5) * 2,
          active: false
        })
      }
    }
    
    // Hidden to Output
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 3; j++) {
        newConnections.push({
          from: `hidden-${i}`,
          to: `output-${j}`,
          weight: (Math.random() - 0.5) * 2,
          active: false
        })
      }
    }

    setNodes(newNodes)
    setConnections(newConnections)
  }, [])

  // 애니메이션 루프
  useEffect(() => {
    const animate = () => {
      setAnimationFrame(prev => prev + 1)
      
      if (isTraining) {
        // Forward propagation simulation
        setNodes(prevNodes => prevNodes.map(node => {
          if (node.layer === 0) {
            // Input layer - 랜덤 입력 시뮬레이션
            return {
              ...node,
              value: 0.3 + 0.4 * Math.sin(animationFrame * 0.05 + parseInt(node.id.split('-')[1]))
            }
          } else if (node.layer === 1) {
            // Hidden layer - 시그모이드 활성화 시뮬레이션
            const activation = 0.5 + 0.3 * Math.sin(animationFrame * 0.03 + parseInt(node.id.split('-')[1]) * 0.5)
            return {
              ...node,
              value: activation,
              activation
            }
          } else {
            // Output layer
            const activation = 0.4 + 0.4 * Math.sin(animationFrame * 0.04 + parseInt(node.id.split('-')[1]) * 0.7)
            return {
              ...node,
              value: activation,
              activation
            }
          }
        }))

        // 연결 활성화 애니메이션
        setConnections(prevConnections => prevConnections.map(conn => ({
          ...conn,
          active: Math.sin(animationFrame * 0.1 + conn.weight) > 0.5
        })))
      }
      
      drawNetwork()
      animationRef.current = requestAnimationFrame(animate)
    }

    animate()
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [nodes, connections, isTraining, animationFrame])

  const drawNetwork = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // 배경 클리어
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // 연결선 그리기
    connections.forEach(conn => {
      const fromNode = nodes.find(n => n.id === conn.from)
      const toNode = nodes.find(n => n.id === conn.to)
      
      if (fromNode && toNode) {
        ctx.beginPath()
        ctx.moveTo(fromNode.x, fromNode.y)
        ctx.lineTo(toNode.x, toNode.y)
        
        if (conn.active && isTraining) {
          ctx.strokeStyle = `rgba(34, 197, 94, ${0.8 * Math.abs(conn.weight)})`
          ctx.lineWidth = 2
          ctx.shadowColor = '#22c55e'
          ctx.shadowBlur = 10
        } else {
          ctx.strokeStyle = `rgba(71, 85, 105, ${0.3 + 0.4 * Math.abs(conn.weight)})`
          ctx.lineWidth = 1
          ctx.shadowBlur = 0
        }
        
        ctx.stroke()
      }
    })

    // 노드 그리기
    nodes.forEach(node => {
      const intensity = isTraining ? node.value : 0.5
      const pulseSize = isTraining ? 2 + 4 * intensity : 0
      
      // 글로우 효과
      if (isTraining && intensity > 0.6) {
        ctx.beginPath()
        ctx.arc(node.x, node.y, 15 + pulseSize, 0, 2 * Math.PI)
        ctx.fillStyle = `rgba(59, 130, 246, ${0.3 * intensity})`
        ctx.fill()
      }
      
      // 노드 본체
      ctx.beginPath()
      ctx.arc(node.x, node.y, 8 + pulseSize, 0, 2 * Math.PI)
      
      if (node.layer === 0) {
        ctx.fillStyle = `rgba(239, 68, 68, ${0.7 + 0.3 * intensity})` // Red for input
      } else if (node.layer === 1) {
        ctx.fillStyle = `rgba(34, 197, 94, ${0.7 + 0.3 * intensity})` // Green for hidden
      } else {
        ctx.fillStyle = `rgba(59, 130, 246, ${0.7 + 0.3 * intensity})` // Blue for output
      }
      
      ctx.fill()
      
      // 노드 테두리
      ctx.beginPath()
      ctx.arc(node.x, node.y, 8 + pulseSize, 0, 2 * Math.PI)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)'
      ctx.lineWidth = 1
      ctx.stroke()
    })

    // 레이어 라벨
    ctx.fillStyle = 'rgba(148, 163, 184, 0.8)'
    ctx.font = '12px monospace'
    ctx.textAlign = 'center'
    
    ctx.fillText('Input', 60, 25)
    ctx.fillText('Hidden', 200, 25)
    ctx.fillText('Output', 340, 25)

    // 에포크 표시
    if (isTraining) {
      ctx.fillStyle = 'rgba(34, 197, 94, 0.9)'
      ctx.font = 'bold 14px monospace'
      ctx.textAlign = 'left'
      ctx.fillText(`Epoch: ${currentEpoch}`, 10, canvas.height - 10)
    }
  }

  const startTraining = () => {
    setIsTraining(true)
    setCurrentEpoch(0)
    
    const epochInterval = setInterval(() => {
      setCurrentEpoch(prev => {
        const next = prev + 1
        if (next >= 100) {
          setIsTraining(false)
          clearInterval(epochInterval)
          return 0
        }
        return next
      })
    }, 50)
  }

  const stopTraining = () => {
    setIsTraining(false)
    setCurrentEpoch(0)
  }

  return (
    <div className="relative bg-slate-900 rounded-lg p-4 border border-slate-700">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-semibold text-sm">Neural Network</h3>
        <div className="flex gap-2">
          {!isTraining ? (
            <button
              onClick={startTraining}
              className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-xs rounded transition-colors"
            >
              Train
            </button>
          ) : (
            <button
              onClick={stopTraining}
              className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded transition-colors"
            >
              Stop
            </button>
          )}
        </div>
      </div>
      
      <canvas
        ref={canvasRef}
        className="w-full border border-slate-600 rounded"
        style={{ maxHeight: '300px' }}
      />
      
      {isTraining && (
        <div className="absolute top-2 right-2 bg-green-600 text-white px-2 py-1 rounded text-xs animate-pulse">
          Training...
        </div>
      )}
      
      <div className="mt-2 text-xs text-slate-400">
        Real-time tokenization and attention mechanisms
      </div>
    </div>
  )
}