'use client'

import { useEffect, useRef, useState } from 'react'

interface GraphNode {
  id: string
  label: string
  x: number
  y: number
  vx: number
  vy: number
  type: 'concept' | 'entity' | 'relation'
  connections: string[]
  active: boolean
}

interface GraphEdge {
  from: string
  to: string
  label: string
  active: boolean
}

export default function KnowledgeGraphSim() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [nodes, setNodes] = useState<GraphNode[]>([])
  const [edges, setEdges] = useState<GraphEdge[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [animationFrame, setAnimationFrame] = useState(0)
  const animationRef = useRef<number>()

  // 그래프 초기화
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const width = canvas.width = 400
    const height = canvas.height = 300

    const centerX = width / 2
    const centerY = height / 2

    const newNodes: GraphNode[] = [
      {
        id: 'ai',
        label: 'AI',
        x: centerX,
        y: centerY,
        vx: 0,
        vy: 0,
        type: 'concept',
        connections: ['ml', 'nlp', 'cv'],
        active: false
      },
      {
        id: 'ml',
        label: 'ML',
        x: centerX - 80,
        y: centerY - 60,
        vx: 0,
        vy: 0,
        type: 'concept',
        connections: ['ai', 'neural'],
        active: false
      },
      {
        id: 'nlp',
        label: 'NLP',
        x: centerX + 80,
        y: centerY - 60,
        vx: 0,
        vy: 0,
        type: 'concept',
        connections: ['ai', 'transformer'],
        active: false
      },
      {
        id: 'cv',
        label: 'Vision',
        x: centerX,
        y: centerY + 80,
        vx: 0,
        vy: 0,
        type: 'concept',
        connections: ['ai', 'cnn'],
        active: false
      },
      {
        id: 'neural',
        label: 'Neural Net',
        x: centerX - 140,
        y: centerY,
        vx: 0,
        vy: 0,
        type: 'entity',
        connections: ['ml'],
        active: false
      },
      {
        id: 'transformer',
        label: 'Transformer',
        x: centerX + 140,
        y: centerY,
        vx: 0,
        vy: 0,
        type: 'entity',
        connections: ['nlp'],
        active: false
      },
      {
        id: 'cnn',
        label: 'CNN',
        x: centerX,
        y: centerY + 140,
        vx: 0,
        vy: 0,
        type: 'entity',
        connections: ['cv'],
        active: false
      }
    ]

    const newEdges: GraphEdge[] = [
      { from: 'ai', to: 'ml', label: 'includes', active: false },
      { from: 'ai', to: 'nlp', label: 'includes', active: false },
      { from: 'ai', to: 'cv', label: 'includes', active: false },
      { from: 'ml', to: 'neural', label: 'uses', active: false },
      { from: 'nlp', to: 'transformer', label: 'uses', active: false },
      { from: 'cv', to: 'cnn', label: 'uses', active: false }
    ]

    setNodes(newNodes)
    setEdges(newEdges)
  }, [])

  // 물리 시뮬레이션 및 애니메이션
  useEffect(() => {
    const animate = () => {
      setAnimationFrame(prev => prev + 1)
      
      if (isRunning) {
        // 노드 위치 업데이트 (스프링 포스)
        setNodes(prevNodes => prevNodes.map(node => {
          let fx = 0, fy = 0
          
          // 중심으로의 복원력
          const centerX = 200, centerY = 150
          const dx = centerX - node.x
          const dy = centerY - node.y
          const distance = Math.sqrt(dx * dx + dy * dy)
          
          if (distance > 0) {
            fx += (dx / distance) * 0.1
            fy += (dy / distance) * 0.1
          }
          
          // 다른 노드들과의 척력
          prevNodes.forEach(other => {
            if (other.id !== node.id) {
              const dx = node.x - other.x
              const dy = node.y - other.y
              const distance = Math.sqrt(dx * dx + dy * dy)
              
              if (distance > 0 && distance < 100) {
                const force = 500 / (distance * distance)
                fx += (dx / distance) * force
                fy += (dy / distance) * force
              }
            }
          })
          
          // 연결된 노드들과의 인력
          node.connections.forEach(connectId => {
            const connectedNode = prevNodes.find(n => n.id === connectId)
            if (connectedNode) {
              const dx = connectedNode.x - node.x
              const dy = connectedNode.y - node.y
              const distance = Math.sqrt(dx * dx + dy * dy)
              const idealDistance = 100
              
              if (distance > 0) {
                const force = (distance - idealDistance) * 0.05
                fx += (dx / distance) * force
                fy += (dy / distance) * force
              }
            }
          })
          
          // 속도 업데이트 (감쇠 포함)
          const newVx = (node.vx + fx) * 0.85
          const newVy = (node.vy + fy) * 0.85
          
          return {
            ...node,
            x: Math.max(30, Math.min(370, node.x + newVx)),
            y: Math.max(30, Math.min(270, node.y + newVy)),
            vx: newVx,
            vy: newVy,
            active: selectedNode ? 
              (node.id === selectedNode || node.connections.includes(selectedNode)) :
              Math.sin(animationFrame * 0.02 + parseInt(node.id.length.toString())) > 0.3
          }
        }))

        // 엣지 활성화
        setEdges(prevEdges => prevEdges.map(edge => ({
          ...edge,
          active: selectedNode ? 
            (edge.from === selectedNode || edge.to === selectedNode) :
            Math.sin(animationFrame * 0.03 + edge.from.length) > 0.5
        })))

        // 자동 노드 선택 (5초마다)
        if (animationFrame % 300 === 0) {
          const nodeIds = nodes.map(n => n.id)
          setSelectedNode(nodeIds[Math.floor(Math.random() * nodeIds.length)])
        }
      }
      
      drawGraph()
      animationRef.current = requestAnimationFrame(animate)
    }

    animate()
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [nodes, edges, isRunning, selectedNode, animationFrame])

  const drawGraph = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // 배경
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // 엣지 그리기
    edges.forEach(edge => {
      const fromNode = nodes.find(n => n.id === edge.from)
      const toNode = nodes.find(n => n.id === edge.to)
      
      if (fromNode && toNode) {
        ctx.beginPath()
        ctx.moveTo(fromNode.x, fromNode.y)
        ctx.lineTo(toNode.x, toNode.y)
        
        if (edge.active) {
          ctx.strokeStyle = 'rgba(34, 197, 94, 0.8)'
          ctx.lineWidth = 2
          ctx.shadowColor = '#22c55e'
          ctx.shadowBlur = 10
          
          // 애니메이션 파티클
          const progress = (animationFrame * 0.02) % 1
          const particleX = fromNode.x + (toNode.x - fromNode.x) * progress
          const particleY = fromNode.y + (toNode.y - fromNode.y) * progress
          
          ctx.beginPath()
          ctx.arc(particleX, particleY, 3, 0, 2 * Math.PI)
          ctx.fillStyle = 'rgba(34, 197, 94, 0.9)'
          ctx.fill()
        } else {
          ctx.strokeStyle = 'rgba(71, 85, 105, 0.4)'
          ctx.lineWidth = 1
          ctx.shadowBlur = 0
        }
        
        ctx.stroke()
        
        // 엣지 레이블
        if (edge.active) {
          const midX = (fromNode.x + toNode.x) / 2
          const midY = (fromNode.y + toNode.y) / 2
          
          // 엣지 레이블 배경
          const textWidth = ctx.measureText(edge.label).width
          ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
          ctx.fillRect(midX - textWidth/2 - 3, midY - 13, textWidth + 6, 18)
          
          // 엣지 레이블 텍스트
          ctx.fillStyle = '#fbbf24'
          ctx.font = 'bold 12px monospace'
          ctx.textAlign = 'center'
          ctx.fillText(edge.label, midX, midY)
        }
      }
    })

    // 노드 그리기
    nodes.forEach(node => {
      const isSelected = node.id === selectedNode
      const intensity = node.active ? 1.0 : 0.5
      const pulseSize = isSelected ? 3 + 2 * Math.sin(animationFrame * 0.1) : 0
      
      // 노드 글로우
      if (node.active) {
        ctx.beginPath()
        ctx.arc(node.x, node.y, 25 + pulseSize, 0, 2 * Math.PI)
        
        let glowColor = 'rgba(59, 130, 246, 0.3)'
        if (node.type === 'concept') glowColor = 'rgba(239, 68, 68, 0.3)'
        else if (node.type === 'entity') glowColor = 'rgba(34, 197, 94, 0.3)'
        
        ctx.fillStyle = glowColor
        ctx.fill()
      }
      
      // 노드 본체
      ctx.beginPath()
      ctx.arc(node.x, node.y, 15 + pulseSize, 0, 2 * Math.PI)
      
      if (node.type === 'concept') {
        ctx.fillStyle = `rgba(239, 68, 68, ${0.7 + 0.3 * intensity})`
      } else if (node.type === 'entity') {
        ctx.fillStyle = `rgba(34, 197, 94, ${0.7 + 0.3 * intensity})`
      } else {
        ctx.fillStyle = `rgba(59, 130, 246, ${0.7 + 0.3 * intensity})`
      }
      
      ctx.fill()
      
      // 노드 테두리
      ctx.beginPath()
      ctx.arc(node.x, node.y, 15 + pulseSize, 0, 2 * Math.PI)
      ctx.strokeStyle = isSelected ? '#fbbf24' : 'rgba(255, 255, 255, 0.8)'
      ctx.lineWidth = isSelected ? 2 : 1
      ctx.stroke()
      
      // 노드 레이블
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 14px monospace'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      // 배경 박스
      const textWidth = ctx.measureText(node.label).width
      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)'
      ctx.fillRect(node.x - textWidth/2 - 4, node.y + 20, textWidth + 8, 20)
      // 텍스트
      ctx.fillStyle = '#ffffff'
      ctx.fillText(node.label, node.x, node.y + 30)
    })

    // 선택된 노드 정보
    if (selectedNode && isRunning) {
      const selected = nodes.find(n => n.id === selectedNode)
      if (selected) {
        ctx.fillStyle = 'rgba(34, 197, 94, 0.9)'
        ctx.font = '12px monospace'
        ctx.textAlign = 'left'
        ctx.fillText(`Focus: ${selected.label}`, 10, canvas.height - 15)
      }
    }

    // 제목
    ctx.fillStyle = 'rgba(148, 163, 184, 0.9)'
    ctx.font = '12px monospace'
    ctx.textAlign = 'left'
    ctx.fillText('$ Rendering 3D knowledge graph...', 10, 20)
  }

  const startGraph = () => {
    setIsRunning(true)
    setSelectedNode(null)
  }

  const stopGraph = () => {
    setIsRunning(false)
    setSelectedNode(null)
    
    // 모든 활성화 상태 리셋
    setNodes(prev => prev.map(node => ({ ...node, active: false })))
    setEdges(prev => prev.map(edge => ({ ...edge, active: false })))
  }

  return (
    <div className="relative bg-slate-900 rounded-lg p-4 border border-slate-700">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-semibold text-sm">3D Knowledge Graph</h3>
        <div className="flex gap-2">
          {!isRunning ? (
            <button
              onClick={startGraph}
              className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-xs rounded transition-colors"
            >
              Render
            </button>
          ) : (
            <button
              onClick={stopGraph}
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
      
      {isRunning && (
        <div className="absolute top-2 right-2 bg-green-600 text-white px-2 py-1 rounded text-xs animate-pulse">
          Rendering...
        </div>
      )}
      
      <div className="mt-2 text-xs text-slate-400">
        Interactive visualization of complex AI concepts
      </div>
    </div>
  )
}