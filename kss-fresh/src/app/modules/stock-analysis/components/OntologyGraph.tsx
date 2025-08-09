'use client'

import { useEffect, useRef, useState } from 'react'
import { ZoomIn, ZoomOut, Maximize2, Move } from 'lucide-react'

interface Node {
  id: string
  label: string
  type: 'main' | 'supplier' | 'competitor' | 'partner' | 'keyword'
  x?: number
  y?: number
  vx?: number
  vy?: number
  fx?: number | null
  fy?: number | null
}

interface Link {
  source: string | Node
  target: string | Node
  type: 'supplier' | 'competitor' | 'partner' | 'keyword'
  strength: number
}

interface OntologyGraphProps {
  company: string
  ticker: string
  relationships: {
    suppliers: string[]
    competitors: string[]
    partners: string[]
  }
  keywords: string[]
  impact?: {
    direct: number
    indirect: number
    sector: number
  }
}

export default function OntologyGraph({
  company,
  ticker,
  relationships,
  keywords,
  impact = { direct: 0, indirect: 0, sector: 0 }
}: OntologyGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const animationRef = useRef<number>()
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [hoveredNode, setHoveredNode] = useState<Node | null>(null)
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)
  const [nodes, setNodes] = useState<Node[]>([])
  const [links, setLinks] = useState<Link[]>([])

  // 노드와 링크 초기화
  useEffect(() => {
    const newNodes: Node[] = []
    const newLinks: Link[] = []

    // 메인 노드 (중앙)
    const mainNode: Node = {
      id: ticker,
      label: company,
      type: 'main',
      x: 0,
      y: 0,
      fx: 0,
      fy: 0
    }
    newNodes.push(mainNode)

    // 공급업체 노드
    if (relationships.suppliers && relationships.suppliers.length > 0) {
      relationships.suppliers.forEach((supplier, i) => {
        if (supplier && supplier.trim()) {  // 빈 문자열 체크
          const angle = (i * 2 * Math.PI) / relationships.suppliers.length - Math.PI / 2
          const radius = 200
          newNodes.push({
            id: `supplier-${i}`,
            label: supplier,
            type: 'supplier',
            x: radius * Math.cos(angle),
            y: radius * Math.sin(angle)
          })
          newLinks.push({
            source: ticker,
            target: `supplier-${i}`,
            type: 'supplier',
            strength: 0.8
          })
        }
      })
    }

    // 경쟁사 노드
    if (relationships.competitors && relationships.competitors.length > 0) {
      relationships.competitors.forEach((competitor, i) => {
        if (competitor && competitor.trim()) {  // 빈 문자열 체크
          const angle = (i * 2 * Math.PI) / relationships.competitors.length + Math.PI / 3
          const radius = 220
          newNodes.push({
            id: `competitor-${i}`,
            label: competitor,
            type: 'competitor',
            x: radius * Math.cos(angle),
            y: radius * Math.sin(angle)
          })
          newLinks.push({
            source: ticker,
            target: `competitor-${i}`,
            type: 'competitor',
            strength: 0.6
          })
        }
      })
    }

    // 파트너사 노드
    if (relationships.partners && relationships.partners.length > 0) {
      relationships.partners.forEach((partner, i) => {
        if (partner && partner.trim()) {  // 빈 문자열 체크
          const angle = (i * 2 * Math.PI) / relationships.partners.length - Math.PI / 3
          const radius = 150
          newNodes.push({
            id: `partner-${i}`,
            label: partner,
            type: 'partner',
            x: radius * Math.cos(angle),
            y: radius * Math.sin(angle)
          })
          newLinks.push({
            source: ticker,
            target: `partner-${i}`,
            type: 'partner',
            strength: 0.9
          })
        }
      })
    }

    // 키워드 노드 (작은 노드로 주변에 배치)
    if (keywords && keywords.length > 0) {
      keywords.slice(0, 5).forEach((keyword, i) => {
        if (keyword && keyword.trim()) {  // 빈 문자열 체크
          const angle = (i * 2 * Math.PI) / 5
          const radius = 100
          newNodes.push({
            id: `keyword-${i}`,
            label: keyword,
            type: 'keyword',
            x: radius * Math.cos(angle),
            y: radius * Math.sin(angle)
          })
          newLinks.push({
            source: ticker,
            target: `keyword-${i}`,
            type: 'keyword',
            strength: 0.3
          })
        }
      })
    }

    setNodes(newNodes)
    setLinks(newLinks)
  }, [company, ticker, relationships, keywords])

  // Force simulation
  useEffect(() => {
    if (!canvasRef.current || nodes.length === 0) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Force simulation parameters
    const alpha = 0.1
    const alphaDecay = 0.99
    const velocityDecay = 0.4
    let currentAlpha = alpha

    // Forces
    const simulation = () => {
      // Apply forces to nodes
      nodes.forEach(node => {
        if (node.fx !== null && node.fx !== undefined) {
          node.x = node.fx
          node.y = node.fy || 0
        } else {
          // Apply velocity
          node.vx = (node.vx || 0) * velocityDecay
          node.vy = (node.vy || 0) * velocityDecay
          node.x = (node.x || 0) + (node.vx || 0)
          node.y = (node.y || 0) + (node.vy || 0)
        }
      })

      // Link force
      links.forEach(link => {
        const source = typeof link.source === 'string' 
          ? nodes.find(n => n.id === link.source)
          : link.source as Node
        const target = typeof link.target === 'string'
          ? nodes.find(n => n.id === link.target)
          : link.target as Node

        if (source && target) {
          const dx = (target.x || 0) - (source.x || 0)
          const dy = (target.y || 0) - (source.y || 0)
          const distance = Math.sqrt(dx * dx + dy * dy)
          const idealDistance = 120
          
          if (distance > 0) {
            const force = (distance - idealDistance) * link.strength * currentAlpha
            const fx = (dx / distance) * force
            const fy = (dy / distance) * force

            if (source.fx === null || source.fx === undefined) {
              source.vx = (source.vx || 0) + fx
              source.vy = (source.vy || 0) + fy
            }
            if (target.fx === null || target.fx === undefined) {
              target.vx = (target.vx || 0) - fx
              target.vy = (target.vy || 0) - fy
            }
          }
        }
      })

      // Repulsion force
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const nodeA = nodes[i]
          const nodeB = nodes[j]
          const dx = (nodeB.x || 0) - (nodeA.x || 0)
          const dy = (nodeB.y || 0) - (nodeA.y || 0)
          const distance = Math.sqrt(dx * dx + dy * dy)
          
          if (distance > 0 && distance < 200) {
            const force = (30 * 30) / (distance * distance) * currentAlpha
            const fx = (dx / distance) * force
            const fy = (dy / distance) * force

            if (nodeA.fx === null || nodeA.fx === undefined) {
              nodeA.vx = (nodeA.vx || 0) - fx
              nodeA.vy = (nodeA.vy || 0) - fy
            }
            if (nodeB.fx === null || nodeB.fx === undefined) {
              nodeB.vx = (nodeB.vx || 0) + fx
              nodeB.vy = (nodeB.vy || 0) + fy
            }
          }
        }
      }

      currentAlpha *= alphaDecay
    }

    // Render function
    const render = () => {
      canvas.width = canvas.offsetWidth
      canvas.height = canvas.offsetHeight

      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.save()

      // Apply transformations
      ctx.translate(canvas.width / 2 + offset.x, canvas.height / 2 + offset.y)
      ctx.scale(zoom, zoom)

      // Draw links
      links.forEach(link => {
        const source = typeof link.source === 'string'
          ? nodes.find(n => n.id === link.source)
          : link.source as Node
        const target = typeof link.target === 'string'
          ? nodes.find(n => n.id === link.target)
          : link.target as Node

        if (source && target) {
          ctx.beginPath()
          ctx.moveTo(source.x || 0, source.y || 0)
          ctx.lineTo(target.x || 0, target.y || 0)
          
          // Link color based on type
          const alpha = selectedNode 
            ? (selectedNode.id === source.id || selectedNode.id === target.id ? 0.8 : 0.2)
            : 0.4
          
          switch (link.type) {
            case 'supplier':
              ctx.strokeStyle = `rgba(59, 130, 246, ${alpha})` // blue
              ctx.lineWidth = 2
              break
            case 'competitor':
              ctx.strokeStyle = `rgba(239, 68, 68, ${alpha})` // red
              ctx.lineWidth = 2
              ctx.setLineDash([5, 5])
              break
            case 'partner':
              ctx.strokeStyle = `rgba(34, 197, 94, ${alpha})` // green
              ctx.lineWidth = 2
              break
            case 'keyword':
              ctx.strokeStyle = `rgba(168, 85, 247, ${alpha})` // purple
              ctx.lineWidth = 1
              ctx.setLineDash([2, 2])
              break
          }
          
          ctx.stroke()
          ctx.setLineDash([])
        }
      })

      // Draw nodes
      nodes.forEach(node => {
        const isHovered = hoveredNode?.id === node.id
        const isSelected = selectedNode?.id === node.id
        const isConnected = selectedNode && links.some(link => 
          (link.source === selectedNode.id && link.target === node.id) ||
          (link.target === selectedNode.id && link.source === node.id)
        )
        
        // Node size and color based on type
        let radius = 8
        let fillColor = '#6B7280'
        let strokeColor = '#374151'
        
        switch (node.type) {
          case 'main':
            radius = 20
            fillColor = impact.direct > 0 ? '#10B981' : impact.direct < 0 ? '#EF4444' : '#6366F1'
            strokeColor = impact.direct > 0 ? '#059669' : impact.direct < 0 ? '#DC2626' : '#4F46E5'
            break
          case 'supplier':
            radius = 12
            fillColor = '#3B82F6'
            strokeColor = '#2563EB'
            break
          case 'competitor':
            radius = 12
            fillColor = '#EF4444'
            strokeColor = '#DC2626'
            break
          case 'partner':
            radius = 12
            fillColor = '#22C55E'
            strokeColor = '#16A34A'
            break
          case 'keyword':
            radius = 6
            fillColor = '#A855F7'
            strokeColor = '#9333EA'
            break
        }

        // Adjust opacity if node is not connected to selected
        const alpha = selectedNode 
          ? (isSelected || isConnected || node.type === 'main' ? 1 : 0.3)
          : 1

        // Draw node shadow if hovered
        if (isHovered || isSelected) {
          ctx.beginPath()
          ctx.arc(node.x || 0, node.y || 0, radius + 4, 0, 2 * Math.PI)
          ctx.fillStyle = `rgba(99, 102, 241, ${isSelected ? 0.3 : 0.2})`
          ctx.fill()
        }

        // Draw node
        ctx.beginPath()
        ctx.arc(node.x || 0, node.y || 0, radius, 0, 2 * Math.PI)
        ctx.fillStyle = fillColor + Math.round(alpha * 255).toString(16).padStart(2, '0')
        ctx.fill()
        ctx.strokeStyle = strokeColor + Math.round(alpha * 255).toString(16).padStart(2, '0')
        ctx.lineWidth = isHovered || isSelected ? 3 : 2
        ctx.stroke()

        // Draw label (텍스트를 더 크고 잘 보이게)
        // 배경 박스 그리기
        ctx.font = node.type === 'main' ? 'bold 14px sans-serif' : 'bold 11px sans-serif'
        const textMetrics = ctx.measureText(node.label)
        const textWidth = textMetrics.width
        const textHeight = node.type === 'main' ? 16 : 14
        const textY = (node.y || 0) + radius + 8
        
        // 텍스트 배경
        ctx.fillStyle = `rgba(255, 255, 255, ${alpha * 0.9})`
        ctx.fillRect(
          (node.x || 0) - textWidth / 2 - 4,
          textY - 2,
          textWidth + 8,
          textHeight
        )
        
        // 텍스트 그리기
        ctx.fillStyle = `rgba(0, 0, 0, ${alpha})`
        ctx.textAlign = 'center'
        ctx.textBaseline = 'top'
        ctx.fillText(node.label, node.x || 0, textY)
      })

      ctx.restore()

      // Continue simulation if needed
      if (currentAlpha > 0.001) {
        simulation()
        animationRef.current = requestAnimationFrame(render)
      }
    }

    render()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [nodes, links, zoom, offset, hoveredNode, selectedNode, impact])

  // Mouse event handlers
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left - canvas.width / 2 - offset.x) / zoom
    const y = (e.clientY - rect.top - canvas.height / 2 - offset.y) / zoom

    if (isDragging) {
      setOffset({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      })
    } else {
      // Check for node hover
      const hoveredNode = nodes.find(node => {
        const dx = (node.x || 0) - x
        const dy = (node.y || 0) - y
        const radius = node.type === 'main' ? 20 : node.type === 'keyword' ? 6 : 12
        return Math.sqrt(dx * dx + dy * dy) < radius
      })
      setHoveredNode(hoveredNode || null)
      canvas.style.cursor = hoveredNode ? 'pointer' : isDragging ? 'grabbing' : 'grab'
    }
  }

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left - canvas.width / 2 - offset.x) / zoom
    const y = (e.clientY - rect.top - canvas.height / 2 - offset.y) / zoom

    // Check if clicking on a node
    const clickedNode = nodes.find(node => {
      const dx = (node.x || 0) - x
      const dy = (node.y || 0) - y
      const radius = node.type === 'main' ? 20 : node.type === 'keyword' ? 6 : 12
      return Math.sqrt(dx * dx + dy * dy) < radius
    })

    if (clickedNode) {
      setSelectedNode(clickedNode)
    } else {
      setSelectedNode(null)
      setIsDragging(true)
      setDragStart({
        x: e.clientX - offset.x,
        y: e.clientY - offset.y
      })
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    setZoom(prevZoom => Math.max(0.5, Math.min(3, prevZoom * delta)))
  }

  const resetView = () => {
    setZoom(1)
    setOffset({ x: 0, y: 0 })
    setSelectedNode(null)
  }

  return (
    <div className="relative w-full h-[500px] bg-gray-50 dark:bg-gray-900 rounded-lg overflow-hidden" ref={containerRef}>
      {/* Canvas */}
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      />

      {/* Controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-2">
        <button
          onClick={() => setZoom(z => Math.min(3, z * 1.2))}
          className="p-2 bg-white dark:bg-gray-800 rounded-lg shadow-lg hover:bg-gray-100 dark:hover:bg-gray-700"
          title="확대"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={() => setZoom(z => Math.max(0.5, z * 0.8))}
          className="p-2 bg-white dark:bg-gray-800 rounded-lg shadow-lg hover:bg-gray-100 dark:hover:bg-gray-700"
          title="축소"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={resetView}
          className="p-2 bg-white dark:bg-gray-800 rounded-lg shadow-lg hover:bg-gray-100 dark:hover:bg-gray-700"
          title="초기화"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-white dark:bg-gray-800 rounded-lg p-3 shadow-lg">
        <h4 className="text-xs font-semibold text-gray-900 dark:text-white mb-2">관계 유형</h4>
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            <span className="text-xs text-gray-600 dark:text-gray-400">공급사</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <span className="text-xs text-gray-600 dark:text-gray-400">경쟁사</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className="text-xs text-gray-600 dark:text-gray-400">파트너</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
            <span className="text-xs text-gray-600 dark:text-gray-400">키워드</span>
          </div>
        </div>
      </div>

      {/* Selected Node Info */}
      {selectedNode && (
        <div className="absolute top-4 left-4 bg-white dark:bg-gray-800 rounded-lg p-3 shadow-lg max-w-xs">
          <h4 className="font-semibold text-gray-900 dark:text-white mb-1">
            {selectedNode.label}
          </h4>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            {selectedNode.type === 'main' ? '분석 대상 기업' :
             selectedNode.type === 'supplier' ? '공급업체' :
             selectedNode.type === 'competitor' ? '경쟁사' :
             selectedNode.type === 'partner' ? '파트너사' :
             '핵심 키워드'}
          </p>
          {selectedNode.type === 'main' && impact && (
            <div className="mt-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">직접 영향:</span>
                <span className={impact.direct > 0 ? 'text-green-600' : 'text-red-600'}>
                  {impact.direct > 0 ? '+' : ''}{impact.direct}
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Instructions */}
      <div className="absolute bottom-4 right-4 text-xs text-gray-500 dark:text-gray-400 bg-white/80 dark:bg-gray-800/80 rounded px-2 py-1">
        <Move className="inline w-3 h-3 mr-1" />
        드래그: 이동 | 스크롤: 확대/축소 | 클릭: 선택
      </div>
    </div>
  )
}