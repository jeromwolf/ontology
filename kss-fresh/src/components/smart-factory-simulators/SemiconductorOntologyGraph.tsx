'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Info, ZoomIn, ZoomOut, Maximize2, RefreshCw } from 'lucide-react'

interface Node {
  id: string
  label: string
  type: 'fab' | 'area' | 'tool' | 'chamber' | 'lot' | 'wafer' | 'die' | 'bin' | 'recipe' | 'step' | 'parameter'
  x: number
  y: number
  color: string
  children?: string[]
  metadata?: Record<string, string | number>
}

interface Edge {
  from: string
  to: string
  label?: string
}

export default function SemiconductorOntologyGraph() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [activeHierarchy, setActiveHierarchy] = useState<'equipment' | 'product' | 'recipe'>('equipment')
  const [hoveredNode, setHoveredNode] = useState<Node | null>(null)

  // 반도체 제조 온톨로지 데이터
  const equipmentNodes: Node[] = [
    { id: 'fab', label: 'Fab\n(공장)', type: 'fab', x: 400, y: 50, color: '#3b82f6', metadata: { capacity: '100K wafers/month', utilization: '92%' } },
    { id: 'photo-area', label: 'Photo Area\n(노광구역)', type: 'area', x: 200, y: 150, color: '#8b5cf6', children: ['litho-tool'], metadata: { tools: 8 } },
    { id: 'etch-area', label: 'Etch Area\n(식각구역)', type: 'area', x: 400, y: 150, color: '#8b5cf6', children: ['etch-tool'], metadata: { tools: 12 } },
    { id: 'depo-area', label: 'Deposition\n(증착구역)', type: 'area', x: 600, y: 150, color: '#8b5cf6', children: ['cvd-tool'], metadata: { tools: 10 } },
    { id: 'litho-tool', label: 'EUV Scanner\n(노광기)', type: 'tool', x: 200, y: 270, color: '#a855f7', children: ['litho-chamber'], metadata: { model: 'ASML Twinscan NXE:3400C', throughput: '170 wph' } },
    { id: 'etch-tool', label: 'Plasma Etcher\n(식각기)', type: 'tool', x: 400, y: 270, color: '#a855f7', children: ['etch-chamber'], metadata: { model: 'LAM Kiyo', chambers: 4 } },
    { id: 'cvd-tool', label: 'CVD System\n(증착기)', type: 'tool', x: 600, y: 270, color: '#a855f7', children: ['cvd-chamber'], metadata: { model: 'Applied Centura', chambers: 6 } },
    { id: 'litho-chamber', label: 'EUV Chamber\n(노광챔버)', type: 'chamber', x: 200, y: 390, color: '#c084fc', metadata: { wavelength: '13.5 nm', power: '250W' } },
    { id: 'etch-chamber', label: 'Process Chamber\n(식각챔버)', type: 'chamber', x: 400, y: 390, color: '#c084fc', metadata: { pressure: '5 mTorr', temp: '350°C' } },
    { id: 'cvd-chamber', label: 'Deposition Chamber\n(증착챔버)', type: 'chamber', x: 600, y: 390, color: '#c084fc', metadata: { thickness: '100 Å', uniformity: '±1%' } },
  ]

  const productNodes: Node[] = [
    { id: 'lot', label: 'Lot\n(로트)', type: 'lot', x: 400, y: 50, color: '#10b981', children: ['wafer-1', 'wafer-2'], metadata: { size: 25, recipe: 'LOGIC-7NM-v3' } },
    { id: 'wafer-1', label: 'Wafer 001\n(웨이퍼)', type: 'wafer', x: 300, y: 170, color: '#34d399', children: ['die-1', 'die-2'], metadata: { diameter: '300mm', thickness: '775µm' } },
    { id: 'wafer-2', label: 'Wafer 002\n(웨이퍼)', type: 'wafer', x: 500, y: 170, color: '#34d399', children: ['die-3', 'die-4'], metadata: { diameter: '300mm', thickness: '775µm' } },
    { id: 'die-1', label: 'Die (0,0)', type: 'die', x: 200, y: 290, color: '#6ee7b7', children: ['bin-pass-1'], metadata: { size: '100mm²', defects: 0 } },
    { id: 'die-2', label: 'Die (0,1)', type: 'die', x: 400, y: 290, color: '#6ee7b7', children: ['bin-pass-2'], metadata: { size: '100mm²', defects: 0 } },
    { id: 'die-3', label: 'Die (1,0)', type: 'die', x: 450, y: 290, color: '#6ee7b7', children: ['bin-fail'], metadata: { size: '100mm²', defects: 2 } },
    { id: 'die-4', label: 'Die (1,1)', type: 'die', x: 600, y: 290, color: '#6ee7b7', children: ['bin-pass-3'], metadata: { size: '100mm²', defects: 0 } },
    { id: 'bin-pass-1', label: 'Bin 1\n(PASS)', type: 'bin', x: 200, y: 410, color: '#10b981', metadata: { yield: 'Good', speed: '3.2GHz' } },
    { id: 'bin-pass-2', label: 'Bin 1\n(PASS)', type: 'bin', x: 400, y: 410, color: '#10b981', metadata: { yield: 'Good', speed: '3.2GHz' } },
    { id: 'bin-fail', label: 'Bin 5\n(FAIL)', type: 'bin', x: 450, y: 410, color: '#ef4444', metadata: { yield: 'Reject', reason: 'Short' } },
    { id: 'bin-pass-3', label: 'Bin 1\n(PASS)', type: 'bin', x: 600, y: 410, color: '#10b981', metadata: { yield: 'Good', speed: '3.2GHz' } },
  ]

  const recipeNodes: Node[] = [
    { id: 'recipe', label: 'LOGIC-7NM-v3\n(레시피)', type: 'recipe', x: 400, y: 50, color: '#f59e0b', children: ['step-1', 'step-2', 'step-3'], metadata: { version: '3.2', layers: 15 } },
    { id: 'step-1', label: 'Step 1: Litho\n(노광단계)', type: 'step', x: 250, y: 180, color: '#fbbf24', children: ['param-1-1', 'param-1-2'], metadata: { duration: '120s', chamber: 'EUV' } },
    { id: 'step-2', label: 'Step 2: Etch\n(식각단계)', type: 'step', x: 400, y: 180, color: '#fbbf24', children: ['param-2-1', 'param-2-2'], metadata: { duration: '90s', chamber: 'Plasma' } },
    { id: 'step-3', label: 'Step 3: Depo\n(증착단계)', type: 'step', x: 550, y: 180, color: '#fbbf24', children: ['param-3-1', 'param-3-2'], metadata: { duration: '180s', chamber: 'CVD' } },
    { id: 'param-1-1', label: 'Dose\n(노광량)', type: 'parameter', x: 200, y: 310, color: '#fcd34d', metadata: { value: '25 mJ/cm²', tolerance: '±2%' } },
    { id: 'param-1-2', label: 'Focus\n(초점)', type: 'parameter', x: 300, y: 310, color: '#fcd34d', metadata: { value: '0 nm', tolerance: '±10nm' } },
    { id: 'param-2-1', label: 'RF Power\n(전력)', type: 'parameter', x: 350, y: 310, color: '#fcd34d', metadata: { value: '1500W', tolerance: '±50W' } },
    { id: 'param-2-2', label: 'Pressure\n(압력)', type: 'parameter', x: 450, y: 310, color: '#fcd34d', metadata: { value: '5 mTorr', tolerance: '±0.5' } },
    { id: 'param-3-1', label: 'Temp\n(온도)', type: 'parameter', x: 500, y: 310, color: '#fcd34d', metadata: { value: '350°C', tolerance: '±5°C' } },
    { id: 'param-3-2', label: 'Flow Rate\n(유량)', type: 'parameter', x: 600, y: 310, color: '#fcd34d', metadata: { value: '200 sccm', tolerance: '±10' } },
  ]

  const nodes = activeHierarchy === 'equipment' ? equipmentNodes : activeHierarchy === 'product' ? productNodes : recipeNodes

  const edges: Edge[] = nodes.filter(n => n.children).flatMap(n =>
    n.children!.map(childId => ({ from: n.id, to: childId, label: '' }))
  )

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    // Clear canvas
    ctx.clearRect(0, 0, rect.width, rect.height)

    // Apply transformations
    ctx.save()
    ctx.translate(pan.x, pan.y)
    ctx.scale(zoom, zoom)

    // Draw edges
    ctx.strokeStyle = '#6b7280'
    ctx.lineWidth = 2
    edges.forEach(edge => {
      const fromNode = nodes.find(n => n.id === edge.from)
      const toNode = nodes.find(n => n.id === edge.to)
      if (fromNode && toNode) {
        ctx.beginPath()
        ctx.moveTo(fromNode.x, fromNode.y + 30)
        ctx.lineTo(toNode.x, toNode.y - 30)
        ctx.stroke()

        // Arrow head
        const angle = Math.atan2(toNode.y - fromNode.y, toNode.x - fromNode.x)
        const arrowSize = 8
        ctx.beginPath()
        ctx.moveTo(toNode.x, toNode.y - 30)
        ctx.lineTo(
          toNode.x - arrowSize * Math.cos(angle - Math.PI / 6),
          toNode.y - 30 - arrowSize * Math.sin(angle - Math.PI / 6)
        )
        ctx.moveTo(toNode.x, toNode.y - 30)
        ctx.lineTo(
          toNode.x - arrowSize * Math.cos(angle + Math.PI / 6),
          toNode.y - 30 - arrowSize * Math.sin(angle + Math.PI / 6)
        )
        ctx.stroke()
      }
    })

    // Draw nodes
    nodes.forEach(node => {
      const isHovered = hoveredNode?.id === node.id
      const isSelected = selectedNode?.id === node.id

      // Node background
      ctx.fillStyle = node.color
      ctx.strokeStyle = isSelected ? '#fff' : isHovered ? '#fff' : node.color
      ctx.lineWidth = isSelected ? 4 : isHovered ? 3 : 0

      const nodeWidth = 140
      const nodeHeight = 60
      const radius = 8

      ctx.beginPath()
      ctx.moveTo(node.x - nodeWidth/2 + radius, node.y - nodeHeight/2)
      ctx.lineTo(node.x + nodeWidth/2 - radius, node.y - nodeHeight/2)
      ctx.quadraticCurveTo(node.x + nodeWidth/2, node.y - nodeHeight/2, node.x + nodeWidth/2, node.y - nodeHeight/2 + radius)
      ctx.lineTo(node.x + nodeWidth/2, node.y + nodeHeight/2 - radius)
      ctx.quadraticCurveTo(node.x + nodeWidth/2, node.y + nodeHeight/2, node.x + nodeWidth/2 - radius, node.y + nodeHeight/2)
      ctx.lineTo(node.x - nodeWidth/2 + radius, node.y + nodeHeight/2)
      ctx.quadraticCurveTo(node.x - nodeWidth/2, node.y + nodeHeight/2, node.x - nodeWidth/2, node.y + nodeHeight/2 - radius)
      ctx.lineTo(node.x - nodeWidth/2, node.y - nodeHeight/2 + radius)
      ctx.quadraticCurveTo(node.x - nodeWidth/2, node.y - nodeHeight/2, node.x - nodeWidth/2 + radius, node.y - nodeHeight/2)
      ctx.closePath()
      ctx.fill()
      if (isSelected || isHovered) ctx.stroke()

      // Node text
      ctx.fillStyle = '#fff'
      ctx.font = isSelected ? 'bold 12px Inter' : '11px Inter'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'

      const lines = node.label.split('\n')
      lines.forEach((line, i) => {
        ctx.fillText(line, node.x, node.y - (lines.length - 1) * 7 + i * 14)
      })
    })

    ctx.restore()
  }, [nodes, edges, zoom, pan, selectedNode, hoveredNode])

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left - pan.x) / zoom
    const y = (e.clientY - rect.top - pan.y) / zoom

    const clickedNode = nodes.find(node => {
      const dx = x - node.x
      const dy = y - node.y
      return Math.abs(dx) < 70 && Math.abs(dy) < 30
    })

    setSelectedNode(clickedNode || null)
  }

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left - pan.x) / zoom
    const y = (e.clientY - rect.top - pan.y) / zoom

    if (isDragging) {
      setPan({
        x: pan.x + (e.clientX - dragStart.x),
        y: pan.y + (e.clientY - dragStart.y)
      })
      setDragStart({ x: e.clientX, y: e.clientY })
    } else {
      const hoveredNode = nodes.find(node => {
        const dx = x - node.x
        const dy = y - node.y
        return Math.abs(dx) < 70 && Math.abs(dy) < 30
      })
      setHoveredNode(hoveredNode || null)
    }
  }

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true)
    setDragStart({ x: e.clientX, y: e.clientY })
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const handleZoomIn = () => setZoom(Math.min(zoom * 1.2, 3))
  const handleZoomOut = () => setZoom(Math.max(zoom / 1.2, 0.5))
  const handleReset = () => {
    setZoom(1)
    setPan({ x: 0, y: 0 })
    setSelectedNode(null)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          🔬 반도체 온톨로지 그래프
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          Equipment, Product, Recipe 계층 구조를 인터랙티브하게 탐색하세요
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-md border border-gray-200 dark:border-gray-700">
        <div className="flex flex-wrap gap-4 items-center justify-between">
          {/* Hierarchy Selector */}
          <div className="flex gap-2">
            <button
              onClick={() => setActiveHierarchy('equipment')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                activeHierarchy === 'equipment'
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              🏭 Equipment
            </button>
            <button
              onClick={() => setActiveHierarchy('product')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                activeHierarchy === 'product'
                  ? 'bg-green-600 text-white shadow-lg'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              💎 Product
            </button>
            <button
              onClick={() => setActiveHierarchy('recipe')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                activeHierarchy === 'recipe'
                  ? 'bg-amber-600 text-white shadow-lg'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              📋 Recipe
            </button>
          </div>

          {/* Zoom Controls */}
          <div className="flex gap-2">
            <button
              onClick={handleZoomOut}
              className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              title="Zoom Out"
            >
              <ZoomOut className="w-5 h-5" />
            </button>
            <button
              onClick={handleReset}
              className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              title="Reset View"
            >
              <RefreshCw className="w-5 h-5" />
            </button>
            <button
              onClick={handleZoomIn}
              className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              title="Zoom In"
            >
              <ZoomIn className="w-5 h-5" />
            </button>
            <div className="px-3 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg font-mono text-sm">
              {Math.round(zoom * 100)}%
            </div>
          </div>
        </div>
      </div>

      {/* Canvas */}
      <div className="bg-gray-50 dark:bg-gray-900 rounded-xl border-2 border-gray-200 dark:border-gray-700 overflow-hidden">
        <canvas
          ref={canvasRef}
          width={800}
          height={500}
          className="w-full h-[500px] cursor-move"
          onClick={handleCanvasClick}
          onMouseMove={handleCanvasMouseMove}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        />
      </div>

      {/* Node Details Panel */}
      {selectedNode && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-1">
                {selectedNode.label.replace('\n', ' ')}
              </h3>
              <span className="inline-block px-3 py-1 rounded-full text-xs font-medium" style={{ backgroundColor: selectedNode.color, color: '#fff' }}>
                {selectedNode.type.toUpperCase()}
              </span>
            </div>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
            >
              ✕
            </button>
          </div>

          {selectedNode.metadata && (
            <div className="grid md:grid-cols-2 gap-3">
              {Object.entries(selectedNode.metadata).map(([key, value]) => (
                <div key={key} className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
                    {key}
                  </p>
                  <p className="text-sm font-semibold text-gray-900 dark:text-white">
                    {value}
                  </p>
                </div>
              ))}
            </div>
          )}

          {selectedNode.children && selectedNode.children.length > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                <strong>하위 노드:</strong> {selectedNode.children.length}개
              </p>
              <div className="flex flex-wrap gap-2">
                {selectedNode.children.map(childId => {
                  const childNode = nodes.find(n => n.id === childId)
                  return childNode ? (
                    <button
                      key={childId}
                      onClick={() => setSelectedNode(childNode)}
                      className="px-3 py-1 rounded-lg text-sm font-medium transition-all hover:scale-105"
                      style={{ backgroundColor: childNode.color, color: '#fff' }}
                    >
                      {childNode.label.split('\n')[0]}
                    </button>
                  ) : null
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Info Box */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-4">
        <div className="flex gap-3">
          <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-blue-900 dark:text-blue-300 space-y-2">
            <p><strong>💡 사용법:</strong></p>
            <ul className="list-disc list-inside space-y-1 ml-2">
              <li>노드를 클릭하면 상세 정보를 확인할 수 있습니다</li>
              <li>캔버스를 드래그하여 이동할 수 있습니다</li>
              <li>줌 버튼으로 확대/축소가 가능합니다</li>
              <li>상단 버튼으로 Equipment/Product/Recipe 계층을 전환하세요</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Statistics */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl p-6 text-white">
          <div className="text-3xl font-bold mb-1">{equipmentNodes.length}</div>
          <div className="text-blue-100">Equipment 노드</div>
        </div>
        <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-xl p-6 text-white">
          <div className="text-3xl font-bold mb-1">{productNodes.length}</div>
          <div className="text-green-100">Product 노드</div>
        </div>
        <div className="bg-gradient-to-br from-amber-500 to-amber-600 rounded-xl p-6 text-white">
          <div className="text-3xl font-bold mb-1">{recipeNodes.length}</div>
          <div className="text-amber-100">Recipe 노드</div>
        </div>
      </div>
    </div>
  )
}
